/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * The Metal half of cross-process sharing (ObjC++/SysV, -fno-objc-arc,
 * manual retain/release).
 *
 *   - Anonymous shared-memory allocation (shm_open + immediate unlink).
 *   - Swizzle install over the Metal device + heap classes, done at dmn_init
 *     BEFORE D3DMetal is dlopen'd so no heap class is missed during its init.
 *   - The substitution: when a hook has armed the current thread, the next
 *     texture creation (device-dedicated OR heap-placed) is replaced by a
 *     linear StorageModeShared texture over our shared memory.
 *
 * No DirectX headers here — the COM side talks to us only through dmn_share.h.
 */

#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "d3dmetal_native.h"
#include "dmn_formats.h"
#include "dmn_private.h"
#include "dmn_share.h"

/* Layout parity with the public PODs — the HANDLE the caller ships is exactly
 * these bytes. */
static_assert(sizeof(DmnShareTexPOD) == sizeof(dmn_shared_texture_handle),
              "texture POD layout drift");
static_assert(sizeof(DmnShareFencePOD) == sizeof(dmn_shared_fence_handle),
              "fence POD layout drift");

namespace {

/* == Anonymous shared file: shm_open a random name, unlink immediately ==== */
int dmn_anon_file(off_t size) {
    const unsigned nonce = arc4random();
    int fd = -1;
    for (unsigned i = 0; i < 32; i++) {
        char name[64];
        snprintf(name, sizeof(name), "/dmn-shm-%d-%x-%x", getpid(), nonce, i);
        fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
        if (fd >= 0) {
            shm_unlink(name); /* anonymous from here on */
            break;
        }
        if (errno != EEXIST)
            break;
    }
    if (fd < 0)
        return -1;
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

size_t page_align(size_t n) {
    size_t pg = (size_t)getpagesize();
    return (n + pg - 1) & ~(pg - 1);
}

/* == Geometry validation ==================================================
 * Width, height, stride and size all originate outside this process: the
 * producer's come from a D3D descriptor the guest supplied, the consumer's
 * come straight off the wire in a POD. Everything below is therefore checked
 * before it reaches size arithmetic — an overflowing stride*height silently
 * wraps into a small allocation that Metal then reads past, and an unbounded
 * `size` is an arbitrary-length mmap.
 *
 * The dimension cap is deliberately well above Metal's own 16384 texture
 * limit: anything beyond it is corruption rather than a surface we failed to
 * anticipate, and refusing early is what keeps the arithmetic in range. */
constexpr size_t kMaxDimension   = 32768;
constexpr size_t kMaxSharedBytes = (size_t)1 << 32; /* 4 GiB */

bool mul_ok(size_t a, size_t b, size_t* out) {
    return !__builtin_mul_overflow(a, b, out);
}

bool add_ok(size_t a, size_t b, size_t* out) {
    return !__builtin_add_overflow(a, b, out);
}

bool dims_ok(size_t width, size_t height, const char* who) {
    if (!width || !height || width > kMaxDimension || height > kMaxDimension) {
        DMN_ERROR("share: %s refusing %zux%zu shared texture (1..%zu)", who,
                  width, height, kMaxDimension);
        return false;
    }
    return true;
}

/* Byte layout of a linear shared surface. The producer derives it from the
 * descriptor; the consumer takes the producer's numbers verbatim and only
 * validates them. `logical` is what the POD reports (and what a peer maps as
 * the surface); `mapped` is the page-aligned span actually handed to mmap. */
struct LinearLayout {
    size_t stride;
    size_t logical;
    size_t mapped;
};

/* Shared backing is reclaimed when the MTLBuffer dies: the deallocator block
 * munmaps the mapping and closes the fd we own (producer side; a consumer's fd
 * belongs to the app, so only the mapping goes). Metal copies the block, so
 * cleanup runs whenever D3DMetal drops its last reference — after any in-flight
 * command buffers, since those retain the buffer. */
id<MTLBuffer> shared_buffer_over(id<MTLDevice> device, void* ptr, size_t aligned,
                                 int owned_fd) {
    return [device newBufferWithBytesNoCopy:ptr
                                     length:aligned
                                    options:MTLResourceStorageModeShared
                                deallocator:^(void* p, NSUInteger len) {
                                    munmap(p, len);
                                    if (owned_fd >= 0)
                                        close(owned_fd);
                                }];
}

/* A linear texture does not necessarily retain its source MTLBuffer across all
 * Metal versions; pin it explicitly so buffer (and backing) lifetime is exactly
 * the texture's. */
const void* kDmnBackingKey = &kDmnBackingKey;

void pin_buffer_to_texture(id<MTLTexture> tex, id<MTLBuffer> buf) {
    objc_setAssociatedObject(tex, kDmnBackingKey, buf,
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

/* == Thread-local arm ===================================================== */
thread_local DmnShareArm t_arm = {};

/* Register a substituted impostor so it is made GPU-resident on every encoder,
 * declaring exactly `usage` (defined with the residency plumbing below). */
void sub_resource_track(id res, MTLResourceUsage usage);

/* The swizzle registry's reader, defined with the rest of the plumbing below. */
IMP lookup_orig(Class c, SEL sel);

/* == Initial-data sentinel ================================================
 * See dmn_share_init_data_sentinel() in dmn_share.h for why a consumer
 * reconstruct passes initial data at all. This is the pointer it passes, and
 * the interception that keeps the data from ever being written.
 *
 * The pointer is a large, lazily-backed read-only zero mapping rather than a
 * made-up address. It has to be recognisable, and it has to be safe to read:
 * if the copy is ever not intercepted, the worst case is an upload of zeros
 * that the import then refuses, instead of a fault inside D3DMetal on a
 * pointer we handed it. MAP_NORESERVE means the reservation costs address
 * space and nothing else.
 *
 * Recognition is a range test, not an equality test: D3DMetal advances the
 * pointer per slice and per mip level before handing it to Metal. */
constexpr size_t kInitSentinelBytes = (size_t)256 << 20; /* 256 MiB */

void* init_sentinel_base() {
    static void* base = [] {
        void* m = mmap(nullptr, kInitSentinelBytes, PROT_READ,
                       MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) {
            DMN_WARN("share: could not reserve the initial-data sentinel: %s",
                     strerror(errno));
            return (void*)nullptr;
        }
        return m;
    }();
    return base;
}

bool is_init_sentinel(const void* bytes) {
    const uint8_t* base = (const uint8_t*)init_sentinel_base();
    const uint8_t* p = (const uint8_t*)bytes;
    return base && p >= base && p < base + kInitSentinelBytes;
}

/* -[MTLTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:]
 * and its 4-argument convenience. Both drop a write whose source is the
 * sentinel and pass everything else through untouched. */
void swz_tex_replace6(id self, SEL _cmd, MTLRegion region, NSUInteger level,
                      NSUInteger slice, const void* bytes, NSUInteger bytesPerRow,
                      NSUInteger bytesPerImage) {
    if (is_init_sentinel(bytes)) {
        t_arm.init_dropped = true;
        return;
    }
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (orig)
        ((void (*)(id, SEL, MTLRegion, NSUInteger, NSUInteger, const void*,
                   NSUInteger, NSUInteger))orig)(self, _cmd, region, level, slice,
                                                 bytes, bytesPerRow, bytesPerImage);
}

void swz_tex_replace4(id self, SEL _cmd, MTLRegion region, NSUInteger level,
                      const void* bytes, NSUInteger bytesPerRow) {
    if (is_init_sentinel(bytes)) {
        t_arm.init_dropped = true;
        return;
    }
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (orig)
        ((void (*)(id, SEL, MTLRegion, NSUInteger, const void*, NSUInteger))orig)(
            self, _cmd, region, level, bytes, bytesPerRow);
}

/* Installed on the impostor's own class the first time one is built, which is
 * the only point at which the concrete buffer-backed texture class is known.
 * That class is shared with every other Metal texture in the process, so this
 * sits on all texture uploads D3DMetal performs, not just ours — hence the
 * sentinel test being a pointer range compare ahead of any other work, and the
 * pass-through being the only other thing either hook does. */
void ensure_texture_class_swizzled(id tex);

/* == The substitution =====================================================
 * Producer and consumer are separate functions on purpose. They agree on the
 * final object but on almost nothing else: who owns the fd, whether the mapped
 * span includes a trailer, and whether the layout is being *derived* or merely
 * *validated*. Folding them into one body left those rules implicit in
 * `alloc_new` tests scattered through a single flow. What they genuinely share
 * — turning a mapped span into a linear MTLTexture — is make_linear_texture()
 * below, which takes ownership decisions as parameters instead of inferring
 * them. */

/* Residency usage for an impostor, derived from the descriptor it was created
 * with — never from which side of the share we are on. An imported surface can
 * legitimately be a render target, so keying this on producer-vs-consumer
 * would strip access that the resource really has. MTLTextureUsageUnknown (0)
 * means "any usage", and anything we cannot classify falls back to the full
 * set: over-declaring costs hazard-tracking precision, under-declaring is a
 * correctness bug. */
MTLResourceUsage residency_usage_for(MTLTextureDescriptor* desc) {
    const MTLResourceUsage both = MTLResourceUsageRead | MTLResourceUsageWrite;
    if (!desc || desc.usage == MTLTextureUsageUnknown)
        return both;

    const MTLTextureUsage u = desc.usage;
    MTLResourceUsage r = 0;
    if (u & MTLTextureUsageShaderRead)
        r |= MTLResourceUsageRead;
    if (u & MTLTextureUsageShaderWrite)
        r |= MTLResourceUsageWrite;
    /* A render target is read as well as written: load actions and blending
     * both sample the existing contents. */
    if (u & MTLTextureUsageRenderTarget)
        r |= both;
    /* MTLTextureUsageShaderAtomic, spelled numerically so this TU still builds
     * against SDKs predating macOS 14 — the same reason dmn_formats.mm spells
     * the GPU families numerically. The value is ABI-stable. */
    if (u & 0x0020)
        r |= both;
    /* PixelFormatView alone says nothing about access. */
    return r ? r : both;
}

/* Wrap an already-mapped span in a linear MTLTexture matching `desc`.
 *
 * `buf` is consumed: on success the returned texture owns the caller's
 * reference, on failure it is released. Returns +1 or nil. */
id<MTLTexture> make_linear_texture(id<MTLBuffer> buf, MTLTextureDescriptor* desc,
                                   const LinearLayout& layout, bool producer) {
    const MTLPixelFormat fmt = desc.pixelFormat;
    const NSUInteger width   = desc.width;
    const NSUInteger height  = desc.height;

    MTLTextureDescriptor* linear =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    /* Honor whatever usage D3DMetal asked for (render target, shader read,
     * etc.) so the substituted texture is accepted wherever the original would
     * have been. PixelFormatView is added unconditionally: the 2DArray view
     * below is only legal on a texture that declared it, it costs nothing on a
     * linear texture, and leaving it to depend on what D3DMetal happened to ask
     * for made view creation fail in a way that only Metal validation caught. */
    linear.usage = (desc.usage ? desc.usage
                               : (MTLTextureUsageRenderTarget |
                                  MTLTextureUsageShaderRead)) |
                   MTLTextureUsagePixelFormatView;
    /* Storage and cache mode must match the backing buffer, which is
     * StorageModeShared / DefaultCache. Set both explicitly rather than
     * inheriting the descriptor default and matching by luck. */
    linear.storageMode  = MTLStorageModeShared;
    linear.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    if (desc.cpuCacheMode != MTLCPUCacheModeDefaultCache)
        DMN_WARN("share: descriptor asked for cpuCacheMode %lu; shared backing "
                 "is DefaultCache", (unsigned long)desc.cpuCacheMode);
    /* Channel swizzle is part of how the surface reads and has no other
     * carrier — dropping it silently permutes B8G8R8X8 / A8 style formats. */
    linear.swizzle = desc.swizzle;
    /* Linear storage is never GPU-compressed anyway; saying so makes it a
     * property of this texture rather than an inference about buffer-backing. */
    linear.allowGPUOptimizedContents = NO;
    /* Default is Tracked for a device resource. If D3DMetal asked for
     * Untracked it is doing its own MTLFence-based tracking, and a tracked
     * impostor will behave differently than it expects. */
    if (desc.hazardTrackingMode == MTLHazardTrackingModeUntracked)
        DMN_WARN("share: descriptor asked for untracked hazards; the impostor "
                 "is tracked");

    id<MTLTexture> tex = [buf newTextureWithDescriptor:linear
                                                offset:0
                                           bytesPerRow:layout.stride];
    if (!tex) {
        DMN_ERROR("share: newTextureWithDescriptor:offset:bytesPerRow: failed "
                  "(fmt=%lu %lux%lu stride=%zu bufLen=%lu) — under-aligned "
                  "stride?",
                  (unsigned long)fmt, (unsigned long)width,
                  (unsigned long)height, layout.stride,
                  (unsigned long)[buf length]);
        [buf release];
        return nil;
    }

    pin_buffer_to_texture(tex, buf);
    [buf release]; /* the texture holds the caller's reference now */
    tex.label = [NSString stringWithFormat:@"dmn-shared-%s-%lux%lu",
                          producer ? "prod" : "cons",
                          (unsigned long)width, (unsigned long)height];

    /* D3DMetal reconstructs an opened MISC_SHARED surface as MTLTextureType2DArray
     * and emits texture2d_array sample code for its SRV, but a buffer-backed
     * linear texture is forced to plain MTLTextureType2D (Metal asserts on a
     * 2DArray linear texture). Sampling the 2D impostor through the array-typed
     * binding reads a garbage array dimension and returns zero for the colour
     * channels — a solid box over every composited XAML glyph, while a
     * CopyResource of the same surface reads correctly. A 2DArray view over the
     * same buffer-backed storage samples correctly, so hand back an array view
     * whenever an array texture was requested. */
    if (desc.textureType == MTLTextureType2DArray && desc.arrayLength <= 1) {
        id<MTLTexture> arrview =
            [tex newTextureViewWithPixelFormat:fmt
                                   textureType:MTLTextureType2DArray
                                        levels:NSMakeRange(0, 1)
                                        slices:NSMakeRange(0, 1)];
        if (arrview) {
            /* the view retains tex, which holds buf; keep the chain explicit
             * (a view need not retain its base across all Metal versions) and
             * hand the caller the view +1 in tex's place. */
            objc_setAssociatedObject(arrview, kDmnBackingKey, tex,
                                     OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            arrview.label = tex.label;
            [tex release];
            tex = arrview;
        } else {
            DMN_WARN("share: 2DArray view of impostor failed; sampling may box");
        }
    }

    /* Keep resident when bound bindlessly, with the access the descriptor
     * actually declared. */
    sub_resource_track(tex, residency_usage_for(desc));
    ensure_texture_class_swizzled(tex);
    return tex; /* +1, caller owns */
}

/* Producer: allocate a fresh shm object sized for `desc` plus any trailer, and
 * substitute a linear texture over it. Returns +1, or nil to let the original
 * creation proceed. */
id<MTLTexture> substitute_producer(id<MTLDevice> device,
                                   MTLTextureDescriptor* desc) {
    const MTLPixelFormat fmt = desc.pixelFormat;
    const size_t width  = (size_t)desc.width;
    const size_t height = (size_t)desc.height;

    const uint32_t bpp = dmn_format_linear_bpp((uint32_t)fmt);
    if (!bpp) {
        DMN_ERROR("share: MTLPixelFormat %lu has no linear layout (compressed, "
                  "depth/stencil or unknown) — refusing to share it rather "
                  "than guessing a stride", (unsigned long)fmt);
        return nil;
    }
    if (!dims_ok(width, height, "producer"))
        return nil;

    NSUInteger row_align =
        [device minimumLinearTextureAlignmentForPixelFormat:fmt];
    if (row_align == 0)
        row_align = 256;

    LinearLayout layout{};
    size_t row = 0, pages = 0;
    /* dims_ok bounds width*bpp well inside size_t, so only the products that
     * involve height and the trailer can actually overflow — check them all
     * the same way rather than reasoning about which. */
    if (!mul_ok(width, (size_t)bpp, &row)) {
        DMN_ERROR("share: row bytes overflow (%zu x %u)", width, bpp);
        return nil;
    }
    layout.stride = (row + (size_t)row_align - 1) & ~((size_t)row_align - 1);
    if (!mul_ok(layout.stride, height, &layout.logical) ||
        layout.logical > kMaxSharedBytes) {
        DMN_ERROR("share: surface bytes out of range (stride %zu x height %zu)",
                  layout.stride, height);
        return nil;
    }
    /* Trailer (e.g. the keyed-mutex page) lives past the page-aligned texture
     * bytes; consumers find it at page_align(pod.size). */
    if ((size_t)t_arm.extra_bytes > kMaxSharedBytes ||
        !add_ok(page_align(layout.logical),
                page_align((size_t)t_arm.extra_bytes), &pages) ||
        pages > kMaxSharedBytes) {
        DMN_ERROR("share: mapped bytes out of range (payload %zu + trailer %llu)",
                  layout.logical, (unsigned long long)t_arm.extra_bytes);
        return nil;
    }
    layout.mapped = pages;

    const int fd = dmn_anon_file((off_t)layout.mapped);
    if (fd < 0) {
        DMN_ERROR("share: anon file (%zu) failed", layout.mapped);
        return nil;
    }
    void* ptr = mmap(nullptr, layout.mapped, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, 0);
    if (ptr == MAP_FAILED) {
        DMN_ERROR("share: mmap(%zu) failed: %s", layout.mapped, strerror(errno));
        close(fd);
        return nil;
    }

    /* The buffer takes ownership of both the mapping and the fd from here, so
     * everything past this point unwinds by releasing the buffer alone —
     * including make_linear_texture's own failure path. */
    id<MTLBuffer> buf = shared_buffer_over(device, ptr, layout.mapped, fd);
    if (!buf) {
        DMN_ERROR("share: newBufferWithBytesNoCopy failed");
        munmap(ptr, layout.mapped);
        close(fd);
        return nil;
    }

    id<MTLTexture> tex = make_linear_texture(buf, desc, layout, /*producer=*/true);
    if (!tex)
        return nil;

    t_arm.captured   = true;
    t_arm.out_fd     = fd;
    t_arm.out_stride = layout.stride;
    t_arm.out_size   = layout.logical;
    DMN_INFO("share: substituted producer texture fmt=%lu %zux%zu stride=%zu "
             "size=%zu fd=%d", (unsigned long)fmt, width, height, layout.stride,
             layout.logical, fd);
    return tex;
}

/* Consumer: reproduce the producer's byte-exact layout over the fd it shipped.
 * The layout is validated, never recomputed — the two processes have to agree
 * on it, so a disagreement is a hard error rather than something to paper
 * over. Returns +1, or nil. */
id<MTLTexture> substitute_consumer(id<MTLDevice> device,
                                   MTLTextureDescriptor* desc) {
    const MTLPixelFormat fmt = desc.pixelFormat;
    const size_t width  = (size_t)desc.width;
    const size_t height = (size_t)desc.height;

    const uint32_t bpp = dmn_format_linear_bpp((uint32_t)fmt);
    if (!bpp) {
        DMN_ERROR("share: MTLPixelFormat %lu has no linear layout — refusing "
                  "the import", (unsigned long)fmt);
        return nil;
    }
    if (!dims_ok(width, height, "consumer"))
        return nil;

    LinearLayout layout{};
    layout.stride  = (size_t)t_arm.existing_stride;
    layout.logical = (size_t)t_arm.existing_size;

    size_t need = 0, row = 0;
    if (!layout.stride || !layout.logical ||
        layout.logical > kMaxSharedBytes) {
        DMN_ERROR("share: import geometry out of range (stride=%zu size=%zu)",
                  layout.stride, layout.logical);
        return nil;
    }
    if (!mul_ok(width, (size_t)bpp, &row) || layout.stride < row) {
        DMN_ERROR("share: import stride %zu is shorter than %zu bytes of pixels",
                  layout.stride, row);
        return nil;
    }
    if (!mul_ok(layout.stride, height, &need) || need > layout.logical) {
        DMN_ERROR("share: import needs %zu bytes but the shared region is %zu",
                  need, layout.logical);
        return nil;
    }
    layout.mapped = page_align(layout.logical);

    /* The fd belongs to the caller on this path — never closed here, and never
     * handed to the buffer's deallocator. */
    const int fd = t_arm.existing_fd;
    void* ptr = mmap(nullptr, layout.mapped, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, 0);
    if (ptr == MAP_FAILED) {
        DMN_ERROR("share: consumer mmap(fd=%d, %zu) failed: %s", fd,
                  layout.mapped, strerror(errno));
        return nil;
    }
    id<MTLBuffer> buf = shared_buffer_over(device, ptr, layout.mapped,
                                           /*owned_fd=*/-1);
    if (!buf) {
        DMN_ERROR("share: consumer newBufferWithBytesNoCopy failed");
        munmap(ptr, layout.mapped);
        return nil;
    }

    id<MTLTexture> tex = make_linear_texture(buf, desc, layout, /*producer=*/false);
    if (!tex)
        return nil;

    t_arm.captured   = true;
    t_arm.out_fd     = fd;
    t_arm.out_stride = layout.stride;
    t_arm.out_size   = layout.logical;
    DMN_INFO("share: substituted consumer texture fmt=%lu %zux%zu stride=%zu "
             "size=%zu fd=%d", (unsigned long)fmt, width, height, layout.stride,
             layout.logical, fd);
    return tex;
}

id<MTLTexture> substitute(id<MTLDevice> device, MTLTextureDescriptor* desc) {
    if (!device || !desc)
        return nil;
    return t_arm.alloc_new ? substitute_producer(device, desc)
                           : substitute_consumer(device, desc);
}

/* Build a shared-memory-backed MTLBuffer (forced to StorageModeShared so the
 * GPU can write it and a peer can read it via mmap). Used for GPU-written
 * fence-value pages. Returns a +1 buffer, or nil.
 *
 * An MTLBuffer carries no usage declaration the way a texture descriptor does,
 * so residency stays Read|Write here — there is nothing to narrow it with. */
id<MTLBuffer> substitute_buffer(id<MTLDevice> device, NSUInteger length) {
    if (!device)
        return nil;

    const MTLResourceUsage both = MTLResourceUsageRead | MTLResourceUsageWrite;

    if (t_arm.alloc_new) {
        /* Two callers have a say in the size: D3DMetal, via the length it asked
         * Metal for, and the COM layer, via the D3D buffer width it armed with.
         * Back the larger — a buffer short of either one is read past. */
        size_t logical = (size_t)length;
        if ((size_t)t_arm.request_bytes > logical)
            logical = (size_t)t_arm.request_bytes;
        if (!logical)
            logical = sizeof(uint64_t);
        if (logical > kMaxSharedBytes) {
            DMN_ERROR("share: refusing %zu-byte shared buffer (limit %zu)",
                      logical, kMaxSharedBytes);
            return nil;
        }
        const size_t mapped = page_align(logical);
        const int fd = dmn_anon_file((off_t)mapped);
        if (fd < 0) {
            DMN_ERROR("share: buffer anon file (%zu) failed", mapped);
            return nil;
        }
        void* ptr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_SHARED,
                         fd, 0);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: buffer mmap(%zu) failed: %s", mapped,
                      strerror(errno));
            close(fd);
            return nil;
        }
        id<MTLBuffer> buf = shared_buffer_over(device, ptr, mapped, fd);
        if (!buf) {
            DMN_ERROR("share: buffer newBufferWithBytesNoCopy failed");
            munmap(ptr, mapped);
            close(fd);
            return nil;
        }
        buf.label = @"dmn-shared-prod-buffer";
        sub_resource_track(buf, both);

        t_arm.captured = true;
        t_arm.out_fd   = fd;
        t_arm.out_size = logical;
        DMN_INFO("share: substituted producer buffer size=%zu fd=%d", logical, fd);
        return buf;
    }

    /* Consumer: the shared region has to cover what the caller asked for. A
     * short buffer is not something to hand back and hope about — D3DMetal
     * would read and write past the mapping. */
    const size_t logical = (size_t)t_arm.existing_size;
    if (!logical || logical > kMaxSharedBytes) {
        DMN_ERROR("share: import buffer size %zu out of range", logical);
        return nil;
    }
    if (length && (size_t)length > logical) {
        DMN_ERROR("share: import buffer is %zu bytes but %lu were requested",
                  logical, (unsigned long)length);
        return nil;
    }
    const size_t mapped = page_align(logical);
    const int fd = t_arm.existing_fd;

    /* Deliberately NOT served from the shared-mapping cache. For a texture the
     * cache hands back a shared MTLBuffer and each import still gets its own
     * MTLTexture on top; here the MTLBuffer *is* the resource, so a cache hit
     * would give two D3D buffers the same Metal object and alias whatever
     * per-resource state D3DMetal keeps on it. Buffer imports are rare (fence
     * pages) and do not churn per frame, so a private mapping each time is the
     * right trade. */
    void* ptr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        DMN_ERROR("share: consumer buffer mmap(fd=%d, %zu) failed: %s", fd,
                  mapped, strerror(errno));
        return nil;
    }
    id<MTLBuffer> buf = shared_buffer_over(device, ptr, mapped, /*owned_fd=*/-1);
    if (!buf) {
        DMN_ERROR("share: consumer buffer newBufferWithBytesNoCopy failed");
        munmap(ptr, mapped);
        return nil;
    }
    buf.label = @"dmn-shared-cons-buffer";
    sub_resource_track(buf, both);

    t_arm.captured = true;
    t_arm.out_fd   = fd;
    t_arm.out_size = logical;
    DMN_INFO("share: substituted consumer buffer size=%zu fd=%d", logical, fd);
    return buf; /* +1, caller owns */
}

/* == Swizzle registry =====================================================
 * One table for every swizzled selector, keyed by (implementing class,
 * selector) and published as an immutable snapshot. Readers — every hooked
 * entry point, some of them per-command-buffer — take an acquire load and no
 * lock; installers copy-mutate-publish under g_install_mutex. The tables are
 * written a handful of times at startup and read for the process lifetime, so
 * a mutex on the read path was pure contention across D3DMetal's free-threaded
 * encoding.
 *
 * Superseded snapshots are deliberately never freed: a reader may still be
 * walking one, there is no safe reclamation point without RCU, and the count
 * is bounded by the number of distinct Metal classes ever swizzled.
 *
 * The key is the class that actually IMPLEMENTS the method, not the class we
 * were handed. class_getInstanceMethod walks the superclass chain and
 * method_setImplementation then mutates the Method wherever it is defined,
 * which affects every sibling subclass at once. Recording under the class we
 * were handed leaves a sibling's lookup finding nothing (and returning nil to
 * D3DMetal), and lets a later install for that sibling record our own
 * replacement as the "original" and recurse forever. */
struct OrigKey {
    Class cls;
    SEL   sel;
    bool operator==(const OrigKey& o) const {
        return cls == o.cls && sel == o.sel;
    }
};
struct OrigKeyHash {
    size_t operator()(const OrigKey& k) const {
        return std::hash<const void*>()((const void*)k.cls) * 31u +
               std::hash<const void*>()((const void*)k.sel);
    }
};
using OrigMap = std::unordered_map<OrigKey, IMP, OrigKeyHash>;

std::mutex                  g_install_mutex;
std::atomic<const OrigMap*> g_origs{nullptr};
/* Every (class, selector) we have already tried, successful or not. Guarding
 * on this rather than on one of the payload maps is what makes a class that
 * lacks a selector stop being retried — retrying re-ran
 * method_setImplementation for its *other* selectors and recorded our own
 * replacement as the original. Guarded by g_install_mutex. */
std::set<std::pair<Class, SEL>> g_attempted;
bool g_installed = false;

IMP lookup_orig(Class c, SEL sel) {
    const OrigMap* m = g_origs.load(std::memory_order_acquire);
    if (!m)
        return nullptr;
    /* Walk up the class chain: the concrete instance class may be a private
     * subclass of the class that implements (and that we recorded) the
     * method. */
    for (Class k = c; k; k = class_getSuperclass(k)) {
        auto it = m->find(OrigKey{k, sel});
        if (it != m->end())
            return it->second;
    }
    return nullptr;
}

/* The class whose method list owns `m`, i.e. the one that really implements
 * `sel` for `cls`. Only runs at install time. */
Class implementing_class(Class cls, Method m) {
    for (Class k = cls; k; k = class_getSuperclass(k)) {
        unsigned n = 0;
        Method* list = class_copyMethodList(k, &n);
        bool here = false;
        for (unsigned i = 0; i < n && !here; i++)
            here = (list[i] == m);
        free(list);
        if (here)
            return k;
    }
    return cls; /* not reachable for a Method the runtime just handed us */
}

/* Install `repl` over `sel` on `cls`, recording the original under the class
 * that implements it. Idempotent both per (cls, sel) and per (implementing
 * class, sel). Returns true if it installed something new. Caller holds
 * g_install_mutex. */
bool install_swizzle(Class cls, SEL sel, IMP repl, const char* what) {
    if (!g_attempted.insert({cls, sel}).second)
        return false;
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) {
        DMN_WARN("share: %s class %s does not implement %s", what,
                 class_getName(cls), sel_getName(sel));
        return false;
    }
    Class owner = implementing_class(cls, m);
    const OrigMap* cur = g_origs.load(std::memory_order_relaxed);
    if (cur && cur->count(OrigKey{owner, sel}))
        return false; /* already swizzled, reached via a sibling subclass */

    /* Publish the original BEFORE swapping the implementation in. The moment
     * method_setImplementation lands, another thread can be inside `repl`
     * looking the original up — and readers no longer share our lock, so
     * recording afterwards leaves a window where that lookup returns nullptr
     * and the hook has no choice but to return nil to D3DMetal. */
    OrigMap* next = cur ? new OrigMap(*cur) : new OrigMap();
    (*next)[OrigKey{owner, sel}] = method_getImplementation(m);
    g_origs.store(next, std::memory_order_release);
    method_setImplementation(m, repl);
    return true;
}

struct SwizzleJob {
    SEL sel;
    IMP repl;
};

/* Install a group of swizzles on `cls`.
 *
 * `memo` is a single-slot lock-free cache of the last class fully processed.
 * In practice there is exactly one concrete class per group, so this keeps
 * g_install_mutex off the per-command-buffer and per-heap paths entirely; a
 * miss just takes the lock and finds everything already attempted. */
void install_swizzles(Class cls, const SwizzleJob* jobs, size_t njobs,
                      const char* what, std::atomic<Class>* memo) {
    if (!cls)
        return;
    if (memo && memo->load(std::memory_order_acquire) == cls)
        return;
    unsigned installed = 0;
    {
        std::lock_guard<std::mutex> lk(g_install_mutex);
        for (size_t i = 0; i < njobs; i++)
            installed += install_swizzle(cls, jobs[i].sel, jobs[i].repl, what);
    }
    if (memo)
        memo->store(cls, std::memory_order_release);
    if (installed)
        DMN_INFO("share: swizzled %u %s selector(s) on %s", installed, what,
                 class_getName(cls));
}

/* Device dedicated path: -[dev newTextureWithDescriptor:] */
id swz_dev_newtex(id self, SEL _cmd, MTLTextureDescriptor* desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (t_arm.armed && t_arm.kind == DMN_SHARE_TEXTURE) {
        id<MTLTexture> sub = substitute((id<MTLDevice>)self, desc);
        if (sub) {
            t_arm.armed = false; /* disarm on first hit */
            return sub;
        }
        DMN_WARN("share: armed dev newTextureWithDescriptor: not substituted; "
                 "passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for dev newTextureWithDescriptor:");
        return nil;
    }
    return ((id (*)(id, SEL, MTLTextureDescriptor*))orig)(self, _cmd, desc);
}

/* Heap placed path: -[heap newTextureWithDescriptor:offset:] (ignore offset:
 * — our texture is backed by its own buffer at offset 0). */
id swz_heap_newtex(id self, SEL _cmd, MTLTextureDescriptor* desc,
                   NSUInteger offset) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (t_arm.armed && t_arm.kind == DMN_SHARE_TEXTURE) {
        id<MTLHeap> heap = (id<MTLHeap>)self;
        id<MTLTexture> sub = substitute([heap device], desc);
        if (sub) {
            t_arm.armed = false;
            return sub;
        }
        DMN_WARN("share: armed heap newTextureWithDescriptor:offset: not "
                 "substituted; passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for heap newTextureWithDescriptor:offset:");
        return nil;
    }
    return ((id (*)(id, SEL, MTLTextureDescriptor*, NSUInteger))orig)(
        self, _cmd, desc, offset);
}

/* Device buffer path: -[dev newBufferWithLength:options:] */
id swz_dev_newbuf(id self, SEL _cmd, NSUInteger length, MTLResourceOptions opts) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (t_arm.armed && t_arm.kind == DMN_SHARE_BUFFER) {
        id<MTLBuffer> sub = substitute_buffer((id<MTLDevice>)self, length);
        if (sub) {
            t_arm.armed = false;
            return sub;
        }
        DMN_WARN("share: armed dev newBufferWithLength:options: not "
                 "substituted; passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for dev newBufferWithLength:options:");
        return nil;
    }
    return ((id (*)(id, SEL, NSUInteger, MTLResourceOptions))orig)(
        self, _cmd, length, opts);
}

/* Heap placed path: -[heap newBufferWithLength:options:offset:] (ignore
 * offset: — our buffer is backed by its own page at offset 0; sub-page slot
 * suballocation is done by the caller within the returned buffer). */
id swz_heap_newbuf(id self, SEL _cmd, NSUInteger length, MTLResourceOptions opts,
                   NSUInteger offset) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (t_arm.armed && t_arm.kind == DMN_SHARE_BUFFER) {
        id<MTLHeap> heap = (id<MTLHeap>)self;
        id<MTLBuffer> sub = substitute_buffer([heap device], length);
        if (sub) {
            t_arm.armed = false;
            return sub;
        }
        DMN_WARN("share: armed heap newBufferWithLength:options:offset: not "
                 "substituted; passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for heap newBufferWithLength:options:offset:");
        return nil;
    }
    return ((id (*)(id, SEL, NSUInteger, MTLResourceOptions, NSUInteger))orig)(
        self, _cmd, length, opts, offset);
}

/* Lazily swizzle a heap class's placed-texture and placed-buffer creators on
 * first sighting. Heap creation is free-threaded through D3DMetal; the
 * check-swap-record is atomic inside install_swizzle. */
void ensure_heap_class_swizzled(Class heapClass) {
    static const SwizzleJob jobs[] = {
        { @selector(newTextureWithDescriptor:offset:), (IMP)swz_heap_newtex },
        { @selector(newBufferWithLength:options:offset:), (IMP)swz_heap_newbuf },
    };
    static std::atomic<Class> memo{nullptr};
    install_swizzles(heapClass, jobs, sizeof(jobs) / sizeof(jobs[0]),
                     "heap-placed", &memo);
}

/* Device heap path: -[dev newHeapWithDescriptor:] — wrap to catch heap classes
 * as they appear, so their placed-texture creator is hooked before D3DMetal
 * ever allocates a placed resource from them. */
id swz_dev_newheap(id self, SEL _cmd, MTLHeapDescriptor* desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id heap = orig
        ? ((id (*)(id, SEL, MTLHeapDescriptor*))orig)(self, _cmd, desc)
        : nil;
    if (heap)
        ensure_heap_class_swizzled(object_getClass(heap));
    return heap;
}

/* == Residency for substituted resources =================================
 * A substituted impostor is a shared-memory MTLBuffer-backed texture/buffer,
 * not placed on one of D3DMetal's heaps. D3DMetal establishes per-encoder GPU
 * residency with useHeap:/useResource: for its OWN heap allocations, so an
 * impostor bound bindlessly through an argument buffer (a D3D descriptor table)
 * never gets a useResource: — the GPU address-faults on the first sample
 * (kIOGPUCommandBufferCallbackErrorPageFault -> SubmissionsIgnored -> the guest
 * device renders permanently black even as flips keep succeeding). Track every
 * substituted resource weakly and useResource: it on every render/compute
 * encoder.
 *
 * The declared usage travels with the resource rather than being a constant:
 * see residency_usage_for(). It is stashed as an associated object so it
 * shares the resource's lifetime exactly — a parallel map keyed by pointer
 * would need pruning and would hand a recycled address the previous
 * resource's access. */
NSHashTable* g_sub_resources;  /* weak MTLResource refs; guarded by g_sub_lock */
std::mutex   g_sub_lock;
const void*  kDmnUsageKey = &kDmnUsageKey;

void sub_resource_track(id res, MTLResourceUsage usage) {
    if (!res)
        return;
    objc_setAssociatedObject(res, kDmnUsageKey,
                             [NSNumber numberWithUnsignedLongLong:usage],
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    std::lock_guard<std::mutex> lk(g_sub_lock);
    if (!g_sub_resources)
        g_sub_resources = [[NSHashTable weakObjectsHashTable] retain];
    [g_sub_resources addObject:res];
}

MTLResourceUsage sub_resource_usage(id res) {
    NSNumber* n = objc_getAssociatedObject(res, kDmnUsageKey);
    /* Untracked resources cannot reach here, but fall back to full access
     * rather than to none if one ever does. */
    return n ? (MTLResourceUsage)[n unsignedLongLongValue]
             : (MTLResourceUsageRead | MTLResourceUsageWrite);
}

/* Snapshot under the lock, useResource: outside it. */
void sub_resources_make_resident(id enc, bool compute) {
    NSArray* snapshot = nil;
    {
        std::lock_guard<std::mutex> lk(g_sub_lock);
        if (!g_sub_resources || g_sub_resources.count == 0)
            return;
        snapshot = [[g_sub_resources allObjects] retain];
    }
    for (id<MTLResource> r in snapshot) {
        const MTLResourceUsage usage = sub_resource_usage(r);
        if (compute) {
            [(id<MTLComputeCommandEncoder>)enc useResource:r usage:usage];
        } else {
            [(id<MTLRenderCommandEncoder>)enc
                useResource:r
                      usage:usage
                     stages:MTLRenderStageVertex | MTLRenderStageFragment];
        }
    }
    [snapshot release];
}

id swz_cb_cceD(id self, SEL _cmd, id desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL, id))orig)(self, _cmd, desc);
    if (enc)
        sub_resources_make_resident(enc, true);
    return enc;
}

id swz_cb_rce(id self, SEL _cmd, MTLRenderPassDescriptor* desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL, MTLRenderPassDescriptor*))orig)(self, _cmd, desc);
    if (enc)
        sub_resources_make_resident(enc, false);
    return enc;
}

id swz_cb_cce(id self, SEL _cmd) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL))orig)(self, _cmd);
    if (enc)
        sub_resources_make_resident(enc, true);
    return enc;
}

id swz_cb_cced(id self, SEL _cmd, NSUInteger dispatchType) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL, NSUInteger))orig)(self, _cmd, dispatchType);
    if (enc)
        sub_resources_make_resident(enc, true);
    return enc;
}

void ensure_texture_class_swizzled(id tex) {
    static const SwizzleJob jobs[] = {
        { @selector(replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:
                    bytesPerImage:), (IMP)swz_tex_replace6 },
        { @selector(replaceRegion:mipmapLevel:withBytes:bytesPerRow:),
          (IMP)swz_tex_replace4 },
    };
    static std::atomic<Class> memo{nullptr};
    install_swizzles(tex ? object_getClass(tex) : nil, jobs,
                     sizeof(jobs) / sizeof(jobs[0]), "texture-upload", &memo);
}

/* Runs for every command buffer, so the memo fast path matters here. */
void ensure_cmdbuf_class_swizzled(Class cbc) {
    static const SwizzleJob jobs[] = {
        { @selector(renderCommandEncoderWithDescriptor:), (IMP)swz_cb_rce },
        { @selector(computeCommandEncoder), (IMP)swz_cb_cce },
        { @selector(computeCommandEncoderWithDispatchType:), (IMP)swz_cb_cced },
        { @selector(computeCommandEncoderWithDescriptor:), (IMP)swz_cb_cceD },
    };
    static std::atomic<Class> memo{nullptr};
    install_swizzles(cbc, jobs, sizeof(jobs) / sizeof(jobs[0]),
                     "encoder-creator", &memo);
}

/* Command-buffer creators on the queue class -> swizzle each cmdbuf class the
 * first time it appears, before D3DMetal encodes anything on it. */
id swz_q_cmdbuf(id self, SEL _cmd) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id cb = orig ? ((id (*)(id, SEL))orig)(self, _cmd) : nil;
    if (cb)
        ensure_cmdbuf_class_swizzled(object_getClass(cb));
    return cb;
}

id swz_q_cmdbuf_unret(id self, SEL _cmd) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id cb = orig ? ((id (*)(id, SEL))orig)(self, _cmd) : nil;
    if (cb)
        ensure_cmdbuf_class_swizzled(object_getClass(cb));
    return cb;
}

id swz_q_cbdesc(id self, SEL _cmd, MTLCommandBufferDescriptor* desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id cb = orig ? ((id (*)(id, SEL, MTLCommandBufferDescriptor*))orig)(self, _cmd,
                                                                        desc)
                 : nil;
    if (cb)
        ensure_cmdbuf_class_swizzled(object_getClass(cb));
    return cb;
}

void ensure_queue_class_swizzled(Class qc) {
    static const SwizzleJob jobs[] = {
        { @selector(commandBuffer), (IMP)swz_q_cmdbuf },
        { @selector(commandBufferWithUnretainedReferences),
          (IMP)swz_q_cmdbuf_unret },
        { @selector(commandBufferWithDescriptor:), (IMP)swz_q_cbdesc },
    };
    static std::atomic<Class> memo{nullptr};
    install_swizzles(qc, jobs, sizeof(jobs) / sizeof(jobs[0]),
                     "command-buffer-creator", &memo);
}

id swz_dev_newq(id self, SEL _cmd) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id q = orig ? ((id (*)(id, SEL))orig)(self, _cmd) : nil;
    if (q)
        ensure_queue_class_swizzled(object_getClass(q));
    return q;
}

id swz_dev_newqmax(id self, SEL _cmd, NSUInteger maxCount) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id q = orig ? ((id (*)(id, SEL, NSUInteger))orig)(self, _cmd, maxCount) : nil;
    if (q)
        ensure_queue_class_swizzled(object_getClass(q));
    return q;
}

/* newCommandQueueWithDescriptor: (macOS 13+). Every queue-creation variant must
 * be hooked or a queue made through the one we miss escapes residency entirely
 * — the impostor-not-resident fault returns for its command buffers. */
id swz_dev_newqdesc(id self, SEL _cmd, id desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    id q = orig ? ((id (*)(id, SEL, id))orig)(self, _cmd, desc) : nil;
    if (q)
        ensure_queue_class_swizzled(object_getClass(q));
    return q;
}

/* Runs once per distinct device class at install time — no memo needed. */
void swizzle_device_class(Class cls) {
    static const SwizzleJob jobs[] = {
        { @selector(newTextureWithDescriptor:), (IMP)swz_dev_newtex },
        { @selector(newBufferWithLength:options:), (IMP)swz_dev_newbuf },
        { @selector(newHeapWithDescriptor:), (IMP)swz_dev_newheap },
        { @selector(newCommandQueue), (IMP)swz_dev_newq },
        { @selector(newCommandQueueWithMaxCommandBufferCount:),
          (IMP)swz_dev_newqmax },
        { @selector(newCommandQueueWithDescriptor:), (IMP)swz_dev_newqdesc },
    };
    install_swizzles(cls, jobs, sizeof(jobs) / sizeof(jobs[0]), "device",
                     /*memo=*/nullptr);
}

} // namespace

/* == dmn_share.h entry points ============================================= */

void dmn_share_arm_producer(uint64_t extra_bytes) {
    t_arm = {};
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_TEXTURE;
    t_arm.alloc_new = true;
    t_arm.extra_bytes = extra_bytes;
}

void dmn_share_arm_consumer(int fd, uint64_t stride, uint64_t size) {
    t_arm = {};
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_TEXTURE;
    t_arm.alloc_new = false;
    t_arm.existing_fd = fd;
    t_arm.existing_stride = stride;
    t_arm.existing_size = size;
}

void dmn_share_arm_producer_buffer(uint64_t size) {
    t_arm = {};
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_BUFFER;
    t_arm.alloc_new = true;
    t_arm.request_bytes = size;
}

void dmn_share_arm_consumer_buffer(int fd, uint64_t size) {
    t_arm = {};
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_BUFFER;
    t_arm.alloc_new = false;
    t_arm.existing_fd = fd;
    t_arm.existing_size = size;
}

bool dmn_share_is_armed(void) { return t_arm.armed; }

const void* dmn_share_init_data_sentinel(void) { return init_sentinel_base(); }

bool dmn_share_disarm(DmnShareArm* out) {
    bool captured = t_arm.captured;
    if (t_arm.armed && !captured)
        DMN_WARN("share: armed create reached NONE of the hooked Metal entry "
                 "points — resource is not shared (add the missing selector)");
    if (out)
        *out = t_arm;
    t_arm = {};
    return captured;
}

void dmn_share_install_swizzles(void) {
    {
        std::lock_guard<std::mutex> lk(g_install_mutex);
        if (g_installed)
            return;
        g_installed = true;
    }
    @autoreleasepool {
        /* Collect the distinct concrete device classes D3DMetal may use — the
         * same set dmn_gfxt_adapter.mm hands it (MTLCopyAllDevices + the system
         * default) — and swizzle each, so an armed create is caught whichever
         * device D3DMetal targets. */
        std::vector<Class> distinct;
        auto note = [&distinct](id<MTLDevice> d) {
            if (!d)
                return;
            Class c = object_getClass(d);
            for (Class seen : distinct)
                if (seen == c)
                    return;
            distinct.push_back(c);
        };

        NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
        for (id<MTLDevice> d in all)
            note(d);
        [all release];
        if (id<MTLDevice> def = MTLCreateSystemDefaultDevice()) {
            note(def);
            [def release];
        }

        for (Class c : distinct)
            swizzle_device_class(c);

        /* The arm record is a single thread-local, disarmed on the first
         * texture creation, so it implicitly assumes ONE active device class.
         * With more than one distinct class present (e.g. an Intel Mac's
         * integrated + discrete GPUs), D3DMetal could create the armed texture
         * on a different device than expected, or a scratch texture on another
         * device could consume the arm first — silently mis-targeting the
         * substitution. Every class is still swizzled, but make the ambiguity
         * loud so it is obvious in the logs. */
        if (distinct.size() > 1) {
            DMN_ERROR("share: %zu distinct Metal device classes present — "
                      "shared-texture arming assumes a single active device "
                      "class; the substitution may target the wrong device or "
                      "capture the wrong texture on multi-GPU systems",
                      distinct.size());
            for (Class c : distinct)
                DMN_ERROR("share:   Metal device class: %s", class_getName(c));
        }
    }
}

/* == plain mmap helpers (also used by dmn_fence.cpp / dmn_kmtx.cpp) ======= */

size_t dmn_share_page_align(size_t n) { return page_align(n); }

int dmn_share_anon_file(size_t size) { return dmn_anon_file((off_t)size); }

void* dmn_share_map_fd(int fd, size_t size) {
    size_t aligned = page_align(size);
    void* p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? nullptr : p;
}

void dmn_share_unmap(void* ptr, size_t size) {
    if (ptr)
        munmap(ptr, page_align(size));
}
