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
#include <string.h>
#include <sys/mman.h>
#include <sys/posix_shm.h>
#include <sys/proc_info.h>
#include <sys/stat.h>
#include <libproc.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <set>
#include <string>
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

/* An app group prefix must leave room for "/", a slot digit and one nonce
 * digit within PSHMNAMLEN. */
constexpr size_t kAppSandboxGroupIdMax = PSHMNAMLEN - 3;

/* == Anonymous shared file: shm_open a random name, unlink immediately ==== */
int dmn_anon_file(off_t size) {
    /* Under App Sandbox a shm name must live in the app group container, i.e.
     * be prefixed with the group identifier instead of '/'; anything else is
     * denied. The PSHMNAMLEN budget then leaves no room for our tag or the
     * pid, so the name is just a slot digit plus as much of the nonce as
     * fits. */
    const char* group = getenv("APP_SANDBOX_GROUP_ID");
    if (group && !group[0])
        group = nullptr;
    if (group && strlen(group) > kAppSandboxGroupIdMax) {
        DMN_ERROR("share: APP_SANDBOX_GROUP_ID is %zu chars, at most %zu fit",
                  strlen(group), kAppSandboxGroupIdMax);
        errno = EINVAL;
        return -1;
    }

    const size_t nonce_room = group ? PSHMNAMLEN - strlen(group) - 2 : 8;
    const unsigned nonce_digits = (unsigned)(nonce_room < 8 ? nonce_room : 8);
    const unsigned nonce = arc4random() >> (32 - 4 * nonce_digits);
    int fd = -1;
    for (unsigned i = 0; i < 32; i++) {
        char name[64];
        if (group) {
            /* the slot must stay a single hex digit to fit the name budget */
            if (i > 0xf)
                break;
            snprintf(name, sizeof(name), "%s/%x%x", group, i, nonce);
        } else {
            snprintf(name, sizeof(name), "/dmn-shm-%d-%x-%x", getpid(), nonce,
                     i);
        }
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

/* The mapping is reclaimed when the MTLBuffer dies: the deallocator block
 * munmaps it (the fd is never the buffer's to close: producer fds belong to
 * the COM layer, consumer fds to the app). Metal copies the block, so cleanup
 * runs whenever the last reference goes. In-flight command buffers are NOT
 * among those references — D3DMetal's are unretained — so a buffer still
 * referencing this one is kept alive by sub_resources_make_resident()
 * instead. */
id<MTLBuffer> shared_buffer_over(id<MTLDevice> device, void* ptr, size_t aligned) {
    return [device newBufferWithBytesNoCopy:ptr
                                     length:aligned
                                    options:MTLResourceStorageModeShared
                                deallocator:^(void* p, NSUInteger len) {
                                    munmap(p, len);
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

/* == Same-process buffer alias cache ======================================
 * Two MTLBuffers over the same shm pages are invisible to Metal's hazard
 * tracking: a GPU write through one and a GPU read through the other have no
 * dependency edge, so Metal may reorder them, and a same-process
 * OpenSharedResource + CopyResource can read the bytes from before the
 * producer's write. D3D11 promises in-order execution on the immediate
 * context, so aliased backings must be ONE Metal object.
 *
 * Every buffer substitution over shared memory therefore registers itself
 * here under (shm object identity, byte offset), and a later substitution of
 * the same window returns the SAME MTLBuffer (+1) when the device matches and
 * the cached mapping covers the requested span. References are weak
 * (objc_storeWeak works without ARC), so the cache never extends a backing's
 * lifetime — a dead entry reads back nil and is replaced.
 *
 * Identity is the pshm_name from proc_pidfdinfo(PROC_PIDFDPSHMINFO): on macOS
 * every POSIX shm fd fstat()s as st_dev=0 st_ino=0, so an inode key would
 * hand unrelated objects at equal offsets the same MTLBuffer. The name
 * survives the immediate shm_unlink and is shared by every dup and every fd
 * received over SCM_RIGHTS. An fd that is neither a POSIX shm object nor a
 * real file (dmn_open_existing_heap_from_fd takes what it is given) is not
 * cached at all: a private mapping per substitution, never a wrong alias. */
struct AliasKey {
    std::string name; /* pshm_name, or "ino:<dev>:<ino>" for a real file */
    uint64_t    off;
    bool operator==(const AliasKey& o) const {
        return off == o.off && name == o.name;
    }
};
struct AliasKeyHash {
    size_t operator()(const AliasKey& k) const {
        return std::hash<std::string>()(k.name) * 31u ^
               std::hash<uint64_t>()(k.off);
    }
};
struct AliasSlot {
    id weak_buf = nil; /* managed via objc_storeWeak / objc_loadWeak */
};
std::mutex g_alias_mtx;
std::unordered_map<AliasKey, AliasSlot*, AliasKeyHash> g_alias_cache;

bool alias_key_of(int fd, uint64_t off, AliasKey* out) {
    struct pshm_fdinfo pi;
    const int n = proc_pidfdinfo(getpid(), fd, PROC_PIDFDPSHMINFO, &pi,
                                 sizeof pi);
    if (n >= (int)sizeof pi && pi.pshminfo.pshm_name[0]) {
        out->name.assign(pi.pshminfo.pshm_name,
                         strnlen(pi.pshminfo.pshm_name,
                                 sizeof pi.pshminfo.pshm_name));
        out->off = off;
        return true;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_ino == 0)
        return false; /* no usable identity: do not cache */
    char b[64];
    snprintf(b, sizeof b, "ino:%llu:%llu", (unsigned long long)st.st_dev,
             (unsigned long long)st.st_ino);
    out->name = b;
    out->off = off;
    return true;
}

/* A live cached buffer for the window, +1, or nil. `need_len` is the mapped
 * span the caller would otherwise create — the cached object must cover it. */
id<MTLBuffer> alias_cache_lookup(id<MTLDevice> device, int fd, uint64_t off,
                                 size_t need_len) {
    AliasKey key;
    if (!alias_key_of(fd, off, &key))
        return nil;
    AliasSlot* slot;
    {
        std::lock_guard<std::mutex> lk(g_alias_mtx);
        auto it = g_alias_cache.find(key);
        if (it == g_alias_cache.end())
            return nil;
        slot = it->second;
    }
    id buf;
    @autoreleasepool { /* objc_loadWeak returns +0 autoreleased; pin it */
        buf = [objc_loadWeak(&slot->weak_buf) retain];
    }
    if (!buf)
        return nil;
    id<MTLBuffer> b = (id<MTLBuffer>)buf;
    if ([b device] != device || [b length] < need_len) {
        [b release];
        return nil;
    }
    return b; /* +1 */
}

void alias_cache_store(int fd, uint64_t off, id<MTLBuffer> buf) {
    AliasKey key;
    if (!alias_key_of(fd, off, &key))
        return;
    std::lock_guard<std::mutex> lk(g_alias_mtx);
    AliasSlot*& slot = g_alias_cache[key];
    if (!slot)
        slot = new AliasSlot();
    objc_storeWeak(&slot->weak_buf, buf);
}

/* == Create-time zero-fill defence ========================================
 * GPTk 1.0, 2.1 and 3.0 memset a new buffer's contents to zero synchronously
 * inside the create; 4.0 does not. For a consumer or window impostor those
 * contents are a PRODUCER'S bytes — a placed buffer's neighbours in a shared
 * heap, an imported guest ring, the surface a peer just wrote — so on those
 * frameworks every import would wipe what it opened. There is nothing to
 * intercept (an inlined memset), so the bytes are snapshotted at capture and
 * put back at disarm.
 *
 * Self-calibrating: the first DECISIVE observation (a nonzero snapshot that
 * either was or was not modified by the create) decides for the process. A
 * framework that does not zero pays for snapshots only until that
 * observation; one that does pays a copy per import. Residual window on a
 * zeroing framework: a peer's GPU write landing between snapshot and restore
 * is reverted — inherent to any CPU-side repair, and far narrower than
 * losing the whole buffer. */
enum ZeroFill { ZF_UNKNOWN, ZF_YES, ZF_NO };
std::atomic<int> g_zero_fill{ZF_UNKNOWN};

void snapshot_for_restore(void* ptr, size_t len) {
    if (g_zero_fill.load(std::memory_order_acquire) == ZF_NO)
        return;
    void* copy = malloc(len);
    if (!copy)
        return; /* nothing to do but let the create through unprotected */
    memcpy(copy, ptr, len);
    t_arm.restore_dst  = ptr;
    t_arm.restore_copy = copy;
    t_arm.restore_len  = len;
}

/* At disarm: compare, restore, and learn. */
void restore_after_create() {
    if (!t_arm.restore_copy)
        return;
    void* dst = t_arm.restore_dst;
    void* copy = t_arm.restore_copy;
    const size_t len = t_arm.restore_len;
    t_arm.restore_dst = nullptr;
    t_arm.restore_copy = nullptr;
    t_arm.restore_len = 0;

    const bool changed = memcmp(dst, copy, len) != 0;
    if (changed)
        memcpy(dst, copy, len);
    if (g_zero_fill.load(std::memory_order_acquire) == ZF_UNKNOWN) {
        /* Decisive only if the snapshot had something to lose: an all-zero
         * snapshot cannot tell a zeroing create from a benign one. */
        bool nonzero = false;
        const uint8_t* b = (const uint8_t*)copy;
        for (size_t i = 0; i < len && !nonzero; i += 64)
            nonzero = b[i] != 0;
        if (!nonzero)
            for (size_t i = 0; i < len && !nonzero; i++)
                nonzero = b[i] != 0;
        if (nonzero) {
            g_zero_fill.store(changed ? ZF_YES : ZF_NO, std::memory_order_release);
            if (changed)
                DMN_WARN("share: this D3DMetal zero-fills new buffers on "
                         "creation; imported/placed buffer contents will be "
                         "snapshotted and restored around every create");
            else
                DMN_INFO("share: this D3DMetal leaves new buffer contents "
                         "alone; no create-time restore needed");
        }
    }
    free(copy);
}

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

/* Wrap an already-mapped span in a linear MTLTexture matching `desc`, with the
 * surface's first byte at `buf_offset` inside the buffer (0 for a surface that
 * owns its whole object; the page-floor delta for a window into a heap's
 * object).
 *
 * `buf` is consumed: on success the returned texture owns the caller's
 * reference, on failure it is released. Returns +1 or nil. */
id<MTLTexture> make_linear_texture(id<MTLBuffer> buf, MTLTextureDescriptor* desc,
                                   const LinearLayout& layout, size_t buf_offset,
                                   bool producer) {
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

    /* Metal requires the in-buffer offset aligned like a row start. Placement
     * offsets are at least 4 KiB-aligned and this alignment is a small power
     * of two, so a violation is a logic error worth stopping on, not a
     * surface shape. */
    NSUInteger off_align =
        [[buf device] minimumLinearTextureAlignmentForPixelFormat:fmt];
    if (off_align && (buf_offset % off_align)) {
        DMN_ERROR("share: texture window offset %zu is not %lu-byte aligned",
                  buf_offset, (unsigned long)off_align);
        [buf release];
        return nil;
    }

    id<MTLTexture> tex = [buf newTextureWithDescriptor:linear
                                                offset:buf_offset
                                           bytesPerRow:layout.stride];
    if (!tex) {
        DMN_ERROR("share: newTextureWithDescriptor:offset:bytesPerRow: failed "
                  "(fmt=%lu %lux%lu stride=%zu off=%zu bufLen=%lu) — "
                  "under-aligned stride?",
                  (unsigned long)fmt, (unsigned long)width,
                  (unsigned long)height, layout.stride, buf_offset,
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

    /* The buffer owns the MAPPING only; the fd travels out through the arm and
     * is closed by the COM layer once the registry holds its dup (or the
     * create fails). Tying it to the deallocator would make fd reclamation
     * wait for D3DMetal's deferred destruction to drop the last Metal
     * reference, which is GPU-timing-dependent. The mapping alone keeps the
     * shm object's pages alive. */
    id<MTLBuffer> buf = shared_buffer_over(device, ptr, layout.mapped);
    if (!buf) {
        DMN_ERROR("share: newBufferWithBytesNoCopy failed");
        munmap(ptr, layout.mapped);
        close(fd);
        return nil;
    }
    alias_cache_store(fd, 0, buf); /* same-process opens alias this object */

    id<MTLTexture> tex = make_linear_texture(buf, desc, layout, /*buf_offset=*/0,
                                             /*producer=*/true);
    if (!tex) {
        close(fd);
        return nil;
    }

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

    /* A placed texture's POD carries its byte offset within the heap's object.
     * mmap needs a page-aligned file offset, so map from the page floor and
     * put the texture at the delta inside the buffer. */
    const size_t offset = (size_t)t_arm.existing_offset;
    const size_t floor  = offset & ~(page_align(1) - 1);
    const size_t delta  = offset - floor;
    size_t span = 0;
    if (offset > kMaxSharedBytes || !add_ok(delta, layout.logical, &span)) {
        DMN_ERROR("share: import window offset %zu out of range", offset);
        return nil;
    }
    layout.mapped = page_align(span);

    /* The fd belongs to the caller on this path — never closed here, and never
     * handed to the buffer's deallocator. Reuse a live impostor over the same
     * span when one exists (see the alias cache): each import still gets its
     * own MTLTexture, but over the SAME buffer, so Metal orders aliased
     * accesses. */
    const int fd = t_arm.existing_fd;
    id<MTLBuffer> buf = alias_cache_lookup(device, fd, floor, layout.mapped);
    if (buf) {
        DMN_INFO("share: consumer texture reuses cached impostor backing");
    } else {
        void* ptr = mmap(nullptr, layout.mapped, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, (off_t)floor);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: consumer mmap(fd=%d, off=%zu, %zu) failed: %s", fd,
                      floor, layout.mapped, strerror(errno));
            return nil;
        }
        buf = shared_buffer_over(device, ptr, layout.mapped);
        if (!buf) {
            DMN_ERROR("share: consumer newBufferWithBytesNoCopy failed");
            munmap(ptr, layout.mapped);
            return nil;
        }
        alias_cache_store(fd, floor, buf);
    }

    id<MTLTexture> tex = make_linear_texture(buf, desc, layout, delta,
                                             /*producer=*/false);
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

/* Texture window: a texture placed in a shared heap. Backed by the heap's
 * EXISTING object like a consumer, but the layout is DERIVED from the
 * descriptor like a producer — the placement is the first sight of this
 * surface, so there is no shipped layout to validate against; the derived one
 * travels out through the arm and into the export POD instead. Overlapping
 * placements alias because every window maps the same object at the offsets
 * the app placed them at. Returns +1, or nil. */
id<MTLTexture> substitute_texture_window(id<MTLDevice> device,
                                         MTLTextureDescriptor* desc) {
    const MTLPixelFormat fmt = desc.pixelFormat;
    const size_t width  = (size_t)desc.width;
    const size_t height = (size_t)desc.height;

    const uint32_t bpp = dmn_format_linear_bpp((uint32_t)fmt);
    if (!bpp) {
        DMN_ERROR("share: MTLPixelFormat %lu has no linear layout — refusing "
                  "to window it into a shared heap", (unsigned long)fmt);
        return nil;
    }
    if (!dims_ok(width, height, "heap-window"))
        return nil;

    NSUInteger row_align =
        [device minimumLinearTextureAlignmentForPixelFormat:fmt];
    if (row_align == 0)
        row_align = 256;

    LinearLayout layout{};
    size_t row = 0;
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
    /* The window is what the heap has left past the placement offset. Our
     * linear layout can legitimately exceed what the app reserved from
     * GetResourceAllocationInfo (a wider aligned stride); that must fail the
     * create rather than run past the heap's object. */
    const size_t avail = (size_t)t_arm.existing_max;
    if (layout.logical > avail) {
        DMN_ERROR("share: placed texture needs %zu bytes but only %zu remain "
                  "in the heap past its offset", layout.logical, avail);
        return nil;
    }

    const size_t offset = (size_t)t_arm.existing_offset;
    const size_t floor  = offset & ~(page_align(1) - 1);
    const size_t delta  = offset - floor;
    size_t span = 0;
    if (offset > kMaxSharedBytes || !add_ok(delta, layout.logical, &span)) {
        DMN_ERROR("share: heap-window offset %zu out of range", offset);
        return nil;
    }
    layout.mapped = page_align(span);

    /* The fd is the heap's — borrowed, never closed here; the deallocator
     * only unmaps this window. Placements at one offset alias, so share the
     * backing MTLBuffer with any live impostor over the same span (see the
     * alias cache) — that is also what gives aliased placements hazard
     * ordering within a process. */
    const int fd = t_arm.existing_fd;
    id<MTLBuffer> buf = alias_cache_lookup(device, fd, floor, layout.mapped);
    if (buf) {
        DMN_INFO("share: heap-window texture reuses cached impostor backing");
    } else {
        void* ptr = mmap(nullptr, layout.mapped, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, (off_t)floor);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: heap-window mmap(fd=%d, off=%zu, %zu) failed: %s",
                      fd, floor, layout.mapped, strerror(errno));
            return nil;
        }
        buf = shared_buffer_over(device, ptr, layout.mapped);
        if (!buf) {
            DMN_ERROR("share: heap-window newBufferWithBytesNoCopy failed");
            munmap(ptr, layout.mapped);
            return nil;
        }
        alias_cache_store(fd, floor, buf);
    }

    id<MTLTexture> tex = make_linear_texture(buf, desc, layout, delta,
                                             /*producer=*/true);
    if (!tex)
        return nil;

    t_arm.captured   = true;
    t_arm.out_fd     = fd;
    t_arm.out_stride = layout.stride;
    t_arm.out_size   = layout.logical;
    DMN_INFO("share: substituted heap-window texture fmt=%lu %zux%zu stride=%zu "
             "size=%zu fd=%d off=%zu", (unsigned long)fmt, width, height,
             layout.stride, layout.logical, fd, offset);
    return tex;
}

id<MTLTexture> substitute(id<MTLDevice> device, MTLTextureDescriptor* desc) {
    if (!device || !desc)
        return nil;
    if (t_arm.alloc_new)
        return substitute_producer(device, desc);
    return t_arm.derive_layout ? substitute_texture_window(device, desc)
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
        /* Mapping-only ownership; the fd goes out through the arm — see the
         * producer texture path for why. */
        id<MTLBuffer> buf = shared_buffer_over(device, ptr, mapped);
        if (!buf) {
            DMN_ERROR("share: buffer newBufferWithBytesNoCopy failed");
            munmap(ptr, mapped);
            close(fd);
            return nil;
        }
        buf.label = @"dmn-shared-prod-buffer";
        sub_resource_track(buf, both);
        alias_cache_store(fd, 0, buf); /* a same-process import must alias
                                          THIS object, not a second mapping */

        t_arm.captured = true;
        t_arm.out_fd   = fd;
        t_arm.out_size = logical;
        DMN_INFO("share: substituted producer buffer size=%zu fd=%d "
                 "(asked %lu) gpuAddress=0x%llx", logical, fd,
                 (unsigned long)length, (unsigned long long)buf.gpuAddress);
        return buf;
    }

    /* Consumer: the shared region has to cover what the caller asked for. A
     * short buffer is not something to hand back and hope about — D3DMetal
     * would read and write past the mapping. With a window (existing_max set,
     * imported-heap placements) a request past the armed size but inside the
     * window is D3DMetal rounding the placed size up, and aliasing more of the
     * window is legal; past the window stays a hard error. */
    size_t logical = (size_t)t_arm.existing_size;
    const size_t window = (size_t)t_arm.existing_max;
    if (!logical || logical > kMaxSharedBytes) {
        DMN_ERROR("share: import buffer size %zu out of range", logical);
        return nil;
    }
    if (length && (size_t)length > logical) {
        if (window && (size_t)length <= window) {
            logical = (size_t)length;
        } else {
            DMN_ERROR("share: import buffer is %zu bytes but %lu were "
                      "requested (window %zu)", logical, (unsigned long)length,
                      window);
            return nil;
        }
    }
    if (t_arm.existing_offset & (page_align(1) - 1)) {
        DMN_ERROR("share: import window offset %llu is not page-aligned",
                  (unsigned long long)t_arm.existing_offset);
        return nil;
    }
    const size_t mapped = page_align(logical);
    if (window && mapped > window) {
        DMN_ERROR("share: import window is %zu bytes but the substitution "
                  "needs %zu mapped — would run past the shm object", window,
                  mapped);
        return nil;
    }
    const int fd = t_arm.existing_fd;

    /* Reuse a live impostor over the same window: two D3D buffers fronting
     * ONE MTLBuffer is what lets Metal's hazard tracking order a write through
     * one against a read through the other (see the alias cache above). A
     * fresh private mapping is the fallback, with the hazard hole it implies —
     * cross-process aliases always take it, and their ordering is the
     * exporting app's business (fences, keyed mutex). */
    id<MTLBuffer> buf =
        alias_cache_lookup(device, fd, t_arm.existing_offset, mapped);
    if (buf) {
        DMN_INFO("share: consumer buffer reuses cached impostor (off=%llu "
                 "len=%zu)", (unsigned long long)t_arm.existing_offset, mapped);
        /* The cached object may have been created as a TEXTURE backing, whose
         * residency rides the texture — as a standalone buffer resource it
         * must be tracked itself. Idempotent for an already-tracked buffer. */
        sub_resource_track(buf, both);
    } else {
        void* ptr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_SHARED,
                         fd, (off_t)t_arm.existing_offset);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: consumer buffer mmap(fd=%d, off=%llu, %zu) "
                      "failed: %s", fd,
                      (unsigned long long)t_arm.existing_offset, mapped,
                      strerror(errno));
            return nil;
        }
        buf = shared_buffer_over(device, ptr, mapped);
        if (!buf) {
            DMN_ERROR("share: consumer buffer newBufferWithBytesNoCopy failed");
            munmap(ptr, mapped);
            return nil;
        }
        buf.label = window ? @"dmn-imported-heap-window"
                           : @"dmn-shared-cons-buffer";
        sub_resource_track(buf, both);
        alias_cache_store(fd, t_arm.existing_offset, buf);
    }
    /* The bytes behind this impostor are somebody's: protect them from a
     * zero-filling create (see the defence above). [buf contents] is the
     * mapping either way — a fresh one, or the cached impostor's. */
    snapshot_for_restore([buf contents], (size_t)[buf length]);

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

/* == Allocation trace (DMN_LOG=trace) =====================================
 * One line per Metal allocation D3DMetal makes through the hooked creators —
 * what it asked for, whether the arm substituted it, and (for buffers/heaps)
 * where it landed. Cheap when disabled, so it is always compiled in. */
void trace_alloc(const char* what, bool substituted, unsigned long long bytes,
                 const char* extra) {
    if (!dmn_log_enabled(DMN_LOG_TRACE))
        return;
    DMN_TRACE("alloc: %-12s %s %llu bytes%s%s", what,
              substituted ? "SUBST" : "orig ", bytes, extra ? " " : "",
              extra ? extra : "");
}

const char* tex_desc_str(MTLTextureDescriptor* d, char* buf, size_t n) {
    snprintf(buf, n, "tex %lux%lu fmt=%lu type=%lu mips=%lu usage=0x%lx "
             "storage=%lu", (unsigned long)d.width, (unsigned long)d.height,
             (unsigned long)d.pixelFormat, (unsigned long)d.textureType,
             (unsigned long)d.mipmapLevelCount, (unsigned long)d.usage,
             (unsigned long)d.storageMode);
    return buf;
}

/* Device dedicated path: -[dev newTextureWithDescriptor:] */
id swz_dev_newtex(id self, SEL _cmd, MTLTextureDescriptor* desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    char tb[160];
    if (t_arm.armed && t_arm.kind == DMN_SHARE_TEXTURE) {
        id<MTLTexture> sub = substitute((id<MTLDevice>)self, desc);
        if (sub) {
            t_arm.armed = false; /* disarm on first hit */
            trace_alloc("dev_newtex", true, 0, tex_desc_str(desc, tb, sizeof tb));
            return sub;
        }
        DMN_WARN("share: armed dev newTextureWithDescriptor: not substituted; "
                 "passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for dev newTextureWithDescriptor:");
        return nil;
    }
    trace_alloc("dev_newtex", false, 0, tex_desc_str(desc, tb, sizeof tb));
    id tex = ((id (*)(id, SEL, MTLTextureDescriptor*))orig)(self, _cmd, desc);
    /* The view guard must be installed before a view is taken of a texture
     * that never went through the share path. Memoized: one atomic load per
     * texture after the first. */
    if (tex)
        ensure_texture_class_swizzled(tex);
    return tex;
}

/* Heap placed path: -[heap newTextureWithDescriptor:offset:] (ignore offset:
 * — our texture is backed by its own buffer at offset 0). */
id swz_heap_newtex(id self, SEL _cmd, MTLTextureDescriptor* desc,
                   NSUInteger offset) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    char tb[160];
    if (t_arm.armed && t_arm.kind == DMN_SHARE_TEXTURE) {
        id<MTLHeap> heap = (id<MTLHeap>)self;
        id<MTLTexture> sub = substitute([heap device], desc);
        if (sub) {
            t_arm.armed = false;
            trace_alloc("heap_newtex", true, 0, tex_desc_str(desc, tb, sizeof tb));
            return sub;
        }
        DMN_WARN("share: armed heap newTextureWithDescriptor:offset: not "
                 "substituted; passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for heap newTextureWithDescriptor:offset:");
        return nil;
    }
    if (dmn_log_enabled(DMN_LOG_TRACE)) {
        char eb[220];
        snprintf(eb, sizeof eb, "%s off=%lu heap=%p", tex_desc_str(desc, tb, sizeof tb),
                 (unsigned long)offset, (void*)self);
        trace_alloc("heap_newtex", false, 0, eb);
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
            trace_alloc("dev_newbuf", true, length, nullptr);
            return sub;
        }
        DMN_WARN("share: armed dev newBufferWithLength:options: not "
                 "substituted; passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for dev newBufferWithLength:options:");
        return nil;
    }
    if (dmn_log_enabled(DMN_LOG_TRACE)) {
        char eb[64];
        snprintf(eb, sizeof eb, "opts=0x%lx", (unsigned long)opts);
        trace_alloc("dev_newbuf", false, length, eb);
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
            trace_alloc("heap_newbuf", true, length, nullptr);
            return sub;
        }
        DMN_WARN("share: armed heap newBufferWithLength:options:offset: not "
                 "substituted; passing through");
    }
    if (!orig) {
        DMN_ERROR("share: no original for heap newBufferWithLength:options:offset:");
        return nil;
    }
    if (dmn_log_enabled(DMN_LOG_TRACE)) {
        char eb[96];
        snprintf(eb, sizeof eb, "opts=0x%lx off=%lu heap=%p", (unsigned long)opts,
                 (unsigned long)offset, (void*)self);
        trace_alloc("heap_newbuf", false, length, eb);
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
    if (dmn_log_enabled(DMN_LOG_TRACE)) {
        char eb[96];
        snprintf(eb, sizeof eb, "type=%lu storage=%lu heap=%p",
                 (unsigned long)desc.type, (unsigned long)desc.storageMode,
                 (void*)heap);
        trace_alloc("dev_newheap", false, (unsigned long long)desc.size, eb);
    }
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

/* Snapshot under the lock, useResource: outside it, and hold the snapshot until
 * `cb` completes rather than until this returns.
 *
 * Residency and lifetime must have the same scope here. g_sub_resources is weak
 * and D3DMetal's command buffers come from commandBufferWithUnretainedReferences,
 * so nothing else keeps an impostor alive between encode and execution — and
 * useResource: has already promised the GPU it will be there. An impostor freed
 * inside that window (the guest destroying an imported surface) aborts the buffer
 * with kIOGPUCommandBufferCallbackErrorInvalidResource, which kills the whole
 * MTLCommandQueue: every later submission on it fails, so its fences never
 * complete and that process renders nothing again.
 *
 * Pinning the entire tracked set rather than the subset a buffer touches is
 * deliberate — useResource: declares the entire set, so that is the promise being
 * backed. It adds no allocation, since allObjects already copies per encoder. */
void sub_resources_make_resident(id<MTLCommandBuffer> cb, id enc, bool compute) {
    NSArray* snapshot = nil;
    {
        std::lock_guard<std::mutex> lk(g_sub_lock);
        if (!g_sub_resources || g_sub_resources.count == 0)
            return;
        snapshot = [[g_sub_resources allObjects] retain];
    }
    /* Residency-only declaration: always Read, never the resource's stored
     * usage.  Writes to impostors ride explicit binding points (render-pass
     * attachments, blit arguments), which hazard-track on their own; a Write
     * declared here would instead put a write hazard on the whole tracked set
     * in every encoder, serializing every encoder against every other one.
     * Read is the weakest usage that still pins residency for the
     * unretained CBs. */
    for (id<MTLResource> r in snapshot) {
        if (compute) {
            [(id<MTLComputeCommandEncoder>)enc useResource:r
                                                     usage:MTLResourceUsageRead];
        } else {
            [(id<MTLRenderCommandEncoder>)enc
                useResource:r
                      usage:MTLResourceUsageRead
                     stages:MTLRenderStageVertex | MTLRenderStageFragment];
        }
    }
    /* Encoders are created before commit, so a completion handler is always
     * still accepted here; it runs on error paths too, so the pin cannot leak
     * on an aborted buffer. */
    [cb addCompletedHandler:^(id<MTLCommandBuffer> unused) {
        (void)unused;
        [snapshot release];
    }];
}

id swz_cb_cceD(id self, SEL _cmd, id desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL, id))orig)(self, _cmd, desc);
    if (enc)
        sub_resources_make_resident(self, enc, true);
    return enc;
}

id swz_cb_rce(id self, SEL _cmd, MTLRenderPassDescriptor* desc) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL, MTLRenderPassDescriptor*))orig)(self, _cmd, desc);
    if (enc)
        sub_resources_make_resident(self, enc, false);
    return enc;
}

id swz_cb_cce(id self, SEL _cmd) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL))orig)(self, _cmd);
    if (enc)
        sub_resources_make_resident(self, enc, true);
    return enc;
}

id swz_cb_cced(id self, SEL _cmd, NSUInteger dispatchType) {
    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    id enc = ((id (*)(id, SEL, NSUInteger))orig)(self, _cmd, dispatchType);
    if (enc)
        sub_resources_make_resident(self, enc, true);
    return enc;
}

/* -newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:
 *
 * An impostor is a single-slice, single-level linear texture, so a view naming
 * a slice or level the parent does not have is reachable from a D3D12 view
 * desc that is correct for the resource the guest created. Metal validates the
 * range itself and calls abort() on a mismatch, which kills the render server
 * inside the call and leaves the guest blocked forever on a reply that never
 * comes. Clamp into range instead: the view then addresses a slice the caller
 * did not ask for, which is wrong but recoverable and logged.
 *
 * This is a backstop for a resource that should not have been substituted at
 * all; the guest driver is responsible for not sharing array resources. */
id swz_tex_new_view(id self, SEL _cmd, MTLPixelFormat fmt,
                    MTLTextureType type, NSRange levels, NSRange slices,
                    MTLTextureSwizzleChannels swz) {
    id<MTLTexture> tex = (id<MTLTexture>)self;
    const NSUInteger have_slices =
        tex.textureType == MTLTextureType3D ? 1 : tex.arrayLength;
    const NSUInteger have_levels = tex.mipmapLevelCount;

    if (slices.location + slices.length > have_slices ||
        levels.location + levels.length > have_levels) {
        DMN_ERROR("share: texture view out of range on a %s texture "
                  "(%lux%lux%lu, arrayLength=%lu, mips=%lu): asked levels "
                  "[%lu,+%lu) slices [%lu,+%lu) -- CLAMPED to keep Metal "
                  "from aborting the worker; the view will address the "
                  "wrong slice/level",
                  tex.textureType == MTLTextureType3D ? "3D" : "non-3D",
                  (unsigned long)tex.width, (unsigned long)tex.height,
                  (unsigned long)tex.depth, (unsigned long)tex.arrayLength,
                  (unsigned long)have_levels,
                  (unsigned long)levels.location, (unsigned long)levels.length,
                  (unsigned long)slices.location, (unsigned long)slices.length);
        if (slices.location >= have_slices)
            slices.location = have_slices ? have_slices - 1 : 0;
        if (slices.location + slices.length > have_slices)
            slices.length = have_slices - slices.location;
        if (!slices.length)
            slices.length = 1;
        if (levels.location >= have_levels)
            levels.location = have_levels ? have_levels - 1 : 0;
        if (levels.location + levels.length > have_levels)
            levels.length = have_levels - levels.location;
        if (!levels.length)
            levels.length = 1;
    }

    IMP orig = lookup_orig(object_getClass(self), _cmd);
    if (!orig)
        return nil;
    return ((id (*)(id, SEL, MTLPixelFormat, MTLTextureType, NSRange, NSRange,
                    MTLTextureSwizzleChannels))orig)(
        self, _cmd, fmt, type, levels, slices, swz);
}

void ensure_texture_class_swizzled(id tex) {
    static const SwizzleJob jobs[] = {
        { @selector(replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:
                    bytesPerImage:), (IMP)swz_tex_replace6 },
        { @selector(replaceRegion:mipmapLevel:withBytes:bytesPerRow:),
          (IMP)swz_tex_replace4 },
        { @selector(newTextureViewWithPixelFormat:textureType:levels:slices:
                    swizzle:), (IMP)swz_tex_new_view },
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

namespace {
/* Fresh arm record; frees a snapshot a never-disarmed arm might still hold. */
void arm_reset() {
    if (t_arm.restore_copy)
        free(t_arm.restore_copy);
    t_arm = {};
}
} // namespace

/* == dmn_share.h entry points ============================================= */

void dmn_share_arm_producer(uint64_t extra_bytes) {
    dmn_dedicated_metal_alloc_begin();
    arm_reset();
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_TEXTURE;
    t_arm.alloc_new = true;
    t_arm.extra_bytes = extra_bytes;
}

void dmn_share_arm_consumer(int fd, uint64_t stride, uint64_t size,
                            uint64_t offset) {
    dmn_dedicated_metal_alloc_begin();
    arm_reset();
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_TEXTURE;
    t_arm.alloc_new = false;
    t_arm.existing_fd = fd;
    t_arm.existing_stride = stride;
    t_arm.existing_size = size;
    t_arm.existing_offset = offset;
}

void dmn_share_arm_texture_window(int fd, uint64_t offset, uint64_t max_size) {
    dmn_dedicated_metal_alloc_begin();
    arm_reset();
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_TEXTURE;
    t_arm.alloc_new = false;
    t_arm.derive_layout = true;
    t_arm.existing_fd = fd;
    t_arm.existing_offset = offset;
    t_arm.existing_max = max_size;
}

void dmn_share_arm_producer_buffer(uint64_t size) {
    dmn_dedicated_metal_alloc_begin();
    arm_reset();
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_BUFFER;
    t_arm.alloc_new = true;
    t_arm.request_bytes = size;
}

void dmn_share_arm_consumer_buffer(int fd, uint64_t size) {
    dmn_dedicated_metal_alloc_begin();
    arm_reset();
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_BUFFER;
    t_arm.alloc_new = false;
    t_arm.existing_fd = fd;
    t_arm.existing_size = size;
}

void dmn_share_arm_import_window(int fd, uint64_t offset, uint64_t size,
                                 uint64_t max_size) {
    dmn_dedicated_metal_alloc_begin();
    arm_reset();
    t_arm.armed = true;
    t_arm.kind = DMN_SHARE_BUFFER;
    t_arm.alloc_new = false;
    t_arm.existing_fd = fd;
    t_arm.existing_size = size;
    t_arm.existing_offset = offset;
    t_arm.existing_max = max_size;
}

bool dmn_share_is_armed(void) { return t_arm.armed; }

const void* dmn_share_init_data_sentinel(void) { return init_sentinel_base(); }

bool dmn_share_disarm(DmnShareArm* out) {
    dmn_dedicated_metal_alloc_end();
    restore_after_create(); /* the create has returned; put back what a
                               zero-filling framework wiped */
    bool captured = t_arm.captured;
    if (t_arm.armed && !captured)
        DMN_WARN("share: armed create reached NONE of the hooked Metal entry "
                 "points — resource is not shared (add the missing selector)");
    if (out)
        *out = t_arm;
    arm_reset();
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
