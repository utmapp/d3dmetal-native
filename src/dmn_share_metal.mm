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

#include <mutex>
#include <unordered_map>
#include <vector>

#include "d3dmetal_native.h"
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

/* Bytes-per-pixel for the linear formats typical shared surfaces use.
 * Unknown formats fall back to 4 with a warning (the stride/size in the
 * POD still describe whatever we built, so a consumer stays byte-consistent). */
uint32_t bpp_for_format(MTLPixelFormat fmt) {
    switch (fmt) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatBGR10A2Unorm:
        case MTLPixelFormatR32Float:
        case MTLPixelFormatRG16Unorm:
            return 4;
        case MTLPixelFormatRGBA16Float:
        case MTLPixelFormatRG32Float:
            return 8;
        case MTLPixelFormatRGBA32Float:
            return 16;
        case MTLPixelFormatR8Unorm:
            return 1;
        case MTLPixelFormatRG8Unorm:
        case MTLPixelFormatR16Float:
            return 2;
        default:
            DMN_WARN("share: unmapped MTLPixelFormat %lu; assuming bpp=4",
                     (unsigned long)fmt);
            return 4;
    }
}

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

/* == The substitution ===================================================== */
/* Build a shared-memory-backed linear texture matching `desc`, honoring the
 * armed allocate-vs-reuse mode, and record what we built into t_arm. Returns a
 * +1 texture (caller owns), or nil to let the original creation proceed. */
id<MTLTexture> substitute(id<MTLDevice> device, MTLTextureDescriptor* desc) {
    if (!device || !desc)
        return nil;

    const MTLPixelFormat fmt = desc.pixelFormat;
    const NSUInteger width  = desc.width;
    const NSUInteger height = desc.height;
    const uint32_t bpp = bpp_for_format(fmt);

    size_t stride, logical, aligned;
    int fd;
    void* ptr;

    if (t_arm.alloc_new) {
        NSUInteger row_align =
            [device minimumLinearTextureAlignmentForPixelFormat:fmt];
        if (row_align == 0)
            row_align = 256;
        stride  = ((size_t)width * bpp + (size_t)row_align - 1) &
                  ~((size_t)row_align - 1);
        logical = stride * (size_t)height;
        /* Trailer (e.g. the keyed-mutex page) lives past the page-aligned
         * texture bytes; consumers find it at page_align(pod.size). */
        aligned = page_align(logical) + page_align((size_t)t_arm.extra_bytes);
        fd = dmn_anon_file((off_t)aligned);
        if (fd < 0) {
            DMN_ERROR("share: anon file (%zu) failed", aligned);
            return nil;
        }
        ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: mmap(%zu) failed: %s", aligned, strerror(errno));
            close(fd);
            return nil;
        }
    } else {
        /* Consumer: reproduce the producer's byte-exact layout. */
        stride  = (size_t)t_arm.existing_stride;
        logical = (size_t)t_arm.existing_size;
        aligned = page_align(logical);
        fd = t_arm.existing_fd;
        ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: consumer mmap(fd=%d, %zu) failed: %s",
                      fd, aligned, strerror(errno));
            return nil;
        }
    }

    id<MTLBuffer> buf =
        shared_buffer_over(device, ptr, aligned, t_arm.alloc_new ? fd : -1);
    if (!buf) {
        DMN_ERROR("share: newBufferWithBytesNoCopy failed");
        munmap(ptr, aligned);
        if (t_arm.alloc_new) close(fd);
        return nil;
    }
    /* From here the buffer owns ptr (and the producer fd) via its
     * deallocator; error paths only release the buffer. */

    MTLTextureDescriptor* linear =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    /* Honor whatever usage D3DMetal asked for (render target, shader read,
     * etc.) so the substituted texture is accepted wherever the original
     * would have been; force Shared storage for cross-process visibility. */
    linear.usage       = desc.usage ? desc.usage
                       : (MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    linear.storageMode = MTLStorageModeShared;

    id<MTLTexture> tex = [buf newTextureWithDescriptor:linear
                                                offset:0
                                           bytesPerRow:stride];
    if (!tex) {
        DMN_ERROR("share: newTextureWithDescriptor:offset:bytesPerRow: failed "
                  "(fmt=%lu %lux%lu stride=%zu) — under-aligned stride?",
                  (unsigned long)fmt, (unsigned long)width,
                  (unsigned long)height, stride);
        [buf release];
        return nil;
    }

    pin_buffer_to_texture(tex, buf);
    [buf release]; /* the texture holds the only reference now */

    t_arm.captured   = true;
    t_arm.out_fd     = fd;
    t_arm.out_stride = stride;
    t_arm.out_size   = logical;

    DMN_INFO("share: substituted %s texture fmt=%lu %lux%lu stride=%zu "
             "size=%zu fd=%d",
             t_arm.alloc_new ? "producer" : "consumer",
             (unsigned long)fmt, (unsigned long)width, (unsigned long)height,
             stride, logical, fd);
    return tex; /* +1, caller owns */
}

/* Build a shared-memory-backed MTLBuffer of at least `length` bytes (forced to
 * StorageModeShared so the GPU can write it and a peer can read it via mmap).
 * Used for GPU-written fence-value pages. Returns a +1 buffer, or nil. */
id<MTLBuffer> substitute_buffer(id<MTLDevice> device, NSUInteger length) {
    if (!device)
        return nil;

    size_t logical, aligned;
    int fd;
    void* ptr;

    if (t_arm.alloc_new) {
        logical = length ? (size_t)length : sizeof(uint64_t);
        aligned = page_align(logical);
        fd = dmn_anon_file((off_t)aligned);
        if (fd < 0) {
            DMN_ERROR("share: buffer anon file (%zu) failed", aligned);
            return nil;
        }
        ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: buffer mmap(%zu) failed: %s", aligned,
                      strerror(errno));
            close(fd);
            return nil;
        }
    } else {
        logical = (size_t)t_arm.existing_size;
        aligned = page_align(logical);
        fd = t_arm.existing_fd;
        ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            DMN_ERROR("share: consumer buffer mmap(fd=%d, %zu) failed: %s",
                      fd, aligned, strerror(errno));
            return nil;
        }
    }

    id<MTLBuffer> buf =
        shared_buffer_over(device, ptr, aligned, t_arm.alloc_new ? fd : -1);
    if (!buf) {
        DMN_ERROR("share: buffer newBufferWithBytesNoCopy failed");
        munmap(ptr, aligned);
        if (t_arm.alloc_new) close(fd);
        return nil;
    }

    t_arm.captured = true;
    t_arm.out_fd   = fd;
    t_arm.out_size = logical;
    DMN_INFO("share: substituted %s buffer size=%zu fd=%d",
             t_arm.alloc_new ? "producer" : "consumer", logical, fd);
    return buf; /* +1, caller owns */
}

/* == Swizzle plumbing ===================================================== */
std::mutex g_swz_mutex;
std::unordered_map<Class, IMP> g_dev_newtex_orig;    /* -newTextureWithDescriptor: */
std::unordered_map<Class, IMP> g_dev_newbuf_orig;    /* -newBufferWithLength:options: */
std::unordered_map<Class, IMP> g_dev_newheap_orig;   /* -newHeapWithDescriptor: */
std::unordered_map<Class, IMP> g_heap_newtex_orig;   /* heap -newTextureWithDescriptor:offset: */
std::unordered_map<Class, IMP> g_heap_newbuf_orig;   /* heap -newBufferWithLength:options:offset: */
bool g_installed = false;

IMP lookup_orig(std::unordered_map<Class, IMP>& m, Class c) {
    std::lock_guard<std::mutex> lk(g_swz_mutex);
    /* Walk up the class chain: the concrete instance class may be a private
     * subclass of the class we swizzled. */
    for (Class k = c; k; k = class_getSuperclass(k)) {
        auto it = m.find(k);
        if (it != m.end())
            return it->second;
    }
    return nullptr;
}

/* Device dedicated path: -[dev newTextureWithDescriptor:] */
id swz_dev_newtex(id self, SEL _cmd, MTLTextureDescriptor* desc) {
    IMP orig = lookup_orig(g_dev_newtex_orig, object_getClass(self));
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
    IMP orig = lookup_orig(g_heap_newtex_orig, object_getClass(self));
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
    IMP orig = lookup_orig(g_dev_newbuf_orig, object_getClass(self));
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
    IMP orig = lookup_orig(g_heap_newbuf_orig, object_getClass(self));
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
 * first sighting. The lock is held across check-swap-record: heap creation is
 * free-threaded through D3DMetal, and a check/record gap would let the losing
 * thread record the winner's replacement IMP as the "original" (infinite
 * recursion on the first placed create). */
void ensure_heap_class_swizzled(Class heapClass) {
    if (!heapClass)
        return;
    struct { SEL sel; IMP repl; std::unordered_map<Class, IMP>* store; } jobs[] = {
        { @selector(newTextureWithDescriptor:offset:), (IMP)swz_heap_newtex,
          &g_heap_newtex_orig },
        { @selector(newBufferWithLength:options:offset:), (IMP)swz_heap_newbuf,
          &g_heap_newbuf_orig },
    };
    std::lock_guard<std::mutex> lk(g_swz_mutex);
    if (g_heap_newtex_orig.count(heapClass))
        return;
    for (auto& j : jobs) {
        Method m = class_getInstanceMethod(heapClass, j.sel);
        if (!m) {
            DMN_WARN("share: heap class %s missing %s", class_getName(heapClass),
                     sel_getName(j.sel));
            continue;
        }
        (*j.store)[heapClass] = method_setImplementation(m, j.repl);
    }
    DMN_INFO("share: swizzled heap-placed resources on %s",
             class_getName(heapClass));
}

/* Device heap path: -[dev newHeapWithDescriptor:] — wrap to catch heap classes
 * as they appear, so their placed-texture creator is hooked before D3DMetal
 * ever allocates a placed resource from them. */
id swz_dev_newheap(id self, SEL _cmd, MTLHeapDescriptor* desc) {
    IMP orig = lookup_orig(g_dev_newheap_orig, object_getClass(self));
    id heap = orig
        ? ((id (*)(id, SEL, MTLHeapDescriptor*))orig)(self, _cmd, desc)
        : nil;
    if (heap)
        ensure_heap_class_swizzled(object_getClass(heap));
    return heap;
}

/* Same locking discipline as ensure_heap_class_swizzled. */
void swizzle_device_class(Class cls) {
    if (!cls)
        return;
    struct { SEL sel; IMP repl; std::unordered_map<Class, IMP>* store; } jobs[] = {
        { @selector(newTextureWithDescriptor:), (IMP)swz_dev_newtex,
          &g_dev_newtex_orig },
        { @selector(newBufferWithLength:options:), (IMP)swz_dev_newbuf,
          &g_dev_newbuf_orig },
        { @selector(newHeapWithDescriptor:), (IMP)swz_dev_newheap,
          &g_dev_newheap_orig },
    };
    std::lock_guard<std::mutex> lk(g_swz_mutex);
    if (g_dev_newtex_orig.count(cls))
        return; /* already done */
    for (auto& j : jobs) {
        Method m = class_getInstanceMethod(cls, j.sel);
        if (!m) {
            DMN_WARN("share: device class %s missing %s", class_getName(cls),
                     sel_getName(j.sel));
            continue;
        }
        (*j.store)[cls] = method_setImplementation(m, j.repl);
    }
    DMN_INFO("share: swizzled Metal device class %s", class_getName(cls));
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
    t_arm.existing_size = size; /* requested length */
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
        std::lock_guard<std::mutex> lk(g_swz_mutex);
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

void* dmn_share_map_fd(int fd, size_t size) {
    size_t aligned = page_align(size);
    void* p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? nullptr : p;
}

void dmn_share_unmap(void* ptr, size_t size) {
    if (ptr)
        munmap(ptr, page_align(size));
}
