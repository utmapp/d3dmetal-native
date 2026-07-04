/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTSwapchain — the presentation core. One instance per
 * CreateSwapchainInterface call; D3DMetal owns it and deletes it through
 * the virtual destructor.
 *
 * Backends:
 *   View/Layer  — drawables come from the window's CAMetalLayer
 *                 (nextDrawable on D3DMetal's worker thread).
 *   Exported    — the library allocates shared-memory-backed linear
 *                 MTLTextures, rotates through them, and wraps each in a
 *                 private CAMetalDrawable-conforming object (the same
 *                 shape as D3DMetal's own NeptuneCAMetalDrawable, which
 *                 its present path expects); the embedder sees only fds,
 *                 strides, and slot indices via the three callbacks in
 *                 dmn_exported_swapchain_config.
 *
 * Ownership contract expected by D3DMetal:
 * GetDrawableForHWND returns +1; D3DMetal releases the drawable.
 *
 * Threading: GFXT calls arrive on D3DMetal worker threads. CALayer
 * property setters are thread-safe; layer *hierarchy* attachment already
 * happened on the app's main thread in dmn_window_create_for_view, so
 * nothing here ever hops to the main thread (deadlock avoidance: the
 * app's main thread may be blocked inside CreateSwapChainForHwnd).
 */

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <objc/runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <mutex>

#include "dmn_gfxt.h"
#include "dmn_share.h"
#include "dmn_window_internal.h"

#if __has_feature(objc_arc)
#error "dmn_gfxt_swapchain.mm must be compiled with -fno-objc-arc"
#endif

/* == Dummy view (GetViewForHWND fallback for view-less windows) ========== */

@interface DmnDummyView : NSView
@end

@implementation DmnDummyView
- (BOOL)isOpaque { return YES; }
- (BOOL)wantsUpdateLayer { return YES; }
- (void)drawRect:(NSRect)dirtyRect { (void)dirtyRect; }
@end

namespace {

NSView* g_dummy_view = nil;
std::once_flag g_dummy_view_once;

void create_dummy_view() {
    g_dummy_view =
        [[DmnDummyView alloc] initWithFrame:NSMakeRect(0, 0, 1, 1)];
}

NSView* ensure_dummy_view() {
    std::call_once(g_dummy_view_once, [] {
        if ([NSThread isMainThread]) {
            create_dummy_view();
        } else {
            /* dispatch_sync to main would deadlock if main is blocked in
             * swapchain creation; async + spin is unnecessary because
             * NSView allocation off-main merely warns. Create directly. */
            create_dummy_view();
        }
    });
    return g_dummy_view;
}

bool present_fallback_enabled() {
    static const bool enabled = getenv("DMN_PRESENT_FALLBACK") != nullptr;
    return enabled;
}

/* DXGI_FORMAT -> MTLPixelFormat for CAMetalLayer. CAMetalLayer only accepts
 * BGRA8Unorm(_sRGB), RGBA16Float, RGB10A2Unorm, BGR10A2Unorm and the XR
 * variants; RGBA8 swapchains ride a BGRA8 layer (D3DMetal blits its
 * internal backbuffer into the drawable, converting as needed). */
MTLPixelFormat layer_format_for_dxgi(DXGI_FORMAT format, bool* known) {
    *known = true;
    switch (format) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return MTLPixelFormatBGRA8Unorm;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return MTLPixelFormatBGRA8Unorm_sRGB;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return MTLPixelFormatRGB10A2Unorm;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return MTLPixelFormatRGBA16Float;
    default:
        *known = false;
        return MTLPixelFormatBGRA8Unorm;
    }
}

/* == Exported backend: format policy ===================================== */
/* The exported image set is a linear shm texture a consumer maps by fd, so
 * the presentable subset is the 32-bit UNORM family (the X8 variant rides
 * an A8 Metal texture with undefined alpha). D3DMetal's drawable blit
 * converts from whatever backbuffer format the app requested. */

bool exported_metal_format(DXGI_FORMAT format, MTLPixelFormat* out) {
    switch (format) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
        *out = MTLPixelFormatBGRA8Unorm;
        return true;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        *out = MTLPixelFormatRGBA8Unorm;
        return true;
    default:
        return false;
    }
}

/* Allocation format: the desc format when presentable, else the embedder's
 * preference, else BGRA8. */
DXGI_FORMAT exported_pick_format(DXGI_FORMAT desc_format,
                                 DXGI_FORMAT preferred) {
    MTLPixelFormat unused;
    if (exported_metal_format(desc_format, &unused))
        return desc_format;
    if (preferred != DXGI_FORMAT_UNKNOWN &&
        exported_metal_format(preferred, &unused))
        return preferred;
    return DXGI_FORMAT_B8G8R8A8_UNORM;
}

/* One shared-memory-backed linear texture. Fills *out (fd owned by the
 * texture through the buffer deallocator). Returns +1 texture or nil. */
id<MTLTexture> exported_alloc_texture(id<MTLDevice> device, uint32_t width,
                                      uint32_t height, MTLPixelFormat fmt,
                                      dmn_exported_image* out) {
    NSUInteger row_align =
        [device minimumLinearTextureAlignmentForPixelFormat:fmt];
    if (row_align == 0)
        row_align = 256;
    const size_t stride = ((size_t)width * 4 + (size_t)row_align - 1) &
                          ~((size_t)row_align - 1);
    const size_t logical = stride * (size_t)height;
    const size_t aligned = dmn_share_page_align(logical);

    int fd = dmn_share_anon_file(aligned);
    if (fd < 0) {
        DMN_ERROR("exported: anon file (%zu) failed", aligned);
        return nil;
    }
    void* ptr = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, 0);
    if (ptr == MAP_FAILED) {
        DMN_ERROR("exported: mmap(%zu) failed", aligned);
        close(fd);
        return nil;
    }

    /* The buffer's deallocator owns the mapping and the fd; both live
     * exactly as long as the texture (which pins the buffer below). */
    id<MTLBuffer> buf =
        [device newBufferWithBytesNoCopy:ptr
                                  length:aligned
                                 options:MTLResourceStorageModeShared
                             deallocator:^(void* p, NSUInteger len) {
                                 munmap(p, len);
                                 close(fd);
                             }];
    if (!buf) {
        DMN_ERROR("exported: newBufferWithBytesNoCopy(%zu) failed", aligned);
        munmap(ptr, aligned);
        close(fd);
        return nil;
    }

    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:fmt
                                     width:width
                                    height:height
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;

    id<MTLTexture> tex = [buf newTextureWithDescriptor:td
                                                offset:0
                                           bytesPerRow:stride];
    if (!tex) {
        DMN_ERROR("exported: linear texture (fmt=%lu %ux%u stride=%zu) failed",
                  (unsigned long)fmt, width, height, stride);
        [buf release];
        return nil;
    }

    /* A linear texture does not necessarily retain its source buffer on all
     * Metal versions; pin it so buffer (and backing) lifetime is exactly
     * the texture's. (Same rule as dmn_share_metal.mm.) */
    static const void* kBackingKey = &kBackingKey;
    objc_setAssociatedObject(tex, kBackingKey, buf,
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [buf release];

    out->fd = fd;
    out->stride = (uint32_t)stride;
    out->size = (uint64_t)logical;
    return tex;
}

/* GPU-done fence fd: the read end of a pipe that a tracking command buffer's
 * completion handler writes one byte to on GPU completion.
 *
 * This fd is handed up to the VMM and passed between processes with SCM_RIGHTS.
 * macOS refuses to send kqueue descriptors that way (sendmsg -> EINVAL), so a
 * kqueue cannot be used here even though it would be the natural primitive; a
 * pipe read end is both pollable and passable.  Once the byte is written the
 * read end stays readable until consumed, matching the one-shot fence contract
 * (the consumer polls for readable, then closes). */
int exported_gpu_done_fd(id<MTLCommandQueue> queue) {
    if (!queue)
        return -1;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb)
        return -1;

    int fds[2];
    if (pipe(fds) < 0) {
        [cb commit];
        return -1;
    }
    /* The write end must outlive this call (until the handler fires); both
     * ends are CLOEXEC so a forked render worker cannot strand the fence. */
    for (int i = 0; i < 2; i++) {
        int flags = fcntl(fds[i], F_GETFD);
        if (flags >= 0)
            fcntl(fds[i], F_SETFD, flags | FD_CLOEXEC);
    }

    const int wr = fds[1];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        (void)done;
        const uint8_t token = 1;
        ssize_t n;
        do { n = write(wr, &token, 1); } while (n < 0 && errno == EINTR);
        close(wr);
    }];
    [cb commit];
    return fds[0];
}

} // namespace

void dmn_swapchain_prepare_dummy_view() {
    if ([NSThread isMainThread]) {
        ensure_dummy_view();
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{ ensure_dummy_view(); });
    }
}

/* == Exported-backend state ============================================== */
/* One per swapchain (created at InitializeForHWND). Drawables retain it, so
 * a late present after the swapchain (or the app's window handle) is gone
 * still resolves. */

@interface DmnExportedSwapchain : NSObject {
@public
    dmn_exported_swapchain_config _cfg;
    id<MTLDevice> _device;      /* retained */
    id<MTLCommandQueue> _queue; /* retained; lazy (gpu-done tracking) */
    NSLock* _lock;              /* guards everything below */
    id<MTLTexture> _textures[DMN_EXPORTED_MAX_IMAGES]; /* retained */
    dmn_exported_image_set _set;
    uint32_t _next;
    uint64_t _frame_id;
}
- (instancetype)initWithDevice:(id<MTLDevice>)device
                        config:(const dmn_exported_swapchain_config*)cfg;
/* count == 0 keeps the current image count (ResizeBacking). Publishes the
 * new set via on_images_changed before returning true. */
- (bool)configureWidth:(uint32_t)width height:(uint32_t)height
                format:(DXGI_FORMAT)format count:(uint32_t)count;
- (id<MTLTexture>)acquireSlot:(uint32_t*)outSlot; /* autoreleased */
- (void)presentSlot:(uint32_t)slot;
@end

@implementation DmnExportedSwapchain

- (instancetype)initWithDevice:(id<MTLDevice>)device
                        config:(const dmn_exported_swapchain_config*)cfg {
    if (!(self = [super init]))
        return nil;
    _device = [device retain];
    _cfg = *cfg;
    _lock = [[NSLock alloc] init];
    return self;
}

- (void)dealloc {
    for (uint32_t i = 0; i < DMN_EXPORTED_MAX_IMAGES; i++)
        [_textures[i] release];
    [_queue release];
    [_device release];
    [_lock release];
    [super dealloc];
}

- (bool)configureWidth:(uint32_t)width height:(uint32_t)height
                format:(DXGI_FORMAT)format count:(uint32_t)count {
    MTLPixelFormat mtl_format;
    if (!exported_metal_format(format, &mtl_format))
        return false; /* callers pass exported_pick_format results */

    if (count == 0) {
        [_lock lock];
        count = _set.num_images;
        [_lock unlock];
        if (count == 0)
            count = 2;
    }
    if (count > DMN_EXPORTED_MAX_IMAGES)
        count = DMN_EXPORTED_MAX_IMAGES;

    /* Build the new generation outside the lock; leave the old set live on
     * failure. */
    dmn_exported_image_set set;
    memset(&set, 0, sizeof(set));
    set.width = width;
    set.height = height;
    set.dxgi_format = (uint32_t)format;
    set.num_images = count;

    id<MTLTexture> textures[DMN_EXPORTED_MAX_IMAGES] = {};
    for (uint32_t i = 0; i < count; i++) {
        textures[i] = exported_alloc_texture(_device, width, height,
                                             mtl_format, &set.images[i]);
        if (!textures[i]) {
            for (uint32_t j = 0; j < i; j++)
                [textures[j] release];
            return false;
        }
    }

    [_lock lock];
    for (uint32_t i = 0; i < DMN_EXPORTED_MAX_IMAGES; i++) {
        [_textures[i] release];
        _textures[i] = textures[i];
    }
    _set = set;
    _next = 0;
    [_lock unlock];

    DMN_INFO("exported: published %u images, %ux%u, dxgi=%u",
             count, width, height, (unsigned)format);
    /* Synchronous publish; the fds stay valid through the callback because
     * _textures holds the references. */
    _cfg.on_images_changed(_cfg.ctx, &set);
    return true;
}

- (id<MTLTexture>)acquireSlot:(uint32_t*)outSlot {
    [_lock lock];
    uint32_t count = _set.num_images;
    if (!count) {
        [_lock unlock];
        return nil;
    }
    uint32_t slot = _next % count;
    _next = (_next + 1) % count;
    [_lock unlock];

    _cfg.on_acquire(_cfg.ctx, slot); /* consumer backpressure; may block */

    /* Re-read under the lock: a concurrent reconfigure may have swapped or
     * shrunk the set while we were blocked. */
    id<MTLTexture> texture = nil;
    [_lock lock];
    if (slot < _set.num_images)
        texture = [[_textures[slot] retain] autorelease];
    [_lock unlock];

    if (texture && outSlot)
        *outSlot = slot;
    return texture;
}

- (void)presentSlot:(uint32_t)slot {
    [_lock lock];
    uint64_t frame_id = ++_frame_id;
    if (!_queue)
        _queue = [_device newCommandQueue];
    id<MTLCommandQueue> queue = _queue;
    [_lock unlock];

    int gpu_done_fd = exported_gpu_done_fd(queue);
    _cfg.on_present(_cfg.ctx, slot, frame_id, gpu_done_fd);
}

@end

/* == Exported-backend drawable =========================================== */

@interface DmnCAMetalDrawable : NSObject <CAMetalDrawable> {
@public
    id<MTLTexture>        _texture;  /* retained */
    NSUInteger            _drawableID;
    DmnExportedSwapchain* _exported; /* retained */
    NSMutableArray*       _handlers;
    NSMutableArray*       _scheduledHandlers;
    CFTimeInterval        _presentedTime;
}
@end

@implementation DmnCAMetalDrawable
- (id<MTLTexture>)texture { return _texture; }
- (CAMetalLayer*)layer { return nil; }
- (NSUInteger)drawableID { return _drawableID; }
- (CFTimeInterval)presentedTime { return _presentedTime; }

- (void)addPresentedHandler:(MTLDrawablePresentedHandler)block {
    if (!_handlers)
        _handlers = [[NSMutableArray alloc] init];
    id copy = [block copy];
    [_handlers addObject:copy];
    [copy release];
}

/* Private MTLDrawable method the Metal command buffer calls from
 * -presentDrawable: on current macOS; without it the present path throws
 * doesNotRecognizeSelector. Fired at present time (we have no scheduled/
 * completed distinction for exported textures). */
- (void)addPresentScheduledHandler:(MTLDrawablePresentedHandler)block {
    if (!_scheduledHandlers)
        _scheduledHandlers = [[NSMutableArray alloc] init];
    id copy = [block copy];
    [_scheduledHandlers addObject:copy];
    [copy release];
}

- (void)present {
    [self presentAtTime:CACurrentMediaTime()];
}

- (void)presentAtTime:(CFTimeInterval)presentationTime {
    _presentedTime = presentationTime;
    for (id b in _scheduledHandlers)
        ((MTLDrawablePresentedHandler)b)(self);
    [_scheduledHandlers removeAllObjects];
    [_exported presentSlot:(uint32_t)_drawableID];
    for (id b in _handlers)
        ((MTLDrawablePresentedHandler)b)(self);
    [_handlers removeAllObjects];
}

- (void)presentAfterMinimumDuration:(CFTimeInterval)duration {
    (void)duration;
    [self presentAtTime:CACurrentMediaTime()];
}

- (void)dealloc {
    [_texture release];
    [_exported release];
    [_handlers release];
    [_scheduledHandlers release];
    [super dealloc];
}
@end

/* == DmnGFXTSwapchain ===================================================== */

DmnGFXTSwapchain::DmnGFXTSwapchain(void* mtlDevice)
    : mtl_device_([(id<MTLDevice>)mtlDevice retain]),
      last_drawable_(nullptr), present_layer_(nullptr), exported_(nullptr) {}

DmnGFXTSwapchain::~DmnGFXTSwapchain() {
    DMN_TRACE("swapchain %p destroyed", (void*)this);
    [(id)last_drawable_ release];
    [(CAMetalLayer*)present_layer_ release];
    [(DmnExportedSwapchain*)exported_ release];
    [(id<MTLDevice>)mtl_device_ release];
}

bool DmnGFXTSwapchain::InitializeForHWND(void* hwnd,
                                         const DXGI_SWAP_CHAIN_DESC1* desc,
                                         D3D12_RESOURCE_DESC& out) {
    DmnWindow* w = dmn_window_lookup(hwnd);
    if (!w) {
        DMN_ERROR("InitializeForHWND(%p): unknown hwnd — pass "
                  "dmn_window_get_hwnd() as the HWND", hwnd);
        return false;
    }
    if (!desc) {
        DMN_ERROR("InitializeForHWND(%p): NULL desc", hwnd);
        return false;
    }

    DMN_INFO("InitializeForHWND(hwnd=%p, %ux%u fmt=%u buffers=%u "
             "swapEffect=%u flags=0x%x)",
             hwnd, desc->Width, desc->Height, (unsigned)desc->Format,
             desc->BufferCount, (unsigned)desc->SwapEffect, desc->Flags);

    /* Resolve format and size, falling back to the window's stored state. */
    DXGI_FORMAT format = desc->Format;
    if (format == DXGI_FORMAT_UNKNOWN) {
        format = (DXGI_FORMAT)w->dxgi_format.load(std::memory_order_relaxed);
        if (format == DXGI_FORMAT_UNKNOWN)
            format = DXGI_FORMAT_B8G8R8A8_UNORM;
    }
    uint32_t width = desc->Width ? desc->Width
                                 : w->width.load(std::memory_order_relaxed);
    uint32_t height = desc->Height ? desc->Height
                                   : w->height.load(std::memory_order_relaxed);
    if (!width || !height) {
        DMN_ERROR("InitializeForHWND(%p): zero-sized backing", hwnd);
        return false;
    }

    if (w->backend == DmnWindowBackend::Exported) {
        DmnExportedSwapchain* ex = (DmnExportedSwapchain*)exported_;
        if (!ex) {
            /* Keep the whole exported state on the swapchain so late
             * presents (after dmn_window_destroy) skip the hwnd lookup —
             * the exported analog of present_layer_ below. */
            ex = [[DmnExportedSwapchain alloc]
                initWithDevice:(id<MTLDevice>)mtl_device_
                        config:&w->cfg];
            if (!ex)
                return false;
            exported_ = ex;
        }
        DXGI_FORMAT alloc_format = exported_pick_format(
            format, (DXGI_FORMAT)w->cfg.preferred_dxgi_format);
        if (alloc_format != format)
            DMN_INFO("InitializeForHWND(%p): exporting fmt=%u as dxgi=%u",
                     hwnd, (unsigned)format, (unsigned)alloc_format);
        uint32_t count = w->cfg.preferred_image_count;
        if (!count) {
            count = desc->BufferCount;
            if (count < 2)
                count = 2;
        }
        if (![ex configureWidth:width height:height format:alloc_format
                          count:count]) {
            DMN_ERROR("InitializeForHWND(%p): exported configure failed",
                      hwnd);
            return false;
        }
    } else {
        CAMetalLayer* layer = w->layer;
        bool known = false;
        MTLPixelFormat mtl_format = layer_format_for_dxgi(format, &known);
        if (!known)
            DMN_WARN("InitializeForHWND(%p): DXGI format %u has no "
                     "CAMetalLayer mapping; using BGRA8",
                     hwnd, (unsigned)format);

        layer.device = (id<MTLDevice>)mtl_device_;
        layer.pixelFormat = mtl_format;
        layer.drawableSize = CGSizeMake(width, height);
        layer.framebufferOnly = NO; /* D3DMetal blits into the drawable */
        NSUInteger buffers = desc->BufferCount;
        layer.maximumDrawableCount =
            buffers < 2 ? 2 : (buffers > 3 ? 3 : buffers);
        if (const char* vsync = getenv("DMN_VSYNC"))
            layer.displaySyncEnabled = atoi(vsync) != 0;
        layer.allowsNextDrawableTimeout = YES;

        /* Take our own reference to the presentation layer so drawable
         * acquisition does not depend on the app's window still being
         * registered (see present_layer_ in the header). */
        if (present_layer_ != (void*)layer) {
            [(CAMetalLayer*)present_layer_ release];
            present_layer_ = [layer retain];
        }
    }

    w->width.store(width, std::memory_order_relaxed);
    w->height.store(height, std::memory_order_relaxed);
    w->dxgi_format.store((uint32_t)format, std::memory_order_relaxed);

    memset(&out, 0, sizeof(out));
    out.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    out.Alignment        = 0;
    out.Width            = width;
    out.Height           = height;
    out.DepthOrArraySize = 1;
    out.MipLevels        = 1;
    out.Format           = format;
    out.SampleDesc.Count = 1;
    out.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    out.Flags            = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    DMN_INFO("InitializeForHWND(%p) -> %ux%u fmt=%u OK", hwnd, width, height,
             (unsigned)format);
    return true;
}

void* DmnGFXTSwapchain::GetViewForHWND(void* hwnd) {
    DmnWindow* w = dmn_window_lookup(hwnd);
    NSView* view = (w && w->view) ? w->view : ensure_dummy_view();
    DMN_TRACE("GetViewForHWND(%p) = %p", hwnd, (void*)view);
    return (void*)view; /* borrowed */
}

void* DmnGFXTSwapchain::GetDrawableForHWND(void* hwnd) {
    id drawable = nil;
    @autoreleasepool {
        /* Prefer the swapchain's own presentation state (the retained layer
         * or the exported state): both outlive the app's window handle, so
         * an async present that fires after dmn_window_destroy still
         * resolves a drawable instead of aborting D3DMetal on a failed
         * lookup. The window table is only needed before InitializeForHWND. */
        CAMetalLayer* layer = (CAMetalLayer*)present_layer_;
        DmnExportedSwapchain* ex = (DmnExportedSwapchain*)exported_;
        if (!layer && !ex) {
            DmnWindow* w = dmn_window_lookup(hwnd);
            if (!w || w->backend == DmnWindowBackend::Exported || !w->layer) {
                DMN_WARN("GetDrawableForHWND(%p): no presentation state "
                         "(before InitializeForHWND?)", hwnd);
                return nullptr;
            }
            layer = w->layer;
        }

        if (ex) {
            uint32_t slot = 0;
            id<MTLTexture> tex = [ex acquireSlot:&slot];
            if (!tex) {
                DMN_WARN("GetDrawableForHWND(%p): no exported image "
                         "available", hwnd);
                return nullptr;
            }
            DmnCAMetalDrawable* d = [[DmnCAMetalDrawable alloc] init];
            d->_texture = [tex retain];
            d->_drawableID = slot;
            d->_exported = [ex retain];
            drawable = d; /* +1 from alloc */
        } else {
            id<CAMetalDrawable> d = [layer nextDrawable];
            if (!d) {
                DMN_WARN("GetDrawableForHWND(%p): nextDrawable returned nil",
                         hwnd);
                return nullptr;
            }
            drawable = [d retain]; /* +1; D3DMetal releases */
        }
    }

    if (present_fallback_enabled()) {
        /* Track the most recent drawable so Present(hwnd) can present it if
         * D3DMetal turns out not to present drawables itself. */
        [(id)last_drawable_ release];
        last_drawable_ = [drawable retain];
    }

    DMN_TRACE("GetDrawableForHWND(%p) = %p", hwnd, (void*)drawable);
    return (void*)drawable;
}

bool DmnGFXTSwapchain::ResizeWindow(void* hwnd, uint32_t width,
                                    uint32_t height) {
    /* The native app owns its NSWindow; D3DMetal cannot drive its size. */
    DMN_INFO("ResizeWindow(%p, %ux%u) ignored", hwnd, width, height);
    return true;
}

bool DmnGFXTSwapchain::ResizeBacking(void* hwnd, uint32_t width,
                                     uint32_t height, DXGI_FORMAT format) {
    DmnWindow* w = dmn_window_lookup(hwnd);
    if (!w) {
        DMN_WARN("ResizeBacking(%p): unknown hwnd", hwnd);
        return false;
    }
    if (!width || !height) {
        DMN_WARN("ResizeBacking(%p): zero size %ux%u", hwnd, width, height);
        return false;
    }

    DXGI_FORMAT resolved = format != DXGI_FORMAT_UNKNOWN
        ? format
        : (DXGI_FORMAT)w->dxgi_format.load(std::memory_order_relaxed);

    DMN_INFO("ResizeBacking(%p, %ux%u, fmt=%u)", hwnd, width, height,
             (unsigned)resolved);

    if (w->backend == DmnWindowBackend::Exported) {
        DmnExportedSwapchain* ex = (DmnExportedSwapchain*)exported_;
        if (!ex) {
            DMN_WARN("ResizeBacking(%p): before InitializeForHWND", hwnd);
            return false;
        }
        DXGI_FORMAT alloc_format = exported_pick_format(
            resolved, (DXGI_FORMAT)w->cfg.preferred_dxgi_format);
        if (![ex configureWidth:width height:height format:alloc_format
                          count:0])
            return false;
    } else {
        CAMetalLayer* layer = w->layer;
        if (format != DXGI_FORMAT_UNKNOWN) {
            bool known = false;
            MTLPixelFormat mtl_format = layer_format_for_dxgi(resolved, &known);
            if (known)
                layer.pixelFormat = mtl_format;
        }
        layer.drawableSize = CGSizeMake(width, height);
    }

    w->width.store(width, std::memory_order_relaxed);
    w->height.store(height, std::memory_order_relaxed);
    w->dxgi_format.store((uint32_t)resolved, std::memory_order_relaxed);
    return true;
}

bool DmnGFXTSwapchain::SetFullscreen(void* hwnd, bool currentlyFullscreen,
                                     bool wantFullscreen, uint32_t x,
                                     uint32_t y, uint32_t width,
                                     uint32_t height) {
    /* Accept-and-ignore: fullscreen transitions belong to the native app. */
    DMN_INFO("SetFullscreen(%p, %d->%d, pos=%u,%u size=%ux%u) ignored", hwnd,
             currentlyFullscreen, wantFullscreen, x, y, width, height);
    return true;
}

bool DmnGFXTSwapchain::Present(void* hwnd) {
    /* D3DMetal presents drawables through its own command buffers; this
     * host-side hook only needs to succeed. The fallback
     * (DMN_PRESENT_FALLBACK=1) presents the last vended drawable here
     * instead, in case that assumption fails on some D3DMetal build. */
    DMN_TRACE("Present(%p)", hwnd);
    if (present_fallback_enabled() && last_drawable_) {
        id<CAMetalDrawable> d = (id<CAMetalDrawable>)last_drawable_;
        last_drawable_ = nullptr;
        [d present];
        [d release];
    }
    return true;
}
