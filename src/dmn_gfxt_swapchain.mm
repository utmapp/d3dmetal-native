/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTSwapchain — the presentation core. One instance per
 * CreateSwapchainInterface call; D3DMetal owns it and deletes it through
 * the virtual destructor.
 *
 * Drawables come from the window's CAMetalLayer (nextDrawable on
 * D3DMetal's worker thread).
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

#include <cstdlib>
#include <cstring>
#include <mutex>

#include "dmn_gfxt.h"
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

} // namespace

void dmn_swapchain_prepare_dummy_view() {
    if ([NSThread isMainThread]) {
        ensure_dummy_view();
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{ ensure_dummy_view(); });
    }
}

/* == DmnGFXTSwapchain ===================================================== */

DmnGFXTSwapchain::DmnGFXTSwapchain(void* mtlDevice)
    : mtl_device_([(id<MTLDevice>)mtlDevice retain]),
      last_drawable_(nullptr), present_layer_(nullptr) {}

DmnGFXTSwapchain::~DmnGFXTSwapchain() {
    DMN_TRACE("swapchain %p destroyed", (void*)this);
    [(id)last_drawable_ release];
    [(CAMetalLayer*)present_layer_ release];
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
        /* Prefer the swapchain's own retained layer: it outlives the app's
         * window handle, so an async present that fires after
         * dmn_window_destroy still resolves a drawable instead of aborting
         * D3DMetal on a failed lookup. The window table is only needed
         * before InitializeForHWND. */
        CAMetalLayer* layer = (CAMetalLayer*)present_layer_;
        if (!layer) {
            DmnWindow* w = dmn_window_lookup(hwnd);
            if (!w || !w->layer) {
                DMN_WARN("GetDrawableForHWND(%p): no presentation state "
                         "(before InitializeForHWND?)", hwnd);
                return nullptr;
            }
            layer = w->layer;
        }

        id<CAMetalDrawable> d = [layer nextDrawable];
        if (!d) {
            DMN_WARN("GetDrawableForHWND(%p): nextDrawable returned nil",
                     hwnd);
            return nullptr;
        }
        drawable = [d retain]; /* +1; D3DMetal releases */
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

    CAMetalLayer* layer = w->layer;
    if (format != DXGI_FORMAT_UNKNOWN) {
        bool known = false;
        MTLPixelFormat mtl_format = layer_format_for_dxgi(resolved, &known);
        if (known)
            layer.pixelFormat = mtl_format;
    }
    layer.drawableSize = CGSizeMake(width, height);

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
