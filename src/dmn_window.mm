/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * dmn_window_* — window objects and the pseudo-HWND table.
 *
 * HWND values handed to D3D/DXGI are small slot-based pseudo-handles
 * ((slot + 1) << 4), never raw pointers: Wine HWNDs are 32-bit, and any
 * Wine-era handle assumption inside D3DMetal would truncate a 64-bit
 * pointer. Lookups validate the slot and the alive flag.
 *
 * Manual reference counting (-fno-objc-arc): every retain/release is
 * explicit because raw +1 pointers cross the GFXT vtable boundary.
 */

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cstdlib>
#include <mutex>
#include <vector>

#include "dmn_window_internal.h"

#if __has_feature(objc_arc)
#error "dmn_window.mm must be compiled with -fno-objc-arc"
#endif

namespace {

std::mutex g_window_mutex;
std::vector<DmnWindow*> g_windows;

void* encode_hwnd(uint32_t slot) {
    return (void*)(uintptr_t)((slot + 1) << 4);
}

DmnWindow* register_window(DmnWindow* w) {
    std::lock_guard<std::mutex> lock(g_window_mutex);
    for (uint32_t i = 0; i < g_windows.size(); i++) {
        if (!g_windows[i]) {
            g_windows[i] = w;
            w->slot = i;
            return w;
        }
    }
    w->slot = (uint32_t)g_windows.size();
    g_windows.push_back(w);
    return w;
}

double initial_scale(NSView* view) {
    if (!getenv("DMN_RETINA"))
        return 1.0; /* 1x for a predictable point==pixel mapping by default */
    NSWindow* win = view.window;
    return win ? win.backingScaleFactor
               : (NSScreen.mainScreen ? NSScreen.mainScreen.backingScaleFactor
                                      : 2.0);
}

} // namespace

DmnWindow* dmn_window_lookup(void* hwnd) {
    uintptr_t v = (uintptr_t)hwnd;
    if (!v || (v & 0xf))
        return nullptr;
    uint32_t slot = (uint32_t)(v >> 4) - 1;
    std::lock_guard<std::mutex> lock(g_window_mutex);
    if (slot >= g_windows.size())
        return nullptr;
    DmnWindow* w = g_windows[slot];
    if (!w || !w->alive.load(std::memory_order_acquire))
        return nullptr;
    return w;
}

extern "C" dmn_window_t dmn_window_create_for_view(dmn_nsview_t view) {
    if (!view) {
        DMN_ERROR("dmn_window_create_for_view: view is NULL");
        return nullptr;
    }
    if (![NSThread isMainThread])
        DMN_WARN("dmn_window_create_for_view called off the main thread; "
                 "AppKit requires layer attachment on the main thread");

    auto* w = new DmnWindow();
    w->backend = DmnWindowBackend::View;
    w->view = [view retain];

    CAMetalLayer* layer = [[CAMetalLayer alloc] init]; /* +1, ours */
    double scale = initial_scale(view);
    layer.contentsScale = scale;
    view.wantsLayer = YES;
    view.layer = layer;

    NSRect bounds = view.bounds;
    uint32_t px_w = (uint32_t)(bounds.size.width * scale + 0.5);
    uint32_t px_h = (uint32_t)(bounds.size.height * scale + 0.5);
    if (px_w && px_h)
        layer.drawableSize = CGSizeMake(px_w, px_h);

    w->layer = layer;
    w->width.store(px_w, std::memory_order_relaxed);
    w->height.store(px_h, std::memory_order_relaxed);
    w->scale.store(scale, std::memory_order_relaxed);

    register_window(w);
    DMN_INFO("window: created for view %p (hwnd=%p, %ux%u px, scale=%.1f)",
             (void*)view, encode_hwnd(w->slot), px_w, px_h, scale);
    return (dmn_window_t)w;
}

extern "C" dmn_window_t dmn_window_create_for_layer(dmn_metal_layer_t layer) {
    if (!layer) {
        DMN_ERROR("dmn_window_create_for_layer: layer is NULL");
        return nullptr;
    }
    auto* w = new DmnWindow();
    w->backend = DmnWindowBackend::Layer;
    w->layer = [layer retain];

    CGSize ds = layer.drawableSize;
    uint32_t px_w = (uint32_t)ds.width;
    uint32_t px_h = (uint32_t)ds.height;
    if (!px_w || !px_h) {
        CGRect b = layer.bounds;
        px_w = (uint32_t)(b.size.width * layer.contentsScale + 0.5);
        px_h = (uint32_t)(b.size.height * layer.contentsScale + 0.5);
    }
    w->width.store(px_w, std::memory_order_relaxed);
    w->height.store(px_h, std::memory_order_relaxed);
    w->scale.store(layer.contentsScale, std::memory_order_relaxed);

    register_window(w);
    DMN_INFO("window: created for layer %p (hwnd=%p, %ux%u px)",
             (void*)layer, encode_hwnd(w->slot), px_w, px_h);
    return (dmn_window_t)w;
}

extern "C" dmn_window_t
dmn_window_create_with_callbacks(const dmn_window_callbacks* callbacks,
                                 uint32_t width, uint32_t height) {
    if (!callbacks || callbacks->struct_size < sizeof(dmn_window_callbacks) ||
        !callbacks->acquire_texture) {
        DMN_ERROR("dmn_window_create_with_callbacks: invalid callbacks");
        return nullptr;
    }
    auto* w = new DmnWindow();
    w->backend = DmnWindowBackend::Callbacks;
    w->cb = *callbacks;
    w->width.store(width, std::memory_order_relaxed);
    w->height.store(height, std::memory_order_relaxed);

    register_window(w);
    DMN_INFO("window: created with callbacks (hwnd=%p, %ux%u px)",
             encode_hwnd(w->slot), width, height);
    return (dmn_window_t)w;
}

extern "C" void* dmn_window_get_hwnd(dmn_window_t window) {
    auto* w = (DmnWindow*)window;
    return w ? encode_hwnd(w->slot) : nullptr;
}

extern "C" dmn_window_t dmn_window_from_hwnd(void* hwnd) {
    return (dmn_window_t)dmn_window_lookup(hwnd);
}

extern "C" dmn_metal_layer_t dmn_window_get_metal_layer(dmn_window_t window) {
    auto* w = (DmnWindow*)window;
    return w ? w->layer : nil;
}

extern "C" void dmn_window_set_size(dmn_window_t window, uint32_t pixel_width,
                                    uint32_t pixel_height,
                                    double contents_scale) {
    auto* w = (DmnWindow*)window;
    if (!w || !pixel_width || !pixel_height)
        return;
    w->width.store(pixel_width, std::memory_order_relaxed);
    w->height.store(pixel_height, std::memory_order_relaxed);
    if (contents_scale > 0.0)
        w->scale.store(contents_scale, std::memory_order_relaxed);
    if (w->layer) {
        /* CALayer property setters are thread-safe. */
        if (contents_scale > 0.0)
            w->layer.contentsScale = contents_scale;
        w->layer.drawableSize = CGSizeMake(pixel_width, pixel_height);
    }
    DMN_DEBUG("window: set_size(hwnd=%p, %ux%u, scale=%.2f)",
              encode_hwnd(w->slot), pixel_width, pixel_height, contents_scale);
}

extern "C" void dmn_window_destroy(dmn_window_t window) {
    auto* w = (DmnWindow*)window;
    if (!w)
        return;
    DMN_INFO("window: destroy(hwnd=%p)", encode_hwnd(w->slot));
    w->alive.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(g_window_mutex);
        if (w->slot < g_windows.size() && g_windows[w->slot] == w)
            g_windows[w->slot] = nullptr;
    }

    NSView* view = w->view;
    CAMetalLayer* layer = w->layer;
    w->view = nil;
    w->layer = nil;
    if (view) {
        /* Detach the library-owned layer on the main thread (AppKit rule);
         * teardown need not be synchronous. */
        dispatch_async(dispatch_get_main_queue(), ^{
            if (view.layer == layer) {
                view.wantsLayer = NO;
                view.layer = nil;
            }
            [layer release];
            [view release];
        });
    } else if (layer) {
        [layer release];
    }
    delete w;
}
