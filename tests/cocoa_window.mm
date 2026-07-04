/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Cocoa shell for the test programs. Manual reference counting.
 */

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <time.h>

#include "cocoa_window.h"

#if __has_feature(objc_arc)
#error "cocoa_window.mm must be compiled with -fno-objc-arc"
#endif

static volatile bool g_quit_requested = false;

/* == Frame-time stats (DMN_FRAMETIME=1) =================================== */
/* Deltas between cocoa_app_poll calls — every demo polls once per rendered
 * frame, so this is the app-observed frame time in either swapchain mode.
 * Summary printed at cocoa_app_shutdown. */
static bool g_ft_enabled = false;
static uint64_t g_ft_last_ns = 0;
static std::vector<double>* g_ft_ms = nullptr;

static uint64_t ft_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void ft_tick(void) {
    if (!g_ft_enabled)
        return;
    uint64_t now = ft_now_ns();
    if (g_ft_last_ns)
        g_ft_ms->push_back((double)(now - g_ft_last_ns) / 1e6);
    g_ft_last_ns = now;
}

static void ft_report(void) {
    if (!g_ft_enabled || !g_ft_ms)
        return;
    std::vector<double>& v = *g_ft_ms;
    const size_t warmup = 30;
    if (v.size() <= warmup + 10) {
        fprintf(stderr, "FRAMETIME: too few frames (%zu)\n", v.size());
        return;
    }
    std::vector<double> s(v.begin() + warmup, v.end());
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double d : s)
        sum += d;
    printf("FRAMETIME: n=%zu avg=%.2fms p50=%.2fms p95=%.2fms max=%.2fms\n",
           s.size(), sum / s.size(), s[s.size() / 2],
           s[(size_t)((double)s.size() * 0.95)], s.back());
}

@interface CocoaTestWindowDelegate : NSObject <NSWindowDelegate>
@end

@implementation CocoaTestWindowDelegate
- (BOOL)windowShouldClose:(NSWindow*)sender {
    (void)sender;
    g_quit_requested = true;
    return YES;
}
@end

@interface CocoaTestAppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation CocoaTestAppDelegate
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)app {
    (void)app;
    g_quit_requested = true;
    return NSTerminateCancel; /* unwind through the test's own loop */
}
@end

@class DmnCbPresenter;

struct cocoa_window {
    NSWindow* window;                  /* retained */
    NSView* view;                      /* retained */
    CocoaTestWindowDelegate* delegate; /* retained */
    DmnCbPresenter* presenter;         /* callback-backed mode only; leaked at
                                          destroy (late async presents may
                                          still call into it) */
};

static CocoaTestAppDelegate* g_app_delegate = nil;

static double scale_policy(NSWindow* window) {
    if (!getenv("DMN_RETINA"))
        return 1.0;
    return window ? window.backingScaleFactor : 2.0;
}

extern "C" bool cocoa_app_init(void) {
    if (getenv("DMN_FRAMETIME")) {
        g_ft_enabled = true;
        g_ft_ms = new std::vector<double>();
        g_ft_ms->reserve(4096);
    }
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        g_app_delegate = [[CocoaTestAppDelegate alloc] init];
        [app setDelegate:g_app_delegate];
        [app finishLaunching];
        [app activateIgnoringOtherApps:YES];
    }
    return true;
}

extern "C" bool cocoa_app_poll(void) {
    ft_tick();
    @autoreleasepool {
        for (;;) {
            NSEvent* ev = [NSApp nextEventMatchingMask:NSEventMaskAny
                                             untilDate:[NSDate distantPast]
                                                inMode:NSDefaultRunLoopMode
                                               dequeue:YES];
            if (ev == nil)
                break;
            [NSApp sendEvent:ev];
        }
    }
    return !g_quit_requested;
}

extern "C" void cocoa_app_shutdown(void) {
    ft_report();
    @autoreleasepool {
        [NSApp setDelegate:nil];
        [g_app_delegate release];
        g_app_delegate = nil;
    }
}

extern "C" cocoa_window_t* cocoa_window_create(const char* utf8_title,
                                               uint32_t width, uint32_t height,
                                               bool resizable) {
    @autoreleasepool {
        NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                           NSWindowStyleMaskMiniaturizable;
        if (resizable)
            style |= NSWindowStyleMaskResizable;

        /* Fixed origin keeps the screencapture rect reproducible. */
        NSRect content = NSMakeRect(100, 100, width, height);
        NSWindow* window =
            [[NSWindow alloc] initWithContentRect:content
                                        styleMask:style
                                          backing:NSBackingStoreBuffered
                                            defer:NO];
        window.releasedWhenClosed = NO;
        if (utf8_title)
            window.title = [NSString stringWithUTF8String:utf8_title];

        NSView* view = [[NSView alloc] initWithFrame:content];
        window.contentView = view;

        auto* delegate = [[CocoaTestWindowDelegate alloc] init];
        window.delegate = delegate;

        [window makeKeyAndOrderFront:nil];

        auto* w = new cocoa_window;
        w->window = window;
        w->view = view;
        w->delegate = delegate;
        w->presenter = nil;
        return w;
    }
}

extern "C" void* cocoa_window_content_view(cocoa_window_t* w) {
    return w ? (void*)w->view : nullptr;
}

extern "C" void cocoa_window_get_content_size(cocoa_window_t* w,
                                              uint32_t* out_w,
                                              uint32_t* out_h) {
    if (!w) {
        if (out_w) *out_w = 0;
        if (out_h) *out_h = 0;
        return;
    }
    @autoreleasepool {
        NSRect bounds = w->view.bounds;
        double scale = scale_policy(w->window);
        if (out_w) *out_w = (uint32_t)(bounds.size.width * scale + 0.5);
        if (out_h) *out_h = (uint32_t)(bounds.size.height * scale + 0.5);
    }
}

extern "C" void cocoa_window_set_title(cocoa_window_t* w,
                                       const char* utf8_title) {
    if (!w || !utf8_title)
        return;
    @autoreleasepool {
        w->window.title = [NSString stringWithUTF8String:utf8_title];
    }
}

extern "C" void cocoa_window_set_content_size(cocoa_window_t* w,
                                              uint32_t width,
                                              uint32_t height) {
    if (!w)
        return;
    @autoreleasepool {
        [w->window setContentSize:NSMakeSize(width, height)];
    }
}

extern "C" void cocoa_window_get_capture_rect(cocoa_window_t* w, int32_t* x,
                                              int32_t* y, uint32_t* out_w,
                                              uint32_t* out_h) {
    if (!w)
        return;
    @autoreleasepool {
        NSRect frame = w->window.frame;
        /* screencapture uses top-left-origin global coordinates; Cocoa is
         * bottom-left relative to the primary screen. */
        NSScreen* primary = NSScreen.screens.firstObject;
        double screen_h = primary ? primary.frame.size.height : 0.0;
        if (x) *x = (int32_t)frame.origin.x;
        if (y) *y = (int32_t)(screen_h - frame.origin.y - frame.size.height);
        if (out_w) *out_w = (uint32_t)frame.size.width;
        if (out_h) *out_h = (uint32_t)frame.size.height;
    }
}

void dmn_cb_presenter_stop(DmnCbPresenter* p); /* defined below */

extern "C" void cocoa_window_destroy(cocoa_window_t* w) {
    if (!w)
        return;
    if (w->presenter)
        dmn_cb_presenter_stop(w->presenter);
    @autoreleasepool {
        w->window.delegate = nil;
        [w->window close];
        [w->delegate release];
        [w->view release];
        [w->window release];
    }
    delete w;
}

extern "C" void* cocoa_create_offscreen_layer(uint32_t pixel_width,
                                              uint32_t pixel_height) {
    CAMetalLayer* layer = [[CAMetalLayer alloc] init];
    layer.bounds = CGRectMake(0, 0, pixel_width, pixel_height);
    layer.drawableSize = CGSizeMake(pixel_width, pixel_height);
    return (void*)layer;
}

extern "C" void cocoa_release_layer(void* layer) {
    [(CAMetalLayer*)layer release];
}

/* == Callback-backed swapchain presenter ================================== */
/* Embedder for dmn_window_create_with_callbacks plus an MTKView that shows
 * the frames. D3DMetal renders into one of kCbSlots BGRA8 textures per frame
 * (acquire/present, on its worker threads); a dedicated render thread calls
 * [MTKView draw], whose delegate blits the latest presented texture into the
 * view's drawable. Pacing: acquire blocks on an in-flight budget that is
 * refunded when a frame is drawn (or replaced unshown), and the render
 * thread itself is throttled by the MTKView drawable pool — together the
 * same backpressure shape as CAMetalLayer's nextDrawable. */

static const uint32_t kCbSlots = 3;

@interface DmnCbPresenter : NSObject <MTKViewDelegate> {
@public
    id<MTLDevice> dev;             /* retained */
    id<MTLCommandQueue> queue;     /* retained */
    MTKView* view;                 /* retained */
    id<MTLTexture> tex[kCbSlots];  /* retained */
    uint32_t texW, texH;
    NSLock* lock;                  /* guards tex/texW/texH (reconfigure) */
    dispatch_semaphore_t budget;   /* in-flight frame budget */
    std::atomic<int> pending;      /* latest presented, undrawn slot; -1 none */
    std::atomic<uint32_t> nextSlot;
    std::atomic<bool> drew;        /* set when drawInMTKView presented */
    std::atomic<bool> stop;
    std::thread renderThread;
}
@end

@implementation DmnCbPresenter

- (void)mtkView:(MTKView*)v drawableSizeWillChange:(CGSize)size {
    (void)v; (void)size;
}

- (void)drawInMTKView:(MTKView*)v {
    int slot = self->pending.exchange(-1, std::memory_order_acq_rel);
    if (slot < 0)
        return;
    id<MTLTexture> src = nil;
    [self->lock lock];
    src = [self->tex[slot] retain];
    [self->lock unlock];
    id<CAMetalDrawable> drawable = v.currentDrawable;
    if (!src || !drawable) {
        [src release];
        dispatch_semaphore_signal(self->budget); /* refund the undrawn frame */
        return;
    }
    id<MTLCommandBuffer> cb = [self->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    MTLSize sz = MTLSizeMake(MIN(src.width, drawable.texture.width),
                             MIN(src.height, drawable.texture.height), 1);
    [blit copyFromTexture:src
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:sz
                toTexture:drawable.texture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [cb presentDrawable:drawable];
    dispatch_semaphore_t sem = self->budget;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        (void)b;
        dispatch_semaphore_signal(sem);
    }];
    [cb commit];
    [src release];
    self->drew.store(true, std::memory_order_release);
}

- (void)dealloc {
    for (uint32_t i = 0; i < kCbSlots; i++)
        [tex[i] release];
    [view release];
    [queue release];
    [lock release];
    [dev release];
    [super dealloc];
}

@end

namespace {

bool cb_configure(void* ctx, uint32_t w, uint32_t h, uint32_t dxgi_format,
                  uint32_t buffers) {
    (void)buffers;
    auto* p = (DmnCbPresenter*)ctx;
    /* All the demos run BGRA8-family swapchains; anything else would need a
     * format map and a matching MTKView colorPixelFormat. */
    (void)dxgi_format;
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:w
                                    height:h
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
               MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    [p->lock lock];
    bool ok = true;
    for (uint32_t i = 0; i < kCbSlots; i++) {
        [p->tex[i] release];
        p->tex[i] = [p->dev newTextureWithDescriptor:td];
        if (!p->tex[i])
            ok = false;
    }
    p->texW = w;
    p->texH = h;
    [p->lock unlock];
    return ok;
}

void* cb_acquire(void* ctx, uint32_t* out_slot) {
    auto* p = (DmnCbPresenter*)ctx;
    dispatch_semaphore_wait(p->budget, DISPATCH_TIME_FOREVER);
    uint32_t s = p->nextSlot.fetch_add(1, std::memory_order_relaxed) % kCbSlots;
    *out_slot = s;
    [p->lock lock];
    id<MTLTexture> t = p->tex[s];
    [p->lock unlock];
    return (void*)t; /* borrowed */
}

void cb_present(void* ctx, uint32_t slot) {
    auto* p = (DmnCbPresenter*)ctx;
    int prev = p->pending.exchange((int)slot, std::memory_order_acq_rel);
    if (prev >= 0)
        dispatch_semaphore_signal(p->budget); /* replaced unshown: refund */
}

void cb_render_loop(DmnCbPresenter* p) {
    while (!p->stop.load(std::memory_order_acquire)) {
        @autoreleasepool {
            p->drew.store(false, std::memory_order_relaxed);
            [p->view draw]; /* blocks in currentDrawable at display rate */
            if (!p->drew.load(std::memory_order_acquire)) {
                struct timespec ns = {0, 2 * 1000 * 1000};
                nanosleep(&ns, nullptr); /* idle: no presented frame yet */
            }
        }
    }
}

} // namespace

/* Stop the render thread. The presenter object itself is leaked: in-flight
 * D3DMetal presents can still invoke the embedder callbacks after the window
 * is gone, and a test process is about to exit anyway. */
void dmn_cb_presenter_stop(DmnCbPresenter* p) {
    p->stop.store(true, std::memory_order_release);
    if (p->renderThread.joinable())
        p->renderThread.join();
    /* Refund the in-flight budget so a late present draining out of D3DMetal
     * can still acquire without blocking its worker (the textures stay alive
     * with the leaked presenter). */
    for (uint32_t i = 0; i < 2 * kCbSlots; i++)
        dispatch_semaphore_signal(p->budget);
}

extern "C" dmn_window_t cocoa_window_create_dmn(cocoa_window_t* w,
                                                bool callback_backed) {
    if (!w)
        return nullptr;
    if (!callback_backed)
        return dmn_window_create_for_view(w->view);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev)
            return nullptr;
        uint32_t cw = 0, ch = 0;
        cocoa_window_get_content_size(w, &cw, &ch);

        DmnCbPresenter* p = [[DmnCbPresenter alloc] init];
        p->dev = dev; /* +1 from create */
        p->queue = [dev newCommandQueue];
        p->lock = [[NSLock alloc] init];
        p->budget = dispatch_semaphore_create(2);
        p->pending.store(-1);
        p->nextSlot.store(0);
        p->stop.store(false);

        MTKView* view = [[MTKView alloc] initWithFrame:w->view.bounds
                                                device:dev];
        view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        view.framebufferOnly = NO;        /* blit destination */
        view.paused = YES;                /* driven by the render thread */
        view.enableSetNeedsDisplay = NO;
        view.autoResizeDrawable = NO;     /* 1x point==pixel policy */
        view.drawableSize = CGSizeMake(cw, ch);
        view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        view.delegate = p;
        [w->view addSubview:view];
        p->view = view; /* +1 from alloc */

        w->presenter = p;
        p->renderThread = std::thread(cb_render_loop, p);

        dmn_window_callbacks cb = {};
        cb.struct_size = sizeof(cb);
        cb.ctx = (void*)p;
        cb.configure = cb_configure;
        cb.acquire_texture = cb_acquire;
        cb.present = cb_present;
        return dmn_window_create_with_callbacks(&cb, cw, ch);
    }
}

extern "C" bool cocoa_arg_callback(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--callback") == 0)
            return true;
    return false;
}
