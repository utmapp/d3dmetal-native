/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Cocoa shell for the test programs. Manual reference counting.
 */

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
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

struct cocoa_window {
    NSWindow* window;                  /* retained */
    NSView* view;                      /* retained */
    CocoaTestWindowDelegate* delegate; /* retained */
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

extern "C" void cocoa_window_destroy(cocoa_window_t* w) {
    if (!w)
        return;
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

extern "C" dmn_window_t cocoa_window_create_dmn(cocoa_window_t* w) {
    return w ? dmn_window_create_for_view(w->view) : nullptr;
}
