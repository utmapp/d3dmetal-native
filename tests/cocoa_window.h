/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Minimal C interface over Cocoa windowing for the test programs.
 * Shaped like a Win32 message loop — create a window, poll events each
 * frame, query the client size — so demos ported from Win32 keep their
 * structure.
 *
 * All functions must be called from the main thread (the render loop
 * of the tests *is* the main thread, matching Win32 conventions).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "d3dmetal_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cocoa_window cocoa_window_t;

/* NSApplication boot: activation policy Regular (so the window can take
 * focus), finishLaunching, activate. */
bool cocoa_app_init(void);

/* Drain all pending events (the PeekMessageW loop equivalent).
 * Returns false once a window close / Cmd-Q requested quit. */
bool cocoa_app_poll(void);

void cocoa_app_shutdown(void);

cocoa_window_t* cocoa_window_create(const char* utf8_title,
                                    uint32_t width, uint32_t height,
                                    bool resizable);

/* The NSView* to hand to dmn_window_create_for_view(). */
void* cocoa_window_content_view(cocoa_window_t* w);

/* Render-target size in pixels. Follows the same scale policy as
 * libd3dmetal-native: 1x unless DMN_RETINA is set. */
void cocoa_window_get_content_size(cocoa_window_t* w,
                                   uint32_t* out_w, uint32_t* out_h);

void cocoa_window_set_title(cocoa_window_t* w, const char* utf8_title);

/* Programmatically resize the window content (for resize-path testing). */
void cocoa_window_set_content_size(cocoa_window_t* w,
                                   uint32_t width, uint32_t height);

/* Window frame in screen-capture coordinates (top-left origin), for
 * `screencapture -R<x>,<y>,<w>,<h>`. */
void cocoa_window_get_capture_rect(cocoa_window_t* w,
                                   int32_t* x, int32_t* y,
                                   uint32_t* out_w, uint32_t* out_h);

void cocoa_window_destroy(cocoa_window_t* w);

/* Create the demo's dmn window over `w` (dmn_window_create_for_view on the
 * content view). */
dmn_window_t cocoa_window_create_dmn(cocoa_window_t* w);

/* Standalone offscreen CAMetalLayer (for windowless swapchain tests).
 * Returns a +1 CAMetalLayer*; release with cocoa_release_layer. */
void* cocoa_create_offscreen_layer(uint32_t pixel_width, uint32_t pixel_height);
void cocoa_release_layer(void* layer);

#ifdef __cplusplus
}
#endif
