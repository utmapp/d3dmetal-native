/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Full DmnWindow definition. Objective-C++ translation units only
 * (dmn_window.mm, dmn_gfxt_swapchain.mm).
 */

#pragma once

#ifndef __OBJC__
#error "dmn_window_internal.h is Objective-C++ only"
#endif

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>

#include "dmn_private.h"

enum class DmnWindowBackend {
    View,      /* NSView + library-owned CAMetalLayer */
    Layer,     /* caller-owned CAMetalLayer */
    Callbacks, /* embedder-provided textures */
};

struct DmnWindow {
    uint32_t          slot;
    DmnWindowBackend  backend;
    NSView*           view;   /* retained; View backend only */
    CAMetalLayer*     layer;  /* retained; View + Layer backends */
    dmn_window_callbacks cb;  /* Callbacks backend; copied */

    std::atomic<uint32_t> width{0};   /* pixels */
    std::atomic<uint32_t> height{0};
    std::atomic<double>   scale{1.0};
    std::atomic<uint32_t> dxgi_format{0}; /* last configured DXGI_FORMAT */
    std::atomic<bool>     alive{true};
};
