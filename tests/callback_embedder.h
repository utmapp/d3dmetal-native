/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Metal-side embedder for the callback-swapchain test. Plain C interface so
 * the D3D side stays a normal (MS-ABI-headers) C++ TU while the MTLTexture
 * pool lives in an ObjC++ TU — the vendored Windows headers cannot be
 * included from Objective-C++.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "d3dmetal_native.h"

#ifdef __cplusplus
extern "C" {
#endif

/* NULL if no Metal device is available. */
void* cbemb_create(void);
void  cbemb_destroy(void* emb);

/* Fill `out` (struct_size/ctx/fn pointers) to hand to
 * dmn_window_create_with_callbacks. */
void cbemb_fill_callbacks(void* emb, dmn_window_callbacks* out);

uint32_t cbemb_configures(void* emb);
uint32_t cbemb_config_w(void* emb);
uint32_t cbemb_config_h(void* emb);
uint32_t cbemb_acquires(void* emb);
uint32_t cbemb_presents(void* emb);
uint32_t cbemb_last_slot(void* emb); /* UINT32_MAX if none presented */

/* Center pixel (BGRA8) of the last-presented slot's texture. */
bool cbemb_read_center(void* emb, uint8_t out[4]);

#ifdef __cplusplus
}
#endif
