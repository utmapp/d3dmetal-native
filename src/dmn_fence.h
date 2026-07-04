/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Internal (NOT public API): a view over a shared fence-value slot,
 * direction-agnostic — dmn_fence_d3d.cpp opens these for imported
 * ID3D11Fence / ID3D12Fence consumers, for the producer's own merge/watch
 * views, and for detached watcher threads. Apps never call these; a shared
 * fence is created and consumed purely through the standard D3D APIs.
 */

#pragma once

#include <cstdint>

#include "d3dmetal_native.h" /* dmn_shared_fence_handle, dmn_wait_status, DMN_WAIT_* */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dmn_shared_fence* dmn_shared_fence_t;

/* Open a consumer view over a received fence handle (fd already patched in). */
dmn_shared_fence_t dmn_shared_fence_open(const dmn_shared_fence_handle* handle);
void               dmn_shared_fence_close(dmn_shared_fence_t fence);

/* Highest GPU-written value seen so far (monotonic). */
uint64_t dmn_shared_fence_get_completed(dmn_shared_fence_t fence);

/* Block until the value reaches >= value, or timeout_ns elapses. */
dmn_wait_status dmn_shared_fence_wait(dmn_shared_fence_t fence, uint64_t value,
                                      uint64_t timeout_ns);

/* CPU-side signal: raise the slot to >= value (monotonic; no-op for lower
 * values). A fence should have a single signaling direction at a time. */
void dmn_shared_fence_signal(dmn_shared_fence_t fence, uint64_t value);

#ifdef __cplusplus
}
#endif
