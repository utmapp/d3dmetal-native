/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * D3D-facing shared-fence implementation (producer and consumer sides).
 * dmn_com_hooks.cpp owns WHERE these are called from (which vtable slots),
 * this module owns WHAT they do, and the fence-value slot protocol itself
 * (mapping, reads, waits) lives in dmn_fence.cpp.
 *
 * Producer: a real D3D fence plus a companion shared buffer holding a uint64
 * value slot the GPU writes on each Signal (see dmn_fence_d3d.cpp for the
 * per-API mechanism). Consumer: OpenShared* creates a REAL fence on the
 * opening device and registers it as an import; the hooked fence/queue/context
 * methods merge the shared slot into reads and arm watcher threads that raise
 * the local fence (waits) or store into the slot (signal-back). Because the
 * imported object is a real D3DMetal fence, unhooked fence-consuming entry
 * points degrade to stale values instead of crashing.
 *
 * MS-ABI TU (built against the vendored DirectX headers), like the hooks.
 */

#pragma once

#include <d3d11_4.h>
#include <d3d12.h>

#include "d3dmetal_native.h" /* the shared-handle PODs */

/* == Producer side ======================================================== */
/* Wrap a freshly created SHARED-flag fence: build the companion buffer and
 * GPU write machinery and register the fence. Returns false on failure (the
 * fence then behaves like a plain unshared fence). */
bool dmn_fd3d_producer_create_d3d11(ID3D11Device5* dev, ID3D11Fence* fence,
                                    UINT64 initial, UINT flags);
bool dmn_fd3d_producer_create_d3d12(ID3D12Device* dev, ID3D12Fence* fence,
                                    UINT64 initial, UINT flags);

/* Tear down producer or import state for a destroyed fence (identity =
 * dmn_com_identity taken while it was alive). Releases the companion buffer /
 * slot view and helper objects; the registered fence pointer is borrowed,
 * never AddRef'd, precisely so destruction can be observed. Called by the
 * hooks' eviction sentinel; no-op for unknown identities. */
void dmn_fd3d_fence_destroy(void* identity);

/* The immediate context the D3D11 producer signals on (borrowed; only valid
 * for a registered D3D11 producer fence). Hooks patch its Signal slot. */
ID3D11DeviceContext* dmn_fd3d_producer_d3d11_ctx(ID3D11Fence* fence);

/* Fill `out` with the fence's shareable POD. False if the fence is neither a
 * producer nor an import (re-export from an opened fence is allowed). */
bool dmn_fd3d_export(IUnknown* fence, dmn_shared_fence_handle* out);

/* GPU signal interception. The producer's D3D11 store must be encoded BEFORE
 * the app's own Signal (same submission ordering); on_ctx_signal is a no-op
 * for imports. The D3D12 producer write rides a helper queue and the import
 * signal-back watchers wait for local completion, so on_queue_signal and
 * after_ctx_signal run after a SUCCESSFUL orig call. */
void dmn_fd3d_on_ctx_signal(ID3D11Fence* fence, UINT64 value);    /* before orig */
void dmn_fd3d_after_ctx_signal(ID3D11Fence* fence, UINT64 value); /* after orig */
void dmn_fd3d_on_queue_signal(ID3D12Fence* fence, UINT64 value);  /* after orig */

/* CPU ID3D12Fence::Signal interception (after a successful orig): Windows CPU
 * signal is immediately visible cross-process, so raise the shared slot now.
 * No-op for fences that are neither producers nor imports. */
void dmn_fd3d_on_cpu_signal(IUnknown* fence, UINT64 value);

/* Merge the shared-slot value into a producer's or import's completed value.
 * Returns `from_fence` unchanged for foreign fences. */
UINT64 dmn_fd3d_completed_merge(IUnknown* fence, UINT64 from_fence);

/* SetEventOnCompletion companion: also release `event` when the shared slot
 * reaches `value` (covers cross-process progress the local fence never
 * sees). Returns false (and does nothing) for foreign fences. */
bool dmn_fd3d_watch_slot(IUnknown* fence, UINT64 value, HANDLE event);

/* == Consumer side ======================================================== */
/* Import a received POD: create a REAL fence on the opening device (initial
 * value = the slot's current merged value) and register it. Returns the +1
 * fence in *out; the caller patches its class methods and attaches the
 * eviction sentinel. */
HRESULT dmn_fd3d_import_d3d12(ID3D12Device* dev,
                              const dmn_shared_fence_handle* pod,
                              ID3D12Fence** out);
HRESULT dmn_fd3d_import_d3d11(ID3D11Device5* dev,
                              const dmn_shared_fence_handle* pod,
                              ID3D11Fence** out);

/* Before a GPU wait on `fence` is enqueued (queue Wait, context Wait, or one
 * entry of SetEventOnMultipleFenceCompletion): no-op unless the fence is an
 * import, in which case a watcher raises the local fence once the shared slot
 * reaches `value`, releasing the native wait. */
void dmn_fd3d_before_queue_wait(ID3D12Fence* fence, UINT64 value);
void dmn_fd3d_before_ctx_wait(ID3D11DeviceContext4* c, ID3D11Fence* fence,
                              UINT64 value);

/* == Provided by dmn_com_hooks.cpp ======================================== */
/* Companion-buffer POD lookup for a resource that just went through the hooked
 * shared-buffer create path (the fence module creates its companion buffers
 * through the standard public APIs, so the hooks record them). */
bool dmn_res_lookup_buffer_pod(IUnknown* res, dmn_shared_buffer_handle* out);

/* Signal through the ORIGINAL (pre-patch) method. Watcher raises of an
 * imported fence must not re-enter the Signal hooks: they are not app
 * signals (no slot mirror wanted), and hook_f12_Signal taking the import
 * lock the raiser already holds would self-deadlock. Falls back to the
 * plain call when the vtable is unpatched. */
HRESULT dmn_hooks_f12_signal_orig(ID3D12Fence* fence, UINT64 value);
HRESULT dmn_hooks_ctx_signal_orig(ID3D11DeviceContext4* c, ID3D11Fence* fence,
                                  UINT64 value);
