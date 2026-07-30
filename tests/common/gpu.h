/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Is the GPU still executing work submitted by this process?
 *
 * The sharing machinery hands Metal buffers whose storage this library owns, and
 * the failure mode when their lifetime is got wrong is not a wrong pixel: a
 * command buffer that references a freed impostor is aborted with
 * kIOGPUCommandBufferCallbackErrorInvalidResource, and that kills the entire
 * MTLCommandQueue. Every later submission on it fails, so no fence ever
 * completes again, and the process typically ends up unkillable in the kernel.
 * A test can only report that as a timeout — the least informative failure there
 * is, and one that also starves whatever runs next.
 *
 * So: after anything that stresses shared-resource lifetime, ask the queue to
 * retire one trivial submission. A wedged queue then produces a named failure in
 * seconds instead of a timeout minutes later.
 *
 * Define T_TAG before including.
 */

#pragma once

#include <stdio.h>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>

#include "common/util.h"

#ifndef T_TAG
#error "define T_TAG before including common/gpu.h"
#endif

/* D3D12: signal a fresh fence on the queue and wait for it. Deliberately a
 * fence this call owns — a fence the caller has been hammering may be held up
 * for reasons of its own. */
static inline bool t_gpu_queue_alive_d3d12(ID3D12Device* dev,
                                           ID3D12CommandQueue* queue,
                                           unsigned ms) {
    if (!dev || !queue)
        return false;
    ID3D12Fence* f = nullptr;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                (void**)&f)) || !f) {
        fprintf(stderr, T_TAG ": queue liveness check could not create a fence\n");
        return false;
    }
    bool alive = false;
    if (SUCCEEDED(queue->Signal(f, 1))) {
        const uint64_t deadline = now_ms() + ms;
        while (now_ms() < deadline) {
            if (f->GetCompletedValue() >= 1) {
                alive = true;
                break;
            }
            sleep_ms(2);
        }
    }
    f->Release();
    if (!alive)
        fprintf(stderr, T_TAG ": the D3D12 command queue did not retire a "
                "trivial signal within %u ms — it is wedged (a command buffer "
                "referencing a freed shared backing kills the whole queue)\n",
                ms);
    return alive;
}

/* D3D11 has no fence before GPTk 3.0, so use an EVENT query — the classic "has
 * the GPU caught up" mechanism, present on every version. */
static inline bool t_gpu_queue_alive_d3d11(ID3D11Device* dev,
                                           ID3D11DeviceContext* ctx,
                                           unsigned ms) {
    if (!dev || !ctx)
        return false;
    D3D11_QUERY_DESC qd = {};
    qd.Query = D3D11_QUERY_EVENT;
    ID3D11Query* q = nullptr;
    if (FAILED(dev->CreateQuery(&qd, &q)) || !q) {
        fprintf(stderr, T_TAG ": queue liveness check could not create an "
                "event query\n");
        return false;
    }
    ctx->End(q);
    ctx->Flush();
    bool alive = false;
    const uint64_t deadline = now_ms() + ms;
    while (now_ms() < deadline) {
        BOOL done = FALSE;
        HRESULT hr = ctx->GetData(q, &done, sizeof(done), 0);
        if (hr == S_OK) {
            alive = true;
            break;
        }
        if (FAILED(hr))
            break;
        sleep_ms(2);
    }
    q->Release();
    if (!alive)
        fprintf(stderr, T_TAG ": the D3D11 immediate context did not retire an "
                "event query within %u ms — the GPU queue is wedged (a command "
                "buffer referencing a freed shared backing kills the whole "
                "queue)\n", ms);
    return alive;
}
