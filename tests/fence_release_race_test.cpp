/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * fence_release_race_test: a fence must outlive its own signal.
 *
 * An app is entitled to signal a fence and drop its last reference immediately —
 * the D3D runtime owes the object deferred destruction until the GPU has passed
 * the value. GPTk 4.0b1 does not provide it: destroying an ID3D11Fence while its
 * signal is still queued kills the Metal command queue outright
 * ("ExecuteCL MTL3 completion error ... IOGPUCommandQueueErrorDomain Code=10"),
 * after which no submission ever completes and the process wedges in the kernel,
 * unkillable. dmn_fence_d3d.cpp therefore holds the reference itself.
 *
 * Two things are asserted, because a fix for one can hide the other:
 *
 *   1. The queue survives. Signal-then-drop, many times, without flushing —
 *      which is the hard case, since a D3D11 context batches and nothing has
 *      retired by the time the app lets go. Then ask the queue to retire one
 *      trivial submission.
 *
 *   2. Nothing accumulates. Holding the fence is only correct if it is given
 *      back: a shared producer fence owns a companion buffer and its fd, so
 *      "hold until the GPU catches up" must not become "hold forever". Measured
 *      as fd growth across rounds, which is where an unbounded hold shows up.
 *
 * Both are run for plain and SHARED fences on both APIs. Shared producer fences
 * add their own hazards to the same rule: their teardown must run outside the
 * framework's fence lock, a D3D12 producer's helper queue must be idle before
 * the fence goes, and completion must be judged by the fence's OWN value (the
 * shared slot can read V while the fence's signal is still queued).
 *
 * Windowless, single process. Prints "FENCERACE: PASS".
 */

#define T_TAG "FENCERACE"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>

#include "d3dmetal_native.h"
#include "common/check.h"
#include "common/com.h"
#include "common/gpu.h"
#include "common/skip.h"
#include "common/util.h"

namespace {

/* Enough signals to outrun any bounded in-flight window, few enough to stay
 * quick. The 4.0b1 wedge reproduced within the first ~20. */
constexpr int kSignals = 32;
constexpr int kRounds = 3;
constexpr unsigned kQueueAliveMs = 5000;

/* Growth is compared between consecutive rounds, so what is allowed is what may
 * legitimately still be outstanding when the count is taken: signals the GPU has
 * not caught up with yet, each holding a shared fence's companion buffer fd and
 * the registry's dup of it. dmn_fence_d3d.cpp bounds that at a handful by
 * flushing. A hold that never ends grows by 2*kSignals every round instead,
 * which is well clear of this. */
constexpr int kFdSlack = 24;

/* == D3D11: signal on the immediate context, drop immediately ============= */
int race_d3d11(ID3D11Device* dev, ID3D11DeviceContext* ctx, bool shared) {
    Com<ID3D11Device5> dev5;
    Com<ID3D11DeviceContext4> ctx4;
    CK(dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5),
       "QI(ID3D11Device5)");
    CK(ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&ctx4),
       "QI(ID3D11DeviceContext4)");

    /* GPTk 2.1 and earlier have no D3D11 fences at all. Stand this half down
     * rather than skipping the test: the D3D12 halves still say something. */
    {
        Com<ID3D11Fence> probe;
        HRESULT hr = dev5->CreateFence(0, D3D11_FENCE_FLAG_NONE,
                                       __uuidof(ID3D11Fence), (void**)&probe);
        if (t_unimplemented(hr)) {
            printf(T_TAG ": D3D11 %s fences skipped (ID3D11Device5::CreateFence "
                   "is not implemented by this D3DMetal)\n",
                   shared ? "shared" : "plain");
            return 0;
        }
        CK(hr, "ID3D11Device5::CreateFence");
    }

    int baseline = 0;
    for (int r = 0; r < kRounds; r++) {
        for (int i = 0; i < kSignals; i++) {
            ID3D11Fence* f = nullptr;
            CK(dev5->CreateFence(0,
                                 shared ? D3D11_FENCE_FLAG_SHARED
                                        : D3D11_FENCE_FLAG_NONE,
                                 __uuidof(ID3D11Fence), (void**)&f),
               "CreateFence");
            CK(ctx4->Signal(f, 1), "context Signal");
            /* The whole point: no Flush, no wait, no GetCompletedValue — the
             * app is done with the fence the instant it has signalled it. */
            f->Release();
        }
        if (!t_gpu_queue_alive_d3d11(dev, ctx, kQueueAliveMs)) {
            fprintf(stderr, T_TAG ": D3D11 %s fences: the queue died after "
                    "round %d of signal-then-drop\n",
                    shared ? "shared" : "plain", r);
            return 1;
        }
        sleep_ms(300); /* let the retired signals be reaped */
        const int fds = t_count_fds();
        if (r == 0)
            baseline = fds;
        if (fds > baseline + kFdSlack) {
            fprintf(stderr, T_TAG ": D3D11 %s fences: %d fd(s) outstanding after "
                    "round %d (was %d after round 0) — signalled fences are "
                    "being held indefinitely, not until the GPU reaches them\n",
                    shared ? "shared" : "plain", fds - baseline, r, baseline);
            return 1;
        }
    }
    printf(T_TAG ": D3D11 %s fence signal-then-drop x%d x%d rounds: queue "
           "alive, nothing accumulating: OK\n",
           shared ? "shared" : "plain", kSignals, kRounds);
    return 0;
}

/* == D3D12: signal on a queue, drop immediately =========================== */
int race_d3d12(ID3D12Device* dev, bool shared) {
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Com<ID3D12CommandQueue> queue;
    CK(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue),
       "CreateCommandQueue");

    int baseline = 0;
    for (int r = 0; r < kRounds; r++) {
        for (int i = 0; i < kSignals; i++) {
            ID3D12Fence* f = nullptr;
            CK(dev->CreateFence(0,
                                shared ? D3D12_FENCE_FLAG_SHARED
                                       : D3D12_FENCE_FLAG_NONE,
                                __uuidof(ID3D12Fence), (void**)&f),
               "CreateFence");
            CK(queue->Signal(f, 1), "queue Signal");
            f->Release();
        }
        if (!t_gpu_queue_alive_d3d12(dev, queue.ptr(), kQueueAliveMs)) {
            fprintf(stderr, T_TAG ": D3D12 %s fences: the queue died after "
                    "round %d of signal-then-drop\n",
                    shared ? "shared" : "plain", r);
            return 1;
        }
        sleep_ms(300); /* let the retired signals be reaped */
        const int fds = t_count_fds();
        if (r == 0)
            baseline = fds;
        if (fds > baseline + kFdSlack) {
            fprintf(stderr, T_TAG ": D3D12 %s fences: %d fd(s) outstanding after "
                    "round %d (was %d after round 0) — signalled fences are "
                    "being held indefinitely\n",
                    shared ? "shared" : "plain", fds - baseline, r, baseline);
            return 1;
        }
    }
    printf(T_TAG ": D3D12 %s fence signal-then-drop x%d x%d rounds: queue "
           "alive, nothing accumulating: OK\n",
           shared ? "shared" : "plain", kSignals, kRounds);
    return 0;
}

int run() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D12Device> d12;
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                         __uuidof(ID3D12Device), (void**)&d12),
       "D3D12CreateDevice");
    /* Plain fences first: they are not this library's business at all, so a
     * failure there says the framework cannot take signal-then-drop even
     * unassisted — worth distinguishing from a fault in the sharing path. */
    if (race_d3d12(d12.ptr(), /*shared=*/false) != 0)
        return 1;
    if (race_d3d12(d12.ptr(), /*shared=*/true) != 0)
        return 1;

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
                         D3D11_SDK_VERSION, &dev, &flo, &ctx),
       "D3D11CreateDevice");
    if (race_d3d11(dev.ptr(), ctx.ptr(), /*shared=*/false) != 0)
        return 1;
    /* FENCERACE_SKIP_SHARED_D3D11=1 stands this leg down if a framework ever
     * needs that while its own fix is pending. */
    if (!getenv("FENCERACE_SKIP_SHARED_D3D11")) {
        if (race_d3d11(dev.ptr(), ctx.ptr(), /*shared=*/true) != 0)
            return 1;
    } else {
        printf(T_TAG ": D3D11 shared fences skipped by request\n");
    }

    T_PASS();
    return 0;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    return run();
}
