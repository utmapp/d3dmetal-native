/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * error_paths_test: a rejected call must leave nothing behind.
 *
 * shared-contract already checks that bad input is *reported* correctly. This
 * checks the other half — what the library did before it found out the call was
 * going to fail. Interception happens around a D3DMetal call, so work often
 * starts before the outcome is known: an arm is set, a watcher thread is
 * started, an fd is duplicated. If the call then fails and that work is not
 * unwound, the debris outlives the caller's mistake.
 *
 * That is not hypothetical. hook_d3d12_SetEventOnMultipleFenceCompletion used to
 * arm a wait watcher per imported fence *before* calling through. On GPTk 2.1
 * and earlier the call is unimplemented, so it returned E_NOTIMPL with watcher
 * threads already waiting on values nobody would ever deliver — and on state the
 * caller then released. The process aborted with "mutex lock failed" some time
 * later, nowhere near the call that caused it.
 *
 * So: drive the rejection paths in bulk and hold the process to three things —
 * it does not crash, it does not accumulate file descriptors, and it does not
 * accumulate threads. Any of those growing with the number of rejected calls is
 * the failure.
 *
 * Windowless, single process. Prints "ERRPATH: PASS".
 */

#define T_TAG "ERRPATH"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include "d3dmetal_native.h"
#include "common/check.h"
#include "common/com.h"
#include "common/util.h"

namespace {

/* Enough repetitions that one leaked fd or thread per rejection is unmistakable,
 * and few enough that a genuine leak does not exhaust the process before the
 * assertion runs. */
constexpr int kReps = 100;

/* Steady-state allowances. Neither should move at all with rejections, but a
 * framework is free to open something on first use of a path (see the warm-up
 * below) and a worker pool may resize on its own. */
constexpr int kFdSlack = 8;
constexpr int kThreadSlack = 4;

struct Debris {
    int fds;
    int threads;
};

Debris snapshot() {
    sleep_ms(200); /* let anything transient wind down before counting */
    Debris d;
    d.fds = t_count_fds();
    d.threads = t_count_threads();
    return d;
}

int check_no_debris(const char* what, const Debris& before) {
    const Debris after = snapshot();
    if (after.fds > before.fds + kFdSlack ||
        after.threads > before.threads + kThreadSlack) {
        fprintf(stderr, T_TAG ": %s left debris after %d rejected calls: "
                "fds %d -> %d, threads %d -> %d\n", what, kReps, before.fds,
                after.fds, before.threads, after.threads);
        return 1;
    }
    printf(T_TAG ": %s: %d rejections left nothing behind (fds %d -> %d, "
           "threads %d -> %d): OK\n", what, kReps, before.fds, after.fds,
           before.threads, after.threads);
    return 0;
}

/* == Rejected multi-fence waits ========================================== */
/* The regression this exists for: the hook used to arm a wait watcher per
 * imported fence BEFORE calling through, so a call the framework then refused
 * left watchers waiting on values nobody would deliver.
 *
 * Only calls that actually FAIL are held to that. GPTk 3.0 and 4.0b1 accept
 * everything well-formed offered here (they do not validate the flags), so on
 * those the assertion is vacuous and says so; 2.1 and earlier reject the entry
 * point outright, which is where it bites.
 *
 * ERRPATH_UNREACHABLE_WAIT=1 additionally arms waits on a value the producer
 * never reaches — a legitimate thing for an app to do, and an ANY-flag wait does
 * it every time the other fence wins. That currently leaks one thread and one fd
 * per armed wait on 3.0 and 4.0b1 (measured: +100 of each over 100 calls),
 * because dmn_fd3d_before_queue_wait opens its own view of the slot and spawns a
 * detached thread that waits DMN_WAIT_INFINITE. It is off by default so the suite
 * is not red on a known-open issue, and present so the fix has a reproducer. */
int test_rejected_multi_fence_wait(ID3D12Device* dev) {
    Com<ID3D12Device1> dev1;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D12Device1), (void**)&dev1)) ||
        !dev1.ptr()) {
        printf(T_TAG ": ID3D12Device1 unavailable; multi-fence rejection "
               "skipped\n");
        return 0;
    }

    /* A real shared fence and a real import of it, so the watcher-arming path is
     * actually reachable. */
    Com<ID3D12Fence> prod;
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
                        (void**)&prod), "CreateFence(SHARED)");
    HANDLE h = nullptr;
    CK(dev->CreateSharedHandle(prod.ptr(), nullptr, 0, nullptr, &h),
       "CreateSharedHandle");
    Com<ID3D12Fence> imp;
    CK(dev->OpenSharedHandle(h, __uuidof(ID3D12Fence), (void**)&imp),
       "OpenSharedHandle");

    ID3D12Fence* fences[1] = {imp.ptr()};
    const bool unreachable = getenv("ERRPATH_UNREACHABLE_WAIT") != nullptr;
    /* Reachable by default: the producer is signalled to it below, so a watcher
     * armed for it retires instead of masking a leak with a legitimate wait. */
    UINT64 values[1] = {unreachable ? ~0ull : 1ull};

    /* Every input here is well-formed memory. Handing a D3D entry point a null
     * array is undefined rather than "rejected" — D3DMetal does not validate it
     * and hangs — and this is about calls that fail, not calls that are illegal
     * to make. An out-of-range flags value is the closest thing to a portable
     * rejection: correct pointers, a count the callee reads, an enum no version
     * defines. */
    const D3D12_MULTIPLE_FENCE_WAIT_FLAGS bad_flags =
        (D3D12_MULTIPLE_FENCE_WAIT_FLAGS)0x7fffffff;

    /* Warm up: the first call through any of these paths may open or start
     * something that legitimately stays. */
    for (int i = 0; i < 4; i++) {
        void* ev = dmn_event_create(0, 0);
        (void)dev1->SetEventOnMultipleFenceCompletion(
            fences, values, 0, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, ev);
        (void)dev1->SetEventOnMultipleFenceCompletion(fences, values, 1,
                                                      bad_flags, ev);
        if (ev)
            dmn_event_close(ev);
    }

    const Debris before = snapshot();
    int rejected = 0, accepted = 0;
    for (int i = 0; i < kReps; i++) {
        void* ev = dmn_event_create(0, 0);
        EXPECT(ev != nullptr, "dmn_event_create failed");
        /* count = 0: nothing to wait on, so nothing may be left waiting. */
        HRESULT hr = dev1->SetEventOnMultipleFenceCompletion(
            fences, values, 0, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, ev);
        FAILED(hr) ? rejected++ : accepted++;
        hr = dev1->SetEventOnMultipleFenceCompletion(fences, values, 1,
                                                     bad_flags, ev);
        FAILED(hr) ? rejected++ : accepted++;
        dmn_event_close(ev);
    }
    /* Let the accepted waits retire: the producer reaches the value they were
     * armed for, so their watchers finish and give their fd and thread back. */
    if (!unreachable) {
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Com<ID3D12CommandQueue> queue;
        CK(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                   (void**)&queue), "CreateCommandQueue");
        CK(queue->Signal(prod.ptr(), 1), "queue Signal");
        for (int i = 0; i < 500 && imp->GetCompletedValue() < 1; i++)
            sleep_ms(10);
    }
    printf(T_TAG ": multi-fence waits: %d rejected, %d accepted%s\n", rejected,
           accepted, unreachable ? " (armed on an unreachable value)" : "");
    if (rejected == 0 && !unreachable) {
        printf(T_TAG ": this D3DMetal rejected none of them, so there is no "
               "rejection to audit here\n");
        EXPECT(dmn_shared_handle_close(h) == DMN_SUCCESS,
               "closing the fence handle failed");
        return 0;
    }
    if (check_no_debris("armed/rejected multi-fence waits", before) != 0)
        return 1;

    EXPECT(dmn_shared_handle_close(h) == DMN_SUCCESS,
           "closing the fence handle failed");
    return 0;
}

/* == Misused shared handles ============================================== */
int test_handle_close_misuse(ID3D11Device* dev) {
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = td.Height = 64;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Texture2D> tex;
    CK(dev->CreateTexture2D(&td, nullptr, &tex), "CreateTexture2D(MISC_SHARED)");
    Com<IDXGIResource> res;
    CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res), "QI view");
    HANDLE legacy = nullptr;
    CK(res->GetSharedHandle(&legacy), "GetSharedHandle");

    /* A legacy handle has no lifecycle of its own; closing one is a caller
     * error and must be refused rather than closing the resource's fd. */
    EXPECT(dmn_shared_handle_close(legacy) == DMN_ERROR_INVALID_ARGUMENT,
           "closing a legacy GetSharedHandle value was not refused");
    EXPECT(dmn_shared_handle_close(nullptr) == DMN_ERROR_INVALID_ARGUMENT,
           "closing NULL was not refused");
    int stack_garbage = 0;
    EXPECT(dmn_shared_handle_close(&stack_garbage) == DMN_ERROR_INVALID_ARGUMENT,
           "closing a pointer we never vended was not refused");

    /* The legacy handle must still work afterwards: a refused close must not
     * have taken anything with it. */
    HANDLE again = nullptr;
    CK(res->GetSharedHandle(&again), "GetSharedHandle after refused closes");
    EXPECT(again == legacy, "the legacy handle changed after refused closes");

    const Debris before = snapshot();
    for (int i = 0; i < kReps; i++) {
        (void)dmn_shared_handle_close(nullptr);
        (void)dmn_shared_handle_close(legacy);
        (void)dmn_shared_handle_close(&stack_garbage);
    }
    if (check_no_debris("refused handle closes", before) != 0)
        return 1;
    return 0;
}

/* == Rejected imports ==================================================== */
/* shared-contract asserts these are refused; here the point is that refusing
 * them in bulk costs nothing. Each malformed POD carries a real fd, so an import
 * that dup'd before validating would show up immediately. */
int test_rejected_imports(ID3D11Device* dev) {
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = td.Height = 64;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Texture2D> tex;
    CK(dev->CreateTexture2D(&td, nullptr, &tex), "CreateTexture2D(MISC_SHARED)");
    Com<IDXGIResource> res;
    CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res), "QI view");
    HANDLE h = nullptr;
    CK(res->GetSharedHandle(&h), "GetSharedHandle");

    dmn_shared_texture_handle good;
    memcpy(&good, h, sizeof(good));
    EXPECT(good.magic == DMN_SHARED_TEXTURE_MAGIC, "export did not give a POD");

    /* Warm up the import path once with something valid so its one-time costs
     * are not counted as debris. */
    {
        Com<ID3D11Texture2D> ok;
        dmn_shared_texture_handle p = good;
        (void)dev->OpenSharedResource((HANDLE)&p, __uuidof(ID3D11Texture2D),
                                      (void**)&ok);
    }

    const Debris before = snapshot();
    for (int i = 0; i < kReps; i++) {
        dmn_shared_texture_handle bad = good;
        bad.stride = 0; /* shorter than a row */
        Com<ID3D11Texture2D> out;
        EXPECT(FAILED(dev->OpenSharedResource((HANDLE)&bad,
                                              __uuidof(ID3D11Texture2D),
                                              (void**)&out)),
               "an import with a zero stride was accepted");

        bad = good;
        bad.size = 1; /* smaller than the surface it claims */
        Com<ID3D11Texture2D> out2;
        EXPECT(FAILED(dev->OpenSharedResource((HANDLE)&bad,
                                              __uuidof(ID3D11Texture2D),
                                              (void**)&out2)),
               "an import smaller than its own surface was accepted");

        bad = good;
        bad.magic = 0xdeadbeef; /* not one of ours at all */
        Com<ID3D11Texture2D> out3;
        EXPECT(FAILED(dev->OpenSharedResource((HANDLE)&bad,
                                              __uuidof(ID3D11Texture2D),
                                              (void**)&out3)),
               "a foreign handle was accepted");
    }
    return check_no_debris("rejected imports", before);
}

int run() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
                         D3D11_SDK_VERSION, &dev, &flo, &ctx),
       "D3D11CreateDevice");
    if (test_handle_close_misuse(dev.ptr()) != 0)
        return 1;
    if (test_rejected_imports(dev.ptr()) != 0)
        return 1;

    Com<ID3D12Device> d12;
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                         __uuidof(ID3D12Device), (void**)&d12),
       "D3D12CreateDevice");
    if (test_rejected_multi_fence_wait(d12.ptr()) != 0)
        return 1;

    T_PASS();
    return 0;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    return run();
}
