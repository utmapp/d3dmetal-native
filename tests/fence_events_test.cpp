/*
 * Event-driven fence paths, windowless, single process:
 *
 *  1. SetEventOnCompletion + dmn_event_wait on the producer fence and on an
 *     imported fence (the slot watchers must release waiters for values that
 *     only ever arrive cross-fence).
 *  2. CPU Signal visibility both directions: a CPU signal on the producer is
 *     seen through the import, and a CPU signal on the import is seen through
 *     the producer (Windows: CPU signals are immediately visible).
 *  3. ID3D12Device1::SetEventOnMultipleFenceCompletion with an imported fence
 *     in the mix, ALL and ANY flavors, against the native multi-wait.
 *  4. Cross-API in-process: the same producer opened as an ID3D11Fence via
 *     OpenSharedFence, waited with SetEventOnCompletion.
 *
 * Prints "FEVENTS: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>

#include <d3d11_4.h>
#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "FEVENTS"
#include "common/check.h"
#include "common/dx11.h"
#include "common/dx12.h"

/* Internal-but-exported event helpers (the HANDLEs SetEventOnCompletion
 * accepts are these; apps normally receive them from D3D APIs). */
extern "C" void* dmn_event_create(int manual_reset, int initial_state);
extern "C" void  dmn_event_close(void* handle);

static const uint64_t kLongNs  = 10ull * 1000 * 1000 * 1000; /* 10 s */

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "FEVENTS: dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");
    Com<ID3D12CommandQueue> queue;
    CK(make_d3d12_queue(dev.ptr(), queue), "CreateCommandQueue");

    Com<ID3D12Fence> prod;
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
                        (void**)&prod), "CreateFence(SHARED)");
    HANDLE h = nullptr;
    CK(dev->CreateSharedHandle(prod.ptr(), nullptr, 0, nullptr, &h),
       "CreateSharedHandle");
    Com<ID3D12Fence> imp;
    CK(dev->OpenSharedHandle(h, __uuidof(ID3D12Fence), (void**)&imp),
       "OpenSharedHandle");

    uint64_t v = 0;

    /* 1) Event waits released by the producer's GPU signal. */
    {
        void* evProd = dmn_event_create(0, 0);
        void* evImp = dmn_event_create(0, 0);
        EXPECT(evProd && evImp, "event create failed");
        v++;
        CK(prod->SetEventOnCompletion(v, evProd), "producer SetEventOnCompletion");
        CK(imp->SetEventOnCompletion(v, evImp), "import SetEventOnCompletion");
        EXPECT(dmn_event_wait(evImp, 50ull * 1000 * 1000) == DMN_WAIT_TIMEOUT,
               "import event fired before any signal");
        CK(queue->Signal(prod.ptr(), v), "queue Signal");
        EXPECT(dmn_event_wait(evProd, kLongNs) == DMN_WAIT_SIGNALED,
               "producer event did not fire");
        EXPECT(dmn_event_wait(evImp, kLongNs) == DMN_WAIT_SIGNALED,
               "import event did not fire");
        dmn_event_close(evProd);
        dmn_event_close(evImp);
        printf("FEVENTS: event waits ok (value %llu)\n", (unsigned long long)v);
    }

    /* 2) CPU Signal visibility, both directions. */
    {
        v++;
        CK(prod->Signal(v), "producer CPU Signal");
        EXPECT(imp->GetCompletedValue() >= v,
               "producer CPU signal invisible through the import");
        v++;
        CK(imp->Signal(v), "import CPU Signal");
        EXPECT(prod->GetCompletedValue() >= v,
               "import CPU signal invisible through the producer");
        printf("FEVENTS: CPU signal visibility ok (value %llu)\n",
               (unsigned long long)v);
    }

    /* 3) Multi-fence waits with an imported fence in the mix. */
    {
        Com<ID3D12Device1> dev1;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device1),
                                          (void**)&dev1)) && dev1) {
            Com<ID3D12Fence> plain;
            CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                (void**)&plain), "CreateFence(plain)");
            ID3D12Fence* fences[2] = {imp.ptr(), plain.ptr()};
            const uint64_t vImp = v + 1, vPlain = 1;
            UINT64 values[2] = {vImp, vPlain};

            /* ALL: must hold until BOTH are signaled. */
            void* evAll = dmn_event_create(0, 0);
            EXPECT(evAll, "event create failed");
            CK(dev1->SetEventOnMultipleFenceCompletion(
                   fences, values, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL, evAll),
               "SetEventOnMultipleFenceCompletion(ALL)");
            CK(queue->Signal(prod.ptr(), vImp), "queue Signal(shared)");
            v = vImp;
            EXPECT(dmn_event_wait(evAll, 200ull * 1000 * 1000) == DMN_WAIT_TIMEOUT,
                   "ALL wait fired with one fence pending");
            CK(queue->Signal(plain.ptr(), vPlain), "queue Signal(plain)");
            EXPECT(dmn_event_wait(evAll, kLongNs) == DMN_WAIT_SIGNALED,
                   "ALL wait did not fire");
            dmn_event_close(evAll);

            /* ANY: the imported fence alone must release it. */
            void* evAny = dmn_event_create(0, 0);
            EXPECT(evAny, "event create failed");
            UINT64 values2[2] = {v + 1, vPlain + 1000};
            CK(dev1->SetEventOnMultipleFenceCompletion(
                   fences, values2, 2, D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, evAny),
               "SetEventOnMultipleFenceCompletion(ANY)");
            CK(queue->Signal(prod.ptr(), v + 1), "queue Signal(shared)");
            v++;
            EXPECT(dmn_event_wait(evAny, kLongNs) == DMN_WAIT_SIGNALED,
                   "ANY wait did not fire on the imported fence");
            dmn_event_close(evAny);
            printf("FEVENTS: multi-fence waits ok (value %llu)\n",
                   (unsigned long long)v);
        } else {
            printf("FEVENTS: ID3D12Device1 unavailable; multi-wait skipped\n");
        }
    }

    /* 4) Cross-API in-process: same POD opened as an ID3D11Fence. */
    {
        Com<ID3D11Device> d11;
        Com<ID3D11DeviceContext> ctx;
        CK(make_d3d11_device(d11, ctx), "D3D11CreateDevice");
        Com<ID3D11Device5> d115;
        CK(d11->QueryInterface(__uuidof(ID3D11Device5), (void**)&d115),
           "ID3D11Device5");
        Com<ID3D11Fence> imp11;
        CK(d115->OpenSharedFence(h, __uuidof(ID3D11Fence), (void**)&imp11),
           "OpenSharedFence");
        EXPECT(imp11->GetCompletedValue() >= v,
               "D3D11 import behind the producer value");
        void* ev11 = dmn_event_create(0, 0);
        EXPECT(ev11, "event create failed");
        v++;
        CK(imp11->SetEventOnCompletion(v, ev11), "D3D11 SetEventOnCompletion");
        CK(queue->Signal(prod.ptr(), v), "queue Signal");
        EXPECT(dmn_event_wait(ev11, kLongNs) == DMN_WAIT_SIGNALED,
               "D3D11 import event did not fire");
        dmn_event_close(ev11);
        printf("FEVENTS: cross-API import event ok (value %llu)\n",
               (unsigned long long)v);
    }

    CK(dmn_shared_handle_close(h) == DMN_SUCCESS ? S_OK : E_FAIL,
       "dmn_shared_handle_close");
    T_PASS();
    return 0;
}
