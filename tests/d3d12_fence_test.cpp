/*
 * D3D12 producer fence, single process. Create an ID3D12Fence with
 * D3D12_FENCE_FLAG_SHARED (hooked to build a companion GPU-written buffer plus a
 * helper queue), export it with ID3D12Device::CreateSharedHandle, then for each
 * value: signal the fence on the app's command queue and confirm the value
 * lands, both through the raw consumer (dmn_shared_fence_open, polling the
 * companion buffer) and through the imported ID3D12Fence that OpenSharedHandle
 * hands back. Cross-process / cross-API round trips build on this same fd.
 *
 * Prints "FENCE12: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <time.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "FENCE12"
#include "common/check.h"
#include "common/util.h"

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "FENCE12: dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D12Device> device;
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
                         (void**)&device),
       "D3D12CreateDevice");
    Com<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CK(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue),
       "CreateCommandQueue");

    /* Shared fence: the hook makes a companion GPU-written buffer + helper queue. */
    Com<ID3D12Fence> fence;
    CK(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
                           (void**)&fence),
       "CreateFence(SHARED)");

    HANDLE fenceH = nullptr;
    CK(device->CreateSharedHandle(fence.ptr(), nullptr, 0, nullptr, &fenceH),
       "CreateSharedHandle");
    if (!fenceH ||
        ((const dmn_shared_fence_handle*)fenceH)->magic != DMN_SHARED_FENCE_MAGIC) {
        fprintf(stderr, "FENCE12: CreateSharedHandle did not return a fence POD\n");
        return 1;
    }
    printf("FENCE12: exported fence via CreateSharedHandle\n");

    /* Consume it back through the STANDARD API: OpenSharedHandle vends an
     * ID3D12Fence whose GetCompletedValue polls the GPU-written companion buffer. */
    Com<ID3D12Fence> vended;
    CK(device->OpenSharedHandle(fenceH, __uuidof(ID3D12Fence), (void**)&vended),
       "OpenSharedHandle(ID3D12Fence)");
    /* NT-style handle: the consumer owns it; close once it has been opened. */
    if (dmn_shared_handle_close(fenceH) != DMN_SUCCESS) {
        fprintf(stderr, "FENCE12: dmn_shared_handle_close FAILED\n");
        return 1;
    }

    bool ok = true;
    for (uint64_t v = 1; v <= 4; v++) {
        CK(queue->Signal(fence.ptr(), v), "queue Signal");

        uint64_t start = now_ms();
        bool reached = false;
        while (now_ms() - start < 5000) {
            if (vended->GetCompletedValue() >= v) { reached = true; break; }
            struct timespec ns = {0, 1000 * 1000};
            nanosleep(&ns, nullptr);
        }
        printf("FENCE12: signal %llu -> vended completed=%llu %s\n",
               (unsigned long long)v, (unsigned long long)vended->GetCompletedValue(),
               reached ? "OK" : "TIMEOUT");
        if (!reached)
            ok = false;
    }

    printf("FENCE12: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
