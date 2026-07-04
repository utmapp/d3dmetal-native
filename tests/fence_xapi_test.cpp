/*
 * Cross-API, cross-process shared fence round trip. One binary that forks:
 * the parent is the producer, the child is the consumer, and the fence POD +
 * its companion-buffer fd travel over a socketpair (SCM_RIGHTS). An argument
 * selects the direction:
 *
 *   d11d12  producer = D3D11 (ID3D11Device5::CreateFence + context4 Signal),
 *           consumer = D3D12 (ID3D12Device::OpenSharedHandle -> ID3D12Fence).
 *   d12d11  producer = D3D12 (ID3D12Device::CreateFence + queue Signal),
 *           consumer = D3D11 (ID3D11Device5::OpenSharedFence -> ID3D11Fence).
 *
 * Either way the producer's GPU writes each value into the shared buffer and
 * the consumer, using the *other* API, observes it through GetCompletedValue.
 * Prints "XFENCE: PASS" and exits 0 on success.
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"
#include "common/ipc.h"
#include "common/util.h"

namespace {

constexpr int kTicks = 4;

/* Poll any *Fence with a GetCompletedValue() until it reaches `v` or times out. */
template <class FenceT>
bool wait_value(FenceT* f, uint64_t v, uint64_t timeout_ms) {
    uint64_t start = now_ms();
    while (f->GetCompletedValue() < v) {
        if (now_ms() - start > timeout_ms)
            return false;
        struct timespec ns = {0, 2 * 1000 * 1000};
        nanosleep(&ns, nullptr);
    }
    return true;
}

/* == Producers ========================================================== */
int producer_d11(int sock) {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "XFENCE: prod D3D11CreateDevice FAILED\n");
        return 1;
    }
    Com<ID3D11Device5> dev5;
    Com<ID3D11DeviceContext4> ctx4;
    Com<ID3D11Fence> fence;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5)) ||
        FAILED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&ctx4)) ||
        FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence),
                                 (void**)&fence))) {
        fprintf(stderr, "XFENCE: prod CreateFence(SHARED) unavailable\n");
        return 1;
    }
    HANDLE h = nullptr;
    if (FAILED(fence->CreateSharedHandle(nullptr, 0, nullptr, &h)) || !h) {
        fprintf(stderr, "XFENCE: prod fence CreateSharedHandle FAILED\n");
        return 1;
    }
    dmn_shared_fence_handle pod;
    memcpy(&pod, h, sizeof(pod));
    if (!send_with_fd(sock, &pod, sizeof(pod), pod.fd)) {
        fprintf(stderr, "XFENCE: prod send FAILED: %s\n", strerror(errno));
        return 1;
    }
    printf("XFENCE: D3D11 producer fence fd=%d\n", pod.fd);
    dmn_shared_handle_close(h); /* NT-style handle; POD + fd are shipped */

    for (int t = 1; t <= kTicks; t++) {
        ctx4->Signal(fence.ptr(), (uint64_t)t);
        char ack = 0;
        if (read(sock, &ack, 1) != 1) {
            fprintf(stderr, "XFENCE: prod ack read FAILED at %d\n", t);
            return 1;
        }
    }
    return 0;
}

int producer_d12(int sock) {
    Com<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&dev))) {
        fprintf(stderr, "XFENCE: prod D3D12CreateDevice FAILED\n");
        return 1;
    }
    Com<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                       (void**)&queue))) {
        fprintf(stderr, "XFENCE: prod CreateCommandQueue FAILED\n");
        return 1;
    }
    Com<ID3D12Fence> fence;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
                                (void**)&fence))) {
        fprintf(stderr, "XFENCE: prod CreateFence(SHARED) FAILED\n");
        return 1;
    }
    HANDLE h = nullptr;
    if (FAILED(dev->CreateSharedHandle(fence.ptr(), nullptr, 0, nullptr, &h)) || !h) {
        fprintf(stderr, "XFENCE: prod CreateSharedHandle FAILED\n");
        return 1;
    }
    dmn_shared_fence_handle pod;
    memcpy(&pod, h, sizeof(pod));
    if (!send_with_fd(sock, &pod, sizeof(pod), pod.fd)) {
        fprintf(stderr, "XFENCE: prod send FAILED: %s\n", strerror(errno));
        return 1;
    }
    printf("XFENCE: D3D12 producer fence fd=%d\n", pod.fd);
    dmn_shared_handle_close(h); /* NT-style handle; POD + fd are shipped */

    for (int t = 1; t <= kTicks; t++) {
        queue->Signal(fence.ptr(), (uint64_t)t);
        char ack = 0;
        if (read(sock, &ack, 1) != 1) {
            fprintf(stderr, "XFENCE: prod ack read FAILED at %d\n", t);
            return 1;
        }
    }
    return 0;
}

/* == Consumers ========================================================== */
/* Receive + patch the POD, then hand it to `open` (an API-specific OpenShared*)
 * and poll the returned fence for each value, acking the producer per tick. */
template <class FenceT, class OpenFn>
int consume(int sock, const char* who, OpenFn open) {
    dmn_shared_fence_handle pod;
    int fd = -1;
    if (!recv_with_fd(sock, &pod, sizeof(pod), &fd)) {
        fprintf(stderr, "XFENCE: %s recv FAILED: %s\n", who, strerror(errno));
        return 1;
    }
    pod.fd = fd; /* the received fd is the resource's token */
    if (pod.magic != DMN_SHARED_FENCE_MAGIC) {
        fprintf(stderr, "XFENCE: %s bad magic 0x%08x\n", who, pod.magic);
        return 1;
    }
    Com<FenceT> fence;
    HRESULT hr = open(&pod, fence);
    if (FAILED(hr) || !fence) {
        fprintf(stderr, "XFENCE: %s open fence FAILED 0x%08x\n", who, (unsigned)hr);
        return 1;
    }
    printf("XFENCE: %s opened fence (fd=%d)\n", who, pod.fd);

    int rc = 0;
    for (int t = 1; t <= kTicks; t++) {
        if (!wait_value(fence.ptr(), (uint64_t)t, 10000)) {
            fprintf(stderr, "XFENCE: %s timed out waiting for %d (completed=%llu)\n",
                    who, t, (unsigned long long)fence->GetCompletedValue());
            rc = 1;
            break;
        }
        printf("XFENCE: %s saw value %d\n", who, t);
        char ack = 1;
        if (write(sock, &ack, 1) != 1) {
            fprintf(stderr, "XFENCE: %s ack write FAILED\n", who);
            rc = 1;
            break;
        }
    }
    if (rc == 0)
        printf("XFENCE: PASS\n");
    return rc;
}

int consumer_d12(int sock) {
    Com<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&dev))) {
        fprintf(stderr, "XFENCE: cons D3D12CreateDevice FAILED\n");
        return 1;
    }
    return consume<ID3D12Fence>(sock, "D3D12 consumer",
        [&](const dmn_shared_fence_handle* pod, Com<ID3D12Fence>& out) {
            return dev->OpenSharedHandle((HANDLE)pod, __uuidof(ID3D12Fence),
                                         (void**)&out);
        });
}

int consumer_d11(int sock) {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "XFENCE: cons D3D11CreateDevice FAILED\n");
        return 1;
    }
    Com<ID3D11Device5> dev5;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5))) {
        fprintf(stderr, "XFENCE: cons QI(ID3D11Device5) FAILED\n");
        return 1;
    }
    return consume<ID3D11Fence>(sock, "D3D11 consumer",
        [&](const dmn_shared_fence_handle* pod, Com<ID3D11Fence>& out) {
            return dev5->OpenSharedFence((HANDLE)pod, __uuidof(ID3D11Fence),
                                         (void**)&out);
        });
}

} // namespace

namespace {
bool g_d11_to_d12 = true;

int producer_main(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS)
        return 1;
    return g_d11_to_d12 ? producer_d11(sock) : producer_d12(sock);
}

int consumer_main(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS)
        return 1;
    return g_d11_to_d12 ? consumer_d12(sock) : consumer_d11(sock);
}
} // namespace

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "d11d12";
    if (strcmp(dir, "d11d12") != 0 && strcmp(dir, "d12d11") != 0) {
        fprintf(stderr, "usage: %s <d11d12|d12d11>\n", argv[0]);
        return 2;
    }
    g_d11_to_d12 = strcmp(dir, "d11d12") == 0;
    printf("XFENCE: direction %s\n", dir);
    return run_fork_pair("XFENCE", producer_main, consumer_main);
}
