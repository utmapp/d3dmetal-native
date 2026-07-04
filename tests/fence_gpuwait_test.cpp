/*
 * GPU-side wait on an IMPORTED shared fence. ID3D12CommandQueue::Wait on an
 * imported fence is a native wait on the real local fence; the wait watcher
 * in dmn_fence_d3d.cpp must release it only when the producer's GPU writes V
 * into the shared slot.
 *
 * Parent = D3D12 producer (shared fence). Child = D3D12 consumer: it imports
 * the fence, and on its own queue enqueues Wait(imported, TARGET) followed by
 * Signal(localFence, DONE). It first proves localFence stays UNsignaled while
 * the producer has not reached TARGET (the wait genuinely blocks), then tells
 * the producer to signal, then confirms localFence reaches DONE (the watcher
 * released the wait). Prints "GPUWAIT: PASS".
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"
#include "common/dx12.h"
#include "common/ipc.h"
#include "common/util.h"

namespace {

constexpr uint64_t kTarget = 7;
constexpr uint64_t kDone = 0xABCD;
constexpr int kCycles = 8; /* repeated wait/release rounds after the first */

int producer(int sock) {
    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev))) { fprintf(stderr, "GPUWAIT: prod device FAILED\n"); return 1; }
    Com<ID3D12CommandQueue> q;
    make_d3d12_queue(dev.ptr(), q);
    Com<ID3D12Fence> fence;
    if (!q || FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                      __uuidof(ID3D12Fence), (void**)&fence))) {
        fprintf(stderr, "GPUWAIT: prod queue/fence FAILED\n"); return 1;
    }
    HANDLE h = nullptr;
    if (FAILED(dev->CreateSharedHandle(fence.ptr(), nullptr, 0, nullptr, &h)) || !h) {
        fprintf(stderr, "GPUWAIT: prod CreateSharedHandle FAILED\n"); return 1;
    }
    dmn_shared_fence_handle pod;
    memcpy(&pod, h, sizeof(pod));
    if (!send_with_fd(sock, &pod, sizeof(pod), pod.fd)) {
        fprintf(stderr, "GPUWAIT: prod send FAILED\n"); return 1;
    }
    dmn_shared_handle_close(h); /* NT-style handle; POD + fd are shipped */

    /* Wait for the consumer to arm its GPU wait + assert not-yet-signaled. */
    char go = 0;
    if (read(sock, &go, 1) != 1) { fprintf(stderr, "GPUWAIT: prod go FAILED\n"); return 1; }

    q->Signal(fence.ptr(), kTarget); /* GPU writes kTarget into the shared slot */
    printf("GPUWAIT: producer signaled target %llu\n", (unsigned long long)kTarget);

    /* Repeated rounds: signal each next value when the consumer asks. */
    for (int i = 1; i <= kCycles; i++) {
        char c = 0;
        if (read(sock, &c, 1) != 1) {
            fprintf(stderr, "GPUWAIT: prod cycle %d read FAILED\n", i);
            return 1;
        }
        q->Signal(fence.ptr(), kTarget + (uint64_t)i);
    }

    char ack = 0;
    if (read(sock, &ack, 1) != 1) { fprintf(stderr, "GPUWAIT: prod ack FAILED\n"); return 1; }
    return 0;
}

int consumer(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS) return 1;
    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev))) { fprintf(stderr, "GPUWAIT: cons device FAILED\n"); return 1; }

    dmn_shared_fence_handle pod;
    int fd = -1;
    if (!recv_with_fd(sock, &pod, sizeof(pod), &fd)) {
        fprintf(stderr, "GPUWAIT: cons recv FAILED\n"); return 1;
    }
    pod.fd = fd;
    if (pod.magic != DMN_SHARED_FENCE_MAGIC) {
        fprintf(stderr, "GPUWAIT: cons bad magic\n"); return 1;
    }
    Com<ID3D12Fence> imported;
    if (FAILED(dev->OpenSharedHandle((HANDLE)&pod, __uuidof(ID3D12Fence),
                                     (void**)&imported)) || !imported) {
        fprintf(stderr, "GPUWAIT: cons OpenSharedHandle FAILED\n"); return 1;
    }
    Com<ID3D12CommandQueue> q;
    make_d3d12_queue(dev.ptr(), q);
    Com<ID3D12Fence> local;
    if (!q || FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                      __uuidof(ID3D12Fence), (void**)&local))) {
        fprintf(stderr, "GPUWAIT: cons queue/local fence FAILED\n"); return 1;
    }

    /* GPU-side: wait for the imported fence, then bump the local fence. The
     * Signal cannot execute until the Wait is satisfied. */
    if (FAILED(q->Wait(imported.ptr(), kTarget))) {
        fprintf(stderr, "GPUWAIT: cons queue Wait FAILED\n"); return 1;
    }
    if (FAILED(q->Signal(local.ptr(), kDone))) {
        fprintf(stderr, "GPUWAIT: cons queue Signal FAILED\n"); return 1;
    }

    /* The producer has not signaled kTarget yet: the local fence must stay
     * unsignaled, proving the wait actually blocks the queue (and that the
     * watcher did not release it early or the wait get ignored). */
    sleep_ms(250);
    if (local->GetCompletedValue() >= kDone) {
        fprintf(stderr, "GPUWAIT: local fence completed BEFORE producer signaled "
                "— wait did not block\n");
        return 1;
    }
    char go = 1;
    if (write(sock, &go, 1) != 1) { fprintf(stderr, "GPUWAIT: cons go FAILED\n"); return 1; }

    /* Now the producer signals; the watcher must release the queue wait and
     * the local fence must reach kDone. */
    uint64_t start = now_ms();
    for (;;) {
        if (local->GetCompletedValue() >= kDone)
            break;
        if (now_ms() - start > 10000) {
            fprintf(stderr, "GPUWAIT: timed out waiting for local fence (watcher "
                    "did not release the GPU wait)\n");
            return 1;
        }
        sleep_ms(2);
    }

    /* Repeated wait/release cycles: a watcher that wedges after its first
     * release (e.g. by dead-locking the import state) hangs round two, not
     * round one. */
    for (int i = 1; i <= kCycles; i++) {
        if (FAILED(q->Wait(imported.ptr(), kTarget + (uint64_t)i)) ||
            FAILED(q->Signal(local.ptr(), kDone + (uint64_t)i))) {
            fprintf(stderr, "GPUWAIT: cycle %d enqueue FAILED\n", i);
            return 1;
        }
        char c = 1;
        if (write(sock, &c, 1) != 1) {
            fprintf(stderr, "GPUWAIT: cycle %d write FAILED\n", i);
            return 1;
        }
        start = now_ms();
        while (local->GetCompletedValue() < kDone + (uint64_t)i) {
            if (now_ms() - start > 10000) {
                fprintf(stderr, "GPUWAIT: cycle %d timed out (watcher wedged "
                        "after an earlier release?)\n", i);
                return 1;
            }
            sleep_ms(2);
        }
    }
    printf("GPUWAIT: PASS\n");
    char ack = 1;
    (void)write(sock, &ack, 1);
    return 0;
}

} // namespace

namespace {
int producer_main(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS)
        return 1;
    return producer(sock);
}
} // namespace

int main() {
    return run_fork_pair("GPUWAIT", producer_main, consumer);
}
