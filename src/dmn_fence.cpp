/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * The shared fence-value slot protocol: a uint64 at the start of an fd-shared
 * companion buffer (the same shm backing as textures/buffers). The producer's
 * GPU writes the reached value once the fence's GPU work completes — no CPU
 * on the producer's signal path — and readers poll the mapped slot. Each
 * dmn_shared_fence_t is an independent view with its own mapping; both the
 * consumer side (imported fences, watcher threads) and the producer's own
 * merge/watch views in dmn_fence_d3d.cpp are built on it.
 *
 * Plain C++ (no Metal, no DirectX); the mapping comes from dmn_share_map_fd.
 */

#include <sched.h>
#include <time.h>

#include <cstdint>

#include "d3dmetal_native.h"
#include "dmn_fence.h"
#include "dmn_log.h"
#include "dmn_share.h"

namespace {

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* The GPU writes the 64-bit value as two 32-bit stores (low at +0, high at +4).
 * Read seqlock-style: if the high word is stable across the read, the pair is
 * consistent. Values are monotonic, so this converges immediately. */
uint64_t read64(const volatile uint32_t* s) {
    for (;;) {
        uint32_t hi1 = s[1];
        uint32_t lo = s[0];
        uint32_t hi2 = s[1];
        if (hi1 == hi2)
            return ((uint64_t)hi1 << 32) | lo;
    }
}

} // namespace

struct dmn_shared_fence {
    void* map;
    size_t map_size;
    volatile uint32_t* slot;
    uint64_t initial;
};

extern "C" dmn_shared_fence_t dmn_shared_fence_open(
        const dmn_shared_fence_handle* handle) {
    if (!handle || handle->magic != DMN_SHARED_FENCE_MAGIC || handle->fd < 0) {
        DMN_WARN("shared_fence_open: invalid handle");
        return nullptr;
    }
    size_t need = sizeof(uint64_t); /* the value slot sits at offset 0 */
    void* map = dmn_share_map_fd(handle->fd, need);
    if (!map) {
        DMN_ERROR("shared_fence_open: map(fd=%d, %zu) failed", handle->fd, need);
        return nullptr;
    }
    auto* f = new dmn_shared_fence();
    f->map = map;
    f->map_size = need;
    f->slot = reinterpret_cast<volatile uint32_t*>(map);
    f->initial = handle->initial_value;
    return f;
}

extern "C" void dmn_shared_fence_close(dmn_shared_fence_t f) {
    if (!f)
        return;
    dmn_share_unmap(f->map, f->map_size);
    delete f;
}

extern "C" uint64_t dmn_shared_fence_get_completed(dmn_shared_fence_t f) {
    if (!f)
        return 0;
    uint64_t v = read64(f->slot);
    return v > f->initial ? v : f->initial;
}

/* Same low-word-first store order as the GPU writers, so a torn cross-process
 * read under-reports rather than over-reports. */
extern "C" void dmn_shared_fence_signal(dmn_shared_fence_t f, uint64_t value) {
    if (!f || read64(f->slot) >= value)
        return;
    __atomic_store_n(&f->slot[0], (uint32_t)(value & 0xffffffffu),
                     __ATOMIC_RELEASE);
    __atomic_store_n(&f->slot[1], (uint32_t)(value >> 32), __ATOMIC_RELEASE);
}

extern "C" dmn_wait_status dmn_shared_fence_wait(dmn_shared_fence_t f,
                                                 uint64_t value,
                                                 uint64_t timeout_ns) {
    if (!f)
        return DMN_WAIT_FAILED;
    if (dmn_shared_fence_get_completed(f) >= value)
        return DMN_WAIT_SIGNALED;

    const bool infinite = (timeout_ns == DMN_WAIT_INFINITE);
    const uint64_t deadline = infinite ? 0 : now_ns() + timeout_ns;
    uint32_t spins = 0;
    for (;;) {
        if (dmn_shared_fence_get_completed(f) >= value)
            return DMN_WAIT_SIGNALED;
        if (!infinite && now_ns() >= deadline)
            return DMN_WAIT_TIMEOUT;
        /* Backoff: spin-yield briefly, then sleep. */
        if (spins < 128) {
            sched_yield();
            spins++;
        } else {
            struct timespec ns = {0, 200 * 1000}; /* 0.2 ms */
            nanosleep(&ns, nullptr);
        }
    }
}
