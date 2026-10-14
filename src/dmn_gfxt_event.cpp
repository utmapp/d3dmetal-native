/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTEvent — Win32-style events and semaphores.
 *
 * Events are directx-native's (dxn_event.h): a HANDLE is a tagged box on a
 * refcounted kqueue core, waitable in-process and pollable across processes
 * through a pipe mirror.  DuplicateEvent shares the core, so a signal on any
 * handle is seen through all of them.  This file only adapts them to the GFXT
 * event interface and re-exports them as the dmn_event_* embedder API.
 *
 * Semaphores use libdispatch, fd-free.  The magic word at the front of both
 * boxes dispatches the public dmn_event_wait() (D3DMetal implements
 * frame-latency waitables as GFXT semaphores, so app-facing waits must accept
 * both).
 */

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>

#include <directx-native/dxn_event.h>

#include "dmn_gfxt.h"
#include "dmn_log.h"

namespace {

constexpr uint32_t kSemaphoreMagic = 0x414D4553; /* 'SEMA' */

/* At GFXT interface version 3 D3DMetal duplicates the event it was handed
 * and fires its asynchronous completion SetEvent on the DUPLICATE; dxn
 * duplicates share one core (kqueue + exported pipe), so that signal still
 * reaches a fence waiter polling the original's fd across the proxy. */

struct DmnSemaphore {
    uint32_t magic;
    std::atomic<uint32_t> refs;
    dispatch_semaphore_t sem;
};

DmnSemaphore* as_semaphore(void* h) {
    auto* s = static_cast<DmnSemaphore*>(h);
    if (!s || s->magic != kSemaphoreMagic) {
        if (h)
            DMN_WARN("semaphore op on non-semaphore handle %p", h);
        return nullptr;
    }
    return s;
}

void destroy_semaphore(DmnSemaphore* s) {
    dispatch_release(s->sem);
    ::free(s);
}

/* == os_sync wait-on-address backed dispatch queue ======================== */

/* os_sync_wait_on_address family (macOS 14.4+).  Resolved at runtime so one
 * build keeps running on older systems.  Signatures per
 * <os/os_sync_wait_on_address.h>; wait blocks the caller while the 4-byte word
 * at addr still equals `value`, flags 0 = process-private (NONE). */
using os_sync_wait_fn = int (*)(void* addr, uint64_t value, size_t size,
                                uint32_t flags);
using os_sync_wake_fn = int (*)(void* addr, size_t size, uint32_t flags);

struct OsSync {
    os_sync_wait_fn wait;
    os_sync_wake_fn wake_any;
    os_sync_wake_fn wake_all;
};

/* The resolved table, or nullptr on macOS < 14.4. */
const OsSync* os_sync() {
    static OsSync table;
    static bool ok = false;
    static std::once_flag once;
    std::call_once(once, [] {
        table.wait =
            (os_sync_wait_fn)dlsym(RTLD_DEFAULT, "os_sync_wait_on_address");
        table.wake_any =
            (os_sync_wake_fn)dlsym(RTLD_DEFAULT, "os_sync_wake_by_address_any");
        table.wake_all =
            (os_sync_wake_fn)dlsym(RTLD_DEFAULT, "os_sync_wake_by_address_all");
        ok = table.wait && table.wake_any && table.wake_all;
    });
    return ok ? &table : nullptr;
}

constexpr uint32_t kSlotCount     = 32; /* one bit per slot in the bitmaps */
constexpr uint32_t kInlinePayload = 32; /* payloads up to this size copy inline */

enum PayloadKind : uint8_t {
    kPayloadInline   = 0, /* bytes live in slot.inln                          */
    kPayloadBorrowed = 1, /* slot.ptr aliases the caller's buffer, or is null */
    kPayloadHeap     = 2, /* slot.ptr is malloc'd; the worker frees it        */
};

struct DispatchSlot {
    void (*fn)(const void*);
    const void*           ptr;      /* payload for kPayloadBorrowed/kPayloadHeap */
    std::atomic<uint32_t> done;     /* blocking handshake: 1 pending, 0 complete */
    PayloadKind           kind;
    bool                  blocking;
    alignas(16) uint8_t   inln[kInlinePayload];
};

/* A bounded lock-free MPSC queue: many producers (_DispatchFunctionInternal
 * callers), one worker.  `alloc` bit i marks slot i owned by a producer;
 * `ready` bit i marks it filled and awaiting the worker. */
struct Dispatcher {
    std::atomic<uint32_t> alloc; /* bit i: slot i in use            */
    std::atomic<uint32_t> ready; /* bit i: slot i filled, unclaimed */
    DispatchSlot          slots[kSlotCount];
};

void dispatcher_worker(Dispatcher* d) {
    const OsSync* os = os_sync();
    for (;;) {
        uint32_t r = d->ready.load(std::memory_order_acquire);
        while (r == 0) {
            os->wait(&d->ready, 0, sizeof(uint32_t), 0);
            r = d->ready.load(std::memory_order_acquire);
        }
        uint32_t i   = (uint32_t)__builtin_ctz(r);
        uint32_t bit = 1u << i;
        d->ready.fetch_and(~bit, std::memory_order_acquire); /* claim the slot */

        DispatchSlot& s        = d->slots[i];
        void (*fn)(const void*) = s.fn;
        const void* arg =
            s.kind == kPayloadInline ? (const void*)s.inln : s.ptr;
        bool blocking = s.blocking;

        if (fn)
            fn(arg);
        if (s.kind == kPayloadHeap)
            ::free(const_cast<void*>(s.ptr));

        if (blocking) {
            /* The producer is parked on done and owns freeing the slot. */
            s.done.store(0, std::memory_order_release);
            os->wake_all(&s.done, sizeof(uint32_t), 0);
        } else {
            /* Async slot is ours: release it and wake any full-queue waiter. */
            d->alloc.fetch_and(~bit, std::memory_order_release);
            os->wake_all(&d->alloc, sizeof(uint32_t), 0);
        }
    }
}

/* Process-wide singleton; the detached worker lives for the process lifetime,
 * matching libdispatch's global queues. */
Dispatcher* dispatcher() {
    static Dispatcher* d = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        d = new Dispatcher();
        d->alloc.store(0, std::memory_order_relaxed);
        d->ready.store(0, std::memory_order_relaxed);
        std::thread(dispatcher_worker, d).detach();
    });
    return d;
}

void dispatcher_submit(void (*fn)(const void*), bool blocking,
                       const void* payload, uint32_t size) {
    const OsSync* os = os_sync();
    Dispatcher*   d  = dispatcher();

    /* 1. Claim a free slot, waiting if all 32 are in flight. */
    uint32_t i;
    for (;;) {
        uint32_t cur = d->alloc.load(std::memory_order_relaxed);
        if (cur == 0xFFFFFFFFu) {
            os->wait(&d->alloc, 0xFFFFFFFFu, sizeof(uint32_t), 0);
            continue;
        }
        i = (uint32_t)__builtin_ctz(~cur);
        if (d->alloc.compare_exchange_weak(cur, cur | (1u << i),
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed))
            break;
    }
    uint32_t      bit = 1u << i;
    DispatchSlot& s   = d->slots[i];

    /* 2. Fill the slot.  Async payloads are copied (the caller's buffer may be
     *    gone before the worker runs); a blocking call borrows the pointer
     *    directly, since we do not return until the callback has run. */
    s.fn       = fn;
    s.blocking = blocking;
    if (blocking)
        s.done.store(1, std::memory_order_relaxed);
    if (!payload || size == 0) {
        s.kind = kPayloadBorrowed;
        s.ptr  = nullptr;
    } else if (blocking) {
        s.kind = kPayloadBorrowed;
        s.ptr  = payload;
    } else if (size <= kInlinePayload) {
        s.kind = kPayloadInline;
        ::memcpy(s.inln, payload, size);
    } else {
        void* heap = ::malloc(size);
        s.kind     = kPayloadHeap;
        s.ptr      = heap;
        if (heap)
            ::memcpy(heap, payload, size);
    }

    /* 3. Publish; wake the worker only if it may be parked (queue was empty). */
    uint32_t prev = d->ready.fetch_or(bit, std::memory_order_release);
    if (prev == 0)
        os->wake_any(&d->ready, sizeof(uint32_t), 0);

    /* 4. Blocking: park until the worker completes, then release the slot. */
    if (blocking) {
        while (s.done.load(std::memory_order_acquire) == 1)
            os->wait(&s.done, 1, sizeof(uint32_t), 0);
        d->alloc.fetch_and(~bit, std::memory_order_release);
        os->wake_all(&d->alloc, sizeof(uint32_t), 0);
    }
}

} // namespace

DmnGFXTEvent::~DmnGFXTEvent() = default;

GFXTInterfaceVersion DmnGFXTEvent::Version() const {
    /* Version 3 advertises the os_sync-backed _DispatchFunctionInternal, which
     * exists only on macOS 14.4+; older systems keep the libdispatch path and
     * report 2. */
    return GFXT_INTERFACE_VERSION(os_sync() ? 3u : 2u);
}

void* DmnGFXTEvent::CreateEvent(uint32_t manualReset, bool initialState) {
    void* e = dxn_event_create(manualReset != 0, initialState);
    DMN_TRACE("CreateEvent(manualReset=%u, initial=%d) = %p",
              manualReset, initialState, e);
    return e;
}

void DmnGFXTEvent::SetEvent(void* handle) {
    dxn_event_signal(handle);
}

void DmnGFXTEvent::ClearEvent(void* handle) {
    dxn_event_clear(handle);
}

void DmnGFXTEvent::PulseEvent(void* handle) {
    /* Trigger-then-clear releases pollers and any waiter the kernel
     * schedules in between, ending non-signaled.  (Win32 PulseEvent is
     * documented-unreliable; this matches its spirit, not its letter.) */
    if (!dxn_event_is_event(handle)) {
        dxn_event_signal(handle); /* warns */
        return;
    }
    dxn_event_signal(handle);
    dxn_event_clear(handle);
}

void DmnGFXTEvent::CloseEvent(void* handle) {
    dxn_event_close(handle);
}

void* DmnGFXTEvent::CreateSemaphore(uint32_t initialCount,
                                    uint32_t maximumCount) {
    (void)maximumCount;
    auto* s = static_cast<DmnSemaphore*>(::calloc(1, sizeof(DmnSemaphore)));
    if (!s)
        return nullptr;
    s->magic = kSemaphoreMagic;
    s->refs.store(1, std::memory_order_relaxed);
    s->sem = dispatch_semaphore_create((long)initialCount);
    if (!s->sem) {
        ::free(s);
        return nullptr;
    }
    DMN_TRACE("CreateSemaphore(initial=%u, max=%u) = %p",
              initialCount, maximumCount, (void*)s);
    return s;
}

void DmnGFXTEvent::WaitSemaphore(void* handle, uint64_t timeoutNs) {
    DmnSemaphore* s = as_semaphore(handle);
    if (!s)
        return;
    dispatch_time_t deadline = timeoutNs == UINT64_MAX
        ? DISPATCH_TIME_FOREVER
        : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeoutNs);
    dispatch_semaphore_wait(s->sem, deadline);
}

void DmnGFXTEvent::SignalSemaphore(void* handle, uint32_t releaseCount) {
    DmnSemaphore* s = as_semaphore(handle);
    if (!s)
        return;
    for (uint32_t i = 0; i < releaseCount; i++)
        dispatch_semaphore_signal(s->sem);
}

void DmnGFXTEvent::CloseSemaphore(void* handle) {
    DmnSemaphore* s = as_semaphore(handle);
    if (!s)
        return;
    if (s->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        destroy_semaphore(s);
}

void* DmnGFXTEvent::DuplicateEvent(void* handle) {
    /* A second HANDLE onto the SAME shared core.  SetEvent, ClearEvent and
     * dmn_event_dup_fd on either box act on one kqueue and one exported pipe,
     * so a signal on the duplicate is visible both to an in-process waiter and
     * to a poller that dup'd the original's fd — the property D3DMetal's
     * version-3 duplicate-then-signal-async completion path relies on. */
    return dxn_event_duplicate(handle);
}

void* DmnGFXTEvent::DuplicateSemaphore(void* handle) {
    DmnSemaphore* s = as_semaphore(handle);
    if (!s)
        return nullptr;
    s->refs.fetch_add(1, std::memory_order_relaxed);
    return s;
}

void DmnGFXTEvent::_DispatchFunctionInternal(void (*fn)(const void*),
                                             bool waitForCompletion,
                                             const void* payload,
                                             uint32_t payloadSize) {
    if (!fn)
        return;

    /* This entry point exists only on the version-3 interface, which Version()
     * reports solely when os_sync is available; a version-aware caller must not
     * reach it otherwise.  Trap rather than silently degrade if that contract
     * is violated. */
    if (!os_sync()) {
        DMN_ERROR("_DispatchFunctionInternal called without os_sync "
                  "(macOS < 14.4); Version() should have gated this");
        abort();
    }

    dispatcher_submit(fn, waitForCompletion, payload, payloadSize);
}

/* == Public C API (also used by dmn_fence_d3d.cpp internally) ============= */

extern "C" void* dmn_event_create(int manual_reset, int initial_state) {
    return dxn_event_create(manual_reset, initial_state);
}

extern "C" void dmn_event_signal(void* handle) { dxn_event_signal(handle); }

extern "C" void dmn_event_clear(void* handle) { dxn_event_clear(handle); }

extern "C" void dmn_event_close(void* handle) { dxn_event_close(handle); }

extern "C" void* dmn_event_duplicate(void* handle) {
    return dxn_event_duplicate(handle);
}

extern "C" int dmn_event_dup_fd(void* handle) {
    return dxn_event_dup_fd(handle);
}

extern "C" dxn_wait_status dmn_event_wait(void* handle, uint64_t timeout_ns) {
    if (!handle)
        return DXN_WAIT_FAILED;
    if (dxn_event_is_event(handle))
        return dxn_event_wait(handle, timeout_ns);
    if (*static_cast<uint32_t*>(handle) == kSemaphoreMagic) {
        auto* s = static_cast<DmnSemaphore*>(handle);
        dispatch_time_t deadline = timeout_ns == UINT64_MAX
            ? DISPATCH_TIME_FOREVER
            : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ns);
        return dispatch_semaphore_wait(s->sem, deadline) == 0
            ? DXN_WAIT_SIGNALED : DXN_WAIT_TIMEOUT;
    }
    DMN_WARN("dmn_event_wait: %p is not a dmn handle", handle);
    return DXN_WAIT_FAILED;
}
