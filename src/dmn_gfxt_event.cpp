/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTEvent — Win32-style events and semaphores.
 *
 * Events are kqueue-backed: the handle is a tiny tagged box around a
 * kqueue fd whose single EVFILT_USER registration IS the signaled
 * state.  Signal = NOTE_TRIGGER.  Manual-reset events register without
 * EV_CLEAR (the trigger stays pending, so every waiter and poller sees
 * it until an explicit clear); auto-reset events register with
 * EV_CLEAR (retrieval consumes the trigger, and the kernel wakes
 * exactly one blocked waiter) — kqueue provides Win32 semantics
 * directly, with no userspace lock: the kqueue is the serializer.
 * Waiting is one blocking kevent(2) call; the pollable view handed out
 * by dmn_event_dup_fd() is a plain dup(2) of the same kqueue.  The one
 * cost is an fd per live event.
 *
 * Clearing deletes and re-adds the registration — the only way to
 * un-trigger EVFILT_USER.
 *
 * Semaphores use libdispatch, fd-free, unchanged.  The magic word at
 * the front of both boxes dispatches the public dmn_event_wait()
 * (D3DMetal implements frame-latency waitables as GFXT semaphores, so
 * app-facing waits must accept both).
 */

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/event.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>

#include "dmn_gfxt.h"
#include "dmn_log.h"

namespace {

constexpr uint32_t kEventMagic     = 0x544E5645; /* 'EVNT' */
constexpr uint32_t kSemaphoreMagic = 0x414D4553; /* 'SEMA' */

struct DmnEvent {
    uint32_t magic;
    bool     manual_reset;
    int      kq;
};

struct DmnSemaphore {
    uint32_t magic;
    std::atomic<uint32_t> refs;
    dispatch_semaphore_t sem;
};

DmnEvent* as_event(void* h) {
    auto* e = static_cast<DmnEvent*>(h);
    if (!e || e->magic != kEventMagic) {
        if (h)
            DMN_WARN("event op on non-event handle %p", h);
        return nullptr;
    }
    return e;
}

DmnSemaphore* as_semaphore(void* h) {
    auto* s = static_cast<DmnSemaphore*>(h);
    if (!s || s->magic != kSemaphoreMagic) {
        if (h)
            DMN_WARN("semaphore op on non-semaphore handle %p", h);
        return nullptr;
    }
    return s;
}

bool event_register(int kq, bool manual_reset) {
    struct kevent ev;
    uint16_t flags = EV_ADD | EV_ENABLE;
    if (!manual_reset)
        flags |= EV_CLEAR;
    EV_SET(&ev, 1, EVFILT_USER, flags, 0, 0, nullptr);
    return kevent(kq, &ev, 1, nullptr, 0, nullptr) == 0;
}

void event_signal(DmnEvent* e) {
    struct kevent ev;
    EV_SET(&ev, 1, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
    kevent(e->kq, &ev, 1, nullptr, 0, nullptr);
}

void event_clear(DmnEvent* e) {
    struct kevent ev;
    EV_SET(&ev, 1, EVFILT_USER, EV_DELETE, 0, 0, nullptr);
    kevent(e->kq, &ev, 1, nullptr, 0, nullptr);
    event_register(e->kq, e->manual_reset);
}

DmnEvent* event_alloc(bool manual_reset, bool initial_state) {
    int kq = kqueue();
    if (kq < 0) {
        DMN_ERROR("event: kqueue() failed (fd exhaustion? raise "
                  "RLIMIT_NOFILE)");
        return nullptr;
    }
    int fdflags = fcntl(kq, F_GETFD);
    if (fdflags >= 0)
        fcntl(kq, F_SETFD, fdflags | FD_CLOEXEC);
    if (!event_register(kq, manual_reset)) {
        close(kq);
        return nullptr;
    }

    auto* e = static_cast<DmnEvent*>(::malloc(sizeof(DmnEvent)));
    if (!e) {
        close(kq);
        return nullptr;
    }
    e->magic = kEventMagic;
    e->manual_reset = manual_reset;
    e->kq = kq;
    if (initial_state)
        event_signal(e);
    return e;
}

/* One blocking kevent, EINTR-safe.  For manual-reset events the
 * registration has no EV_CLEAR, so retrieval leaves the trigger pending
 * (all waiters wake, later waits return immediately); for auto-reset
 * EV_CLEAR makes the retrieval the consumption. */
dmn_wait_status event_wait(DmnEvent* e, uint64_t timeout_ns) {
    uint64_t deadline = 0;
    if (timeout_ns != UINT64_MAX)
        deadline = clock_gettime_nsec_np(CLOCK_MONOTONIC) + timeout_ns;

    for (;;) {
        struct timespec ts;
        struct timespec* tsp = nullptr;
        if (timeout_ns != UINT64_MAX) {
            uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC);
            uint64_t left = deadline > now ? deadline - now : 0;
            ts.tv_sec  = (time_t)(left / 1000000000ull);
            ts.tv_nsec = (long)(left % 1000000000ull);
            tsp = &ts;
        }
        struct kevent out;
        int n = kevent(e->kq, nullptr, 0, &out, 1, tsp);
        if (n > 0)
            return DMN_WAIT_SIGNALED;
        if (n == 0)
            return DMN_WAIT_TIMEOUT;
        if (errno != EINTR) {
            DMN_WARN("event: kevent wait failed: %d", errno);
            return DMN_WAIT_FAILED;
        }
    }
}

void destroy_semaphore(DmnSemaphore* s) {
    dispatch_release(s->sem);
    ::free(s);
}

} // namespace

DmnGFXTEvent::~DmnGFXTEvent() = default;

void* DmnGFXTEvent::CreateEvent(uint32_t manualReset, bool initialState) {
    DmnEvent* e = event_alloc(manualReset != 0, initialState);
    DMN_TRACE("CreateEvent(manualReset=%u, initial=%d) = %p",
              manualReset, initialState, (void*)e);
    return e;
}

void DmnGFXTEvent::SetEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_signal(e);
}

void DmnGFXTEvent::ClearEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_clear(e);
}

void DmnGFXTEvent::PulseEvent(void* handle) {
    /* Trigger-then-clear releases pollers and any waiter the kernel
     * schedules in between, ending non-signaled.  (Win32 PulseEvent is
     * documented-unreliable; this matches its spirit, not its letter.) */
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    event_signal(e);
    event_clear(e);
}

void DmnGFXTEvent::CloseEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    close(e->kq);
    e->magic = 0;
    ::free(e);
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
    /* A new box with its own dup of the kqueue: the kernel refcounts the
     * kqueue object, so either handle may be closed independently while
     * trigger state stays shared. */
    DmnEvent* e = as_event(handle);
    if (!e)
        return nullptr;
    int kq = dup(e->kq);
    if (kq < 0)
        return nullptr;
    auto* d = static_cast<DmnEvent*>(::malloc(sizeof(DmnEvent)));
    if (!d) {
        close(kq);
        return nullptr;
    }
    *d = *e;
    d->kq = kq;
    return d;
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
    (void)payloadSize;
    if (!fn)
        return;
    dispatch_queue_t queue =
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    if (waitForCompletion) {
        dispatch_sync(queue, ^{ fn(payload); });
    } else {
        dispatch_async(queue, ^{ fn(payload); });
    }
}

/* == Public C API (also used by dmn_fence_d3d.cpp internally) ============= */

extern "C" void* dmn_event_create(int manual_reset, int initial_state) {
    return event_alloc(manual_reset != 0, initial_state != 0);
}

extern "C" void dmn_event_signal(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_signal(e);
}

extern "C" void dmn_event_clear(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_clear(e);
}

extern "C" void dmn_event_close(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    close(e->kq);
    e->magic = 0;
    ::free(e);
}

extern "C" int dmn_event_dup_fd(void* handle) {
    DmnEvent* e = as_event(handle);
    return e ? dup(e->kq) : -1;
}

extern "C" dmn_wait_status dmn_event_wait(void* handle, uint64_t timeout_ns) {
    if (!handle)
        return DMN_WAIT_FAILED;
    uint32_t magic = *static_cast<uint32_t*>(handle);
    if (magic == kEventMagic)
        return event_wait(static_cast<DmnEvent*>(handle), timeout_ns);
    if (magic == kSemaphoreMagic) {
        auto* s = static_cast<DmnSemaphore*>(handle);
        dispatch_time_t deadline = timeout_ns == UINT64_MAX
            ? DISPATCH_TIME_FOREVER
            : dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ns);
        return dispatch_semaphore_wait(s->sem, deadline) == 0
            ? DMN_WAIT_SIGNALED : DMN_WAIT_TIMEOUT;
    }
    DMN_WARN("dmn_event_wait: %p is not a dmn handle", handle);
    return DMN_WAIT_FAILED;
}
