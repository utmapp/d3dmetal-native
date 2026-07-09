/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTEvent — Win32-style events and semaphores.
 *
 * Events are kqueue-backed: a handle is a tiny tagged box referencing a
 * refcounted DmnEventCore that owns a kqueue fd whose single EVFILT_USER
 * registration IS the signaled state.  DuplicateEvent shares the core (see
 * DmnEventCore), so a signal on any handle is seen through all of them.
 * Signal = NOTE_TRIGGER.  Manual-reset events register without
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
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/event.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>

#include "dmn_gfxt.h"
#include "dmn_log.h"

namespace {

constexpr uint32_t kEventMagic     = 0x544E5645; /* 'EVNT' */
constexpr uint32_t kSemaphoreMagic = 0x414D4553; /* 'SEMA' */

/* Shared, refcounted event state.  A Win32 event HANDLE and every
 * DuplicateEvent of it are separate DmnEvent boxes that all reference one
 * DmnEventCore, so a SetEvent on any of them is visible through all of them:
 * both the in-process kqueue wait AND the cross-process pollable pipe.
 *
 * Why the sharing matters: at GFXT interface version 3 D3DMetal duplicates
 * the event it was handed and fires its asynchronous completion SetEvent on
 * the DUPLICATE.  If the pipe lived on the original box alone, that signal
 * would reach the shared kqueue (a dup shares the kernel object) but never
 * the pipe the other process polls -- so a fence waiter across the proxy
 * would hang.  Keeping kqueue + pipe in one shared core closes that gap. */
struct DmnEventCore {
    std::atomic<uint32_t> refs;
    bool       manual_reset; /* immutable after creation */
    int        kq;           /* the signaled state; immutable after creation */
    /* Exported pollable view.  dmn_event_dup_fd() must hand out an fd that
     * survives SCM_RIGHTS to another process, and macOS refuses to pass
     * kqueue fds across processes (sendmsg -> EINVAL).  So the cross-process
     * view is a pipe, created lazily on first dup (many events are in-process
     * only and never need one), whose readable state mirrors the kqueue
     * trigger: signal writes a token, clear drains it.  pipe_mtx guards the
     * lazy creation and every token write/drain: a signal sets the kqueue
     * trigger BEFORE taking the lock and creation primes the pipe under the
     * lock AFTER peeking, so a first-dup racing a signal never loses the edge
     * (whichever critical section runs second observes the other's effect). */
    std::mutex pipe_mtx;
    int        pipe_r; /* read end handed out (dup'd) by dmn_event_dup_fd */
    int        pipe_w; /* write end poked alongside the kqueue trigger */
};

/* A HANDLE: a magic tag plus a reference to the shared core. */
struct DmnEvent {
    uint32_t      magic;
    DmnEventCore* core;
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

/* Mark an fd close-on-exec and non-blocking (macOS has no pipe2). */
void set_cloexec_nonblock(int fd) {
    int fdflags = fcntl(fd, F_GETFD);
    if (fdflags >= 0)
        fcntl(fd, F_SETFD, fdflags | FD_CLOEXEC);
    int stflags = fcntl(fd, F_GETFL);
    if (stflags >= 0)
        fcntl(fd, F_SETFL, stflags | O_NONBLOCK);
}

/* Non-consuming peek for whether the trigger is pending.  Only valid for
 * manual-reset events: their registration has no EV_CLEAR, so a zero-timeout
 * kevent read does not consume the trigger.  Auto-reset would be consumed. */
bool event_peek_signaled(DmnEventCore* c) {
    if (!c->manual_reset)
        return false;
    struct kevent out;
    struct timespec zero = {0, 0};
    return kevent(c->kq, nullptr, 0, &out, 1, &zero) > 0;
}

void pipe_write_token(int w) {
    if (w < 0)
        return;
    const uint8_t b = 1;
    ssize_t n;
    do { n = ::write(w, &b, 1); } while (n < 0 && errno == EINTR);
    /* EAGAIN: the pipe already holds a token and is still readable — the
     * "signaled" edge is preserved, which is all a poller needs. */
}

void pipe_drain(int r) {
    if (r < 0)
        return;
    uint8_t buf[64];
    ssize_t n;
    do { n = ::read(r, buf, sizeof(buf)); } while (n > 0 || (n < 0 && errno == EINTR));
}

/* Lazily create the exported pipe under pipe_mtx, priming it if the event is
 * already signaled so a dup taken after the signal still reports readable. */
bool event_ensure_pipe(DmnEventCore* c) {
    std::lock_guard<std::mutex> lk(c->pipe_mtx);
    if (c->pipe_r >= 0)
        return true;
    int fds[2];
    if (::pipe(fds) != 0) {
        DMN_ERROR("event: pipe() failed for exported fd (fd exhaustion?)");
        return false;
    }
    set_cloexec_nonblock(fds[0]);
    set_cloexec_nonblock(fds[1]);
    c->pipe_r = fds[0];
    c->pipe_w = fds[1];
    if (event_peek_signaled(c))
        pipe_write_token(c->pipe_w);
    return true;
}

void event_signal(DmnEventCore* c) {
    struct kevent ev;
    EV_SET(&ev, 1, EVFILT_USER, 0, NOTE_TRIGGER, 0, nullptr);
    kevent(c->kq, &ev, 1, nullptr, 0, nullptr);
    /* The kqueue trigger is now set; take the lock only to poke the pipe.  A
     * first-dup that publishes the pipe after this trigger will observe it via
     * its post-creation peek, so the token is never lost. */
    std::lock_guard<std::mutex> lk(c->pipe_mtx);
    pipe_write_token(c->pipe_w);
}

void event_clear(DmnEventCore* c) {
    struct kevent ev;
    EV_SET(&ev, 1, EVFILT_USER, EV_DELETE, 0, 0, nullptr);
    kevent(c->kq, &ev, 1, nullptr, 0, nullptr);
    event_register(c->kq, c->manual_reset);
    std::lock_guard<std::mutex> lk(c->pipe_mtx);
    pipe_drain(c->pipe_r);
}

/* Allocate the shared core: a fresh kqueue with its EVFILT_USER registration,
 * no pipe yet (created on first dmn_event_dup_fd), one reference. */
DmnEventCore* core_alloc(bool manual_reset, bool initial_state) {
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

    auto* c = new (std::nothrow) DmnEventCore();
    if (!c) {
        close(kq);
        return nullptr;
    }
    c->refs.store(1, std::memory_order_relaxed);
    c->manual_reset = manual_reset;
    c->kq     = kq;
    c->pipe_r = -1;
    c->pipe_w = -1;
    if (initial_state)
        event_signal(c);
    return c;
}

void core_ref(DmnEventCore* c) {
    c->refs.fetch_add(1, std::memory_order_relaxed);
}

/* Drop a reference; the last one closes the kqueue and pipe and frees it. */
void core_unref(DmnEventCore* c) {
    if (c->refs.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    if (c->pipe_r >= 0) close(c->pipe_r);
    if (c->pipe_w >= 0) close(c->pipe_w);
    close(c->kq);
    delete c;
}

/* Hand out a dup of the exported pipe's read end (created on demand). */
int event_dup_fd_core(DmnEventCore* c) {
    if (!event_ensure_pipe(c))
        return -1;
    std::lock_guard<std::mutex> lk(c->pipe_mtx);
    return c->pipe_r >= 0 ? dup(c->pipe_r) : -1;
}

/* Wrap a fresh core in a HANDLE box. */
DmnEvent* event_alloc(bool manual_reset, bool initial_state) {
    DmnEventCore* c = core_alloc(manual_reset, initial_state);
    if (!c)
        return nullptr;
    auto* e = static_cast<DmnEvent*>(::malloc(sizeof(DmnEvent)));
    if (!e) {
        core_unref(c);
        return nullptr;
    }
    e->magic = kEventMagic;
    e->core  = c;
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
        int n = kevent(e->core->kq, nullptr, 0, &out, 1, tsp);
        if (n > 0) {
            /* Auto-reset: the EV_CLEAR read just consumed the single kqueue
             * trigger, so the pollable pipe must clear too — this waiter is the
             * consumer that won the race (Win32 semantics; see the header's
             * dmn_event_dup_fd caveat). Manual-reset events stay signaled until
             * an explicit clear, so their pipe is left alone. */
            if (!e->core->manual_reset) {
                std::lock_guard<std::mutex> lk(e->core->pipe_mtx);
                pipe_drain(e->core->pipe_r);
            }
            return DMN_WAIT_SIGNALED;
        }
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
    DmnEvent* e = event_alloc(manualReset != 0, initialState);
    DMN_TRACE("CreateEvent(manualReset=%u, initial=%d) = %p",
              manualReset, initialState, (void*)e);
    return e;
}

void DmnGFXTEvent::SetEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_signal(e->core);
}

void DmnGFXTEvent::ClearEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_clear(e->core);
}

void DmnGFXTEvent::PulseEvent(void* handle) {
    /* Trigger-then-clear releases pollers and any waiter the kernel
     * schedules in between, ending non-signaled.  (Win32 PulseEvent is
     * documented-unreliable; this matches its spirit, not its letter.) */
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    event_signal(e->core);
    event_clear(e->core);
}

void DmnGFXTEvent::CloseEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    core_unref(e->core);
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
    /* A second HANDLE onto the SAME shared core.  SetEvent, ClearEvent and
     * dmn_event_dup_fd on either box act on one kqueue and one exported pipe,
     * so a signal on the duplicate is visible both to an in-process waiter and
     * to a poller that dup'd the original's fd — the property D3DMetal's
     * version-3 duplicate-then-signal-async completion path relies on. */
    DmnEvent* e = as_event(handle);
    if (!e)
        return nullptr;
    auto* d = static_cast<DmnEvent*>(::malloc(sizeof(DmnEvent)));
    if (!d)
        return nullptr;
    core_ref(e->core);
    d->magic = kEventMagic;
    d->core  = e->core;
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
    return event_alloc(manual_reset != 0, initial_state != 0);
}

extern "C" void dmn_event_signal(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_signal(e->core);
}

extern "C" void dmn_event_clear(void* handle) {
    DmnEvent* e = as_event(handle);
    if (e)
        event_clear(e->core);
}

extern "C" void dmn_event_close(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    core_unref(e->core);
    e->magic = 0;
    ::free(e);
}

extern "C" int dmn_event_dup_fd(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return -1;
    /* Hand out a dup of the pipe read end, not the kqueue: the caller
     * passes this fd to another process via SCM_RIGHTS, which macOS
     * permits for pipes but rejects (EINVAL) for kqueues. */
    return event_dup_fd_core(e->core);
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
