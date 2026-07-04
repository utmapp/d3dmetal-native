/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTEvent — Win32-style events and semaphores as magic-tagged,
 * refcounted heap handles. Events use pthread mutex+cond (manual/auto
 * reset semantics map directly); semaphores use libdispatch. Handles
 * stay in-process, so no fds are needed (games create many events).
 *
 * The same handles back the public dmn_event_wait(), which lets apps
 * wait on HANDLEs surfaced through D3D APIs (frame-latency objects).
 */

#include <dispatch/dispatch.h>
#include <pthread.h>

#include <atomic>
#include <cstdlib>

#include "dmn_gfxt.h"
#include "dmn_log.h"

namespace {

constexpr uint32_t kEventMagic     = 0x544E5645; /* 'EVNT' */
constexpr uint32_t kSemaphoreMagic = 0x414D4553; /* 'SEMA' */

struct DmnHandleHeader {
    uint32_t magic;
    std::atomic<uint32_t> refs;
};

struct DmnEvent : DmnHandleHeader {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    bool manual_reset;
    bool signaled;
};

struct DmnSemaphore : DmnHandleHeader {
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

void destroy_event(DmnEvent* e) {
    pthread_mutex_destroy(&e->mutex);
    pthread_cond_destroy(&e->cond);
    ::free(e);
}

void destroy_semaphore(DmnSemaphore* s) {
    dispatch_release(s->sem);
    ::free(s);
}

/* Returns DMN_WAIT_SIGNALED or DMN_WAIT_TIMEOUT. */
dmn_wait_status event_wait(DmnEvent* e, uint64_t timeout_ns) {
    dmn_wait_status status = DMN_WAIT_SIGNALED;
    pthread_mutex_lock(&e->mutex);
    if (timeout_ns == UINT64_MAX) {
        while (!e->signaled)
            pthread_cond_wait(&e->cond, &e->mutex);
    } else {
        struct timespec rel;
        rel.tv_sec  = (time_t)(timeout_ns / 1000000000ull);
        rel.tv_nsec = (long)(timeout_ns % 1000000000ull);
        while (!e->signaled) {
            if (pthread_cond_timedwait_relative_np(&e->cond, &e->mutex, &rel)
                    != 0) {
                /* macOS recomputes nothing for us; treat any wake past the
                 * relative deadline as a timeout check point. */
                if (!e->signaled)
                    status = DMN_WAIT_TIMEOUT;
                break;
            }
        }
    }
    if (status == DMN_WAIT_SIGNALED && !e->manual_reset)
        e->signaled = false; /* auto-reset consumes the signal */
    pthread_mutex_unlock(&e->mutex);
    return status;
}

} // namespace

DmnGFXTEvent::~DmnGFXTEvent() = default;

void* DmnGFXTEvent::CreateEvent(uint32_t manualReset, bool initialState) {
    auto* e = static_cast<DmnEvent*>(::calloc(1, sizeof(DmnEvent)));
    if (!e)
        return nullptr;
    e->magic = kEventMagic;
    e->refs.store(1, std::memory_order_relaxed);
    pthread_mutex_init(&e->mutex, nullptr);
    pthread_cond_init(&e->cond, nullptr);
    e->manual_reset = manualReset != 0;
    e->signaled = initialState;
    DMN_TRACE("CreateEvent(manualReset=%u, initial=%d) = %p",
              manualReset, initialState, (void*)e);
    return e;
}

void DmnGFXTEvent::SetEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    pthread_mutex_lock(&e->mutex);
    e->signaled = true;
    if (e->manual_reset)
        pthread_cond_broadcast(&e->cond);
    else
        pthread_cond_signal(&e->cond);
    pthread_mutex_unlock(&e->mutex);
}

void DmnGFXTEvent::ClearEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    pthread_mutex_lock(&e->mutex);
    e->signaled = false;
    pthread_mutex_unlock(&e->mutex);
}

void DmnGFXTEvent::PulseEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    pthread_mutex_lock(&e->mutex);
    /* Release current waiters, end up non-signaled (Win32 PulseEvent). */
    if (e->manual_reset)
        pthread_cond_broadcast(&e->cond);
    else
        pthread_cond_signal(&e->cond);
    e->signaled = false;
    pthread_mutex_unlock(&e->mutex);
}

void DmnGFXTEvent::CloseEvent(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        destroy_event(e);
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
    DmnEvent* e = as_event(handle);
    if (!e)
        return nullptr;
    e->refs.fetch_add(1, std::memory_order_relaxed);
    return e;
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

/* Internal: create/destroy a bare DmnEvent (kEventMagic, so D3DMetal's GFXT
 * event interface will SetEvent it) for dmn_fence_d3d.cpp to hand to
 * ID3D11Fence/ID3D12Fence::SetEventOnCompletion. */
extern "C" void* dmn_event_create(int manual_reset, int initial_state) {
    auto* e = static_cast<DmnEvent*>(::calloc(1, sizeof(DmnEvent)));
    if (!e)
        return nullptr;
    e->magic = kEventMagic;
    e->refs.store(1, std::memory_order_relaxed);
    pthread_mutex_init(&e->mutex, nullptr);
    pthread_cond_init(&e->cond, nullptr);
    e->manual_reset = manual_reset != 0;
    e->signaled = initial_state != 0;
    return e;
}

extern "C" void dmn_event_close(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
        destroy_event(e);
}

/* Internal: signal a DmnEvent handle (used by the shared-fence watchers in
 * dmn_fence_d3d.cpp). Same semantics as SetEvent. */
extern "C" void dmn_event_signal(void* handle) {
    DmnEvent* e = as_event(handle);
    if (!e)
        return;
    pthread_mutex_lock(&e->mutex);
    e->signaled = true;
    if (e->manual_reset)
        pthread_cond_broadcast(&e->cond);
    else
        pthread_cond_signal(&e->cond);
    pthread_mutex_unlock(&e->mutex);
}

/* == Public wait API ====================================================== */

extern "C" dmn_wait_status dmn_event_wait(void* handle, uint64_t timeout_ns) {
    if (!handle)
        return DMN_WAIT_FAILED;
    auto* hdr = static_cast<DmnHandleHeader*>(handle);
    if (hdr->magic == kEventMagic)
        return event_wait(static_cast<DmnEvent*>(handle), timeout_ns);
    if (hdr->magic == kSemaphoreMagic) {
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
