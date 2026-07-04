/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Public event API (dmn_event_create/signal/clear/close/wait/dup_fd),
 * windowless and without dmn_init — events are self-contained kqueue
 * handles. Covers Win32 CreateEvent semantics (manual/auto reset, initial
 * state, auto-reset consuming exactly one waiter), the pollable dup_fd view
 * (readability tracking signal/clear transitions, fds surviving
 * dmn_event_close, auto-reset waits clearing readability), and the
 * non-event-handle failure paths.
 *
 * Prints "EVENTS: PASS" and exits 0.
 */

#include <cstdint>
#include <cstdio>

#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <thread>

#include "d3dmetal_native.h"

#define T_TAG "EVENTS"
#include "common/check.h"

static const uint64_t kShortNs = 50ull * 1000 * 1000;         /* 50 ms */
static const uint64_t kLongNs  = 5ull * 1000 * 1000 * 1000;   /* 5 s */

/* poll() one fd for POLLIN with a millisecond timeout. */
static bool fd_readable(int fd, int timeout_ms) {
    struct pollfd pfd = {fd, POLLIN, 0};
    return poll(&pfd, 1, timeout_ms) == 1 && (pfd.revents & POLLIN);
}

int main() {
    /* 1) Auto-reset: one signal releases one wait, then it is consumed. */
    {
        void* ev = dmn_event_create(0, 0);
        EXPECT(ev, "auto-reset create failed");
        EXPECT(dmn_event_wait(ev, kShortNs) == DMN_WAIT_TIMEOUT,
               "unsignaled auto event did not time out");
        dmn_event_signal(ev);
        EXPECT(dmn_event_wait(ev, kLongNs) == DMN_WAIT_SIGNALED,
               "signaled auto event did not release the wait");
        EXPECT(dmn_event_wait(ev, kShortNs) == DMN_WAIT_TIMEOUT,
               "auto event was not consumed by the wait");
        dmn_event_close(ev);
        printf("EVENTS: auto-reset ok\n");
    }

    /* 2) initial_state: signaled at birth, both flavors. */
    {
        void* evAuto = dmn_event_create(0, 1);
        void* evMan = dmn_event_create(1, 1);
        EXPECT(evAuto && evMan, "initial-state create failed");
        EXPECT(dmn_event_wait(evAuto, kShortNs) == DMN_WAIT_SIGNALED,
               "initially-signaled auto event not signaled");
        EXPECT(dmn_event_wait(evAuto, kShortNs) == DMN_WAIT_TIMEOUT,
               "initially-signaled auto event not consumed");
        EXPECT(dmn_event_wait(evMan, kShortNs) == DMN_WAIT_SIGNALED,
               "initially-signaled manual event not signaled");
        dmn_event_close(evAuto);
        dmn_event_close(evMan);
        printf("EVENTS: initial state ok\n");
    }

    /* 3) Manual-reset: stays signaled across waits, clear rearms, and a
     *    signal after clear works (clear re-registers the kqueue filter). */
    {
        void* ev = dmn_event_create(1, 0);
        EXPECT(ev, "manual-reset create failed");
        dmn_event_signal(ev);
        EXPECT(dmn_event_wait(ev, kLongNs) == DMN_WAIT_SIGNALED,
               "manual event did not release the first wait");
        EXPECT(dmn_event_wait(ev, kShortNs) == DMN_WAIT_SIGNALED,
               "manual event did not stay signaled");
        dmn_event_clear(ev);
        EXPECT(dmn_event_wait(ev, kShortNs) == DMN_WAIT_TIMEOUT,
               "cleared manual event still signaled");
        dmn_event_signal(ev);
        EXPECT(dmn_event_wait(ev, kLongNs) == DMN_WAIT_SIGNALED,
               "signal after clear did not release the wait");
        dmn_event_close(ev);
        printf("EVENTS: manual-reset ok\n");
    }

    /* 4) Cross-thread wake, and auto-reset releasing exactly one of two
     *    blocked waiters per signal. */
    {
        void* ev = dmn_event_create(0, 0);
        EXPECT(ev, "create failed");
        std::atomic<int> signaled{0}, timed_out{0};
        auto waiter = [&] {
            dmn_wait_status s = dmn_event_wait(ev, 1500ull * 1000 * 1000);
            if (s == DMN_WAIT_SIGNALED)
                signaled.fetch_add(1);
            else if (s == DMN_WAIT_TIMEOUT)
                timed_out.fetch_add(1);
        };
        std::thread t1(waiter), t2(waiter);
        /* Let both block, then signal once. */
        usleep(200 * 1000);
        dmn_event_signal(ev);
        t1.join();
        t2.join();
        EXPECT(signaled.load() == 1 && timed_out.load() == 1,
               "one auto-reset signal must wake exactly one of two waiters");
        dmn_event_close(ev);
        printf("EVENTS: cross-thread auto-reset wake-one ok\n");
    }

    /* 5) dup_fd on a manual event: readability follows every signal/clear
     *    transition, on fds dup'd both before and after the signal. */
    {
        void* ev = dmn_event_create(1, 0);
        EXPECT(ev, "create failed");
        int fd1 = dmn_event_dup_fd(ev);
        EXPECT(fd1 >= 0, "dup_fd failed");
        EXPECT(!fd_readable(fd1, 0), "fd readable while event is clear");
        dmn_event_signal(ev);
        EXPECT(fd_readable(fd1, 1000), "fd not readable after signal");
        int fd2 = dmn_event_dup_fd(ev);
        EXPECT(fd2 >= 0 && fd_readable(fd2, 1000),
               "fd dup'd after the signal not readable");
        dmn_event_clear(ev);
        EXPECT(!fd_readable(fd1, 0) && !fd_readable(fd2, 0),
               "fds still readable after clear");
        dmn_event_signal(ev);
        EXPECT(fd_readable(fd1, 1000) && fd_readable(fd2, 1000),
               "fds did not track a signal after clear");
        close(fd1);
        close(fd2);
        dmn_event_close(ev);
        printf("EVENTS: dup_fd transition tracking ok\n");
    }

    /* 6) dup_fd on an auto event: a dmn_event_wait consumer clears
     *    readability too (the header's auto-reset caveat). */
    {
        void* ev = dmn_event_create(0, 0);
        EXPECT(ev, "create failed");
        int fd = dmn_event_dup_fd(ev);
        EXPECT(fd >= 0, "dup_fd failed");
        dmn_event_signal(ev);
        EXPECT(fd_readable(fd, 1000), "fd not readable after signal");
        EXPECT(dmn_event_wait(ev, kLongNs) == DMN_WAIT_SIGNALED,
               "wait did not consume the signal");
        EXPECT(!fd_readable(fd, 0),
               "fd readable after an auto-reset wait consumed the signal");
        close(fd);
        dmn_event_close(ev);
        printf("EVENTS: dup_fd auto-reset consumption ok\n");
    }

    /* 7) dup_fd survives dmn_event_close with its state intact. */
    {
        void* ev = dmn_event_create(1, 0);
        EXPECT(ev, "create failed");
        dmn_event_signal(ev);
        int fd = dmn_event_dup_fd(ev);
        EXPECT(fd >= 0, "dup_fd failed");
        dmn_event_close(ev);
        EXPECT(fd_readable(fd, 1000), "fd lost the signal across close");
        close(fd);
        printf("EVENTS: dup_fd survives close ok\n");
    }

    /* 8) Non-event handles: rejected without crashing. */
    {
        uint32_t bogus = 0;
        EXPECT(dmn_event_dup_fd(&bogus) == -1, "dup_fd accepted a non-event");
        dmn_event_signal(&bogus); /* no-op */
        dmn_event_clear(&bogus);  /* no-op */
        EXPECT(dmn_event_wait(&bogus, 0) == DMN_WAIT_FAILED,
               "wait accepted a non-event");
        EXPECT(dmn_event_wait(nullptr, 0) == DMN_WAIT_FAILED,
               "wait accepted NULL");
        printf("EVENTS: non-event handle rejection ok\n");
    }

    T_PASS();
    return 0;
}
