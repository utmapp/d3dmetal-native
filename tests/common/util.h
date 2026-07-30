/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Small time and process helpers for the test programs.
 */

#pragma once

#include <fcntl.h>
#include <stdint.h>
#include <time.h>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline uint64_t now_ms(void) {
    return now_ns() / 1000000ull;
}

static inline void sleep_ms(unsigned ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, nullptr);
}

/* Open file descriptors in this process. Every shared resource owns one (the
 * shared-memory object its backing lives in), so this is how a test observes
 * that the library reclaimed one — no private-data slot required, and it works
 * on object kinds whose destruction the framework defers.
 *
 * The scan bound is well past anything the suite opens; a test that hits it
 * would be leaking on a completely different scale. */
static inline int t_count_fds(void) {
    int n = 0;
    for (int fd = 0; fd < 4096; fd++)
        if (fcntl(fd, F_GETFD) != -1)
            n++;
    return n;
}
