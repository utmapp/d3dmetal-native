/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Shared check macros for the test programs. Define T_TAG (the test's log
 * prefix, e.g. "XFENCE") before including. All failure paths `return 1` from
 * the enclosing function.
 *
 *   CK(hr-expr, what)     — fail on FAILED(hr), quiet on success.
 *   CK_OK(hr-expr, what)  — like CK but also prints "<TAG>: <what>: OK".
 *   EXPECT(cond, msg)     — fail with msg when cond is false.
 *   T_PASS()              — print the conventional "<TAG>: PASS" line.
 */

#pragma once

#include <stdio.h>

#ifndef T_TAG
#error "define T_TAG before including common/check.h"
#endif

#define CK(expr, what)                                                       \
    do {                                                                     \
        HRESULT hr_ = (expr);                                                \
        if (FAILED(hr_)) {                                                   \
            fprintf(stderr, T_TAG ": %s FAILED 0x%08x\n", what,              \
                    (unsigned)hr_);                                          \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define CK_OK(expr, what)                                                    \
    do {                                                                     \
        HRESULT hr_ = (expr);                                                \
        if (FAILED(hr_)) {                                                   \
            fprintf(stderr, T_TAG ": %s FAILED 0x%08x\n", what,              \
                    (unsigned)hr_);                                          \
            return 1;                                                        \
        }                                                                    \
        printf(T_TAG ": %s: OK\n", what);                                    \
    } while (0)

#define EXPECT(cond, msg)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, T_TAG ": %s\n", msg);                            \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define T_PASS() printf(T_TAG ": PASS\n")
