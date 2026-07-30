/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Capability skips. The suite runs against whichever D3DMetal.framework the
 * environment points at, and the older Game Porting Toolkit releases simply do
 * not implement parts of D3D11/D3D12 that the newer ones do — GPTk 1.0 and 2.1
 * have no D3D11 fences (ID3D11Device5::CreateFence), no
 * ID3D11Device4::CreateTexture2D1, no
 * ID3D12Device1::SetEventOnMultipleFenceCompletion, and ship no D3DCompile at
 * all. A test that needs one of those has nothing to say about that framework,
 * so it reports SKIP rather than FAIL: a red suite should mean this library is
 * broken, not that Apple had not written the entry point yet.
 *
 * Define T_TAG before including (as for common/check.h).
 *
 *   T_SKIP("why", ...)         — print the reason and exit 77 (meson's "skip").
 *   T_SKIP_IF_UNIMPL(hr, what) — skip when `hr` says "not implemented".
 *   t_unimplemented(hr)        — the predicate on its own.
 *   T_SKIP_WITHOUT(entry)      — skip unless D3DMetal exports `entry`.
 *
 * In a forked producer/consumer test, check capabilities BEFORE forking:
 * exit 77 from a child is just an exit code to the parent, not a suite skip.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "d3dmetal_native.h"

#ifndef T_TAG
#error "define T_TAG before including common/skip.h"
#endif

/* meson's exit code for "this test did not run". */
#define T_SKIP_CODE 77

#define T_SKIP(...)                                                          \
    do {                                                                     \
        printf(T_TAG ": SKIP — ");                                           \
        printf(__VA_ARGS__);                                                 \
        printf("\n");                                                        \
        fflush(stdout);                                                      \
        exit(T_SKIP_CODE);                                                   \
    } while (0)

/* The two ways D3DMetal reports an entry point it does not implement:
 * E_NOTIMPL from the D3D11/D3D12 methods, DXGI_ERROR_UNSUPPORTED from a few
 * DXGI-flavoured ones. Both come with an "Unsupported API: ..." line in
 * D3DMetal's own log. */
static inline bool t_unimplemented(HRESULT hr) {
    return hr == (HRESULT)0x80004001L ||  /* E_NOTIMPL */
           hr == (HRESULT)0x887A0004L;    /* DXGI_ERROR_UNSUPPORTED */
}

#define T_SKIP_IF_UNIMPL(expr, what)                                         \
    do {                                                                     \
        HRESULT hr_ = (expr);                                                \
        if (t_unimplemented(hr_))                                            \
            T_SKIP("%s is not implemented by this D3DMetal (0x%08x)", what,   \
                   (unsigned)hr_);                                           \
        if (FAILED(hr_)) {                                                   \
            fprintf(stderr, T_TAG ": %s FAILED 0x%08x\n", what,              \
                    (unsigned)hr_);                                          \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define T_SKIP_WITHOUT(entry)                                                \
    do {                                                                     \
        if (!dmn_framework_has_entry_point(entry))                           \
            T_SKIP("this D3DMetal does not export %s", entry);                \
    } while (0)
