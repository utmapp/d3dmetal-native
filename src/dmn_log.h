/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Leveled logging. Threshold from the DMN_LOG environment variable
 * (error|warn|info|debug|trace, default warn); sink is stderr or the
 * user callback installed via dmn_init options.
 */

#pragma once

#include "d3dmetal_native.h"

#ifdef __cplusplus
extern "C" {
#endif

void dmn_log_impl(dmn_log_level level, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
bool dmn_log_enabled(dmn_log_level level);
void dmn_log_set_sink(void (*cb)(dmn_log_level, const char*, void*), void* ctx);

#ifdef __cplusplus
}
#endif

#define DMN_ERROR(...) dmn_log_impl(DMN_LOG_ERROR, __VA_ARGS__)
#define DMN_WARN(...)  dmn_log_impl(DMN_LOG_WARN,  __VA_ARGS__)
#define DMN_INFO(...)  dmn_log_impl(DMN_LOG_INFO,  __VA_ARGS__)
#define DMN_DEBUG(...) dmn_log_impl(DMN_LOG_DEBUG, __VA_ARGS__)
#define DMN_TRACE(...) dmn_log_impl(DMN_LOG_TRACE, __VA_ARGS__)
