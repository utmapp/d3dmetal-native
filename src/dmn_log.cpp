/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 */

#include "dmn_log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

std::atomic<int> g_level{-1};
void (*g_sink)(dmn_log_level, const char*, void*) = nullptr;
void* g_sink_ctx = nullptr;
std::mutex g_sink_mutex;

int resolve_level() {
    int lv = g_level.load(std::memory_order_relaxed);
    if (lv >= 0)
        return lv;
    lv = DMN_LOG_WARN;
    if (const char* env = getenv("DMN_LOG")) {
        if (!strcasecmp(env, "error")) lv = DMN_LOG_ERROR;
        else if (!strcasecmp(env, "warn"))  lv = DMN_LOG_WARN;
        else if (!strcasecmp(env, "info"))  lv = DMN_LOG_INFO;
        else if (!strcasecmp(env, "debug")) lv = DMN_LOG_DEBUG;
        else if (!strcasecmp(env, "trace")) lv = DMN_LOG_TRACE;
        else if (!strcasecmp(env, "quiet")) lv = -2; /* below error */
    }
    g_level.store(lv, std::memory_order_relaxed);
    return lv;
}

const char* level_tag(dmn_log_level level) {
    switch (level) {
    case DMN_LOG_ERROR: return "err";
    case DMN_LOG_WARN:  return "warn";
    case DMN_LOG_INFO:  return "info";
    case DMN_LOG_DEBUG: return "dbg";
    case DMN_LOG_TRACE: return "trace";
    }
    return "?";
}

} // namespace

extern "C" bool dmn_log_enabled(dmn_log_level level) {
    return (int)level <= resolve_level();
}

extern "C" void dmn_log_set_sink(void (*cb)(dmn_log_level, const char*, void*),
                                 void* ctx) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    g_sink = cb;
    g_sink_ctx = ctx;
}

extern "C" void dmn_log_impl(dmn_log_level level, const char* fmt, ...) {
    if (!dmn_log_enabled(level))
        return;

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    std::lock_guard<std::mutex> lock(g_sink_mutex);
    if (g_sink) {
        g_sink(level, buf, g_sink_ctx);
    } else {
        fprintf(stderr, "d3dmetal-native:%s: %s\n", level_tag(level), buf);
        fflush(stderr);
    }
}
