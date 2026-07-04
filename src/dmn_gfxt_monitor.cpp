/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTMonitor — CGDisplay-backed monitor enumeration. CoreGraphics
 * display APIs are plain C and thread-safe, so no AppKit/main-thread
 * involvement. HMONITOR values are CGDirectDisplayID + 1 (never 0;
 * display IDs are 32-bit so the +1 cannot truncate).
 */

#include <CoreGraphics/CoreGraphics.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "dmn_gfxt.h"
#include "dmn_log.h"

namespace {

constexpr uint32_t kMaxDisplays = 16;

uint64_t to_hmonitor(CGDirectDisplayID id) {
    return (uint64_t)id + 1;
}

CGDirectDisplayID from_hmonitor(uint64_t hmonitor) {
    return (CGDirectDisplayID)(hmonitor - 1);
}

uint32_t active_displays(CGDirectDisplayID* ids) {
    uint32_t count = 0;
    if (CGGetActiveDisplayList(kMaxDisplays, ids, &count) != kCGErrorSuccess)
        count = 0;
    if (count == 0) {
        /* Some session types (e.g. remote shells) report an empty active
         * list even though per-display queries against the main display
         * work fine. A monitor-less host hard-crashes D3DMetal (null
         * DXGIOutput), so always report at least the main display. */
        ids[0] = CGMainDisplayID();
        return 1;
    }
    return count;
}

/* Refresh rate of a display mode; 0.0 (virtual/external panels) -> 60. */
double mode_refresh(CGDisplayModeRef mode) {
    double rate = mode ? CGDisplayModeGetRefreshRate(mode) : 0.0;
    return rate > 0.0 ? rate : 60.0;
}

/* Cache of CGDisplayCopyAllDisplayModes per display (mode-list identity is
 * stable enough for our purposes; no invalidation in v1). */
std::mutex g_modes_mutex;
std::unordered_map<uint32_t, CFArrayRef> g_modes_cache;

CFArrayRef modes_for_display(CGDirectDisplayID id) {
    std::lock_guard<std::mutex> lock(g_modes_mutex);
    auto it = g_modes_cache.find(id);
    if (it != g_modes_cache.end())
        return it->second;
    CFArrayRef modes = CGDisplayCopyAllDisplayModes(id, nullptr);
    if (modes)
        g_modes_cache.emplace(id, modes); /* retained for process lifetime */
    return modes;
}

} // namespace

DmnGFXTMonitor::~DmnGFXTMonitor() = default;

bool DmnGFXTMonitor::QueryMonitorInfo(uint32_t index, MonitorInfo& out) {
    CGDirectDisplayID ids[kMaxDisplays];
    uint32_t count = active_displays(ids);
    if (index >= count) {
        DMN_TRACE("monitor: QueryMonitorInfo(%u) -> end (count=%u)", index,
                  count);
        return false;
    }

    CGDirectDisplayID id = ids[index];
    CGDisplayModeRef mode = CGDisplayCopyDisplayMode(id);
    int32_t width = mode ? (int32_t)CGDisplayModeGetPixelWidth(mode)
                         : (int32_t)CGDisplayPixelsWide(id);
    int32_t height = mode ? (int32_t)CGDisplayModeGetPixelHeight(mode)
                          : (int32_t)CGDisplayPixelsHigh(id);
    int32_t hz = (int32_t)(mode_refresh(mode) + 0.5);
    if (mode)
        CGDisplayModeRelease(mode);

    out.handle         = (void*)(uintptr_t)to_hmonitor(id);
    out.refreshRateHz  = hz;
    out.physicalWidth  = width;
    out.physicalHeight = height;
    out.logicalWidth   = width;
    out.logicalHeight  = height;
    out._pad1c         = 0;
    DMN_TRACE("monitor: QueryMonitorInfo(%u) -> id=%u %dx%d@%d", index, id,
              width, height, hz);
    return true;
}

bool DmnGFXTMonitor::QueryDisplayMode(uint64_t hmonitor,
                                      const char16_t* deviceName,
                                      uint32_t modeIndex,
                                      DisplayModeInfo& out) {
    (void)deviceName;
    CGDirectDisplayID id = from_hmonitor(hmonitor);
    CFArrayRef modes = modes_for_display(id);
    if (!modes || modeIndex >= (uint32_t)CFArrayGetCount(modes)) {
        DMN_TRACE("monitor: QueryDisplayMode(%llu, %u) -> end",
                  (unsigned long long)hmonitor, modeIndex);
        return false;
    }

    auto mode =
        (CGDisplayModeRef)CFArrayGetValueAtIndex(modes, (CFIndex)modeIndex);
    out.bitsPerPixel           = 32;
    out.pelsWidth              = (uint32_t)CGDisplayModeGetPixelWidth(mode);
    out.pelsHeight             = (uint32_t)CGDisplayModeGetPixelHeight(mode);
    out.refreshRateNumerator   = (uint32_t)(mode_refresh(mode) * 1000.0 + 0.5);
    out.refreshRateDenominator = 1000;
    DMN_TRACE("monitor: QueryDisplayMode(%llu, %u) -> %ux%u@%u/%u",
              (unsigned long long)hmonitor, modeIndex, out.pelsWidth,
              out.pelsHeight, out.refreshRateNumerator,
              out.refreshRateDenominator);
    return true;
}

bool DmnGFXTMonitor::QueryDescription(uint64_t hmonitor,
                                      MonitorDescription& out) {
    CGDirectDisplayID id = from_hmonitor(hmonitor);

    CGRect bounds = CGDisplayBounds(id); /* global display space, points */
    CGDisplayModeRef mode = CGDisplayCopyDisplayMode(id);
    int32_t width = mode ? (int32_t)CGDisplayModeGetPixelWidth(mode)
                         : (int32_t)bounds.size.width;
    int32_t height = mode ? (int32_t)CGDisplayModeGetPixelHeight(mode)
                          : (int32_t)bounds.size.height;
    if (mode)
        CGDisplayModeRelease(mode);

    out.monitorTop    = (int32_t)bounds.origin.y;
    out.monitorLeft   = (int32_t)bounds.origin.x;
    out.monitorRight  = out.monitorLeft + width;
    out.monitorBottom = out.monitorTop + height;

    /* "\\.\DISPLAYn" where n is the 1-based index in the active list. */
    uint32_t display_number = 1;
    CGDirectDisplayID ids[kMaxDisplays];
    uint32_t count = active_displays(ids);
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] == id) {
            display_number = i + 1;
            break;
        }
    }
    char ascii[32];
    snprintf(ascii, sizeof(ascii), "\\\\.\\DISPLAY%u", display_number);
    memset(out.szDevice, 0, sizeof(out.szDevice));
    for (size_t i = 0; ascii[i] && i < 31; i++)
        out.szDevice[i] = (char16_t)ascii[i];

    DMN_TRACE("monitor: QueryDescription(%llu) -> (%d,%d)-(%d,%d) %s",
              (unsigned long long)hmonitor, out.monitorLeft, out.monitorTop,
              out.monitorRight, out.monitorBottom, ascii);
    return true;
}

bool DmnGFXTMonitor::ChangeDisplayMode(uint64_t hmonitor,
                                       const char16_t* deviceName,
                                       const DisplayModeInfo& mode) {
    (void)deviceName;
    /* Accept-and-ignore: native apps own their windows; D3DMetal only asks
     * for mode switches in exclusive fullscreen. */
    DMN_INFO("monitor: ChangeDisplayMode(%llu, %ux%u@%u/%u) ignored",
             (unsigned long long)hmonitor, mode.pelsWidth, mode.pelsHeight,
             mode.refreshRateNumerator, mode.refreshRateDenominator);
    return true;
}
