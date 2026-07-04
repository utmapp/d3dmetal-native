/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * GFXT interface from D3DMetal
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "dmn_directx_types.h"

struct GFXTInterfaceVersion {
    uint32_t version;
    uint32_t _pad;
};

#define GFXT_INTERFACE_VERSION(v) GFXTInterfaceVersion{v, 0}

// ---------------------------------------------------------------------------
// Abstract GFXT interfaces. Method order and layout must match what
// D3DMetal expects exactly — do not reorder, insert, or remove entries.
// ---------------------------------------------------------------------------

class GFXTMonitorInterface {
public:
    // 32-byte record populated by WineMonitor::MonitorEnumProc.
    //
    // The binary writes the fields in two steps:
    //   1. RSI (HMONITOR) -> +0x00
    //   2. EBX (DEVMODEW.dmDisplayFrequency, devmode + 0xb8) -> +0x08
    //   3. XMM7 = [width, height, width, height] -> +0x0c via MOVDQU (16 bytes)
    // where width  = rcMonitor.right  - rcMonitor.left and
    //       height = rcMonitor.bottom - rcMonitor.top.
    //
    // The "logical" pair is initialized to the same value as the "physical"
    // pair – the duplication is preserved here so the struct lays out at the
    // exact offsets the rest of the library expects.
    struct MonitorInfo {
        void*   handle;            // +0x00  HMONITOR
        int32_t refreshRateHz;     // +0x08  DEVMODEW.dmDisplayFrequency
        int32_t physicalWidth;     // +0x0c  rcMonitor width
        int32_t physicalHeight;    // +0x10  rcMonitor height
        int32_t logicalWidth;      // +0x14  duplicate of physicalWidth
        int32_t logicalHeight;     // +0x18  duplicate of physicalHeight
        int32_t _pad1c;            // +0x1c
    };
    static_assert(sizeof(MonitorInfo) == 0x20, "MonitorInfo must be 32 bytes");

    // 20-byte projection of the DEVMODEW fields the binary cares about.
    //
    // QueryDisplayMode reads them out of EnumDisplaySettingsExW's output buffer:
    //     bitsPerPixel           = DEVMODEW.dmBitsPerPel        (devmode + 0xa8)
    //     pelsWidth              = DEVMODEW.dmPelsWidth         (devmode + 0xac)
    //     pelsHeight             = DEVMODEW.dmPelsHeight        (devmode + 0xb0)
    //     refreshRateNumerator   = DEVMODEW.dmDisplayFrequency * 1000  (devmode + 0xb8)
    //     refreshRateDenominator = 1000  (literal)
    //
    // ChangeDisplayMode pushes the same fields back into a DEVMODEW before
    // calling ChangeDisplaySettingsExW.
    struct DisplayModeInfo {
        uint32_t bitsPerPixel;            // +0x00
        uint32_t pelsWidth;               // +0x04
        uint32_t pelsHeight;              // +0x08
        uint32_t refreshRateNumerator;    // +0x0c  (dmDisplayFrequency * 1000)
        uint32_t refreshRateDenominator;  // +0x10  (always 1000)
    };
    static_assert(sizeof(DisplayModeInfo) == 20, "DisplayModeInfo must be 20 bytes");

    // 80-byte projection used by WineMonitor::QueryDescription.
    //
    // The binary builds a local MONITORINFOEXW (cbSize = 0x68), calls
    // GetMonitorInfoW into it, then re-shapes the result:
    //
    //   1. The four int32s of rcMonitor are loaded with MOVDQU and rearranged
    //      by `PSHUFD xmm, 0xe1`, which only swaps the *first two* lanes –
    //      i.e. (left, top, right, bottom) becomes (top, left, right, bottom).
    //   2. The 64-byte szDevice block is copied verbatim into the output
    //      with four MOVUPS.
    //
    // rcWork and dwFlags from MONITORINFOEXW are intentionally NOT copied.
    struct MonitorDescription {
        int32_t  monitorTop;        // +0x00  rcMonitor.top  (lane-swapped)
        int32_t  monitorLeft;       // +0x04  rcMonitor.left (lane-swapped)
        int32_t  monitorRight;      // +0x08  rcMonitor.right
        int32_t  monitorBottom;     // +0x0c  rcMonitor.bottom
        char16_t szDevice[32];      // +0x10..+0x4f  CCHDEVICENAME wchars
    };
    static_assert(sizeof(MonitorDescription) == 0x50,
                  "MonitorDescription must be 80 bytes");

    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTMonitorInterface() = default;
    virtual bool QueryMonitorInfo(uint32_t index, MonitorInfo& out) = 0;
    virtual bool QueryDisplayMode(uint64_t hmonitor, const char16_t* deviceName,
                                  uint32_t modeIndex, DisplayModeInfo& out) = 0;
    virtual bool QueryDescription(uint64_t hmonitor, MonitorDescription& out) = 0;
    virtual bool ChangeDisplayMode(uint64_t hmonitor, const char16_t* deviceName,
                                   const DisplayModeInfo& mode) = 0;
};

class GFXTRegistryInterface {
public:
    enum RegistryMainKey : uint32_t {
        HKEY_CLASSES_ROOT_  = 0,
        HKEY_CURRENT_USER_  = 1,
        HKEY_LOCAL_MACHINE_ = 2,
        HKEY_USERS_         = 3,
    };

    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTRegistryInterface() = default;
    virtual void* OpenKey(RegistryMainKey root, const char* subkey) = 0;
    virtual void* CreateKey(RegistryMainKey root, const char* subkey,
                            bool unused, bool* createdNew) = 0;
    virtual void  CloseKey(void* key) = 0;
    virtual void  SetValue(void* key, const char* name, uint32_t value) = 0;
    virtual void  SetValue(void* key, const char* name, uint64_t value) = 0;
    virtual void  SetValue(void* key, const char* name, const std::string& value) = 0;
    virtual void  SetValue(void* key, const char* name,
                           const std::vector<uint8_t>& value) = 0;
    virtual bool  GetValue(void* key, const char* name, uint32_t& out) = 0;
    virtual bool  GetValue(void* key, const char* name, uint64_t& out) = 0;
    virtual bool  GetValue(void* key, const char* name, std::string& out) = 0;
    virtual bool  GetValue(void* key, const char* name,
                           std::vector<uint8_t>& out) = 0;
    virtual bool  DeleteValue(void* key, const char* name, const char* unused) = 0;
};

class GFXTEventInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTEventInterface() = default;
    virtual void* CreateEvent(uint32_t manualReset, bool initialState) = 0;
    virtual void  SetEvent(void* handle) = 0;
    virtual void  ClearEvent(void* handle) = 0;
    virtual void  PulseEvent(void* handle) = 0;
    virtual void  CloseEvent(void* handle) = 0;
    virtual void* CreateSemaphore(uint32_t initialCount, uint32_t maximumCount) = 0;
    virtual void  WaitSemaphore(void* handle, uint64_t timeoutNs) = 0;
    virtual void  SignalSemaphore(void* handle, uint32_t releaseCount) = 0;
    virtual void  CloseSemaphore(void* handle) = 0;
    virtual void* DuplicateEvent(void* handle) = 0;
    virtual void* DuplicateSemaphore(void* handle) = 0;
    virtual void  _DispatchFunctionInternal(void (*fn)(const void*), bool waitForCompletion,
                                            const void* payload, uint32_t payloadSize) = 0;
};

class GFXTAdapterInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTAdapterInterface() = default;
    virtual uint32_t getAdapterLUIDs(LUID* outArray, size_t capacity) = 0;
};

class GFXTPathInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTPathInterface() = default;
    virtual bool windowsToUnixPath(const char16_t* in, char* out, size_t* inOutBytes) = 0;
    virtual bool unixToWindowsPath(const char* in, char16_t* out, size_t* inOutChars) = 0;
    virtual bool windowsSystemDirectoryPath(char16_t* out, size_t* inOutChars) = 0;
    virtual void getExecutablePath(char* out, uint32_t bytes) = 0;
    virtual void getModulePath(void* module, char* out, uint32_t bytes) = 0;
};

class GFXTAllocationInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTAllocationInterface() = default;
    virtual void* allocateBytesFromNewPages(size_t bytes) = 0;
    virtual void  freeBytesFromAllocatedPages(void* p, size_t bytes) = 0;
    virtual void* malloc(size_t bytes) = 0;
    virtual void  free(void* p) = 0;
    virtual void  makeExecutable(void* p, size_t bytes) = 0;
    virtual void* allocateBytesFromImage(const char* moduleName, size_t bytes) = 0;
};

class GFXTLibraryInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTLibraryInterface() = default;
    virtual void* loadLibrary(const char* name) = 0;
    virtual void* getModuleHandle(const char* name) = 0;
    virtual void* getProcAddress(void* module, const char* name) = 0;
    virtual void  freeLibrary(void* module) = 0;
    virtual void* loadLibraryFromSystemDirectory(const char* name) = 0;
};

class GFXTSwapchainInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTSwapchainInterface() = default;
    virtual bool InitializeForHWND(void* hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                                   D3D12_RESOURCE_DESC& outBackBufferDesc) = 0;
    virtual void* GetViewForHWND(void* hwnd) = 0;        // NSView
    virtual void* GetDrawableForHWND(void* hwnd) = 0;    // CAMetalDrawable
    virtual bool  ResizeWindow(void* hwnd, uint32_t width, uint32_t height) = 0;
    virtual bool  ResizeBacking(void* hwnd, uint32_t width, uint32_t height,
                                DXGI_FORMAT format) = 0;
    // The four numeric arguments are forwarded as `(X, Y, cx, cy)` to
    // SetWindowPos / MoveWindow: position first, then size — not size
    // followed by a monitor origin.
    virtual bool  SetFullscreen(void* hwnd, bool currentlyFullscreen, bool wantFullscreen,
                                uint32_t x, uint32_t y,
                                uint32_t width, uint32_t height) = 0;
    virtual bool  Present(void* hwnd) = 0;
};

class GFXTOSInterface {
public:
    virtual GFXTInterfaceVersion Version() const = 0;
    virtual ~GFXTOSInterface() = default;
    virtual GFXTMonitorInterface*    CreateMonitorInterface(const GFXTInterfaceVersion& v) = 0;
    virtual GFXTRegistryInterface*   CreateRegistryInterface(const GFXTInterfaceVersion& v) = 0;
    virtual GFXTEventInterface*      CreateEventInterface(const GFXTInterfaceVersion& v) = 0;
    virtual GFXTSwapchainInterface*  CreateSwapchainInterface(const GFXTInterfaceVersion& v,
                                                              void* mtlDevice) = 0;
    virtual GFXTAdapterInterface*    CreateAdapterInterface(const GFXTInterfaceVersion& v) = 0;
    virtual GFXTPathInterface*       CreatePathInterface(const GFXTInterfaceVersion& v) = 0;
    virtual GFXTAllocationInterface* CreateAllocationInterface(const GFXTInterfaceVersion& v) = 0;
    virtual GFXTLibraryInterface*    CreateLibraryInterface(const GFXTInterfaceVersion& v) = 0;
};
