/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * d3dmetal-native implementations of the GFXT host interfaces.
 * Plain C++ declarations (Objective-C objects held as void*) so both
 * .cpp and .mm translation units can include this header.
 *
 * Interface versions mirror the ones libd3dshared.dylib reports.
 */

#pragma once

#include <memory>

#include "d3dmetal_gfxt.h"
#include "d3dmetal_native.h" /* dmn_wait_status */

class DmnGFXTMonitor final : public GFXTMonitorInterface {
public:
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(2); }
    ~DmnGFXTMonitor() override;
    bool QueryMonitorInfo(uint32_t index, MonitorInfo& out) override;
    bool QueryDisplayMode(uint64_t hmonitor, const char16_t* deviceName,
                          uint32_t modeIndex, DisplayModeInfo& out) override;
    bool QueryDescription(uint64_t hmonitor, MonitorDescription& out) override;
    bool ChangeDisplayMode(uint64_t hmonitor, const char16_t* deviceName,
                           const DisplayModeInfo& mode) override;
};

class DmnGFXTRegistry final : public GFXTRegistryInterface {
public:
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(2); }
    ~DmnGFXTRegistry() override;
    void* OpenKey(RegistryMainKey root, const char* subkey) override;
    void* CreateKey(RegistryMainKey root, const char* subkey,
                    bool unused, bool* createdNew) override;
    void  CloseKey(void* key) override;
    void  SetValue(void* key, const char* name, uint32_t value) override;
    void  SetValue(void* key, const char* name, uint64_t value) override;
    void  SetValue(void* key, const char* name, const std::string& value) override;
    void  SetValue(void* key, const char* name,
                   const std::vector<uint8_t>& value) override;
    bool  GetValue(void* key, const char* name, uint32_t& out) override;
    bool  GetValue(void* key, const char* name, uint64_t& out) override;
    bool  GetValue(void* key, const char* name, std::string& out) override;
    bool  GetValue(void* key, const char* name,
                   std::vector<uint8_t>& out) override;
    bool  DeleteValue(void* key, const char* name, const char* unused) override;
};

class DmnGFXTEvent final : public GFXTEventInterface {
public:
    GFXTInterfaceVersion Version() const override;
    ~DmnGFXTEvent() override;
    void* CreateEvent(uint32_t manualReset, bool initialState) override;
    void  SetEvent(void* handle) override;
    void  ClearEvent(void* handle) override;
    void  PulseEvent(void* handle) override;
    void  CloseEvent(void* handle) override;
    void* CreateSemaphore(uint32_t initialCount, uint32_t maximumCount) override;
    void  WaitSemaphore(void* handle, uint64_t timeoutNs) override;
    void  SignalSemaphore(void* handle, uint32_t releaseCount) override;
    void  CloseSemaphore(void* handle) override;
    void* DuplicateEvent(void* handle) override;
    void* DuplicateSemaphore(void* handle) override;
    void  _DispatchFunctionInternal(void (*fn)(const void*), bool waitForCompletion,
                                    const void* payload, uint32_t payloadSize) override;
};

class DmnGFXTAdapter final : public GFXTAdapterInterface {
public:
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(1); }
    ~DmnGFXTAdapter() override;
    uint32_t getAdapterLUIDs(LUID* outArray, size_t capacity) override;
};

class DmnGFXTPath final : public GFXTPathInterface {
public:
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(3); }
    ~DmnGFXTPath() override;
    bool windowsToUnixPath(const char16_t* in, char* out, size_t* inOutBytes) override;
    bool unixToWindowsPath(const char* in, char16_t* out, size_t* inOutChars) override;
    bool windowsSystemDirectoryPath(char16_t* out, size_t* inOutChars) override;
    void getExecutablePath(char* out, uint32_t bytes) override;
    void getModulePath(void* module, char* out, uint32_t bytes) override;
};

class DmnGFXTAllocation final : public GFXTAllocationInterface {
public:
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(2); }
    ~DmnGFXTAllocation() override;
    void* allocateBytesFromNewPages(size_t bytes) override;
    void  freeBytesFromAllocatedPages(void* p, size_t bytes) override;
    void* malloc(size_t bytes) override;
    void  free(void* p) override;
    void  makeExecutable(void* p, size_t bytes) override;
    void* allocateBytesFromImage(const char* moduleName, size_t bytes) override;
};

class DmnGFXTLibrary final : public GFXTLibraryInterface {
public:
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(2); }
    ~DmnGFXTLibrary() override;
    void* loadLibrary(const char* name) override;
    void* getModuleHandle(const char* name) override;
    void* getProcAddress(void* module, const char* name) override;
    void  freeLibrary(void* module) override;
    void* loadLibraryFromSystemDirectory(const char* name) override;
};

class DmnGFXTSwapchain final : public GFXTSwapchainInterface {
public:
    /* mtlDevice is the id<MTLDevice> D3DMetal passed to
     * CreateSwapchainInterface (retained for the swapchain's lifetime). */
    explicit DmnGFXTSwapchain(void* mtlDevice);
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(3); }
    ~DmnGFXTSwapchain() override;

    bool InitializeForHWND(void* hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                           D3D12_RESOURCE_DESC& outBackBufferDesc) override;
    void* GetViewForHWND(void* hwnd) override;
    void* GetDrawableForHWND(void* hwnd) override;
    bool  ResizeWindow(void* hwnd, uint32_t width, uint32_t height) override;
    bool  ResizeBacking(void* hwnd, uint32_t width, uint32_t height,
                        DXGI_FORMAT format) override;
    bool  SetFullscreen(void* hwnd, bool currentlyFullscreen, bool wantFullscreen,
                        uint32_t x, uint32_t y,
                        uint32_t width, uint32_t height) override;
    bool  Present(void* hwnd) override;

private:
    void* mtl_device_;     /* id<MTLDevice>, retained */
    void* last_drawable_;  /* id<CAMetalDrawable>, retained; only when the
                              DMN_PRESENT_FALLBACK env toggle is active */
    void* present_layer_;  /* id<CAMetalLayer>, retained at InitializeForHWND.
                              D3DMetal owns this swapchain and may fire an
                              async present after the app has already called
                              dmn_window_destroy; holding our own reference
                              lets that late present resolve a drawable
                              instead of failing the hwnd lookup (which makes
                              D3DMetal abort). */
};

class DmnGFXT final : public GFXTOSInterface {
public:
    DmnGFXT();
    GFXTInterfaceVersion Version() const override { return GFXT_INTERFACE_VERSION(5); }
    ~DmnGFXT() override;

    GFXTMonitorInterface*    CreateMonitorInterface(const GFXTInterfaceVersion& v) override;
    GFXTRegistryInterface*   CreateRegistryInterface(const GFXTInterfaceVersion& v) override;
    GFXTEventInterface*      CreateEventInterface(const GFXTInterfaceVersion& v) override;
    GFXTSwapchainInterface*  CreateSwapchainInterface(const GFXTInterfaceVersion& v,
                                                      void* mtlDevice) override;
    GFXTAdapterInterface*    CreateAdapterInterface(const GFXTInterfaceVersion& v) override;
    GFXTPathInterface*       CreatePathInterface(const GFXTInterfaceVersion& v) override;
    GFXTAllocationInterface* CreateAllocationInterface(const GFXTInterfaceVersion& v) override;
    GFXTLibraryInterface*    CreateLibraryInterface(const GFXTInterfaceVersion& v) override;

private:
    std::unique_ptr<DmnGFXTEvent>      events_;
    std::unique_ptr<DmnGFXTMonitor>    monitor_;
    std::unique_ptr<DmnGFXTRegistry>   registry_;
    std::unique_ptr<DmnGFXTAdapter>    adapters_;
    std::unique_ptr<DmnGFXTPath>       paths_;
    std::unique_ptr<DmnGFXTAllocation> allocations_;
    std::unique_ptr<DmnGFXTLibrary>    libraries_;
};
