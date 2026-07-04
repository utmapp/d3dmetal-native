/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Internal shared declarations. Plain C++ (safe for .cpp and .mm TUs);
 * Objective-C types appear only as void* here.
 */

#pragma once

#include <cstdint>
#include <string>

#include "d3dmetal_native.h"
#include "dmn_log.h"

/* == Framework API table (dmn_init.cpp) ================================= */

/* D3DMetal's D3D/DXGI/D3DCompile exports use the Microsoft x64 calling
 * convention (they are called directly from Wine PE code). GFXT_Initialize
 * and the whole GFXT host interface are plain SysV. */
#define DMN_MS_ABI __attribute__((ms_abi))

class GFXTOSInterface;

struct DmnFrameworkApi {
    void    (*GFXT_Initialize)(GFXTOSInterface*);
    int32_t (DMN_MS_ABI *D3D11CreateDevice)(void*, uint32_t, void*, uint32_t,
                                 const uint32_t*, uint32_t, uint32_t,
                                 void**, uint32_t*, void**);
    int32_t (DMN_MS_ABI *D3D11CreateDeviceAndSwapChain)(void*, uint32_t, void*, uint32_t,
                                             const uint32_t*, uint32_t, uint32_t,
                                             const void*, void**, void**,
                                             uint32_t*, void**);
    int32_t (DMN_MS_ABI *CreateDXGIFactory)(const void*, void**);
    int32_t (DMN_MS_ABI *CreateDXGIFactory1)(const void*, void**);
    int32_t (DMN_MS_ABI *CreateDXGIFactory2)(uint32_t, const void*, void**);
    int32_t (DMN_MS_ABI *D3D12CreateDevice)(void*, uint32_t, const void*, void**);
    int32_t (DMN_MS_ABI *D3D12GetDebugInterface)(const void*, void**);
    int32_t (DMN_MS_ABI *D3D12SerializeRootSignature)(const void*, uint32_t,
                                                      void**, void**);
    int32_t (DMN_MS_ABI *D3D12SerializeVersionedRootSignature)(const void*,
                                                               void**, void**);
    int32_t (DMN_MS_ABI *D3D12CreateRootSignatureDeserializer)(const void*, size_t,
                                                               const void*, void**);
    int32_t (DMN_MS_ABI *D3D12CreateVersionedRootSignatureDeserializer)(
        const void*, size_t, const void*, void**);
    int32_t (DMN_MS_ABI *D3D12EnableExperimentalFeatures)(uint32_t, const void*,
                                                          void*, uint32_t*);
    int32_t (DMN_MS_ABI *DXGIDeclareAdapterRemovalSupport)(void);
    int32_t (DMN_MS_ABI *D3DCompile)(const void*, size_t, const char*, const void*, void*,
                          const char*, const char*, uint32_t, uint32_t,
                          void**, void**);
    int32_t (DMN_MS_ABI *D3DCompileFromFile)(const void*, const void*, void*,
                                  const char*, const char*, uint32_t, uint32_t,
                                  void**, void**);
    int32_t (DMN_MS_ABI *D3D10CreateBlob)(size_t, void**);
};

extern DmnFrameworkApi g_dmn_api;

/* Absolute path of the loaded framework binary ("" before init). */
const std::string& dmn_framework_binary_path();
/* Directory containing the framework's bundled dylibs (Versions/A/Resources). */
const std::string& dmn_framework_resources_dir();

/* Lazy default init used by the exported D3D forwarders. */
dmn_result dmn_ensure_init();

/* == Window table (dmn_window.mm) ======================================== */

struct DmnWindow; /* full definition in dmn_window_internal.h (ObjC++ only) */

/* Decode + validate an HWND pseudo-handle; NULL if unknown/dead. */
DmnWindow* dmn_window_lookup(void* hwnd);

/* == Swapchain helpers (dmn_gfxt_swapchain.mm) =========================== */

/* Create the shared fallback NSView ahead of time (main thread if possible,
 * else dispatched); called once from dmn_init. */
void dmn_swapchain_prepare_dummy_view();

/* == Cross-process sharing hooks (dmn_com_hooks.cpp, dmn_share_metal.mm) == */

extern "C" {
/* Install the Metal device/heap swizzles (dmn_share_metal.mm). Must run before
 * D3DMetal is dlopen'd. Idempotent. */
void dmn_share_install_swizzles(void);
/* Patch the shared-resource/fence vtables reachable from a freshly created
 * device. No-ops when the object is null. */
void dmn_hooks_after_d3d11_device(void* device);
void dmn_hooks_after_d3d12_device(void* device);
}

/* == Registry backend (dmn_gfxt_registry.cpp) ============================ */

bool dmn_registry_store_u32(uint32_t root, const char* subkey,
                            const char* name, uint32_t value);
bool dmn_registry_store_u64(uint32_t root, const char* subkey,
                            const char* name, uint64_t value);
bool dmn_registry_store_string(uint32_t root, const char* subkey,
                               const char* name, const char* value);
