/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * dmn_init: locate and dlopen D3DMetal.framework, resolve its entry
 * points, and install our GFXTOSInterface implementation through
 * GFXT_Initialize. Also defines the exported D3D11/D3D12/DXGI/D3DCompile
 * forwarders so consumers link only against libd3dmetal-native.
 *
 * The forwarders use pointer/integer signatures (ABI-identical to the
 * real prototypes on x86_64 SysV) so this TU needs no Windows headers.
 */

#include <dlfcn.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "dmn_gfxt.h"
#include "dmn_private.h"

DmnFrameworkApi g_dmn_api = {};

namespace {

std::mutex g_init_mutex;
bool g_initialized = false;
dmn_result g_init_result = DMN_ERROR_NOT_INITIALIZED;
std::string g_framework_binary;
std::string g_resources_dir;
DmnGFXT* g_gfxt = nullptr;

bool file_exists(const std::string& p) {
    struct stat st;
    return !p.empty() && stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const std::string& p) {
    struct stat st;
    return !p.empty() && stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/* Accept either a path to the framework directory or to the binary itself,
 * and resolve to the real Versions/A binary when possible so that
 * @loader_path/Resources (the framework's own LC_RPATH) lands next to the
 * bundled dylibs. */
std::string resolve_framework_binary(const std::string& p) {
    if (p.empty())
        return {};
    if (file_exists(p))
        return p;
    if (dir_exists(p)) {
        std::string versioned = p + "/Versions/A/D3DMetal";
        if (file_exists(versioned))
            return versioned;
        std::string flat = p + "/D3DMetal";
        if (file_exists(flat))
            return flat;
    }
    return {};
}

std::string own_dylib_dir() {
    Dl_info info{};
    if (!dladdr((void*)&dmn_init, &info) || !info.dli_fname)
        return {};
    char buf[PATH_MAX];
    strlcpy(buf, info.dli_fname, sizeof(buf));
    return dirname(buf);
}

std::string find_framework_binary(const char* explicit_path) {
    if (explicit_path && *explicit_path) {
        std::string r = resolve_framework_binary(explicit_path);
        if (r.empty())
            DMN_ERROR("framework_path '%s' does not contain a D3DMetal binary",
                      explicit_path);
        return r;
    }

    if (const char* env = getenv("D3DMETAL_FRAMEWORK_PATH")) {
        std::string r = resolve_framework_binary(env);
        if (!r.empty())
            return r;
        DMN_WARN("D3DMETAL_FRAMEWORK_PATH '%s' has no D3DMetal binary; "
                 "continuing search", env);
    }

    std::string dylib_dir = own_dylib_dir();
    if (!dylib_dir.empty()) {
        for (const char* rel : {"/D3DMetal.framework",
                                "/../Frameworks/D3DMetal.framework"}) {
            std::string r = resolve_framework_binary(dylib_dir + rel);
            if (!r.empty())
                return r;
        }
    }

#ifdef DMN_DEV_FRAMEWORK_PATH
    {
        std::string r = resolve_framework_binary(DMN_DEV_FRAMEWORK_PATH);
        if (!r.empty())
            return r;
    }
#endif
    return {};
}

template <typename T>
void load_sym(void* handle, T& slot, const char* name, bool* missing_any) {
    slot = reinterpret_cast<T>(dlsym(handle, name));
    if (!slot) {
        DMN_WARN("D3DMetal does not export %s", name);
        if (missing_any)
            *missing_any = true;
    }
}

dmn_result init_locked(const dmn_options* options) {
    const char* explicit_path =
        (options && options->struct_size >= sizeof(dmn_options))
            ? options->framework_path : nullptr;

    if (options && options->struct_size >= sizeof(dmn_options) &&
        options->log_callback)
        dmn_log_set_sink(options->log_callback, options->log_ctx);

    std::string binary = find_framework_binary(explicit_path);
    if (binary.empty()) {
        DMN_ERROR("D3DMetal.framework not found (set D3DMETAL_FRAMEWORK_PATH "
                  "or dmn_options::framework_path)");
        return DMN_ERROR_FRAMEWORK_NOT_FOUND;
    }

    /* Install the Metal swizzles BEFORE D3DMetal loads, so no heap class it
     * creates during GFXT_Initialize is missed. */
    dmn_share_install_swizzles();

    DMN_INFO("loading D3DMetal: %s", binary.c_str());
    void* handle = dlopen(binary.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        DMN_ERROR("dlopen failed: %s", err ? err : "(no error string)");
        if (err && (strstr(err, "not valid") || strstr(err, "killed")))
            DMN_ERROR("hint: the framework may be quarantined; try "
                      "'xattr -dr com.apple.quarantine <D3DMetal.framework>'");
        return DMN_ERROR_FRAMEWORK_NOT_FOUND;
    }

    g_dmn_api.GFXT_Initialize = reinterpret_cast<void (*)(GFXTOSInterface*)>(
        dlsym(handle, "GFXT_Initialize"));
    if (!g_dmn_api.GFXT_Initialize) {
        DMN_ERROR("dlsym(GFXT_Initialize) failed: %s", dlerror());
        return DMN_ERROR_SYMBOL_NOT_FOUND;
    }

    load_sym(handle, g_dmn_api.D3D11CreateDevice, "D3D11CreateDevice", nullptr);
    load_sym(handle, g_dmn_api.D3D11CreateDeviceAndSwapChain,
             "D3D11CreateDeviceAndSwapChain", nullptr);
    load_sym(handle, g_dmn_api.CreateDXGIFactory, "CreateDXGIFactory", nullptr);
    load_sym(handle, g_dmn_api.CreateDXGIFactory1, "CreateDXGIFactory1", nullptr);
    load_sym(handle, g_dmn_api.CreateDXGIFactory2, "CreateDXGIFactory2", nullptr);
    load_sym(handle, g_dmn_api.D3D12CreateDevice, "D3D12CreateDevice", nullptr);
    load_sym(handle, g_dmn_api.D3D12GetDebugInterface,
             "D3D12GetDebugInterface", nullptr);
    load_sym(handle, g_dmn_api.D3D12SerializeRootSignature,
             "D3D12SerializeRootSignature", nullptr);
    load_sym(handle, g_dmn_api.D3D12SerializeVersionedRootSignature,
             "D3D12SerializeVersionedRootSignature", nullptr);
    load_sym(handle, g_dmn_api.D3D12CreateRootSignatureDeserializer,
             "D3D12CreateRootSignatureDeserializer", nullptr);
    load_sym(handle, g_dmn_api.D3D12CreateVersionedRootSignatureDeserializer,
             "D3D12CreateVersionedRootSignatureDeserializer", nullptr);
    load_sym(handle, g_dmn_api.D3D12EnableExperimentalFeatures,
             "D3D12EnableExperimentalFeatures", nullptr);
    load_sym(handle, g_dmn_api.DXGIDeclareAdapterRemovalSupport,
             "DXGIDeclareAdapterRemovalSupport", nullptr);
    load_sym(handle, g_dmn_api.D3DCompile, "D3DCompile", nullptr);
    load_sym(handle, g_dmn_api.D3DCompileFromFile, "D3DCompileFromFile", nullptr);
    load_sym(handle, g_dmn_api.D3D10CreateBlob, "D3D10CreateBlob", nullptr);

    g_framework_binary = binary;
    {
        char buf[PATH_MAX];
        strlcpy(buf, binary.c_str(), sizeof(buf));
        g_resources_dir = std::string(dirname(buf)) + "/Resources";
        if (!dir_exists(g_resources_dir))
            DMN_WARN("framework Resources directory missing: %s",
                     g_resources_dir.c_str());
    }

    /* libdxccontainer.dylib (linked by D3DMetal) loads its compiler
     * backends with bare leaf names — dlopen("libdxcompiler.dylib") — which
     * only resolve inside Wine's environment via DYLD_LIBRARY_PATH.
     * Pre-loading them by absolute path registers the images, so the
     * leaf-name dlopen resolves to the already-loaded handle. Without
     * this, D3DCompile dies with an uncaught hlsl::Exception. */
    for (const char* leaf : {"libdxcompiler.dylib", "libdxilconv.dylib"}) {
        std::string p = g_resources_dir + "/" + leaf;
        if (file_exists(p)) {
            if (!dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL))
                DMN_WARN("preload of %s failed: %s", p.c_str(), dlerror());
            else
                DMN_DEBUG("preloaded %s", leaf);
        }
    }

    g_gfxt = new DmnGFXT();
    DMN_INFO("calling GFXT_Initialize");
    g_dmn_api.GFXT_Initialize(g_gfxt);
    DMN_INFO("GFXT_Initialize returned");

    dmn_swapchain_prepare_dummy_view();
    return DMN_SUCCESS;
}

} // namespace

const std::string& dmn_framework_binary_path() { return g_framework_binary; }
const std::string& dmn_framework_resources_dir() { return g_resources_dir; }

extern "C" dmn_result dmn_init(const dmn_options* options) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialized)
        return g_init_result;
    g_init_result = init_locked(options);
    /* A missing framework is retryable with different options; anything
     * past dlopen is latched (GFXT_Initialize must run at most once). */
    if (g_init_result != DMN_ERROR_FRAMEWORK_NOT_FOUND)
        g_initialized = true;
    return g_init_result;
}

extern "C" bool dmn_is_initialized(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    return g_initialized && g_init_result == DMN_SUCCESS;
}

extern "C" const char* dmn_framework_path(void) {
    return g_framework_binary.empty() ? nullptr : g_framework_binary.c_str();
}

dmn_result dmn_ensure_init() {
    return dmn_init(nullptr);
}

/* == Exported D3D entry points =========================================== */

namespace {
constexpr int32_t kEFail = (int32_t)0x80004005; /* E_FAIL */
}

#define DMN_FORWARD(fn, args)                                  \
    do {                                                       \
        if (dmn_ensure_init() != DMN_SUCCESS || !g_dmn_api.fn) \
            return kEFail;                                     \
        return g_dmn_api.fn args;                              \
    } while (0)

extern "C" {

DMN_MS_ABI int32_t D3D11CreateDevice(void* adapter, uint32_t driverType, void* software,
                          uint32_t flags, const uint32_t* featureLevels,
                          uint32_t numLevels, uint32_t sdkVersion,
                          void** outDevice, uint32_t* outLevel,
                          void** outContext) {
    if (dmn_ensure_init() != DMN_SUCCESS || !g_dmn_api.D3D11CreateDevice)
        return kEFail;
    int32_t hr = g_dmn_api.D3D11CreateDevice(adapter, driverType, software, flags,
                                             featureLevels, numLevels, sdkVersion,
                                             outDevice, outLevel, outContext);
    if (hr >= 0 && outDevice && *outDevice)
        dmn_hooks_after_d3d11_device(*outDevice);
    return hr;
}

DMN_MS_ABI int32_t D3D11CreateDeviceAndSwapChain(
        void* adapter, uint32_t driverType, void* software, uint32_t flags,
        const uint32_t* featureLevels, uint32_t numLevels, uint32_t sdkVersion,
        const void* swapChainDesc, void** outSwapChain, void** outDevice,
        uint32_t* outLevel, void** outContext) {
    if (dmn_ensure_init() != DMN_SUCCESS ||
        !g_dmn_api.D3D11CreateDeviceAndSwapChain)
        return kEFail;
    int32_t hr = g_dmn_api.D3D11CreateDeviceAndSwapChain(
        adapter, driverType, software, flags, featureLevels, numLevels,
        sdkVersion, swapChainDesc, outSwapChain, outDevice, outLevel, outContext);
    if (hr >= 0 && outDevice && *outDevice)
        dmn_hooks_after_d3d11_device(*outDevice);
    return hr;
}

DMN_MS_ABI int32_t CreateDXGIFactory(const void* riid, void** factory) {
    DMN_FORWARD(CreateDXGIFactory, (riid, factory));
}

DMN_MS_ABI int32_t CreateDXGIFactory1(const void* riid, void** factory) {
    DMN_FORWARD(CreateDXGIFactory1, (riid, factory));
}

DMN_MS_ABI int32_t CreateDXGIFactory2(uint32_t flags, const void* riid, void** factory) {
    DMN_FORWARD(CreateDXGIFactory2, (flags, riid, factory));
}

DMN_MS_ABI int32_t D3D12CreateDevice(void* adapter, uint32_t minFeatureLevel,
                          const void* riid, void** device) {
    if (dmn_ensure_init() != DMN_SUCCESS || !g_dmn_api.D3D12CreateDevice)
        return kEFail;
    int32_t hr = g_dmn_api.D3D12CreateDevice(adapter, minFeatureLevel, riid, device);
    if (hr >= 0 && device && *device)
        dmn_hooks_after_d3d12_device(*device);
    return hr;
}

DMN_MS_ABI int32_t D3D12SerializeRootSignature(const void* rootSignature,
                          uint32_t version, void** blob, void** errorBlob) {
    DMN_FORWARD(D3D12SerializeRootSignature,
                (rootSignature, version, blob, errorBlob));
}

DMN_MS_ABI int32_t D3D12GetDebugInterface(const void* riid, void** debug) {
    DMN_FORWARD(D3D12GetDebugInterface, (riid, debug));
}

DMN_MS_ABI int32_t D3D12SerializeVersionedRootSignature(
        const void* rootSignature, void** blob, void** errorBlob) {
    DMN_FORWARD(D3D12SerializeVersionedRootSignature,
                (rootSignature, blob, errorBlob));
}

DMN_MS_ABI int32_t D3D12CreateRootSignatureDeserializer(
        const void* srcData, size_t srcDataSize, const void* riid,
        void** deserializer) {
    DMN_FORWARD(D3D12CreateRootSignatureDeserializer,
                (srcData, srcDataSize, riid, deserializer));
}

DMN_MS_ABI int32_t D3D12CreateVersionedRootSignatureDeserializer(
        const void* srcData, size_t srcDataSize, const void* riid,
        void** deserializer) {
    DMN_FORWARD(D3D12CreateVersionedRootSignatureDeserializer,
                (srcData, srcDataSize, riid, deserializer));
}

DMN_MS_ABI int32_t D3D12EnableExperimentalFeatures(uint32_t numFeatures,
                          const void* iids, void* configs, uint32_t* sizes) {
    DMN_FORWARD(D3D12EnableExperimentalFeatures,
                (numFeatures, iids, configs, sizes));
}

DMN_MS_ABI int32_t DXGIDeclareAdapterRemovalSupport(void) {
    DMN_FORWARD(DXGIDeclareAdapterRemovalSupport, ());
}

DMN_MS_ABI int32_t D3DCompile(const void* srcData, size_t srcDataSize,
                   const char* sourceName, const void* defines, void* include,
                   const char* entrypoint, const char* target,
                   uint32_t flags1, uint32_t flags2,
                   void** code, void** errorMsgs) {
    DMN_FORWARD(D3DCompile,
                (srcData, srcDataSize, sourceName, defines, include,
                 entrypoint, target, flags1, flags2, code, errorMsgs));
}

DMN_MS_ABI int32_t D3DCompileFromFile(const void* fileName, const void* defines,
                           void* include, const char* entrypoint,
                           const char* target, uint32_t flags1, uint32_t flags2,
                           void** code, void** errorMsgs) {
    DMN_FORWARD(D3DCompileFromFile,
                (fileName, defines, include, entrypoint, target,
                 flags1, flags2, code, errorMsgs));
}

DMN_MS_ABI int32_t D3D10CreateBlob(size_t numBytes, void** blob) {
    DMN_FORWARD(D3D10CreateBlob, (numBytes, blob));
}

} /* extern "C" */
