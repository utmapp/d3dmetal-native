/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * COM vtable interception for cross-process sharing: this TU decides WHICH
 * vtable slots are patched and dispatches into the implementation modules —
 * dmn_share (texture/buffer backing), dmn_fence_d3d (shared fences),
 * dmn_kmtx (keyed mutex). Plain C++ (no ObjC) built against the vendored
 * DirectX headers; every method here is STDMETHODCALLTYPE == ms_abi, matching
 * the vtables D3DMetal hands out.
 *
 * Slot indices come from the headers' own declarations via dmn_vindex
 * (dmn_hook.h); interfaces past the vendored headers' horizon are declared in
 * dmn_d3d12_up.h. The member-pointer slot decoding and D3DMetal's private-data
 * release-on-destroy contract are validated by meson tests (abi-check,
 * evict-contract) rather than at runtime — we ship pinned to the vendored
 * D3DMetal.framework, so environment drift is a test-time concern.
 *
 * Lifecycle: every registry entry (and its vended handle POD, keyed mutex,
 * and producer-fence state) is evicted when the tracked object is destroyed,
 * observed via a sentinel IUnknown planted with SetPrivateDataInterface — see
 * the eviction section below. The shared-memory backing itself is reclaimed
 * by the MTLBuffer deallocator in dmn_share_metal.mm.
 */

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include "d3dmetal_native.h"
#include "dmn_d3d12_up.h"
#include "dmn_fence_d3d.h"
#include "dmn_formats.h"
#include "dmn_hook.h"
#include "dmn_kmtx.h"
#include "dmn_log.h"
#include "dmn_share.h"

/* == Format-support bit mirrors ============================================ */
/* dmn_formats.h re-declares the D3D11_FORMAT_SUPPORT bits because its
 * implementation TU is ObjC++ and cannot include d3d11.h. This is the only TU
 * that sees both, so it is the only place the mirror can be checked. */
#define DMN_FMT_ASSERT(name) \
    static_assert((uint32_t)DMN_FMT_SUPPORT_##name == \
                  (uint32_t)D3D11_FORMAT_SUPPORT_##name, \
                  "DMN_FMT_SUPPORT_" #name " drifted from d3d11.h")
DMN_FMT_ASSERT(BUFFER);
DMN_FMT_ASSERT(IA_VERTEX_BUFFER);
DMN_FMT_ASSERT(IA_INDEX_BUFFER);
DMN_FMT_ASSERT(SO_BUFFER);
DMN_FMT_ASSERT(TEXTURE1D);
DMN_FMT_ASSERT(TEXTURE2D);
DMN_FMT_ASSERT(TEXTURE3D);
DMN_FMT_ASSERT(TEXTURECUBE);
DMN_FMT_ASSERT(SHADER_LOAD);
DMN_FMT_ASSERT(SHADER_SAMPLE);
DMN_FMT_ASSERT(SHADER_SAMPLE_COMPARISON);
DMN_FMT_ASSERT(MIP);
DMN_FMT_ASSERT(MIP_AUTOGEN);
DMN_FMT_ASSERT(RENDER_TARGET);
DMN_FMT_ASSERT(BLENDABLE);
DMN_FMT_ASSERT(DEPTH_STENCIL);
DMN_FMT_ASSERT(CPU_LOCKABLE);
DMN_FMT_ASSERT(MULTISAMPLE_RESOLVE);
DMN_FMT_ASSERT(DISPLAY);
DMN_FMT_ASSERT(CAST_WITHIN_BIT_LAYOUT);
DMN_FMT_ASSERT(MULTISAMPLE_RENDERTARGET);
DMN_FMT_ASSERT(MULTISAMPLE_LOAD);
DMN_FMT_ASSERT(SHADER_GATHER);
DMN_FMT_ASSERT(TYPED_UNORDERED_ACCESS_VIEW);
DMN_FMT_ASSERT(SHADER_GATHER_COMPARISON);
#undef DMN_FMT_ASSERT

#define DMN_FMT2_ASSERT(name) \
    static_assert((uint32_t)DMN_FMT_SUPPORT2_##name == \
                  (uint32_t)D3D11_FORMAT_SUPPORT2_##name, \
                  "DMN_FMT_SUPPORT2_" #name " drifted from d3d11.h")
DMN_FMT2_ASSERT(UAV_ATOMIC_ADD);
DMN_FMT2_ASSERT(UAV_ATOMIC_BITWISE_OPS);
DMN_FMT2_ASSERT(UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE);
DMN_FMT2_ASSERT(UAV_ATOMIC_EXCHANGE);
DMN_FMT2_ASSERT(UAV_ATOMIC_SIGNED_MIN_OR_MAX);
DMN_FMT2_ASSERT(UAV_ATOMIC_UNSIGNED_MIN_OR_MAX);
DMN_FMT2_ASSERT(UAV_TYPED_LOAD);
DMN_FMT2_ASSERT(UAV_TYPED_STORE);
DMN_FMT2_ASSERT(OUTPUT_MERGER_LOGIC_OP);
DMN_FMT2_ASSERT(TILED);
DMN_FMT2_ASSERT(SHAREABLE);
#undef DMN_FMT2_ASSERT

namespace {

/* == Live-handle registries =============================================== */
std::mutex g_reg_mtx;
std::unordered_map<void*, dmn_shared_texture_handle> g_tex_reg; /* identity -> POD */
std::unordered_map<void*, dmn_shared_buffer_handle>  g_buf_reg;
std::unordered_set<void*> g_shared_heaps; /* identities of SHARED-flag heaps */
std::unordered_map<void*, int> g_owned_fds; /* identity -> registry-owned dup */

/* Every registration stores its OWN duplicate of the backing fd in its POD
 * (closed at eviction). This is what makes re-export from an OPENED resource
 * outlive the creator — MSDN ties legacy-handle validity to the underlying
 * memory, extended by ANY resource object that refers to it; without the dup,
 * an opened resource's POD would carry the creator's fd number, which dies
 * (or gets recycled) with the creator's Metal backing. Also hardens imports
 * against the app closing its transport fd right after OpenShared*. Returns
 * the borrowed original if dup fails (entry then simply doesn't own an fd). */
int register_owned_fd(void* id, int fd) {
    if (fd < 0)
        return fd;
    int owned = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (owned < 0) {
        DMN_WARN("hooks: dup(fd=%d) for identity=%p failed; registration "
                 "borrows the fd (re-export may not survive the creator)",
                 fd, id);
        return fd;
    }
    std::lock_guard<std::mutex> lk(g_reg_mtx);
    auto it = g_owned_fds.find(id);
    if (it != g_owned_fds.end())
        close(it->second); /* re-registration of the identity: drop the old dup */
    g_owned_fds[id] = owned;
    return owned;
}

void close_owned_fd(void* id) {
    int fd = -1;
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        auto it = g_owned_fds.find(id);
        if (it != g_owned_fds.end()) {
            fd = it->second;
            g_owned_fds.erase(it);
        }
    }
    if (fd >= 0)
        close(fd);
}

/* Heap-allocated PODs handed back as HANDLEs. One allocation per resource
 * identity, refreshed and returned again on repeated exports (so per-frame
 * GetSharedHandle calls do not grow the heap), freed when the resource is
 * evicted — the handle is documented as valid only while the resource lives. */
std::mutex g_pod_mtx;
std::unordered_map<void*, void*> g_pods; /* identity -> malloc'd POD */

HRESULT vend_pod(void* id, const void* src, size_t n, HANDLE* out) {
    std::lock_guard<std::mutex> lk(g_pod_mtx);
    void*& slot = g_pods[id];
    if (!slot)
        slot = std::malloc(n);
    if (!slot) {
        g_pods.erase(id);
        return E_OUTOFMEMORY;
    }
    std::memcpy(slot, src, n);
    *out = (HANDLE)slot;
    return S_OK;
}

void free_pod(void* id) {
    std::lock_guard<std::mutex> lk(g_pod_mtx);
    auto it = g_pods.find(id);
    if (it != g_pods.end()) {
        std::free(it->second);
        g_pods.erase(it);
    }
}

/* == NT-style handles (the CreateSharedHandle family) ===================== */
/* Windows gives these their own lifecycle: the NT handle holds a reference
 * on the allocation until CloseHandle. Mirrored here: each CreateSharedHandle
 * returns a FRESH POD owning its own dup of the fd, valid even after the
 * resource is destroyed, until the app calls dmn_shared_handle_close() (the
 * public CloseHandle analog). Legacy GetSharedHandle handles stay cached on
 * the resource above (Windows: legacy handles are never closed). */
std::mutex g_nt_mtx;
std::unordered_set<void*> g_nt_handles;

/* dmn_shared_handle_close() reads the fd through this common prefix. */
constexpr size_t kPodFdOffset = 8;
static_assert(offsetof(dmn_shared_texture_handle, fd) == kPodFdOffset &&
              offsetof(dmn_shared_buffer_handle, fd) == kPodFdOffset &&
              offsetof(dmn_shared_fence_handle, fd) == kPodFdOffset,
              "shared-handle PODs must start {u32 magic, u32 version, i32 fd}");

template <class Pod>
HRESULT vend_pod_owned(Pod pod, HANDLE* out) {
    if (pod.fd >= 0) {
        int owned = fcntl(pod.fd, F_DUPFD_CLOEXEC, 0);
        if (owned < 0) {
            DMN_ERROR("hooks: dup(fd=%d) for an NT-style handle failed", pod.fd);
            return E_FAIL;
        }
        pod.fd = owned;
    }
    void* p = std::malloc(sizeof(Pod));
    if (!p) {
        if (pod.fd >= 0)
            close(pod.fd);
        return E_OUTOFMEMORY;
    }
    std::memcpy(p, &pod, sizeof(Pod));
    {
        std::lock_guard<std::mutex> lk(g_nt_mtx);
        g_nt_handles.insert(p);
    }
    *out = (HANDLE)p;
    return S_OK;
}

/* == Destruction-driven eviction ========================================== */
/* Everything above is keyed by COM identity, so entries MUST leave the
 * registries when the object dies: the allocator recycles identity pointers,
 * and a stale hit would vend dead fds or misroute creates. There is no
 * Release hook; instead a sentinel IUnknown rides each tracked object via
 * SetPrivateDataInterface — D3D releases private-data interfaces at object
 * destruction, and the sentinel's final Release runs the eviction. That
 * D3DMetal honors the contract (stores the interface, releases it on destroy)
 * is validated by the evict-contract meson test against the vendored
 * framework, not re-probed at runtime. */

enum EvictKind { EV_TEX, EV_BUF, EV_HEAP, EV_FENCE };

/* {7D3A1F7B-9C44-4A6E-8F21-5D0E6B3CA942} — private-data slot for the sentinel. */
const GUID kDmnEvictGuid = {0x7d3a1f7b, 0x9c44, 0x4a6e,
                            {0x8f, 0x21, 0x5d, 0x0e, 0x6b, 0x3c, 0xa9, 0x42}};

const char* evict_kind_name(int kind) {
    switch (kind) {
    case EV_TEX:   return "texture";
    case EV_BUF:   return "buffer";
    case EV_HEAP:  return "heap";
    case EV_FENCE: return "fence";
    default:       return "?";
    }
}

void evict_identity(void* id, int kind) {
    switch (kind) {
    case EV_TEX:
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            g_tex_reg.erase(id);
        }
        dmn_kmtx_unregister(id);
        break;
    case EV_BUF:
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            g_buf_reg.erase(id);
        }
        break;
    case EV_HEAP:
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            g_shared_heaps.erase(id);
        }
        break;
    case EV_FENCE:
        dmn_fd3d_fence_destroy(id);
        break;
    }
    close_owned_fd(id);
    free_pod(id);
    DMN_INFO("hooks: evicted shared %s identity=%p", evict_kind_name(kind), id);
}

struct EvictSentinel {
    void** vtbl;
    std::atomic<ULONG> refs;
    void* identity;
    int kind;
    std::atomic<bool> armed; /* evict only once actually attached */
};

HRESULT STDMETHODCALLTYPE sent_QueryInterface(EvictSentinel* self, REFIID riid,
                                              void** ppv) {
    if (!ppv)
        return E_POINTER;
    if (dmn_iid_eq(riid, __uuidof(IUnknown))) {
        self->refs.fetch_add(1, std::memory_order_relaxed);
        *ppv = self;
        return S_OK;
    }
    *ppv = nullptr;
    return E_NOINTERFACE;
}
ULONG STDMETHODCALLTYPE sent_AddRef(EvictSentinel* self) {
    return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
}
ULONG STDMETHODCALLTYPE sent_Release(EvictSentinel* self) {
    ULONG r = self->refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (r == 0) {
        if (self->armed.load(std::memory_order_acquire))
            evict_identity(self->identity, self->kind);
        delete self;
    }
    return r;
}

void* g_sent_vtbl[3] = {
    (void*)sent_QueryInterface, (void*)sent_AddRef, (void*)sent_Release,
};

/* Attach a sentinel to `obj` (D3D11 or D3D12 object, resolved by QI). The
 * caller must hold `obj` alive across the call. Returns false when tracking
 * is unavailable — the entry then stays until process exit */
bool attach_evict_sentinel(IUnknown* obj, void* id, int kind) {
    ID3D11DeviceChild* c11 = nullptr;
    ID3D12Object* o12 = nullptr;
    if (FAILED(obj->QueryInterface(__uuidof(ID3D11DeviceChild),
                                   reinterpret_cast<void**>(&c11))) || !c11) {
        if (FAILED(obj->QueryInterface(__uuidof(ID3D12Object),
                                       reinterpret_cast<void**>(&o12))) || !o12) {
            DMN_WARN("hooks: identity=%p has no private-data interface; its "
                     "registry entry lives until process exit", id);
            return false;
        }
    }
    bool ok = false;
    auto* s = new (std::nothrow) EvictSentinel();
    if (s) {
        s->vtbl = g_sent_vtbl;
        s->refs.store(1, std::memory_order_relaxed);
        s->identity = id;
        s->kind = kind;
        s->armed.store(false, std::memory_order_relaxed);
        auto* si = reinterpret_cast<IUnknown*>(s);
        HRESULT hr = c11 ? c11->SetPrivateDataInterface(kDmnEvictGuid, si)
                         : o12->SetPrivateDataInterface(kDmnEvictGuid, si);
        if (SUCCEEDED(hr)) {
            s->armed.store(true, std::memory_order_release);
            ok = true;
        }
        /* Drop the construction reference; on success the object's
         * private-data slot holds the surviving one. */
        sent_Release(s);
    }
    if (c11) c11->Release();
    if (o12) o12->Release();
    return ok;
}

/* == Thunk forward declarations =========================================== */
/* D3D11 device + context. */
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture2D(ID3D11Device*, const D3D11_TEXTURE2D_DESC*,
    const D3D11_SUBRESOURCE_DATA*, ID3D11Texture2D**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture2D1(ID3D11Device3*, const D3D11_TEXTURE2D_DESC1*,
    const D3D11_SUBRESOURCE_DATA*, ID3D11Texture2D1**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture1D(ID3D11Device*, const D3D11_TEXTURE1D_DESC*,
    const D3D11_SUBRESOURCE_DATA*, ID3D11Texture1D**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture3D(ID3D11Device*, const D3D11_TEXTURE3D_DESC*,
    const D3D11_SUBRESOURCE_DATA*, ID3D11Texture3D**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateBuffer(ID3D11Device*, const D3D11_BUFFER_DESC*,
    const D3D11_SUBRESOURCE_DATA*, ID3D11Buffer**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckFormatSupport(ID3D11Device*, DXGI_FORMAT, UINT*);
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckFeatureSupport(ID3D11Device*, D3D11_FEATURE, void*,
    UINT);
HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResource(ID3D11Device*, HANDLE, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResource1(ID3D11Device1*, HANDLE, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResourceByName(ID3D11Device1*, LPCWSTR, DWORD,
    REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateFence(ID3D11Device5*, UINT64, D3D11_FENCE_FLAG,
    REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedFence(ID3D11Device5*, HANDLE, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_ctx_Signal(ID3D11DeviceContext4*, ID3D11Fence*, UINT64);
HRESULT STDMETHODCALLTYPE hook_ctx_Wait(ID3D11DeviceContext4*, ID3D11Fence*, UINT64);

/* DXGI resource exports + keyed-mutex QI (patched per shared resource). */
HRESULT STDMETHODCALLTYPE hook_GetSharedHandle(IDXGIResource*, HANDLE*);
HRESULT STDMETHODCALLTYPE hook_res_CreateSharedHandle(IDXGIResource1*,
    const SECURITY_ATTRIBUTES*, DWORD, const WCHAR*, HANDLE*);
HRESULT STDMETHODCALLTYPE hook_res_QueryInterface(IUnknown*, REFIID, void**);

/* D3D11 fence object (patched per producer/imported fence). */
HRESULT STDMETHODCALLTYPE hook_f11_CreateSharedHandle(ID3D11Fence*,
    const SECURITY_ATTRIBUTES*, DWORD, const WCHAR*, HANDLE*);
UINT64 STDMETHODCALLTYPE hook_f11_GetCompletedValue(ID3D11Fence*);
HRESULT STDMETHODCALLTYPE hook_f11_SetEventOnCompletion(ID3D11Fence*, UINT64, HANDLE);

/* D3D12 device. */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource(ID3D12Device*,
    const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource1(ID3D12Device4*,
    const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
    DMN_ID3D12ProtectedResourceSession*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource2(ID3D12Device8*,
    const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC1*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
    DMN_ID3D12ProtectedResourceSession*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource3(ID3D12Device10*,
    const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC1*,
    DMN_D3D12_BARRIER_LAYOUT, const D3D12_CLEAR_VALUE*,
    DMN_ID3D12ProtectedResourceSession*, UINT32, const DXGI_FORMAT*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateHeap(ID3D12Device*, const D3D12_HEAP_DESC*,
    REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateHeap1(ID3D12Device4*, const D3D12_HEAP_DESC*,
    DMN_ID3D12ProtectedResourceSession*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource(ID3D12Device*, ID3D12Heap*,
    UINT64, const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource1(ID3D12Device8*, ID3D12Heap*,
    UINT64, const D3D12_RESOURCE_DESC1*, D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource2(ID3D12Device10*, ID3D12Heap*,
    UINT64, const D3D12_RESOURCE_DESC1*, DMN_D3D12_BARRIER_LAYOUT,
    const D3D12_CLEAR_VALUE*, UINT32, const DXGI_FORMAT*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateFence(ID3D12Device*, UINT64,
    D3D12_FENCE_FLAGS, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateSharedHandle(ID3D12Device*,
    ID3D12DeviceChild*, const SECURITY_ATTRIBUTES*, DWORD, const WCHAR*, HANDLE*);
HRESULT STDMETHODCALLTYPE hook_d3d12_OpenSharedHandle(ID3D12Device*, HANDLE, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_OpenSharedHandleByName(ID3D12Device*, const WCHAR*,
    DWORD, HANDLE*);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandQueue(ID3D12Device*,
    const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandQueue1(ID3D12Device9*,
    const D3D12_COMMAND_QUEUE_DESC*, REFIID, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_SetEventOnMultipleFenceCompletion(ID3D12Device1*,
    ID3D12Fence* const*, const UINT64*, UINT, D3D12_MULTIPLE_FENCE_WAIT_FLAGS, HANDLE);

/* D3D12 queue + fence object. */
HRESULT STDMETHODCALLTYPE hook_queue_Signal(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
HRESULT STDMETHODCALLTYPE hook_queue_Wait(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
UINT64 STDMETHODCALLTYPE hook_f12_GetCompletedValue(ID3D12Fence*);
HRESULT STDMETHODCALLTYPE hook_f12_SetEventOnCompletion(ID3D12Fence*, UINT64, HANDLE);
HRESULT STDMETHODCALLTYPE hook_f12_Signal(ID3D12Fence*, UINT64);

/* == Hook state (typed original-fn maps) ================================== */
DMN_HOOK_STATE(d3d11_CreateTexture2D);
DMN_HOOK_STATE(d3d11_CreateTexture2D1);
DMN_HOOK_STATE(d3d11_CreateTexture1D);
DMN_HOOK_STATE(d3d11_CreateTexture3D);
DMN_HOOK_STATE(d3d11_CreateBuffer);
DMN_HOOK_STATE(d3d11_CheckFormatSupport);
DMN_HOOK_STATE(d3d11_CheckFeatureSupport);
DMN_HOOK_STATE(d3d11_OpenSharedResource);
DMN_HOOK_STATE(d3d11_OpenSharedResource1);
DMN_HOOK_STATE(d3d11_OpenSharedResourceByName);
DMN_HOOK_STATE(d3d11_CreateFence);
DMN_HOOK_STATE(d3d11_OpenSharedFence);
DMN_HOOK_STATE(ctx_Signal);
DMN_HOOK_STATE(ctx_Wait);
DMN_HOOK_STATE(GetSharedHandle);
DMN_HOOK_STATE(res_CreateSharedHandle);
DMN_HOOK_STATE(res_QueryInterface);
DMN_HOOK_STATE(f11_CreateSharedHandle);
DMN_HOOK_STATE(f11_GetCompletedValue);
DMN_HOOK_STATE(f11_SetEventOnCompletion);
DMN_HOOK_STATE(d3d12_CreateCommittedResource);
DMN_HOOK_STATE(d3d12_CreateCommittedResource1);
DMN_HOOK_STATE(d3d12_CreateCommittedResource2);
DMN_HOOK_STATE(d3d12_CreateCommittedResource3);
DMN_HOOK_STATE(d3d12_CreateHeap);
DMN_HOOK_STATE(d3d12_CreateHeap1);
DMN_HOOK_STATE(d3d12_CreatePlacedResource);
DMN_HOOK_STATE(d3d12_CreatePlacedResource1);
DMN_HOOK_STATE(d3d12_CreatePlacedResource2);
DMN_HOOK_STATE(d3d12_CreateFence);
DMN_HOOK_STATE(d3d12_CreateSharedHandle);
DMN_HOOK_STATE(d3d12_OpenSharedHandle);
DMN_HOOK_STATE(d3d12_OpenSharedHandleByName);
DMN_HOOK_STATE(d3d12_CreateCommandQueue);
DMN_HOOK_STATE(d3d12_CreateCommandQueue1);
DMN_HOOK_STATE(d3d12_SetEventOnMultipleFenceCompletion);
DMN_HOOK_STATE(queue_Signal);
DMN_HOOK_STATE(queue_Wait);
DMN_HOOK_STATE(f12_GetCompletedValue);
DMN_HOOK_STATE(f12_SetEventOnCompletion);
DMN_HOOK_STATE(f12_Signal);

/* == Shared-resource recording ============================================ */
bool is_shared_misc(UINT misc) {
    return (misc & (D3D11_RESOURCE_MISC_SHARED |
                    D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                    D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) != 0;
}

bool wants_kmtx(UINT misc) {
    return (misc & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) != 0;
}

/* Extra shm to request from the producer arm: one page for the keyed-mutex
 * state when the flag asks for it. */
uint64_t kmtx_extra(UINT misc) {
    return wants_kmtx(misc) ? (uint64_t)dmn_share_page_align(1) : 0;
}

/* Patch the per-class fence methods and attach the eviction sentinel. Shared
 * producer fences and imports get identical treatment: slot-merged reads,
 * re-export, and (D3D12) CPU-signal mirroring. */
void track_fence_d3d11(ID3D11Fence* f) {
    DMN_PATCH(f, ID3D11Fence, CreateSharedHandle, f11_CreateSharedHandle);
    DMN_PATCH(f, ID3D11Fence, GetCompletedValue, f11_GetCompletedValue);
    DMN_PATCH(f, ID3D11Fence, SetEventOnCompletion, f11_SetEventOnCompletion);
    attach_evict_sentinel(f, dmn_com_identity(f), EV_FENCE);
}

void track_fence_d3d12(ID3D12Fence* f) {
    DMN_PATCH(f, ID3D12Fence, GetCompletedValue, f12_GetCompletedValue);
    DMN_PATCH(f, ID3D12Fence, SetEventOnCompletion, f12_SetEventOnCompletion);
    DMN_PATCH(f, ID3D12Fence, Signal, f12_Signal);
    attach_evict_sentinel(f, dmn_com_identity(f), EV_FENCE);
}

/* Patch the export methods (and keyed-mutex QI) onto a shared resource's
 * vtables. Patches both the interface pointer the app holds and its
 * IDXGIResource view, since QueryInterface sits at slot 0 of each. */
void patch_shared_resource_exports(IUnknown* res) {
    DMN_PATCH_QI(res, res_QueryInterface);
    IDXGIResource1* r1 = nullptr;
    if (SUCCEEDED(res->QueryInterface(__uuidof(IDXGIResource1),
                                      reinterpret_cast<void**>(&r1))) && r1) {
        DMN_PATCH_QI(r1, res_QueryInterface);
        DMN_PATCH(r1, IDXGIResource, GetSharedHandle, GetSharedHandle);
        DMN_PATCH(r1, IDXGIResource1, CreateSharedHandle, res_CreateSharedHandle);
        r1->Release();
        return;
    }
    IDXGIResource* r0 = nullptr;
    if (SUCCEEDED(res->QueryInterface(__uuidof(IDXGIResource),
                                      reinterpret_cast<void**>(&r0))) && r0) {
        DMN_PATCH_QI(r0, res_QueryInterface);
        DMN_PATCH(r0, IDXGIResource, GetSharedHandle, GetSharedHandle);
        r0->Release();
    }
}

/* Record a shared texture (producer, or consumer re-registration for
 * re-export). `init_kmtx` distinguishes who initializes the mutex page. */
void record_texture_pod(IUnknown* tex, const dmn_shared_texture_handle& pod,
                        bool init_kmtx) {
    void* id = dmn_com_identity(tex);
    dmn_shared_texture_handle rec = pod;
    rec.fd = register_owned_fd(id, pod.fd);
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        g_tex_reg[id] = rec;
    }
    patch_shared_resource_exports(tex);
    if (wants_kmtx(rec.misc_flags))
        dmn_kmtx_register(id, rec.fd, dmn_share_page_align((size_t)rec.size),
                          init_kmtx);
    attach_evict_sentinel(tex, id, EV_TEX);
    DMN_INFO("hooks: recorded shared texture identity=%p fd=%d %ux%u stride=%llu",
             id, rec.fd, rec.width, rec.height, (unsigned long long)rec.stride);
}

void record_texture(IUnknown* tex, const D3D11_TEXTURE2D_DESC& d,
                    const DmnShareArm& arm) {
    dmn_shared_texture_handle pod{};
    pod.magic = DMN_SHARED_TEXTURE_MAGIC;
    pod.version = DMN_SHARED_HANDLE_VERSION;
    pod.fd = arm.out_fd;
    pod.width = d.Width;
    pod.height = d.Height;
    pod.dxgi_format = d.Format;
    pod.mip_levels = d.MipLevels ? d.MipLevels : 1;
    pod.array_size = d.ArraySize ? d.ArraySize : 1;
    pod.sample_count = d.SampleDesc.Count ? d.SampleDesc.Count : 1;
    pod.bind_flags = d.BindFlags;
    pod.misc_flags = d.MiscFlags;
    pod.cpu_access = d.CPUAccessFlags;
    pod.stride = arm.out_stride;
    pod.size = arm.out_size;
    record_texture_pod(tex, pod, /*init_kmtx=*/true);
}

/* Register `res` as a shared buffer under `pod` (whose fd is the transport
 * fd; the registry takes its own dup), patch its export methods, and attach
 * the eviction sentinel. Producer captures and consumer imports, both APIs.
 * The export patch is a no-op for D3D12 buffers, which have no IDXGIResource
 * and export device-side. */
void register_buffer_pod(IUnknown* res, dmn_shared_buffer_handle pod) {
    void* id = dmn_com_identity(res);
    pod.fd = register_owned_fd(id, pod.fd);
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        g_buf_reg[id] = pod;
    }
    patch_shared_resource_exports(res);
    attach_evict_sentinel(res, id, EV_BUF);
    DMN_INFO("hooks: registered shared buffer identity=%p fd=%d size=%llu", id,
             pod.fd, (unsigned long long)pod.size);
}

void record_buffer(IUnknown* res, uint64_t size, uint32_t bind, uint32_t misc,
                   uint32_t cpu, const DmnShareArm& arm) {
    dmn_shared_buffer_handle pod{};
    pod.magic = DMN_SHARED_BUFFER_MAGIC;
    pod.version = DMN_SHARED_HANDLE_VERSION;
    pod.fd = arm.out_fd;
    pod.size = arm.out_size ? arm.out_size : size;
    pod.bind_flags = bind;
    pod.misc_flags = misc;
    pod.cpu_access = cpu;
    register_buffer_pod(res, pod);
}

/* Track a SHARED-flag heap so placed creates on it are intercepted. */
void record_shared_heap(IUnknown* heap) {
    void* id = dmn_com_identity(heap);
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        g_shared_heaps.insert(id);
    }
    attach_evict_sentinel(heap, id, EV_HEAP);
    DMN_INFO("hooks: tracking SHARED heap %p (placed resources on it will "
             "each get their own shared backing)", id);
}

/* == Export (handle vending) ============================================== */
/* `nt` selects the handle regime: false = legacy GetSharedHandle (cached on
 * the resource, never closed, dies with it); true = CreateSharedHandle (a
 * fresh owned handle the app must dmn_shared_handle_close). */
HRESULT return_texture_pod(IUnknown* res, HANDLE* out, bool nt) {
    if (!out)
        return E_INVALIDARG;
    void* id = dmn_com_identity(res);
    dmn_shared_texture_handle pod;
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        auto it = g_tex_reg.find(id);
        if (it == g_tex_reg.end())
            return E_FAIL; /* not one of ours; caller falls back */
        pod = it->second;
    }
    HRESULT hr = nt ? vend_pod_owned(pod, out)
                    : vend_pod(id, &pod, sizeof(pod), out);
    if (SUCCEEDED(hr))
        DMN_INFO("hooks: exported %s texture handle %p (fd=%d)",
                 nt ? "NT" : "legacy", *out, pod.fd);
    return hr;
}

HRESULT return_buffer_pod(IUnknown* res, HANDLE* out, bool nt) {
    if (!out)
        return E_INVALIDARG;
    void* id = dmn_com_identity(res);
    dmn_shared_buffer_handle pod;
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        auto it = g_buf_reg.find(id);
        if (it == g_buf_reg.end())
            return E_FAIL;
        pod = it->second;
    }
    HRESULT hr = nt ? vend_pod_owned(pod, out)
                    : vend_pod(id, &pod, sizeof(pod), out);
    if (SUCCEEDED(hr))
        DMN_INFO("hooks: exported %s buffer handle %p (fd=%d)",
                 nt ? "NT" : "legacy", *out, pod.fd);
    return hr;
}

/* Fences only export through the CreateSharedHandle family — always NT. */
HRESULT return_fence_pod(IUnknown* fenceObj, HANDLE* out) {
    if (!out)
        return E_INVALIDARG;
    dmn_shared_fence_handle pod;
    if (!dmn_fd3d_export(fenceObj, &pod))
        return E_FAIL; /* not one of ours */
    HRESULT hr = vend_pod_owned(pod, out);
    if (SUCCEEDED(hr))
        DMN_INFO("hooks: exported NT fence handle %p (fd=%d)", *out, pod.fd);
    return hr;
}

/* == Import (consumer reconstruction) ===================================== */
HRESULT import_texture_d3d11(ID3D11Device* dev, const dmn_shared_texture_handle* pod,
                             REFIID iid, void** out) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = pod->width;
    td.Height = pod->height;
    td.MipLevels = pod->mip_levels ? pod->mip_levels : 1;
    td.ArraySize = pod->array_size ? pod->array_size : 1;
    td.Format = (DXGI_FORMAT)pod->dxgi_format;
    td.SampleDesc.Count = pod->sample_count ? pod->sample_count : 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = pod->bind_flags
                       ? pod->bind_flags
                       : (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
    td.CPUAccessFlags = pod->cpu_access;
    td.MiscFlags = pod->misc_flags;

    ID3D11Texture2D* tex = nullptr;
    dmn_share_arm_consumer(pod->fd, pod->stride, pod->size);
    HRESULT hr = dev->CreateTexture2D(&td, nullptr, &tex);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (FAILED(hr) || !tex) {
        DMN_WARN("hooks: D3D11 texture import CreateTexture2D 0x%08x", (unsigned)hr);
        return FAILED(hr) ? hr : E_FAIL;
    }
    if (!captured)
        DMN_WARN("hooks: D3D11 texture import did not route through the swizzle");
    /* Register the imported view so re-export (GetSharedHandle on an opened
     * resource — valid on Windows) and keyed-mutex QI work here too. */
    record_texture_pod(tex, *pod, /*init_kmtx=*/false);
    hr = tex->QueryInterface(iid, out);
    tex->Release();
    return hr;
}

HRESULT import_buffer_d3d11(ID3D11Device* dev, const dmn_shared_buffer_handle* pod,
                            REFIID iid, void** out) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = (UINT)pod->size;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = pod->bind_flags ? pod->bind_flags
                                   : D3D11_BIND_UNORDERED_ACCESS;
    bd.CPUAccessFlags = pod->cpu_access;
    bd.MiscFlags = pod->misc_flags;

    ID3D11Buffer* buf = nullptr;
    dmn_share_arm_consumer_buffer(pod->fd, pod->size);
    HRESULT hr = dev->CreateBuffer(&bd, nullptr, &buf);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (FAILED(hr) || !buf) {
        DMN_WARN("hooks: D3D11 buffer import CreateBuffer 0x%08x", (unsigned)hr);
        return FAILED(hr) ? hr : E_FAIL;
    }
    if (!captured)
        DMN_WARN("hooks: D3D11 buffer import did not route through the swizzle");
    register_buffer_pod(buf, *pod);
    hr = buf->QueryInterface(iid, out);
    buf->Release();
    return hr;
}

HRESULT import_texture_d3d12(ID3D12Device* dev, const dmn_shared_texture_handle* pod,
                             REFIID riid, void** out) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = pod->width;
    rd.Height = pod->height;
    rd.DepthOrArraySize = (UINT16)(pod->array_size ? pod->array_size : 1);
    rd.MipLevels = (UINT16)(pod->mip_levels ? pod->mip_levels : 1);
    rd.Format = (DXGI_FORMAT)pod->dxgi_format;
    rd.SampleDesc.Count = pod->sample_count ? pod->sample_count : 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    dmn_share_arm_consumer(pod->fd, pod->stride, pod->size);
    HRESULT hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, riid, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (!captured)
        DMN_WARN("hooks: D3D12 import did not route through the Metal swizzle");
    if (SUCCEEDED(hr) && out && *out) {
        /* Register for re-export via ID3D12Device::CreateSharedHandle. */
        auto* res = reinterpret_cast<IUnknown*>(*out);
        void* id = dmn_com_identity(res);
        dmn_shared_texture_handle rec = *pod;
        rec.fd = register_owned_fd(id, pod->fd);
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            g_tex_reg[id] = rec;
        }
        attach_evict_sentinel(res, id, EV_TEX);
    }
    return hr;
}

HRESULT import_buffer_d3d12(ID3D12Device* dev, const dmn_shared_buffer_handle* pod,
                            REFIID riid, void** out) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = pod->size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    dmn_share_arm_consumer_buffer(pod->fd, pod->size);
    HRESULT hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, riid, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (!captured)
        DMN_WARN("hooks: D3D12 buffer import did not route through the swizzle");
    if (SUCCEEDED(hr) && out && *out)
        register_buffer_pod(reinterpret_cast<IUnknown*>(*out), *pod);
    return hr;
}

/* Shared import dispatch for the D3D11 OpenSharedResource variants. */
HRESULT open_shared_d3d11(ID3D11Device* dev, HANDLE handle, REFIID iid, void** out) {
    uint32_t magic = *reinterpret_cast<uint32_t*>(handle);
    if (magic == DMN_SHARED_BUFFER_MAGIC)
        return import_buffer_d3d11(
            dev, reinterpret_cast<dmn_shared_buffer_handle*>(handle), iid, out);
    if (magic == DMN_SHARED_TEXTURE_MAGIC)
        return import_texture_d3d11(
            dev, reinterpret_cast<dmn_shared_texture_handle*>(handle), iid, out);
    return E_FAIL; /* not ours; caller falls back to orig */
}

/* == D3D11 producer thunks ================================================= */
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture2D(
        ID3D11Device* This, const D3D11_TEXTURE2D_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture2D** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture2D, This);
    if (!orig)
        return E_FAIL;
    /* is_armed: a consumer reconstruct is mid-flight; pass straight to the
     * swizzle instead of re-detecting a producer create. */
    if (!desc || dmn_share_is_armed() || !is_shared_misc(desc->MiscFlags))
        return orig(This, desc, init, out);

    DMN_INFO("hooks: CreateTexture2D MISC_SHARED %ux%u fmt=%u", desc->Width,
             desc->Height, desc->Format);
    dmn_share_arm_producer(kmtx_extra(desc->MiscFlags));
    HRESULT hr = orig(This, desc, init, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (SUCCEEDED(hr) && captured && out && *out)
        record_texture(*out, *desc, arm);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture2D1(
        ID3D11Device3* This, const D3D11_TEXTURE2D_DESC1* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture2D1** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture2D1, This);
    if (!orig)
        return E_FAIL;
    if (!desc || dmn_share_is_armed() || !is_shared_misc(desc->MiscFlags))
        return orig(This, desc, init, out);

    DMN_INFO("hooks: CreateTexture2D1 MISC_SHARED %ux%u fmt=%u", desc->Width,
             desc->Height, desc->Format);
    dmn_share_arm_producer(kmtx_extra(desc->MiscFlags));
    HRESULT hr = orig(This, desc, init, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (SUCCEEDED(hr) && captured && out && *out) {
        /* Flatten DESC1 -> DESC for the POD (drops only TextureLayout). */
        D3D11_TEXTURE2D_DESC d{};
        d.Width = desc->Width; d.Height = desc->Height;
        d.MipLevels = desc->MipLevels; d.ArraySize = desc->ArraySize;
        d.Format = desc->Format; d.SampleDesc = desc->SampleDesc;
        d.Usage = desc->Usage; d.BindFlags = desc->BindFlags;
        d.CPUAccessFlags = desc->CPUAccessFlags; d.MiscFlags = desc->MiscFlags;
        record_texture(*out, d, arm);
    }
    return hr;
}

/* 1D/3D shared textures have no backing implementation: warn loudly instead
 * of letting the app discover a silently unshared resource at export time. */
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture1D(
        ID3D11Device* This, const D3D11_TEXTURE1D_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture1D** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture1D, This);
    if (!orig)
        return E_FAIL;
    if (desc && is_shared_misc(desc->MiscFlags))
        DMN_WARN("hooks: CreateTexture1D with a SHARED misc flag is not "
                 "supported; the texture is created UNSHARED");
    return orig(This, desc, init, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture3D(
        ID3D11Device* This, const D3D11_TEXTURE3D_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture3D** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture3D, This);
    if (!orig)
        return E_FAIL;
    if (desc && is_shared_misc(desc->MiscFlags))
        DMN_WARN("hooks: CreateTexture3D with a SHARED misc flag is not "
                 "supported; the texture is created UNSHARED");
    return orig(This, desc, init, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateBuffer(
        ID3D11Device* This, const D3D11_BUFFER_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Buffer** out) {
    auto orig = DMN_ORIG(d3d11_CreateBuffer, This);
    if (!orig)
        return E_FAIL;
    if (!desc || !is_shared_misc(desc->MiscFlags) || dmn_share_is_armed())
        return orig(This, desc, init, out);

    DMN_INFO("hooks: CreateBuffer MISC_SHARED size=%u", desc->ByteWidth);
    dmn_share_arm_producer_buffer(desc->ByteWidth);
    HRESULT hr = orig(This, desc, init, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (SUCCEEDED(hr) && captured && out && *out)
        record_buffer(reinterpret_cast<IUnknown*>(*out), desc->ByteWidth,
                      desc->BindFlags, desc->MiscFlags, desc->CPUAccessFlags, arm);
    return hr;
}

/* == D3D11 format-capability thunks ======================================== */
/* D3DMetal answers CheckFormatSupport with 0xffffffff for anything it knows
 * and 0 otherwise — those are the only two values it returns, always with
 * S_OK. We replace the answer outright rather than filtering one; there is
 * nothing in its reply worth preserving. dmn_formats.mm has the tables.
 *
 * One deliberate behavioural difference: for a valid-but-unrepresentable
 * DXGI_FORMAT (NV12, P8, R1_UNORM, ...) D3DMetal returns S_OK with 0, whereas
 * we follow dxmt and return E_INVALIDARG. MSDN reserves E_INVALIDARG for
 * values that are not valid DXGI_FORMATs at all, so S_OK-with-0 is arguably
 * the more conformant answer; this is the first thing to try if an app trips
 * over a video-format query. */
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckFormatSupport(
        ID3D11Device* This, DXGI_FORMAT format, UINT* out) {
    (void)This;
    if (!out)
        return E_INVALIDARG;
    uint32_t support = 0;
    if (!dmn_format_support((uint32_t)format, &support)) {
        *out = 0;
        return E_INVALIDARG;
    }
    *out = support;
    return S_OK;
}

/* FORMAT_SUPPORT and FORMAT_SUPPORT2 must agree with CheckFormatSupport, so
 * they come from the same tables. Every other feature is D3DMetal's to answer
 * — forward it untouched. */
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckFeatureSupport(
        ID3D11Device* This, D3D11_FEATURE feature, void* data, UINT size) {
    if (feature == D3D11_FEATURE_FORMAT_SUPPORT) {
        auto* info = static_cast<D3D11_FEATURE_DATA_FORMAT_SUPPORT*>(data);
        if (!info || size != sizeof(*info))
            return E_INVALIDARG;
        return hook_d3d11_CheckFormatSupport(This, info->InFormat,
                                             &info->OutFormatSupport);
    }
    if (feature == D3D11_FEATURE_FORMAT_SUPPORT2) {
        auto* info = static_cast<D3D11_FEATURE_DATA_FORMAT_SUPPORT2*>(data);
        if (!info || size != sizeof(*info))
            return E_INVALIDARG;
        uint32_t support2 = 0;
        if (!dmn_format_support2((uint32_t)info->InFormat, &support2)) {
            info->OutFormatSupport2 = 0;
            return E_INVALIDARG;
        }
        info->OutFormatSupport2 = support2;
        return S_OK;
    }
    auto orig = DMN_ORIG(d3d11_CheckFeatureSupport, This);
    return orig ? orig(This, feature, data, size) : E_FAIL;
}

/* == D3D11 consumer thunks ================================================= */
HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResource(
        ID3D11Device* This, HANDLE handle, REFIID iid, void** out) {
    if (handle && out) {
        HRESULT hr = open_shared_d3d11(This, handle, iid, out);
        if (hr != E_FAIL)
            return hr;
    }
    auto orig = DMN_ORIG(d3d11_OpenSharedResource, This);
    return orig ? orig(This, handle, iid, out) : E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResource1(
        ID3D11Device1* This, HANDLE handle, REFIID iid, void** out) {
    if (handle && out) {
        HRESULT hr = open_shared_d3d11(reinterpret_cast<ID3D11Device*>(This),
                                       handle, iid, out);
        if (hr != E_FAIL)
            return hr;
    }
    auto orig = DMN_ORIG(d3d11_OpenSharedResource1, This);
    return orig ? orig(This, handle, iid, out) : E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResourceByName(
        ID3D11Device1* This, LPCWSTR name, DWORD access, REFIID iid, void** out) {
    (void)This; (void)name; (void)access; (void)iid;
    if (out)
        *out = nullptr;
    DMN_ERROR("hooks: OpenSharedResourceByName is not supported — named "
              "handles do not exist here; ship the handle POD + fd instead");
    return E_NOTIMPL;
}

/* == D3D11 fences ========================================================== */
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateFence(
        ID3D11Device5* This, UINT64 initial, D3D11_FENCE_FLAG flags,
        REFIID iid, void** out) {
    auto orig = DMN_ORIG(d3d11_CreateFence, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, initial, flags, iid, out);
    if (FAILED(hr) || !out || !*out)
        return hr;
    /* Only shared fences get a companion buffer; leave plain fences alone. */
    if (!(flags & D3D11_FENCE_FLAG_SHARED))
        return hr;
    ID3D11Fence* f = nullptr;
    if (SUCCEEDED(reinterpret_cast<IUnknown*>(*out)->QueryInterface(
            __uuidof(ID3D11Fence), reinterpret_cast<void**>(&f))) && f) {
        if (dmn_fd3d_producer_create_d3d11(This, f, initial, (UINT)flags)) {
            /* Producer signal interception + export + slot-merged readback. */
            if (ID3D11DeviceContext* ctx = dmn_fd3d_producer_d3d11_ctx(f)) {
                ID3D11DeviceContext4* c4 = nullptr;
                if (SUCCEEDED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext4),
                                                  reinterpret_cast<void**>(&c4))) && c4) {
                    DMN_PATCH(c4, ID3D11DeviceContext4, Signal, ctx_Signal);
                    DMN_PATCH(c4, ID3D11DeviceContext4, Wait, ctx_Wait);
                    c4->Release();
                }
            }
            track_fence_d3d11(f);
        } else {
            DMN_ERROR("fence: D3D11 producer setup failed");
        }
        f->Release();
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedFence(
        ID3D11Device5* This, HANDLE handle, REFIID iid, void** out) {
    if (handle && out &&
        *reinterpret_cast<uint32_t*>(handle) == DMN_SHARED_FENCE_MAGIC) {
        ID3D11Fence* f = nullptr;
        HRESULT hr = dmn_fd3d_import_d3d11(
            This, reinterpret_cast<dmn_shared_fence_handle*>(handle), &f);
        if (FAILED(hr) || !f)
            return hr;
        track_fence_d3d11(f); /* a consumer-only process patches only here */
        hr = f->QueryInterface(iid, out);
        f->Release();
        return hr;
    }
    auto orig = DMN_ORIG(d3d11_OpenSharedFence, This);
    return orig ? orig(This, handle, iid, out) : E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE hook_ctx_Signal(
        ID3D11DeviceContext4* This, ID3D11Fence* fence, UINT64 value) {
    auto orig = DMN_ORIG(ctx_Signal, This);
    /* Producer: encode the value store BEFORE the app's Signal so it rides
     * the same submission as the immediate context's pending render (the
     * work this value represents), making the store GPU-ordered strictly
     * after it. No-op for imports. */
    dmn_fd3d_on_ctx_signal(fence, value);
    HRESULT hr = orig ? orig(This, fence, value) : E_FAIL;
    /* Import signal-back: store into the slot once the local fence completes
     * the value. */
    if (SUCCEEDED(hr))
        dmn_fd3d_after_ctx_signal(fence, value);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_ctx_Wait(
        ID3D11DeviceContext4* This, ID3D11Fence* fence, UINT64 value) {
    /* Import: arm a watcher that raises the local fence when the shared slot
     * reaches the value, releasing the native wait enqueued below. */
    dmn_fd3d_before_ctx_wait(This, fence, value);
    auto orig = DMN_ORIG(ctx_Wait, This);
    return orig ? orig(This, fence, value) : E_FAIL;
}

/* == DXGI resource exports + keyed-mutex QI ================================ */
HRESULT STDMETHODCALLTYPE hook_GetSharedHandle(IDXGIResource* This, HANDLE* out) {
    if (SUCCEEDED(return_texture_pod(This, out, /*nt=*/false)) ||
        SUCCEEDED(return_buffer_pod(This, out, /*nt=*/false)))
        return S_OK;
    auto orig = DMN_ORIG(GetSharedHandle, This);
    return orig ? orig(This, out) : E_FAIL;
}

HRESULT STDMETHODCALLTYPE hook_res_CreateSharedHandle(
        IDXGIResource1* This, const SECURITY_ATTRIBUTES* attr, DWORD access,
        const WCHAR* name, HANDLE* out) {
    if (SUCCEEDED(return_texture_pod(This, out, /*nt=*/true)) ||
        SUCCEEDED(return_buffer_pod(This, out, /*nt=*/true)))
        return S_OK;
    auto orig = DMN_ORIG(res_CreateSharedHandle, This);
    return orig ? orig(This, attr, access, name, out) : E_FAIL;
}

HRESULT STDMETHODCALLTYPE hook_res_QueryInterface(
        IUnknown* This, REFIID riid, void** ppv) {
    /* Vend the keyed mutex for textures created/imported with the flag. The
     * identity lookup below QIs IUnknown, which re-enters this thunk with a
     * different IID and falls straight through to orig. */
    if (ppv && dmn_iid_eq(riid, __uuidof(IDXGIKeyedMutex))) {
        if (IDXGIKeyedMutex* km = dmn_kmtx_lookup(dmn_com_identity(This))) {
            *ppv = km; /* AddRef'd by lookup */
            return S_OK;
        }
    }
    auto orig = DMN_ORIG(res_QueryInterface, This);
    return orig ? orig(This, riid, ppv) : E_NOINTERFACE;
}

/* == D3D11 producer-fence object thunks ==================================== */
HRESULT STDMETHODCALLTYPE hook_f11_CreateSharedHandle(
        ID3D11Fence* This, const SECURITY_ATTRIBUTES* attr, DWORD access,
        const WCHAR* name, HANDLE* out) {
    if (SUCCEEDED(return_fence_pod(This, out)))
        return S_OK;
    auto orig = DMN_ORIG(f11_CreateSharedHandle, This);
    return orig ? orig(This, attr, access, name, out) : E_FAIL;
}

UINT64 STDMETHODCALLTYPE hook_f11_GetCompletedValue(ID3D11Fence* This) {
    auto orig = DMN_ORIG(f11_GetCompletedValue, This);
    UINT64 v = orig ? orig(This) : 0;
    /* Consumer signal-back lands in the shared slot, not the real fence. */
    return dmn_fd3d_completed_merge(This, v);
}

HRESULT STDMETHODCALLTYPE hook_f11_SetEventOnCompletion(
        ID3D11Fence* This, UINT64 value, HANDLE ev) {
    auto orig = DMN_ORIG(f11_SetEventOnCompletion, This);
    HRESULT hr = orig ? orig(This, value, ev) : E_FAIL;
    /* Also watch the shared slot so consumer signal-back releases waiters. */
    if (SUCCEEDED(hr))
        dmn_fd3d_watch_slot(This, value, ev);
    return hr;
}

/* == D3D12 committed/placed creates ======================================== */
/* Common shared-create wrapper: arm by dimension, run the actual create, and
 * record what the swizzle captured. `DescT` is D3D12_RESOURCE_DESC or
 * D3D12_RESOURCE_DESC1 (layout-compatible for the fields used here). */
template <class DescT, class OrigCall>
HRESULT d12_create_shared(const DescT* desc, void** out, OrigCall&& call) {
    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        DMN_INFO("hooks: D3D12 SHARED buffer create size=%llu",
                 (unsigned long long)desc->Width);
        dmn_share_arm_producer_buffer(desc->Width);
        HRESULT hr = call();
        DmnShareArm arm{};
        bool captured = dmn_share_disarm(&arm);
        if (SUCCEEDED(hr) && captured && out && *out)
            record_buffer(reinterpret_cast<IUnknown*>(*out), desc->Width,
                          0, 0, 0, arm);
        return hr;
    }
    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        DMN_INFO("hooks: D3D12 SHARED texture create %llux%u fmt=%u",
                 (unsigned long long)desc->Width, desc->Height, desc->Format);
        dmn_share_arm_producer(0);
        HRESULT hr = call();
        DmnShareArm arm{};
        bool captured = dmn_share_disarm(&arm);
        if (SUCCEEDED(hr) && captured && out && *out) {
            /* Flatten the D3D12 desc into the POD's D3D11 shape. */
            D3D11_TEXTURE2D_DESC d{};
            d.Width = (UINT)desc->Width;
            d.Height = desc->Height;
            d.MipLevels = desc->MipLevels ? desc->MipLevels : 1;
            d.ArraySize = desc->DepthOrArraySize ? desc->DepthOrArraySize : 1;
            d.Format = desc->Format;
            d.SampleDesc.Count = desc->SampleDesc.Count ? desc->SampleDesc.Count : 1;
            record_texture(reinterpret_cast<IUnknown*>(*out), d, arm);
        }
        return hr;
    }
    DMN_WARN("hooks: D3D12 SHARED create with unsupported dimension %d passes "
             "through UNSHARED", (int)desc->Dimension);
    return call();
}

bool heap_is_shared(ID3D12Heap* heap) {
    if (!heap)
        return false;
    void* id = dmn_com_identity(reinterpret_cast<IUnknown*>(heap));
    std::lock_guard<std::mutex> lk(g_reg_mtx);
    return g_shared_heaps.count(id) != 0;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource(
        ID3D12Device* This, const D3D12_HEAP_PROPERTIES* hp,
        D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear,
        REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommittedResource, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, hp, flags, desc, state, clear, riid, out); };
    if (dmn_share_is_armed() || !desc || !(flags & D3D12_HEAP_FLAG_SHARED))
        return call();
    return d12_create_shared(desc, out, call);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource1(
        ID3D12Device4* This, const D3D12_HEAP_PROPERTIES* hp,
        D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear,
        DMN_ID3D12ProtectedResourceSession* session, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommittedResource1, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, hp, flags, desc, state, clear, session,
                                  riid, out); };
    if (dmn_share_is_armed() || !desc || !(flags & D3D12_HEAP_FLAG_SHARED))
        return call();
    return d12_create_shared(desc, out, call);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource2(
        ID3D12Device8* This, const D3D12_HEAP_PROPERTIES* hp,
        D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC1* desc,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear,
        DMN_ID3D12ProtectedResourceSession* session, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommittedResource2, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, hp, flags, desc, state, clear, session,
                                  riid, out); };
    if (dmn_share_is_armed() || !desc || !(flags & D3D12_HEAP_FLAG_SHARED))
        return call();
    return d12_create_shared(desc, out, call);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource3(
        ID3D12Device10* This, const D3D12_HEAP_PROPERTIES* hp,
        D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC1* desc,
        DMN_D3D12_BARRIER_LAYOUT layout, const D3D12_CLEAR_VALUE* clear,
        DMN_ID3D12ProtectedResourceSession* session, UINT32 num_castable,
        const DXGI_FORMAT* castable, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommittedResource3, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, hp, flags, desc, layout, clear, session,
                                  num_castable, castable, riid, out); };
    if (dmn_share_is_armed() || !desc || !(flags & D3D12_HEAP_FLAG_SHARED))
        return call();
    return d12_create_shared(desc, out, call);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateHeap(
        ID3D12Device* This, const D3D12_HEAP_DESC* desc, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateHeap, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, desc, riid, out);
    if (SUCCEEDED(hr) && desc && (desc->Flags & D3D12_HEAP_FLAG_SHARED) &&
        out && *out)
        record_shared_heap(reinterpret_cast<IUnknown*>(*out));
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateHeap1(
        ID3D12Device4* This, const D3D12_HEAP_DESC* desc,
        DMN_ID3D12ProtectedResourceSession* session, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateHeap1, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, desc, session, riid, out);
    if (SUCCEEDED(hr) && desc && (desc->Flags & D3D12_HEAP_FLAG_SHARED) &&
        out && *out)
        record_shared_heap(reinterpret_cast<IUnknown*>(*out));
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource(
        ID3D12Device* This, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state,
        const D3D12_CLEAR_VALUE* clear, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreatePlacedResource, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, heap, offset, desc, state, clear,
                                  riid, out); };
    if (dmn_share_is_armed() || !desc || !heap_is_shared(heap))
        return call();
    return d12_create_shared(desc, out, call);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource1(
        ID3D12Device8* This, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC1* desc, D3D12_RESOURCE_STATES state,
        const D3D12_CLEAR_VALUE* clear, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreatePlacedResource1, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, heap, offset, desc, state, clear,
                                  riid, out); };
    if (dmn_share_is_armed() || !desc || !heap_is_shared(heap))
        return call();
    return d12_create_shared(desc, out, call);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource2(
        ID3D12Device10* This, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC1* desc, DMN_D3D12_BARRIER_LAYOUT layout,
        const D3D12_CLEAR_VALUE* clear, UINT32 num_castable,
        const DXGI_FORMAT* castable, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreatePlacedResource2, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, heap, offset, desc, layout, clear,
                                  num_castable, castable, riid, out); };
    if (dmn_share_is_armed() || !desc || !heap_is_shared(heap))
        return call();
    return d12_create_shared(desc, out, call);
}

/* == D3D12 fences ========================================================== */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateFence(
        ID3D12Device* This, UINT64 initial, D3D12_FENCE_FLAGS flags,
        REFIID iid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateFence, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, initial, flags, iid, out);
    if (FAILED(hr) || !out || !*out)
        return hr;
    /* Only shared fences get a companion buffer + helper queue; the export
     * path is ID3D12Device::CreateSharedHandle (device-side). */
    if (!(flags & D3D12_FENCE_FLAG_SHARED))
        return hr;
    ID3D12Fence* f = nullptr;
    if (SUCCEEDED(reinterpret_cast<IUnknown*>(*out)->QueryInterface(
            __uuidof(ID3D12Fence), reinterpret_cast<void**>(&f))) && f) {
        if (dmn_fd3d_producer_create_d3d12(This, f, initial, (UINT)flags)) {
            track_fence_d3d12(f);
        } else {
            DMN_ERROR("fence: D3D12 producer setup failed");
        }
        f->Release();
    }
    return hr;
}

UINT64 STDMETHODCALLTYPE hook_f12_GetCompletedValue(ID3D12Fence* This) {
    auto orig = DMN_ORIG(f12_GetCompletedValue, This);
    UINT64 v = orig ? orig(This) : 0;
    return dmn_fd3d_completed_merge(This, v);
}

HRESULT STDMETHODCALLTYPE hook_f12_SetEventOnCompletion(
        ID3D12Fence* This, UINT64 value, HANDLE ev) {
    auto orig = DMN_ORIG(f12_SetEventOnCompletion, This);
    HRESULT hr = orig ? orig(This, value, ev) : E_FAIL;
    if (SUCCEEDED(hr))
        dmn_fd3d_watch_slot(This, value, ev);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_f12_Signal(ID3D12Fence* This, UINT64 value) {
    auto orig = DMN_ORIG(f12_Signal, This);
    HRESULT hr = orig ? orig(This, value) : E_FAIL;
    /* Windows: a CPU fence signal is immediately visible cross-process. */
    if (SUCCEEDED(hr))
        dmn_fd3d_on_cpu_signal(reinterpret_cast<IUnknown*>(This), value);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_queue_Signal(
        ID3D12CommandQueue* This, ID3D12Fence* fence, UINT64 value) {
    auto orig = DMN_ORIG(queue_Signal, This);
    HRESULT hr = orig ? orig(This, fence, value) : E_FAIL;
    /* Producer: GPU-write the slot via the helper queue. Import signal-back:
     * store into the slot once the local fence completes the value. */
    if (SUCCEEDED(hr))
        dmn_fd3d_on_queue_signal(fence, value);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_queue_Wait(
        ID3D12CommandQueue* This, ID3D12Fence* fence, UINT64 value) {
    /* Import: arm a watcher that raises the local fence when the shared slot
     * reaches the value, releasing the native wait enqueued below. */
    dmn_fd3d_before_queue_wait(fence, value);
    auto orig = DMN_ORIG(queue_Wait, This);
    return orig ? orig(This, fence, value) : E_FAIL;
}

void patch_d3d12_queue(IUnknown* obj) {
    ID3D12CommandQueue* q = nullptr;
    if (SUCCEEDED(obj->QueryInterface(__uuidof(ID3D12CommandQueue),
                                      reinterpret_cast<void**>(&q))) && q) {
        DMN_PATCH(q, ID3D12CommandQueue, Signal, queue_Signal);
        DMN_PATCH(q, ID3D12CommandQueue, Wait, queue_Wait);
        q->Release();
    }
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandQueue(
        ID3D12Device* This, const D3D12_COMMAND_QUEUE_DESC* desc,
        REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommandQueue, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, desc, riid, out);
    if (SUCCEEDED(hr) && out && *out)
        patch_d3d12_queue(reinterpret_cast<IUnknown*>(*out));
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandQueue1(
        ID3D12Device9* This, const D3D12_COMMAND_QUEUE_DESC* desc,
        REFIID creator, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommandQueue1, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, desc, creator, riid, out);
    if (SUCCEEDED(hr) && out && *out)
        patch_d3d12_queue(reinterpret_cast<IUnknown*>(*out));
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_SetEventOnMultipleFenceCompletion(
        ID3D12Device1* This, ID3D12Fence* const* fences, const UINT64* values,
        UINT count, D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE ev) {
    /* Imports are real fences, so the native multi-wait works as-is; each
     * imported entry just needs its wait watcher armed so the local fence
     * reaches the awaited value. */
    for (UINT i = 0; fences && values && i < count; i++)
        dmn_fd3d_before_queue_wait(fences[i], values[i]);
    auto orig = DMN_ORIG(d3d12_SetEventOnMultipleFenceCompletion, This);
    return orig ? orig(This, fences, values, count, flags, ev) : E_NOTIMPL;
}

/* == D3D12 share entry points ============================================== */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateSharedHandle(
        ID3D12Device* This, ID3D12DeviceChild* object,
        const SECURITY_ATTRIBUTES* attr, DWORD access, const WCHAR* name,
        HANDLE* out) {
    if (object && out) {
        auto* unk = reinterpret_cast<IUnknown*>(object);
        if (SUCCEEDED(return_fence_pod(unk, out)))
            return S_OK;
        if (SUCCEEDED(return_texture_pod(unk, out, /*nt=*/true)))
            return S_OK;
        if (SUCCEEDED(return_buffer_pod(unk, out, /*nt=*/true)))
            return S_OK;
    }
    DMN_WARN("hooks: CreateSharedHandle on an object that never went through "
             "the shared create path; falling through to D3DMetal (stubbed)");
    auto orig = DMN_ORIG(d3d12_CreateSharedHandle, This);
    return orig ? orig(This, object, attr, access, name, out) : E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_OpenSharedHandle(
        ID3D12Device* This, HANDLE handle, REFIID riid, void** out) {
    if (handle && out) {
        uint32_t magic = *reinterpret_cast<uint32_t*>(handle);
        if (magic == DMN_SHARED_FENCE_MAGIC) {
            ID3D12Fence* f = nullptr;
            HRESULT hr = dmn_fd3d_import_d3d12(
                This, reinterpret_cast<dmn_shared_fence_handle*>(handle), &f);
            if (FAILED(hr) || !f)
                return hr;
            track_fence_d3d12(f); /* a consumer-only process patches only here */
            hr = f->QueryInterface(riid, out);
            f->Release();
            return hr;
        }
        if (magic == DMN_SHARED_TEXTURE_MAGIC)
            return import_texture_d3d12(
                This, reinterpret_cast<dmn_shared_texture_handle*>(handle), riid, out);
        if (magic == DMN_SHARED_BUFFER_MAGIC)
            return import_buffer_d3d12(
                This, reinterpret_cast<dmn_shared_buffer_handle*>(handle), riid, out);
    }
    auto orig = DMN_ORIG(d3d12_OpenSharedHandle, This);
    return orig ? orig(This, handle, riid, out) : E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_OpenSharedHandleByName(
        ID3D12Device* This, const WCHAR* name, DWORD access, HANDLE* out) {
    (void)This; (void)name; (void)access;
    if (out)
        *out = nullptr;
    DMN_ERROR("hooks: OpenSharedHandleByName is not supported — named handles "
              "do not exist here; ship the handle POD + fd instead");
    return E_NOTIMPL;
}

} // namespace

/* == Public API: close an NT-style shared handle ========================== */
/* Contract documented on the declaration in d3dmetal_native.h. */
extern "C" dmn_result dmn_shared_handle_close(void* handle) {
    if (!handle)
        return DMN_ERROR_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lk(g_nt_mtx);
        if (!g_nt_handles.erase(handle))
            return DMN_ERROR_INVALID_ARGUMENT;
    }
    int32_t fd;
    std::memcpy(&fd, static_cast<char*>(handle) + kPodFdOffset, sizeof(fd));
    if (fd >= 0)
        close(fd);
    std::free(handle);
    return DMN_SUCCESS;
}

/* == Provided to dmn_fence_d3d.cpp ======================================== */
HRESULT dmn_hooks_f12_signal_orig(ID3D12Fence* fence, UINT64 value) {
    auto orig = DMN_ORIG(f12_Signal, fence);
    return orig ? orig(fence, value) : fence->Signal(value);
}

HRESULT dmn_hooks_ctx_signal_orig(ID3D11DeviceContext4* c, ID3D11Fence* fence,
                                  UINT64 value) {
    auto orig = DMN_ORIG(ctx_Signal, c);
    return orig ? orig(c, fence, value) : c->Signal(fence, value);
}

bool dmn_res_lookup_buffer_pod(IUnknown* res, dmn_shared_buffer_handle* out) {
    void* id = dmn_com_identity(res);
    std::lock_guard<std::mutex> lk(g_reg_mtx);
    auto it = g_buf_reg.find(id);
    if (it == g_buf_reg.end())
        return false;
    *out = it->second;
    return true;
}

/* == Post-create patch entry points (called from dmn_init.cpp) =========== */

extern "C" void dmn_hooks_after_d3d11_device(void* device) {
    if (!device)
        return;
    auto* dev = reinterpret_cast<IUnknown*>(device);

    ID3D11Device* d0 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device),
                                      reinterpret_cast<void**>(&d0))) && d0) {
        DMN_PATCH(d0, ID3D11Device, CreateTexture2D, d3d11_CreateTexture2D);
        DMN_PATCH(d0, ID3D11Device, CreateTexture1D, d3d11_CreateTexture1D);
        DMN_PATCH(d0, ID3D11Device, CreateTexture3D, d3d11_CreateTexture3D);
        DMN_PATCH(d0, ID3D11Device, CreateBuffer, d3d11_CreateBuffer);
        DMN_PATCH(d0, ID3D11Device, OpenSharedResource, d3d11_OpenSharedResource);
        /* Format capability reporting (dmn_formats.mm) — replaces D3DMetal's
         * all-bits-or-nothing answer. */
        DMN_PATCH(d0, ID3D11Device, CheckFormatSupport, d3d11_CheckFormatSupport);
        DMN_PATCH(d0, ID3D11Device, CheckFeatureSupport, d3d11_CheckFeatureSupport);
        /* The context Wait hook must exist before a consumer waits on an
         * imported fence, so patch at device creation (per-class vtable). */
        ID3D11DeviceContext* ctx = nullptr;
        d0->GetImmediateContext(&ctx);
        if (ctx) {
            ID3D11DeviceContext4* c4 = nullptr;
            if (SUCCEEDED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext4),
                                              reinterpret_cast<void**>(&c4))) && c4) {
                DMN_PATCH(c4, ID3D11DeviceContext4, Signal, ctx_Signal);
                DMN_PATCH(c4, ID3D11DeviceContext4, Wait, ctx_Wait);
                c4->Release();
            }
            ctx->Release();
        }
        d0->Release();
    }
    ID3D11Device1* d1 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device1),
                                      reinterpret_cast<void**>(&d1))) && d1) {
        DMN_PATCH(d1, ID3D11Device1, OpenSharedResource1, d3d11_OpenSharedResource1);
        DMN_PATCH(d1, ID3D11Device1, OpenSharedResourceByName, d3d11_OpenSharedResourceByName);
        d1->Release();
    }
    ID3D11Device3* d3 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device3),
                                      reinterpret_cast<void**>(&d3))) && d3) {
        DMN_PATCH(d3, ID3D11Device3, CreateTexture2D1, d3d11_CreateTexture2D1);
        d3->Release();
    }
    ID3D11Device5* d5 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device5),
                                      reinterpret_cast<void**>(&d5))) && d5) {
        DMN_PATCH(d5, ID3D11Device5, CreateFence, d3d11_CreateFence);
        DMN_PATCH(d5, ID3D11Device5, OpenSharedFence, d3d11_OpenSharedFence);
        d5->Release();
    }
    DMN_INFO("hooks: patched D3D11 device %p", device);
}

extern "C" void dmn_hooks_after_d3d12_device(void* device) {
    if (!device)
        return;
    auto* dev = reinterpret_cast<IUnknown*>(device);
    ID3D12Device* d = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(&d))) && d) {
        DMN_PATCH(d, ID3D12Device, OpenSharedHandle, d3d12_OpenSharedHandle);
        DMN_PATCH(d, ID3D12Device, OpenSharedHandleByName, d3d12_OpenSharedHandleByName);
        DMN_PATCH(d, ID3D12Device, CreateFence, d3d12_CreateFence);
        DMN_PATCH(d, ID3D12Device, CreateSharedHandle, d3d12_CreateSharedHandle);
        DMN_PATCH(d, ID3D12Device, CreateCommittedResource, d3d12_CreateCommittedResource);
        DMN_PATCH(d, ID3D12Device, CreateHeap, d3d12_CreateHeap);
        DMN_PATCH(d, ID3D12Device, CreatePlacedResource, d3d12_CreatePlacedResource);
        DMN_PATCH(d, ID3D12Device, CreateCommandQueue, d3d12_CreateCommandQueue);
        d->Release();
    }
    /* Newer device interfaces (declared in dmn_d3d12_up.h) that add shared
     * create variants; D3DMetal exposes all of them via QI. */
    ID3D12Device1* d1 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device1),
                                      reinterpret_cast<void**>(&d1))) && d1) {
        DMN_PATCH(d1, ID3D12Device1, SetEventOnMultipleFenceCompletion,
                  d3d12_SetEventOnMultipleFenceCompletion);
        d1->Release();
    }
    ID3D12Device4* d4 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device4),
                                      reinterpret_cast<void**>(&d4))) && d4) {
        DMN_PATCH(d4, ID3D12Device4, CreateCommittedResource1,
                  d3d12_CreateCommittedResource1);
        DMN_PATCH(d4, ID3D12Device4, CreateHeap1, d3d12_CreateHeap1);
        d4->Release();
    }
    ID3D12Device8* d8 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device8),
                                      reinterpret_cast<void**>(&d8))) && d8) {
        DMN_PATCH(d8, ID3D12Device8, CreateCommittedResource2,
                  d3d12_CreateCommittedResource2);
        DMN_PATCH(d8, ID3D12Device8, CreatePlacedResource1,
                  d3d12_CreatePlacedResource1);
        d8->Release();
    }
    ID3D12Device9* d9 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device9),
                                      reinterpret_cast<void**>(&d9))) && d9) {
        DMN_PATCH(d9, ID3D12Device9, CreateCommandQueue1, d3d12_CreateCommandQueue1);
        d9->Release();
    }
    ID3D12Device10* d10 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device10),
                                      reinterpret_cast<void**>(&d10))) && d10) {
        DMN_PATCH(d10, ID3D12Device10, CreateCommittedResource3,
                  d3d12_CreateCommittedResource3);
        DMN_PATCH(d10, ID3D12Device10, CreatePlacedResource2,
                  d3d12_CreatePlacedResource2);
        d10->Release();
    }
    DMN_INFO("hooks: patched D3D12 device %p", device);
}
