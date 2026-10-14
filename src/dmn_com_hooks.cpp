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
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include "d3dmetal_native.h"
#include "dmn_d3d12_up.h"
#include "dmn_dxil_reflect.h"
#include "dmn_fence_d3d.h"
#include "dmn_formats.h"
#include "dmn_hook.h"
#include "dmn_kmtx.h"
#include "dmn_log.h"
#include "dmn_private.h"
#include "dmn_share.h"
#include "dmn_sparse.h"

/* D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, absent from the vendored d3d12.h.
 * Passed on every committed create whose allocation is substituted with a
 * window over EXISTING bytes (placed translations, buffer imports): the
 * framework zero-fills committed resources inside the create, and those
 * bytes are a producer's — live guest pages, a peer's surface. The flag
 * removes the memset at the source; the matching arm carries
 * create_not_zeroed so the zero-fill detection (dmn_share_metal.mm)
 * skips these creates. */
static const D3D12_HEAP_FLAGS kHeapCreateNotZeroed = (D3D12_HEAP_FLAGS)0x1000;

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
std::unordered_map<void*, dxn_shared_texture> g_tex_reg; /* identity -> POD */
std::unordered_map<void*, dxn_shared_buffer>  g_buf_reg;
std::unordered_map<void*, int> g_owned_fds; /* identity -> registry-owned dup */

/* D3D12 resources created SHARED (committed, or placed in a SHARED heap) with
 * a dimension the shared-memory machinery cannot back (1D/3D). D3D12 lets any
 * resource be shared, so the create stands as a real private resource, but it
 * must not export: a handle for it would hand a peer memory that is not the
 * resource's. Registered so CreateSharedHandle fails them loudly instead of
 * falling through to D3DMetal's fake-passthrough handle. */
std::unordered_set<void*> g_unshared; /* under g_reg_mtx */

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

/* dmn_shared_handle_close() reads the fd through the common descriptor
 * prefix (dxn_shared_header); directx-native asserts every descriptor's
 * layout, which is the cross-process, cross-library ABI. */
constexpr size_t kPodFdOffset = offsetof(dxn_shared_header, fd);

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

enum EvictKind { EV_TEX, EV_BUF, EV_FENCE, EV_UNSHARED, EV_SPARSE };

/* {7D3A1F7B-9C44-4A6E-8F21-5D0E6B3CA942} — private-data slot for the sentinel. */
const GUID kDmnEvictGuid = {0x7d3a1f7b, 0x9c44, 0x4a6e,
                            {0x8f, 0x21, 0x5d, 0x0e, 0x6b, 0x3c, 0xa9, 0x42}};

const char* evict_kind_name(int kind) {
    switch (kind) {
    case EV_TEX:   return "texture";
    case EV_BUF:   return "buffer";
    case EV_FENCE: return "fence";
    case EV_UNSHARED: return "unshareable";
    case EV_SPARSE: return "sparse-reserved";
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
    case EV_FENCE:
        dmn_fd3d_fence_destroy(id);
        break;
    case EV_UNSHARED:
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            g_unshared.erase(id);
        }
        break;
    case EV_SPARSE:
        dmn_sparse_unregister(id);
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

/* == The synthetic ID3D12Heap ============================================= */
/* A SHARED heap (and every heap minted by dmn_open_existing_heap_from_fd) is
 * OUR COM object, never D3DMetal's, holding nothing but the shm object that
 * backs it. Placed creates on it are translated into armed
 * CreateCommittedResource calls whose Metal allocation is substituted with a
 * window into that object.
 *
 * Why the framework must not see the heap: D3DMetal implements an ID3D12Heap
 * as a buffer over its own MTLHeapTypePlacement heap, and a placed resource as
 * another view of the SAME allocation. Substituting the placed resource's
 * backing (which sharing requires) breaks that identity, and the framework's
 * heap memory becomes a live second copy — every placed byte then costs 2x
 * resident. With no framework heap there is nothing to double. The framework
 * also must never RECEIVE this pointer: the placed hooks translate instead of
 * forwarding, and the residency hooks below filter it out of pageable arrays.
 *
 * The one D3D12 fidelity loss: a "placed" resource here is a committed one, so
 * ID3D12Resource::GetHeapProperties reports the heap's Properties (passed
 * through) but D3D12_HEAP_FLAG_NONE instead of SHARED. */

struct DmnSynthHeap;
HRESULT STDMETHODCALLTYPE sheap_QueryInterface(DmnSynthHeap*, REFIID, void**);
ULONG STDMETHODCALLTYPE sheap_AddRef(DmnSynthHeap*);
ULONG STDMETHODCALLTYPE sheap_Release(DmnSynthHeap*);
HRESULT STDMETHODCALLTYPE sheap_GetPrivateData(DmnSynthHeap*, REFGUID, UINT*, void*);
HRESULT STDMETHODCALLTYPE sheap_SetPrivateData(DmnSynthHeap*, REFGUID, UINT, const void*);
HRESULT STDMETHODCALLTYPE sheap_SetPrivateDataInterface(DmnSynthHeap*, REFGUID, const IUnknown*);
HRESULT STDMETHODCALLTYPE sheap_SetName(DmnSynthHeap*, LPCWSTR);
HRESULT STDMETHODCALLTYPE sheap_GetDevice(DmnSynthHeap*, REFIID, void**);
D3D12_HEAP_DESC* STDMETHODCALLTYPE sheap_GetDesc(DmnSynthHeap*, D3D12_HEAP_DESC*);

void* g_sheap_vtbl[9] = {
    (void*)sheap_QueryInterface, (void*)sheap_AddRef, (void*)sheap_Release,
    (void*)sheap_GetPrivateData, (void*)sheap_SetPrivateData,
    (void*)sheap_SetPrivateDataInterface, (void*)sheap_SetName,
    (void*)sheap_GetDevice, (void*)sheap_GetDesc,
};

struct DmnSynthHeap {
    void** vtbl;
    std::atomic<ULONG> refs;
    D3D12_HEAP_DESC desc; /* what the app asked for; GetDesc answers it as-is */
    int      fd;          /* owned; closed at destruction */
    uint64_t fd_off;      /* window start inside fd (page-aligned) */
    uint64_t size;        /* usable bytes (== desc.SizeInBytes) */
    bool     imported;    /* dmn_open_existing_heap_from_fd: the pages belong to
                             the caller, so placements are buffers only and are
                             not registered for export */
    IUnknown* device;     /* retained; GetDevice resolves through it */
    /* Honest private-data storage: D3D releases stored interfaces at object
     * destruction, and callers (our own eviction machinery included, were it
     * ever pointed here) rely on that. Linear scan — a handful of entries. */
    std::mutex pd_mtx;
    struct PdEntry {
        GUID guid;
        std::vector<uint8_t> blob;
        IUnknown* iface; /* non-null: interface entry; blob unused */
    };
    std::vector<PdEntry> pd;
};

DmnSynthHeap* as_synth_heap(ID3D12Heap* h) {
    if (!h || *reinterpret_cast<void**>(h) != (void*)g_sheap_vtbl)
        return nullptr;
    return reinterpret_cast<DmnSynthHeap*>(h);
}

HRESULT STDMETHODCALLTYPE sheap_QueryInterface(DmnSynthHeap* self, REFIID riid,
                                               void** ppv) {
    if (!ppv)
        return E_POINTER;
    if (dmn_iid_eq(riid, __uuidof(IUnknown)) ||
        dmn_iid_eq(riid, __uuidof(ID3D12Object)) ||
        dmn_iid_eq(riid, __uuidof(ID3D12DeviceChild)) ||
        dmn_iid_eq(riid, __uuidof(ID3D12Pageable)) ||
        dmn_iid_eq(riid, __uuidof(ID3D12Heap))) {
        self->refs.fetch_add(1, std::memory_order_relaxed);
        *ppv = self;
        return S_OK;
    }
    *ppv = nullptr;
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE sheap_AddRef(DmnSynthHeap* self) {
    return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
}

ULONG STDMETHODCALLTYPE sheap_Release(DmnSynthHeap* self) {
    ULONG r = self->refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (r == 0) {
        /* Placed resources do not hold the heap: each one's MTLBuffer owns its
         * own mmap window, and a mapping keeps the shm object's pages alive on
         * its own — so closing the fd here strands nothing even when the heap
         * is released before its placements. */
        for (auto& e : self->pd)
            if (e.iface)
                e.iface->Release();
        if (self->device)
            self->device->Release();
        if (self->fd >= 0)
            close(self->fd);
        DMN_INFO("hooks: synthetic heap %p destroyed", (void*)self);
        delete self;
    }
    return r;
}

HRESULT STDMETHODCALLTYPE sheap_GetPrivateData(DmnSynthHeap* self, REFGUID guid,
                                               UINT* size, void* data) {
    if (!size)
        return E_INVALIDARG;
    std::lock_guard<std::mutex> lk(self->pd_mtx);
    for (auto& e : self->pd) {
        if (!dmn_iid_eq(e.guid, guid))
            continue;
        if (e.iface) {
            if (!data) {
                *size = (UINT)sizeof(IUnknown*);
                return S_OK;
            }
            if (*size < sizeof(IUnknown*)) {
                *size = (UINT)sizeof(IUnknown*);
                return E_INVALIDARG;
            }
            e.iface->AddRef(); /* the caller receives a reference */
            std::memcpy(data, &e.iface, sizeof(IUnknown*));
            *size = (UINT)sizeof(IUnknown*);
            return S_OK;
        }
        if (!data) {
            *size = (UINT)e.blob.size();
            return S_OK;
        }
        if (*size < e.blob.size()) {
            *size = (UINT)e.blob.size();
            return E_INVALIDARG;
        }
        std::memcpy(data, e.blob.data(), e.blob.size());
        *size = (UINT)e.blob.size();
        return S_OK;
    }
    *size = 0;
    return DXGI_ERROR_NOT_FOUND;
}

HRESULT STDMETHODCALLTYPE sheap_SetPrivateData(DmnSynthHeap* self, REFGUID guid,
                                               UINT size, const void* data) {
    std::lock_guard<std::mutex> lk(self->pd_mtx);
    for (size_t i = 0; i < self->pd.size(); i++) {
        if (!dmn_iid_eq(self->pd[i].guid, guid))
            continue;
        if (self->pd[i].iface)
            self->pd[i].iface->Release();
        if (!data) {
            self->pd.erase(self->pd.begin() + (ptrdiff_t)i);
            return S_OK;
        }
        self->pd[i].iface = nullptr;
        self->pd[i].blob.assign((const uint8_t*)data,
                                (const uint8_t*)data + size);
        return S_OK;
    }
    if (!data)
        return S_OK; /* clearing an absent slot */
    DmnSynthHeap::PdEntry e;
    e.guid = guid;
    e.iface = nullptr;
    e.blob.assign((const uint8_t*)data, (const uint8_t*)data + size);
    self->pd.push_back(std::move(e));
    return S_OK;
}

HRESULT STDMETHODCALLTYPE sheap_SetPrivateDataInterface(DmnSynthHeap* self,
                                                        REFGUID guid,
                                                        const IUnknown* iface) {
    auto* unk = const_cast<IUnknown*>(iface);
    if (unk)
        unk->AddRef(); /* before dropping any previous holder of the slot */
    std::lock_guard<std::mutex> lk(self->pd_mtx);
    for (size_t i = 0; i < self->pd.size(); i++) {
        if (!dmn_iid_eq(self->pd[i].guid, guid))
            continue;
        if (self->pd[i].iface)
            self->pd[i].iface->Release();
        if (!unk) {
            self->pd.erase(self->pd.begin() + (ptrdiff_t)i);
            return S_OK;
        }
        self->pd[i].blob.clear();
        self->pd[i].iface = unk;
        return S_OK;
    }
    if (!unk)
        return S_OK;
    DmnSynthHeap::PdEntry e;
    e.guid = guid;
    e.iface = unk;
    self->pd.push_back(std::move(e));
    return S_OK;
}

HRESULT STDMETHODCALLTYPE sheap_SetName(DmnSynthHeap*, LPCWSTR) {
    return S_OK; /* nothing reads names off a heap that never meets a debugger */
}

HRESULT STDMETHODCALLTYPE sheap_GetDevice(DmnSynthHeap* self, REFIID riid,
                                          void** ppv) {
    if (!ppv)
        return E_POINTER;
    if (!self->device) {
        *ppv = nullptr;
        return E_FAIL;
    }
    return self->device->QueryInterface(riid, ppv);
}

D3D12_HEAP_DESC* STDMETHODCALLTYPE sheap_GetDesc(DmnSynthHeap* self,
                                                 D3D12_HEAP_DESC* out) {
    *out = self->desc;
    return out;
}

/* Build a synthetic heap over `owned_fd` (consumed — closed here on failure)
 * and hand back the `riid` view of it, +1. */
HRESULT synth_heap_new(IUnknown* device_unk, const D3D12_HEAP_DESC& desc,
                       int owned_fd, uint64_t fd_off, uint64_t size,
                       bool imported, REFIID riid, void** out) {
    auto* h = new (std::nothrow) DmnSynthHeap();
    if (!h) {
        if (owned_fd >= 0)
            close(owned_fd);
        return E_OUTOFMEMORY;
    }
    h->vtbl = g_sheap_vtbl;
    h->refs.store(1, std::memory_order_relaxed);
    h->desc = desc;
    h->fd = owned_fd;
    h->fd_off = fd_off;
    h->size = size;
    h->imported = imported;
    h->device = device_unk;
    if (h->device)
        h->device->AddRef();
    HRESULT hr = sheap_QueryInterface(h, riid, out);
    sheap_Release(h); /* construction ref; success leaves the QI ref */
    return hr;
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
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckMultisampleQualityLevels(ID3D11Device*, DXGI_FORMAT,
    UINT, UINT*);
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckMultisampleQualityLevels1(ID3D11Device2*,
    DXGI_FORMAT, UINT, UINT, UINT*);
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckFeatureSupport(ID3D11Device*, D3D11_FEATURE, void*,
    UINT);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateBlendState1(ID3D11Device1*,
    const D3D11_BLEND_DESC1*, ID3D11BlendState1**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateShaderResourceView1(ID3D11Device3*,
    ID3D11Resource*, const D3D11_SHADER_RESOURCE_VIEW_DESC1*,
    ID3D11ShaderResourceView1**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateRenderTargetView1(ID3D11Device3*,
    ID3D11Resource*, const D3D11_RENDER_TARGET_VIEW_DESC1*,
    ID3D11RenderTargetView1**);
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateUnorderedAccessView1(ID3D11Device3*,
    ID3D11Resource*, const D3D11_UNORDERED_ACCESS_VIEW_DESC1*,
    ID3D11UnorderedAccessView1**);
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
HRESULT STDMETHODCALLTYPE hook_d3d12_MakeResident(ID3D12Device*, UINT,
    ID3D12Pageable* const*);
HRESULT STDMETHODCALLTYPE hook_d3d12_Evict(ID3D12Device*, UINT,
    ID3D12Pageable* const*);
HRESULT STDMETHODCALLTYPE hook_d3d12_SetResidencyPriority(ID3D12Device1*, UINT,
    ID3D12Pageable* const*, const D3D12_RESIDENCY_PRIORITY*);
HRESULT STDMETHODCALLTYPE hook_d3d12_EnqueueMakeResident(ID3D12Device3*, UINT,
    UINT, ID3D12Pageable* const*, ID3D12Fence*, UINT64);
HRESULT STDMETHODCALLTYPE hook_d3d12_CheckFeatureSupport(ID3D12Device*, D3D12_FEATURE,
    void*, UINT);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateReservedResource(ID3D12Device*,
        const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
        REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateReservedResource1(ID3D12Device4*,
        const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
        DMN_ID3D12ProtectedResourceSession*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateReservedResource2(ID3D12Device10*,
        const D3D12_RESOURCE_DESC*, DMN_D3D12_BARRIER_LAYOUT, const D3D12_CLEAR_VALUE*,
        DMN_ID3D12ProtectedResourceSession*, UINT32, const DXGI_FORMAT*, REFIID, void**);
void STDMETHODCALLTYPE hook_d3d12_GetResourceTiling(ID3D12Device*, ID3D12Resource*,
        UINT*, D3D12_PACKED_MIP_INFO*, D3D12_TILE_SHAPE*, UINT*, UINT,
        D3D12_SUBRESOURCE_TILING*);
D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo(
        ID3D12Device*, D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT,
        const D3D12_RESOURCE_DESC*);
D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo1(
        ID3D12Device4*, D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT,
        const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_ALLOCATION_INFO1*);
D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo2(
        ID3D12Device8*, D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT,
        const D3D12_RESOURCE_DESC1*, D3D12_RESOURCE_ALLOCATION_INFO1*);
D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo3(
        ID3D12Device12*, D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT,
        const D3D12_RESOURCE_DESC1*, const UINT32*, const DXGI_FORMAT* const*,
        D3D12_RESOURCE_ALLOCATION_INFO1*);
static bool dmn_desc_format_ok(ID3D12Device* dev, const D3D12_RESOURCE_DESC* d);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandList(ID3D12Device*, UINT,
        D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator*, ID3D12PipelineState*, REFIID,
        void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandList1(ID3D12Device4*, UINT,
        D3D12_COMMAND_LIST_TYPE, UINT, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandQueue(ID3D12Device*,
    const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateGraphicsPipelineState(ID3D12Device*,
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateComputePipelineState(ID3D12Device*,
    const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandQueue1(ID3D12Device9*,
    const D3D12_COMMAND_QUEUE_DESC*, REFIID, REFIID, void**);
HRESULT STDMETHODCALLTYPE hook_d3d12_SetEventOnMultipleFenceCompletion(ID3D12Device1*,
    ID3D12Fence* const*, const UINT64*, UINT, D3D12_MULTIPLE_FENCE_WAIT_FLAGS, HANDLE);

/* D3D12 queue + fence object. */
HRESULT STDMETHODCALLTYPE hook_queue_Signal(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
HRESULT STDMETHODCALLTYPE hook_queue_Wait(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
/* Declared with the REAL 10-parameter signature (MSDN: pHeap is the 5th
 * parameter). The vendored d3d12.h mis-declares this method WITHOUT the heap
 * parameter — only its slot index is borrowed from that declaration; calling
 * through the vendored prototype would pass every argument after the fourth
 * in the wrong register. */
void STDMETHODCALLTYPE hook_queue_UpdateTileMappings(ID3D12CommandQueue*,
    ID3D12Resource*, UINT, const D3D12_TILED_RESOURCE_COORDINATE*,
    const D3D12_TILE_REGION_SIZE*, ID3D12Heap*, UINT,
    const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*,
    D3D12_TILE_MAPPING_FLAGS);
void STDMETHODCALLTYPE hook_queue_CopyTileMappings(ID3D12CommandQueue*,
    ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*, ID3D12Resource*,
    const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*,
    D3D12_TILE_MAPPING_FLAGS);
/* D3D12 graphics command list. */
void STDMETHODCALLTYPE hook_cmdlist_CopyTiles(ID3D12GraphicsCommandList*,
    ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*,
    const D3D12_TILE_REGION_SIZE*, ID3D12Resource*, UINT64, D3D12_TILE_COPY_FLAGS);
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
DMN_HOOK_STATE(d3d11_CheckMultisampleQualityLevels);
DMN_HOOK_STATE(d3d11_CheckMultisampleQualityLevels1);
DMN_HOOK_STATE(d3d11_CreateBlendState1);
DMN_HOOK_STATE(d3d11_CreateShaderResourceView1);
DMN_HOOK_STATE(d3d11_CreateRenderTargetView1);
DMN_HOOK_STATE(d3d11_CreateUnorderedAccessView1);
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
DMN_HOOK_STATE(d3d12_MakeResident);
DMN_HOOK_STATE(d3d12_Evict);
DMN_HOOK_STATE(d3d12_SetResidencyPriority);
DMN_HOOK_STATE(d3d12_EnqueueMakeResident);
DMN_HOOK_STATE(d3d12_CreateCommandQueue);
DMN_HOOK_STATE(d3d12_CreateReservedResource);
DMN_HOOK_STATE(d3d12_CreateReservedResource1);
DMN_HOOK_STATE(d3d12_CreateReservedResource2);
DMN_HOOK_STATE(d3d12_GetResourceTiling);
DMN_HOOK_STATE(d3d12_GetResourceAllocationInfo);
DMN_HOOK_STATE(d3d12_GetResourceAllocationInfo1);
DMN_HOOK_STATE(d3d12_GetResourceAllocationInfo2);
DMN_HOOK_STATE(d3d12_GetResourceAllocationInfo3);
DMN_HOOK_STATE(d3d12_CheckFeatureSupport);
DMN_HOOK_STATE(d3d12_CreateCommandList);
DMN_HOOK_STATE(d3d12_CreateCommandList1);
DMN_HOOK_STATE(d3d12_CreateCommandQueue1);
DMN_HOOK_STATE(d3d12_CreateGraphicsPipelineState);
DMN_HOOK_STATE(d3d12_CreateComputePipelineState);
DMN_HOOK_STATE(d3d12_SetEventOnMultipleFenceCompletion);
DMN_HOOK_STATE(queue_Signal);
DMN_HOOK_STATE(queue_Wait);
DMN_HOOK_STATE(queue_UpdateTileMappings);
DMN_HOOK_STATE(queue_CopyTileMappings);
DMN_HOOK_STATE(cmdlist_CopyTiles);
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

/* The POD carries D3D11 bind flags whichever API made the resource, so a
 * consumer of either API is created with the views the producer allowed. */
UINT d12_flags_to_bind(D3D12_RESOURCE_FLAGS f) {
    UINT bind = 0;
    if (!(f & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
        bind |= D3D11_BIND_SHADER_RESOURCE;
    if (f & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        bind |= D3D11_BIND_RENDER_TARGET;
    if (f & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        bind |= D3D11_BIND_DEPTH_STENCIL;
    if (f & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        bind |= D3D11_BIND_UNORDERED_ACCESS;
    return bind;
}

D3D12_RESOURCE_FLAGS bind_to_d12_flags(UINT bind) {
    UINT f = 0;
    if (bind & D3D11_BIND_RENDER_TARGET)
        f |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (bind & D3D11_BIND_DEPTH_STENCIL)
        f |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if (bind & D3D11_BIND_UNORDERED_ACCESS)
        f |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    /* DENY_SHADER_RESOURCE is only legal alongside ALLOW_DEPTH_STENCIL. */
    if (!(bind & D3D11_BIND_SHADER_RESOURCE) && (bind & D3D11_BIND_DEPTH_STENCIL))
        f |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    return (D3D12_RESOURCE_FLAGS)f;
}

/* A shared surface is backed by one linear 2D Metal texture over shared
 * memory, which has one mip, one slice and one sample.  That is also the
 * shape D3D11 guarantees to share without ExtendedResourceSharing (which the
 * CheckFeatureSupport hook therefore clears).  Anything else is refused
 * before it is armed: a substitute of the wrong shape leaves D3DMetal
 * addressing mips, slices or samples the texture does not have.  A MipLevels
 * of 0 asks for the full chain. */
bool share_shape_ok(UINT mips, UINT array, UINT samples, const char* what) {
    if (mips == 1 && array <= 1 && samples <= 1)
        return true;
    DMN_ERROR("hooks: %s: a shared texture is a single linear 2D surface; "
              "mips=%u array=%u samples=%u cannot be shared", what, mips,
              array, samples);
    return false;
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

/* == DXGI view identity ===================================================
 * A DXGI resource view is a distinct object fronting a D3D resource, minted
 * fresh by each QueryInterface(IDXGIResource). Two of its properties vary by
 * D3DMetal version and both break identity resolution on GPTk 4.0b1:
 *
 *   - QueryInterface(IID_IUnknown) on a view returns the VIEW, not the resource
 *     it fronts, so IUnknown is no longer a canonical key (handled by asking
 *     ID3D11DeviceChild / ID3D12Object first — see dmn_com_identity).
 *   - QueryInterface for those interfaces hands back the resource but AddRefs
 *     the VIEW, so the reference is on the wrong object. Releasing the returned
 *     resource destroys it under the app (whose own next Release then faults on
 *     freed memory); releasing nothing instead leaves the view alive forever,
 *     and the view holds a reference on the resource — so the resource is never
 *     destroyed, its registry entry is never evicted, and its fd and shared
 *     mapping leak.
 *
 * Both are properties of the view, so identity resolution has to know it is
 * looking at one. A view's vtable is exactly a vtable we patched GetSharedHandle
 * onto, which the hook state already records — no extra bookkeeping.
 *
 * Which object the QueryInterface counts against is probed once, the first time
 * a view is in hand next to the resource it fronts. */
std::atomic<int> g_view_qi_ref_target{-1}; /* -1 unknown, else DmnQiRefTarget */

void probe_view_qi_ref_target(IUnknown* primary, IUnknown* view) {
    if (g_view_qi_ref_target.load(std::memory_order_acquire) >= 0)
        return;
    const unsigned long pri_before = dmn_com_refs(primary);
    const unsigned long view_before = dmn_com_refs(view);
    ID3D11DeviceChild* c = nullptr;
    if (FAILED(view->QueryInterface(__uuidof(ID3D11DeviceChild),
                                    reinterpret_cast<void**>(&c))) || !c)
        return; /* nothing to learn from; leave it unknown and probe again */

    DmnQiRefTarget target;
    if (dmn_com_refs(primary) > pri_before) {
        target = DMN_QI_REF_RESULT;
        c->Release();
    } else if (dmn_com_refs(view) > view_before) {
        target = DMN_QI_REF_SOURCE;
        view->Release();
        DMN_WARN("hooks: this D3DMetal's DXGI views AddRef themselves when their "
                 "QueryInterface hands back the resource they front; identity "
                 "lookups will release the view instead of the result");
    } else {
        target = DMN_QI_REF_NONE;
        DMN_WARN("hooks: this D3DMetal's DXGI views take no reference at all in "
                 "QueryInterface; identity lookups will release nothing");
    }
    g_view_qi_ref_target.store((int)target, std::memory_order_release);
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
        probe_view_qi_ref_target(res, r1);
        r1->Release();
        return;
    }
    IDXGIResource* r0 = nullptr;
    if (SUCCEEDED(res->QueryInterface(__uuidof(IDXGIResource),
                                      reinterpret_cast<void**>(&r0))) && r0) {
        DMN_PATCH_QI(r0, res_QueryInterface);
        DMN_PATCH(r0, IDXGIResource, GetSharedHandle, GetSharedHandle);
        probe_view_qi_ref_target(res, r0);
        r0->Release();
    }
}

/* Record a shared texture (producer, or consumer re-registration for
 * re-export). `init_kmtx` distinguishes who initializes the mutex page. */
void record_texture_pod(IUnknown* tex, const dxn_shared_texture& pod,
                        bool init_kmtx) {
    void* id = dmn_com_identity(tex);
    dxn_shared_texture rec = pod;
    rec.fd = register_owned_fd(id, pod.fd);
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        g_tex_reg[id] = rec;
    }
    patch_shared_resource_exports(tex);
    if (wants_kmtx(rec.meta.misc_flags))
        dmn_kmtx_register(id, rec.fd, dmn_share_page_align((size_t)rec.size),
                          init_kmtx);
    attach_evict_sentinel(tex, id, EV_TEX);
    DMN_INFO("hooks: recorded shared texture identity=%p fd=%d %ux%u stride=%llu",
             id, rec.fd, rec.meta.width, rec.meta.height, (unsigned long long)rec.planes[0].pitch);
}

void record_texture(IUnknown* tex, const D3D11_TEXTURE2D_DESC& d,
                    const DmnShareArm& arm, uint64_t offset = 0) {
    dxn_shared_texture pod{};
    pod.magic = DXN_SHARED_TEXTURE_MAGIC;
    pod.version = DXN_SHARED_VERSION;
    pod.struct_size = sizeof(pod);
    pod.planes[0].offset = offset;
    pod.fd = arm.out_fd;
    pod.meta.width = d.Width;
    pod.meta.height = d.Height;
    pod.meta.dxgi_format = d.Format;
    pod.meta.mip_levels = d.MipLevels ? d.MipLevels : 1;
    pod.meta.array_size = d.ArraySize ? d.ArraySize : 1;
    pod.meta.sample_count = d.SampleDesc.Count ? d.SampleDesc.Count : 1;
    pod.meta.sample_quality = d.SampleDesc.Quality;
    pod.meta.usage = d.Usage;
    pod.meta.bind_flags = d.BindFlags;
    pod.meta.misc_flags = d.MiscFlags;
    pod.meta.cpu_access_flags = d.CPUAccessFlags;
    /* One linear plane of shared memory. */
    pod.plane_count = 1;
    pod.planes[0].pitch = arm.out_stride;
    pod.size = arm.out_size;
    record_texture_pod(tex, pod, /*init_kmtx=*/true);
}

/* Register `res` as a shared buffer under `pod` (whose fd is the transport
 * fd; the registry takes its own dup), patch its export methods, and attach
 * the eviction sentinel. Producer captures and consumer imports, both APIs.
 * The export patch is a no-op for D3D12 buffers, which have no IDXGIResource
 * and export device-side. */
void register_buffer_pod(IUnknown* res, dxn_shared_buffer pod) {
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
                   uint32_t cpu, const DmnShareArm& arm, uint64_t offset = 0) {
    dxn_shared_buffer pod{};
    pod.magic = DXN_SHARED_BUFFER_MAGIC;
    pod.version = DXN_SHARED_VERSION;
    pod.struct_size = sizeof(pod);
    pod.fd = arm.out_fd;
    pod.size = arm.out_size ? arm.out_size : size;
    pod.bind_flags = bind;
    pod.misc_flags = misc;
    pod.cpu_access_flags = cpu;
    pod.offset = offset;
    register_buffer_pod(res, pod);
}

/* A handle from a peer built against another POD layout must not be parsed:
 * the fields past the header would be read from wherever that peer's struct
 * ends. */
/* The caller has dispatched on the magic; the version and size must match
 * this build's directx-native exactly. */
template <class Pod>
bool pod_version_ok(const Pod* pod, const char* what) {
    if (pod->version == DXN_SHARED_VERSION && pod->struct_size == sizeof(Pod))
        return true;
    DMN_ERROR("hooks: %s handle has descriptor version %u size %u, this build "
              "speaks %u size %zu; refusing to import it", what, pod->version,
              pod->struct_size, (unsigned)DXN_SHARED_VERSION, sizeof(Pod));
    return false;
}

/* A shm-backed texture is exactly one linear plane. */
bool texture_planes_ok(const dxn_shared_texture* pod) {
    if (pod->plane_count == 1)
        return true;
    DMN_ERROR("hooks: texture handle has %u planes; only single-plane shm "
              "textures can be imported", pod->plane_count);
    return false;
}

/* == Export (handle vending) ============================================== */
/* `nt` selects the handle regime: false = legacy GetSharedHandle (cached on
 * the resource, never closed, dies with it); true = CreateSharedHandle (a
 * fresh owned handle the app must dmn_shared_handle_close). */
HRESULT return_texture_pod(IUnknown* res, HANDLE* out, bool nt) {
    if (!out)
        return E_INVALIDARG;
    void* id = dmn_com_identity(res);
    dxn_shared_texture pod;
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
    dxn_shared_buffer pod;
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
    dxn_shared_fence pod;
    if (!dmn_fd3d_export(fenceObj, &pod))
        return E_FAIL; /* not one of ours */
    HRESULT hr = vend_pod_owned(pod, out);
    if (SUCCEEDED(hr))
        DMN_INFO("hooks: exported NT fence handle %p (fd=%d)", *out, pod.fd);
    return hr;
}

/* Close a PRODUCER arm's fd. The producer substitution hands its fd's
 * ownership out through the arm (the impostor MTLBuffer owns only the
 * mapping), so every producer create site must sink it exactly once: after
 * the registration has taken its dup, or when the create failed with a
 * capture. Consumer/window arms borrow their fd and are excluded by
 * alloc_new. */
void arm_fd_close(const DmnShareArm& arm) {
    if (arm.captured && arm.alloc_new && arm.out_fd >= 0)
        close(arm.out_fd);
}

/* A shared create that did not route through the Metal swizzle produced an
 * ordinary, process-private resource. Returning it with a success HRESULT
 * hands the app a valid-looking handle whose peer can never see anything, and
 * the mistake surfaces much later as a blank surface with no failing call to
 * point at. Fail the create instead and drop the resource nobody may use.
 *
 * `out` is optional throughout D3D: a create with a null out-pointer is a
 * parameter-validation call that allocates nothing, so callers must only reach
 * here when a resource was actually produced. */
HRESULT fail_unshared(const char* what, void** out) {
    DMN_ERROR("hooks: %s did not route through the Metal swizzle", what);
    if (out && *out) {
        reinterpret_cast<IUnknown*>(*out)->Release();
        *out = nullptr;
    }
    return E_FAIL;
}

/* Record a successfully created resource that carries a SHARED flag but has
 * no shared backing (see g_unshared). */
void register_unshared(void** out) {
    if (!out || !*out)
        return;
    auto* res = reinterpret_cast<IUnknown*>(*out);
    void* id = dmn_com_identity(res);
    {
        std::lock_guard<std::mutex> lk(g_reg_mtx);
        g_unshared.insert(id);
    }
    attach_evict_sentinel(res, id, EV_UNSHARED);
}

/* == Import (consumer reconstruction) ===================================== */
HRESULT import_texture_d3d11(ID3D11Device* dev, const dxn_shared_texture* pod,
                             REFIID iid, void** out) {
    if (!pod_version_ok(pod, "texture") || !texture_planes_ok(pod) ||
        !share_shape_ok(pod->meta.mip_levels ? pod->meta.mip_levels : 1,
                        pod->meta.array_size, pod->meta.sample_count,
                        "D3D11 texture import"))
        return E_INVALIDARG;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = pod->meta.width;
    td.Height = pod->meta.height;
    td.MipLevels = pod->meta.mip_levels ? pod->meta.mip_levels : 1;
    td.ArraySize = pod->meta.array_size ? pod->meta.array_size : 1;
    td.Format = (DXGI_FORMAT)pod->meta.dxgi_format;
    td.SampleDesc.Count = pod->meta.sample_count ? pod->meta.sample_count : 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = pod->meta.bind_flags
                       ? pod->meta.bind_flags
                       : (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
    td.CPUAccessFlags = pod->meta.cpu_access_flags;
    td.MiscFlags = pod->meta.misc_flags;

    /* Initial data, pointing at the sentinel. Creating the texture with none
     * leaves D3DMetal treating it as undefined, and the first render pass that
     * attaches it loads with a clear — discarding everything the producer put
     * in the surface we are aliasing. The copy itself is intercepted and
     * dropped in the Metal half, so this marks the texture defined without
     * writing anything. */
    const void* sentinel = dmn_share_init_data_sentinel();
    const UINT subresources = td.MipLevels * td.ArraySize;
    std::vector<D3D11_SUBRESOURCE_DATA> init;
    if (sentinel && subresources) {
        init.resize(subresources);
        for (auto& s : init) {
            s.pSysMem = sentinel;
            s.SysMemPitch = (UINT)pod->planes[0].pitch;
            s.SysMemSlicePitch = (UINT)pod->size;
        }
    }

    ID3D11Texture2D* tex = nullptr;
    dmn_share_arm_consumer(pod->fd, pod->planes[0].pitch, pod->size, pod->planes[0].offset);
    HRESULT hr = dev->CreateTexture2D(&td, init.empty() ? nullptr : init.data(),
                                      &tex);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (FAILED(hr) || !tex) {
        DMN_WARN("hooks: D3D11 texture import CreateTexture2D 0x%08x", (unsigned)hr);
        return FAILED(hr) ? hr : E_FAIL;
    }
    if (!captured)
        return fail_unshared("D3D11 texture import",
                             reinterpret_cast<void**>(&tex));
    /* If the sentinel upload was not intercepted, D3DMetal took a path that
     * copies it, and the surface now holds the sentinel's zeros instead of the
     * producer's pixels. */
    if (!init.empty() && !arm.init_dropped)
        return fail_unshared("D3D11 texture import (sentinel initial data was "
                             "not intercepted)",
                             reinterpret_cast<void**>(&tex));
    /* Register the imported view so re-export (GetSharedHandle on an opened
     * resource — valid on Windows) and keyed-mutex QI work here too. */
    record_texture_pod(tex, *pod, /*init_kmtx=*/false);
    hr = tex->QueryInterface(iid, out);
    tex->Release();
    return hr;
}

HRESULT import_buffer_d3d11(ID3D11Device* dev, const dxn_shared_buffer* pod,
                            REFIID iid, void** out) {
    if (!pod_version_ok(pod, "buffer"))
        return E_INVALIDARG;
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = (UINT)pod->size;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = pod->bind_flags ? pod->bind_flags
                                   : D3D11_BIND_UNORDERED_ACCESS;
    bd.CPUAccessFlags = pod->cpu_access_flags;
    bd.MiscFlags = pod->misc_flags;

    ID3D11Buffer* buf = nullptr;
    if (pod->offset)
        dmn_share_arm_import_window(pod->fd, pod->offset, pod->size, 0,
                                    /*create_not_zeroed=*/false);
    else
        dmn_share_arm_consumer_buffer(pod->fd, pod->size,
                                      /*create_not_zeroed=*/false);
    HRESULT hr = dev->CreateBuffer(&bd, nullptr, &buf);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (FAILED(hr) || !buf) {
        DMN_WARN("hooks: D3D11 buffer import CreateBuffer 0x%08x", (unsigned)hr);
        return FAILED(hr) ? hr : E_FAIL;
    }
    if (!captured)
        return fail_unshared("D3D11 buffer import",
                             reinterpret_cast<void**>(&buf));
    /* The create zeroed the bytes it aliases (unsupported zeroing
     * framework — see the detection in dmn_share_metal.mm): the
     * producer's data is gone, so hand back a loud failure, never a
     * silently-wiped alias. */
    if (arm.zero_filled)
        return fail_unshared("D3D11 buffer import (create zero-filled the "
                             "imported bytes)",
                             reinterpret_cast<void**>(&buf));
    register_buffer_pod(buf, *pod);
    hr = buf->QueryInterface(iid, out);
    buf->Release();
    return hr;
}

HRESULT import_texture_d3d12(ID3D12Device* dev, const dxn_shared_texture* pod,
                             REFIID riid, void** out) {
    if (!pod_version_ok(pod, "texture") || !texture_planes_ok(pod) ||
        !share_shape_ok(pod->meta.mip_levels ? pod->meta.mip_levels : 1,
                        pod->meta.array_size, pod->meta.sample_count,
                        "D3D12 texture import"))
        return E_INVALIDARG;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = pod->meta.width;
    rd.Height = pod->meta.height;
    rd.DepthOrArraySize = (UINT16)(pod->meta.array_size ? pod->meta.array_size : 1);
    rd.MipLevels = (UINT16)(pod->meta.mip_levels ? pod->meta.mip_levels : 1);
    rd.Format = (DXGI_FORMAT)pod->meta.dxgi_format;
    rd.SampleDesc.Count = pod->meta.sample_count ? pod->meta.sample_count : 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = bind_to_d12_flags(pod->meta.bind_flags);

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    dmn_share_arm_consumer(pod->fd, pod->planes[0].pitch, pod->size, pod->planes[0].offset);
    HRESULT hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, riid, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (SUCCEEDED(hr) && out && *out && !captured)
        return fail_unshared("D3D12 texture import", out);
    if (SUCCEEDED(hr) && out && *out) {
        /* Register for re-export via ID3D12Device::CreateSharedHandle. */
        auto* res = reinterpret_cast<IUnknown*>(*out);
        void* id = dmn_com_identity(res);
        dxn_shared_texture rec = *pod;
        rec.fd = register_owned_fd(id, pod->fd);
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            g_tex_reg[id] = rec;
        }
        attach_evict_sentinel(res, id, EV_TEX);
    }
    return hr;
}

HRESULT import_buffer_d3d12(ID3D12Device* dev, const dxn_shared_buffer* pod,
                            REFIID riid, void** out) {
    if (!pod_version_ok(pod, "buffer"))
        return E_INVALIDARG;
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

    if (pod->offset)
        dmn_share_arm_import_window(pod->fd, pod->offset, pod->size, 0,
                                    /*create_not_zeroed=*/true);
    else
        dmn_share_arm_consumer_buffer(pod->fd, pod->size,
                                      /*create_not_zeroed=*/true);
    HRESULT hr = dev->CreateCommittedResource(&hp, kHeapCreateNotZeroed, &rd,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, riid, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (SUCCEEDED(hr) && out && *out && !captured)
        return fail_unshared("D3D12 buffer import", out);
    if (SUCCEEDED(hr) && out && *out)
        register_buffer_pod(reinterpret_cast<IUnknown*>(*out), *pod);
    return hr;
}

/* Open a shared-heap handle (a buffer-magic POD carrying the heap's whole
 * window; see hook_d3d12_CreateSharedHandle) as a synthetic heap of our own.
 * The framework is never involved, and placements on the opened heap window
 * the same shm object as the exporter's, so cross-process placements at one
 * offset alias exactly like local ones. */
HRESULT import_heap_d3d12(ID3D12Device* dev, const dxn_shared_buffer* pod,
                          REFIID riid, void** out) {
    if (!pod_version_ok(pod, "heap"))
        return E_INVALIDARG;
    if (!pod->size || pod->size > (1ull << 32) ||
        (pod->offset & (dmn_share_page_align(1) - 1))) {
        DMN_ERROR("hooks: shared-heap import geometry out of range "
                  "(size=%llu offset=%llu)", (unsigned long long)pod->size,
                  (unsigned long long)pod->offset);
        return E_INVALIDARG;
    }
    const int owned = pod->fd >= 0 ? fcntl(pod->fd, F_DUPFD_CLOEXEC, 0) : -1;
    if (owned < 0) {
        DMN_ERROR("hooks: shared-heap import dup(fd=%d) failed", pod->fd);
        return E_FAIL;
    }
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = pod->size;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = 0;
    hd.Flags = D3D12_HEAP_FLAG_SHARED;
    HRESULT hr = synth_heap_new(reinterpret_cast<IUnknown*>(dev), hd, owned,
                                pod->offset, pod->size, /*imported=*/false,
                                riid, out);
    if (SUCCEEDED(hr))
        DMN_INFO("hooks: opened shared heap size=%llu off=%llu as %p",
                 (unsigned long long)pod->size,
                 (unsigned long long)pod->offset, *out);
    return hr;
}

/* Shared import dispatch for the D3D11 OpenSharedResource variants.
 *
 * Returns false when `handle` is not one of ours, which is the caller's cue to
 * fall through to D3DMetal. Otherwise *out_hr holds the import's own result —
 * including a failure, which must be reported as-is. The two cases have to
 * stay distinguishable: falling through on a failed import would hand D3DMetal
 * a pointer that is not a Windows HANDLE at all, and would turn a precise
 * "this handle's geometry is invalid" into whatever D3DMetal says about a
 * handle it cannot parse. */
bool open_shared_d3d11(ID3D11Device* dev, HANDLE handle, REFIID iid, void** out,
                       HRESULT* out_hr) {
    uint32_t magic = *reinterpret_cast<uint32_t*>(handle);
    if (magic == DXN_SHARED_BUFFER_MAGIC) {
        *out_hr = import_buffer_d3d11(
            dev, reinterpret_cast<dxn_shared_buffer*>(handle), iid, out);
        return true;
    }
    if (magic == DXN_SHARED_TEXTURE_MAGIC) {
        *out_hr = import_texture_d3d11(
            dev, reinterpret_cast<dxn_shared_texture*>(handle), iid, out);
        return true;
    }
    return false;
}

/* Whether D3DMetal itself supports textures of `format`: its own
 * CheckFormatSupport answers 0 for a format it cannot make. */
static bool d3dmetal_has_format(ID3D11Device* dev, DXGI_FORMAT format) {
    auto orig = DMN_ORIG(d3d11_CheckFormatSupport, dev);
    UINT support = 0;
    return orig && SUCCEEDED(orig(dev, format, &support)) && support;
}

/* A texture in a format Metal has no pixel format for (the planar and packed
 * video formats, P8, R1_UNORM) cannot exist here; D3DMetal's D3D12 creates
 * abort on such a format inside Metal validation (dmn_desc_format_ok), and its
 * D3D11 ones take the same descriptor path.  Refuse it the way a driver
 * refuses an unsupported format.  The format table decides, so that whether
 * a create is attempted never depends on what a given D3DMetal version
 * reports about itself. */
static bool d3d11_format_ok(DXGI_FORMAT format) {
    uint32_t support = 0;
    if (dmn_format_support((uint32_t)format, &support))
        return true;
    DMN_WARN("refusing a D3D11 texture in DXGI format %d, which has no Metal "
             "pixel format", (int)format);
    return false;
}

/* == D3D11 producer thunks ================================================= */
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture2D(
        ID3D11Device* This, const D3D11_TEXTURE2D_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture2D** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture2D, This);
    if (!orig)
        return E_FAIL;
    if (desc && !d3d11_format_ok(desc->Format))
        return E_INVALIDARG;
    /* is_armed: a consumer reconstruct is mid-flight; pass straight to the
     * swizzle instead of re-detecting a producer create. */
    if (!desc || dmn_share_is_armed() || !is_shared_misc(desc->MiscFlags))
        return orig(This, desc, init, out);
    if (!share_shape_ok(desc->MipLevels, desc->ArraySize, desc->SampleDesc.Count,
                        "CreateTexture2D MISC_SHARED"))
        return E_INVALIDARG;

    DMN_INFO("hooks: CreateTexture2D MISC_SHARED %ux%u fmt=%u", desc->Width,
             desc->Height, desc->Format);
    dmn_share_arm_producer(kmtx_extra(desc->MiscFlags));
    HRESULT hr = orig(This, desc, init, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    /* A null `out` is a parameter-validation call: nothing was allocated, so
     * there is nothing to have shared and nothing to fail. */
    if (FAILED(hr) || !out || !*out) {
        arm_fd_close(arm);
        return hr;
    }
    if (!captured)
        return fail_unshared("CreateTexture2D MISC_SHARED",
                             reinterpret_cast<void**>(out));
    record_texture(*out, *desc, arm);
    arm_fd_close(arm);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture2D1(
        ID3D11Device3* This, const D3D11_TEXTURE2D_DESC1* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture2D1** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture2D1, This);
    if (!orig)
        return E_FAIL;
    if (desc && !d3d11_format_ok(desc->Format))
        return E_INVALIDARG;
    if (!desc || dmn_share_is_armed() || !is_shared_misc(desc->MiscFlags))
        return orig(This, desc, init, out);
    if (!share_shape_ok(desc->MipLevels, desc->ArraySize, desc->SampleDesc.Count,
                        "CreateTexture2D1 MISC_SHARED"))
        return E_INVALIDARG;

    DMN_INFO("hooks: CreateTexture2D1 MISC_SHARED %ux%u fmt=%u", desc->Width,
             desc->Height, desc->Format);
    dmn_share_arm_producer(kmtx_extra(desc->MiscFlags));
    HRESULT hr = orig(This, desc, init, out);
    DmnShareArm arm{};
    bool captured = dmn_share_disarm(&arm);
    if (FAILED(hr) || !out || !*out) {
        arm_fd_close(arm);
        return hr;
    }
    if (!captured)
        return fail_unshared("CreateTexture2D1 MISC_SHARED",
                             reinterpret_cast<void**>(out));
    /* Flatten DESC1 -> DESC for the POD (drops only TextureLayout). */
    D3D11_TEXTURE2D_DESC d{};
    d.Width = desc->Width; d.Height = desc->Height;
    d.MipLevels = desc->MipLevels; d.ArraySize = desc->ArraySize;
    d.Format = desc->Format; d.SampleDesc = desc->SampleDesc;
    d.Usage = desc->Usage; d.BindFlags = desc->BindFlags;
    d.CPUAccessFlags = desc->CPUAccessFlags; d.MiscFlags = desc->MiscFlags;
    record_texture(*out, d, arm);
    arm_fd_close(arm);
    return hr;
}

/* == CreateBlendState1 repair ==============================================
 * GPTk 4.0b1's ID3D11Device1::CreateBlendState1 returns S_OK and writes NULL.
 * It builds and caches the state correctly, then fetches the COM pointer to
 * hand back through the 4.0-era D3DMExternalInterface lookup
 * (D3D11BlendState::AsInterface(GUID)), which has no case for
 * IID_ID3D11BlendState1 and answers NULL. It neither checks that NULL nor drops
 * the reference it took, so the caller sees success with an empty
 * out-parameter. Sibling CreateBlendState asks for IID_ID3D11BlendState, which
 * the table does have.
 *
 * Repair: satisfy the call through CreateBlendState. The object it returns is
 * the one CreateBlendState1 would have cached — D3DMetal keys blend states on
 * the widened DESC1 and vends one external interface per object, whose vtable
 * is the full ID3D11BlendState1 one (GetDesc1 included) — so the pointer is a
 * valid ID3D11BlendState1 and repeat creates still dedupe to one state.
 *
 * The pass-through comes first, so a framework whose CreateBlendState1 works
 * (3.0 and earlier) is untouched and keeps full DESC1 fidelity. Once a call has
 * shown the entry point broken the original is skipped: the missing AsInterface
 * case does not depend on the desc, and calling it anyway would strand another
 * reference on the cached state. */

/* Narrow a DESC1 to the DESC that CreateBlendState takes. D3DMetal widens it
 * straight back, so the only fields that cannot survive the round trip are the
 * two this drops. */
D3D11_BLEND_DESC narrow_blend_desc(const D3D11_BLEND_DESC1& d1, bool* lost_logic_op) {
    D3D11_BLEND_DESC d{};
    d.AlphaToCoverageEnable = d1.AlphaToCoverageEnable;
    d.IndependentBlendEnable = d1.IndependentBlendEnable;
    for (int i = 0; i < 8; i++) {
        const D3D11_RENDER_TARGET_BLEND_DESC1& s = d1.RenderTarget[i];
        d.RenderTarget[i].BlendEnable = s.BlendEnable;
        d.RenderTarget[i].SrcBlend = s.SrcBlend;
        d.RenderTarget[i].DestBlend = s.DestBlend;
        d.RenderTarget[i].BlendOp = s.BlendOp;
        d.RenderTarget[i].SrcBlendAlpha = s.SrcBlendAlpha;
        d.RenderTarget[i].DestBlendAlpha = s.DestBlendAlpha;
        d.RenderTarget[i].BlendOpAlpha = s.BlendOpAlpha;
        d.RenderTarget[i].RenderTargetWriteMask = s.RenderTargetWriteMask;
        if (s.LogicOpEnable)
            *lost_logic_op = true;
        /* Only RenderTarget[0] is read unless IndependentBlendEnable. */
        if (!d1.IndependentBlendEnable)
            break;
    }
    return d;
}

std::atomic<bool> g_blend_state1_broken{false};

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateBlendState1(
        ID3D11Device1* This, const D3D11_BLEND_DESC1* desc,
        ID3D11BlendState1** out) {
    auto orig = DMN_ORIG(d3d11_CreateBlendState1, This);
    if (!orig)
        return E_FAIL;
    if (!g_blend_state1_broken.load(std::memory_order_acquire)) {
        HRESULT hr = orig(This, desc, out);
        /* Working framework, a real failure, or a parameter-validation call
         * with no out-parameter to fill: nothing to repair. */
        if (FAILED(hr) || !out || *out || !desc)
            return hr;
        g_blend_state1_broken.store(true, std::memory_order_release);
    } else if (!out || !desc) {
        /* Preserve the original's own parameter validation. */
        return orig(This, desc, out);
    }

    ID3D11Device* d0 = nullptr;
    if (FAILED(This->QueryInterface(__uuidof(ID3D11Device),
                                    reinterpret_cast<void**>(&d0))) || !d0)
        return E_NOINTERFACE;

    bool lost_logic_op = false;
    const D3D11_BLEND_DESC d = narrow_blend_desc(*desc, &lost_logic_op);
    ID3D11BlendState* bs = nullptr;
    HRESULT hr = d0->CreateBlendState(&d, &bs);
    d0->Release();
    if (FAILED(hr) || !bs)
        return FAILED(hr) ? hr : E_NOINTERFACE;

    /* Same object, and its external vtable IS ID3D11BlendState1's. */
    *out = reinterpret_cast<ID3D11BlendState1*>(bs);

    if (lost_logic_op)
        DMN_WARN("hooks: CreateBlendState1 asked for logic-op blending, which "
                 "the CreateBlendState retry path cannot express; the state was "
                 "created without it");
    return S_OK;
}

/* == Create*View1 repair ===================================================
 * D3DMetal answers E_NOTIMPL for the ID3D11Device3 view creates
 * (CreateShaderResourceView1 / CreateRenderTargetView1 /
 * CreateUnorderedAccessView1). The D3D11.3 runtime hands a UMD the DESC1
 * forms, so a driver at that DDI tier reasonably calls the ...1 entry
 * points, and a caller that cannot see the failure goes on to bind a view
 * that was never created.
 *
 * Satisfy the call through the base create. Every DESC is a layout prefix
 * of its DESC1 (PlaneSlice is appended to the Texture2D/Texture2DArray
 * union members), so narrowing is a DESC-sized copy; only a nonzero
 * PlaneSlice cannot survive, and a request for one warns. The base create
 * vends the framework's per-object external interface, same as the
 * blend-state repair above. */

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateShaderResourceView1(
        ID3D11Device3* This, ID3D11Resource* res,
        const D3D11_SHADER_RESOURCE_VIEW_DESC1* desc1,
        ID3D11ShaderResourceView1** out) {
    auto orig = DMN_ORIG(d3d11_CreateShaderResourceView1, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, res, desc1, out);
    if (hr != E_NOTIMPL)
        return hr;

    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    D3D11_SHADER_RESOURCE_VIEW_DESC* pd = nullptr;
    if (desc1) {
        memcpy(&d, desc1, sizeof(d));
        pd = &d;
        if ((desc1->ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D &&
             desc1->Texture2D.PlaneSlice) ||
            (desc1->ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2DARRAY &&
             desc1->Texture2DArray.PlaneSlice))
            DMN_WARN("hooks: CreateShaderResourceView1 asked for PlaneSlice, "
                     "which the CreateShaderResourceView retry path cannot "
                     "express; the view was created for plane 0");
    }
    ID3D11Device* d0 = nullptr;
    if (FAILED(This->QueryInterface(__uuidof(ID3D11Device),
                                    reinterpret_cast<void**>(&d0))) || !d0)
        return E_NOTIMPL;
    ID3D11ShaderResourceView* v = nullptr;
    hr = d0->CreateShaderResourceView(res, pd, out ? &v : nullptr);
    d0->Release();
    if (SUCCEEDED(hr) && out)
        *out = reinterpret_cast<ID3D11ShaderResourceView1*>(v);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateRenderTargetView1(
        ID3D11Device3* This, ID3D11Resource* res,
        const D3D11_RENDER_TARGET_VIEW_DESC1* desc1,
        ID3D11RenderTargetView1** out) {
    auto orig = DMN_ORIG(d3d11_CreateRenderTargetView1, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, res, desc1, out);
    if (hr != E_NOTIMPL)
        return hr;

    D3D11_RENDER_TARGET_VIEW_DESC d{};
    D3D11_RENDER_TARGET_VIEW_DESC* pd = nullptr;
    if (desc1) {
        memcpy(&d, desc1, sizeof(d));
        pd = &d;
        if ((desc1->ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D &&
             desc1->Texture2D.PlaneSlice) ||
            (desc1->ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2DARRAY &&
             desc1->Texture2DArray.PlaneSlice))
            DMN_WARN("hooks: CreateRenderTargetView1 asked for PlaneSlice, "
                     "which the CreateRenderTargetView retry path cannot "
                     "express; the view was created for plane 0");
    }
    ID3D11Device* d0 = nullptr;
    if (FAILED(This->QueryInterface(__uuidof(ID3D11Device),
                                    reinterpret_cast<void**>(&d0))) || !d0)
        return E_NOTIMPL;
    ID3D11RenderTargetView* v = nullptr;
    hr = d0->CreateRenderTargetView(res, pd, out ? &v : nullptr);
    d0->Release();
    if (SUCCEEDED(hr) && out)
        *out = reinterpret_cast<ID3D11RenderTargetView1*>(v);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateUnorderedAccessView1(
        ID3D11Device3* This, ID3D11Resource* res,
        const D3D11_UNORDERED_ACCESS_VIEW_DESC1* desc1,
        ID3D11UnorderedAccessView1** out) {
    auto orig = DMN_ORIG(d3d11_CreateUnorderedAccessView1, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, res, desc1, out);
    if (hr != E_NOTIMPL)
        return hr;

    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
    D3D11_UNORDERED_ACCESS_VIEW_DESC* pd = nullptr;
    if (desc1) {
        memcpy(&d, desc1, sizeof(d));
        pd = &d;
        if ((desc1->ViewDimension == D3D11_UAV_DIMENSION_TEXTURE2D &&
             desc1->Texture2D.PlaneSlice) ||
            (desc1->ViewDimension == D3D11_UAV_DIMENSION_TEXTURE2DARRAY &&
             desc1->Texture2DArray.PlaneSlice))
            DMN_WARN("hooks: CreateUnorderedAccessView1 asked for PlaneSlice, "
                     "which the CreateUnorderedAccessView retry path cannot "
                     "express; the view was created for plane 0");
    }
    ID3D11Device* d0 = nullptr;
    if (FAILED(This->QueryInterface(__uuidof(ID3D11Device),
                                    reinterpret_cast<void**>(&d0))) || !d0)
        return E_NOTIMPL;
    ID3D11UnorderedAccessView* v = nullptr;
    hr = d0->CreateUnorderedAccessView(res, pd, out ? &v : nullptr);
    d0->Release();
    if (SUCCEEDED(hr) && out)
        *out = reinterpret_cast<ID3D11UnorderedAccessView1*>(v);
    return hr;
}

/* D3D11 shares only 2D textures (D3D11_RESOURCE_MISC_SHARED: "The only
 * resources that can be shared are 2D non-mipmapped textures"). */
HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture1D(
        ID3D11Device* This, const D3D11_TEXTURE1D_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture1D** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture1D, This);
    if (!orig)
        return E_FAIL;
    if (desc && !d3d11_format_ok(desc->Format))
        return E_INVALIDARG;
    if (desc && is_shared_misc(desc->MiscFlags)) {
        DMN_ERROR("hooks: CreateTexture1D with a SHARED misc flag: only 2D "
                  "textures can be shared");
        return E_INVALIDARG;
    }
    return orig(This, desc, init, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CreateTexture3D(
        ID3D11Device* This, const D3D11_TEXTURE3D_DESC* desc,
        const D3D11_SUBRESOURCE_DATA* init, ID3D11Texture3D** out) {
    auto orig = DMN_ORIG(d3d11_CreateTexture3D, This);
    if (!orig)
        return E_FAIL;
    if (desc && !d3d11_format_ok(desc->Format))
        return E_INVALIDARG;
    if (desc && is_shared_misc(desc->MiscFlags)) {
        DMN_ERROR("hooks: CreateTexture3D with a SHARED misc flag: only 2D "
                  "textures can be shared");
        return E_INVALIDARG;
    }
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
    if (FAILED(hr) || !out || !*out) {
        arm_fd_close(arm);
        return hr;
    }
    if (!captured)
        return fail_unshared("CreateBuffer MISC_SHARED",
                             reinterpret_cast<void**>(out));
    record_buffer(reinterpret_cast<IUnknown*>(*out), desc->ByteWidth,
                  desc->BindFlags, desc->MiscFlags, desc->CPUAccessFlags, arm);
    arm_fd_close(arm);
    return hr;
}

/* == D3D11 format-capability thunks ======================================== */
/* D3DMetal answers CheckFormatSupport with 0xffffffff for anything it knows
 * and 0 otherwise — those are the only two values it returns, always with
 * S_OK. We replace the answer outright rather than filtering one; there is
 * nothing in its reply worth preserving. dmn_formats.mm has the tables.
 * A format Metal cannot represent (NV12, P8, R1_UNORM, ...), or that D3DMetal
 * cannot make, is E_FAIL, MSDN's answer for a format the device does not
 * support. */
HRESULT STDMETHODCALLTYPE hook_d3d11_CheckFormatSupport(
        ID3D11Device* This, DXGI_FORMAT format, UINT* out) {
    if (!out)
        return E_INVALIDARG;
    uint32_t support = 0;
    if (!dmn_format_support((uint32_t)format, &support) ||
        (format != DXGI_FORMAT_UNKNOWN && !d3dmetal_has_format(This, format))) {
        *out = 0;
        return E_FAIL;
    }
    *out = support;
    return S_OK;
}

/* D3DMetal answers quality levels for formats it does not support and for
 * sample counts Metal cannot give a format (BC included), and a caller that
 * trusts it creates a texture Metal aborts on.  Its answer stands except
 * where it claims such a combination. */
static HRESULT filter_quality_levels(ID3D11Device* dev, HRESULT hr,
                                     DXGI_FORMAT format, UINT sample_count,
                                     UINT* out) {
    if (SUCCEEDED(hr) && *out &&
        (!d3dmetal_has_format(dev, format) ||
         !dmn_format_msaa_ok((uint32_t)format, sample_count)))
        *out = 0;
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CheckMultisampleQualityLevels(
        ID3D11Device* This, DXGI_FORMAT format, UINT sample_count,
        UINT* out) {
    if (!out)
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d11_CheckMultisampleQualityLevels, This);
    return filter_quality_levels(
        This, orig ? orig(This, format, sample_count, out) : E_FAIL,
        format, sample_count, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d11_CheckMultisampleQualityLevels1(
        ID3D11Device2* This, DXGI_FORMAT format, UINT sample_count,
        UINT flags, UINT* out) {
    if (!out)
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d11_CheckMultisampleQualityLevels1, This);
    return filter_quality_levels(
        This, orig ? orig(This, format, sample_count, flags, out) : E_FAIL,
        format, sample_count, out);
}

/* FORMAT_SUPPORT and FORMAT_SUPPORT2 must agree with CheckFormatSupport, so
 * they come from the same tables. Every other feature is D3DMetal's to answer,
 * less what the sharing layer cannot back. */
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
            return E_FAIL;
        }
        info->OutFormatSupport2 = support2;
        return S_OK;
    }
    auto orig = DMN_ORIG(d3d11_CheckFeatureSupport, This);
    HRESULT hr = orig ? orig(This, feature, data, size) : E_FAIL;
    /* Extended sharing (mips, arrays, more formats) is what share_shape_ok
     * refuses; only the baseline D3D11.1 shared shape can be honoured. */
    if (SUCCEEDED(hr) && feature == D3D11_FEATURE_D3D11_OPTIONS && data &&
        size == sizeof(D3D11_FEATURE_DATA_D3D11_OPTIONS))
        static_cast<D3D11_FEATURE_DATA_D3D11_OPTIONS*>(data)
            ->ExtendedResourceSharing = FALSE;
    return hr;
}

/* == D3D11 consumer thunks ================================================= */
HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResource(
        ID3D11Device* This, HANDLE handle, REFIID iid, void** out) {
    if (handle && out) {
        HRESULT hr = E_FAIL;
        if (open_shared_d3d11(This, handle, iid, out, &hr))
            return hr;
    }
    auto orig = DMN_ORIG(d3d11_OpenSharedResource, This);
    return orig ? orig(This, handle, iid, out) : E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE hook_d3d11_OpenSharedResource1(
        ID3D11Device1* This, HANDLE handle, REFIID iid, void** out) {
    if (handle && out) {
        HRESULT hr = E_FAIL;
        if (open_shared_d3d11(reinterpret_cast<ID3D11Device*>(This), handle, iid,
                              out, &hr))
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
    /* The framework never sees the SHARED flag: sharing is implemented
     * entirely here (companion slot), so D3DMetal's own D3D11 shared-fence
     * machinery would only add an unused second mechanism with its own
     * failure modes. The fence is created PLAIN and our sharing is layered on
     * top, exactly as imports already do (dmn_fd3d_import_d3d11 creates with
     * FLAG_NONE). */
    const D3D11_FENCE_FLAG fwd =
        (D3D11_FENCE_FLAG)(flags & ~D3D11_FENCE_FLAG_SHARED);
    HRESULT hr = orig(This, initial, fwd, iid, out);
    if (FAILED(hr) || !out || !*out)
        return hr;
    /* Only shared fences get a companion buffer; leave plain fences alone. */
    if (!(flags & D3D11_FENCE_FLAG_SHARED))
        return hr;
    ID3D11Fence* f = nullptr;
    if (SUCCEEDED(reinterpret_cast<IUnknown*>(*out)->QueryInterface(
            __uuidof(ID3D11Fence), reinterpret_cast<void**>(&f))) && f) {
        if (dmn_fd3d_producer_create_d3d11(f, initial, (UINT)flags)) {
            /* Producer signal interception + export + slot-merged readback. */
            ID3D11DeviceContext* ctx = nullptr;
            This->GetImmediateContext(&ctx);
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
        *reinterpret_cast<uint32_t*>(handle) == DXN_SHARED_FENCE_MAGIC) {
        ID3D11Fence* f = nullptr;
        HRESULT hr = dmn_fd3d_import_d3d11(
            This, reinterpret_cast<dxn_shared_fence*>(handle), &f);
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
    HRESULT hr = orig ? orig(This, fence, value) : E_FAIL;
    if (SUCCEEDED(hr)) {
        /* The signalled fence must outlive its own signal (see
         * dmn_fd3d_keepalive_d3d11); the app may drop it right here. */
        dmn_fd3d_keepalive_d3d11(This, fence, value);
        /* Producer or import: publish the value into the shared slot once
         * the fence itself completes it. */
        dmn_fd3d_after_ctx_signal(fence, value);
    }
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
    /* Pin ev BEFORE the registration: the moment orig registers it,
     * D3DMetal's completion (fired on its own safe duplicate) can land and
     * the caller may close ev in response -- while the slot watcher below
     * still has to signal it.  The watcher owns and closes the pin. */
    void* pin = ev ? dmn_event_duplicate(ev) : nullptr;
    HRESULT hr = orig ? orig(This, value, ev) : E_FAIL;
    /* Also watch the shared slot so consumer signal-back releases waiters. */
    if (SUCCEEDED(hr) && pin)
        dmn_fd3d_watch_slot(This, value, pin);
    else if (pin)
        dmn_event_close(pin);
    return hr;
}

/* == D3D12 committed/placed creates ======================================== */

/* Flatten a D3D12 texture desc into the POD's D3D11 shape. `DescT` is
 * D3D12_RESOURCE_DESC or D3D12_RESOURCE_DESC1 (layout-compatible here). */
template <class DescT>
D3D11_TEXTURE2D_DESC d12_desc_to_11(const DescT& desc) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = (UINT)desc.Width;
    d.Height = desc.Height;
    d.MipLevels = desc.MipLevels ? desc.MipLevels : 1;
    d.ArraySize = desc.DepthOrArraySize ? desc.DepthOrArraySize : 1;
    d.Format = desc.Format;
    d.SampleDesc.Count = desc.SampleDesc.Count ? desc.SampleDesc.Count : 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = d12_flags_to_bind(desc.Flags);
    d.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    return d;
}

/* Common shared-create wrapper: arm by dimension, run the actual create, and
 * record what the swizzle captured. */
template <class DescT, class OrigCall>
HRESULT d12_create_shared(const DescT* desc, void** out, OrigCall&& call) {
    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        DMN_INFO("hooks: D3D12 SHARED buffer create size=%llu",
                 (unsigned long long)desc->Width);
        dmn_share_arm_producer_buffer(desc->Width);
        HRESULT hr = call();
        DmnShareArm arm{};
        bool captured = dmn_share_disarm(&arm);
        if (FAILED(hr) || !out || !*out) {
            arm_fd_close(arm);
            return hr;
        }
        if (!captured)
            return fail_unshared("D3D12 SHARED buffer create", out);
        record_buffer(reinterpret_cast<IUnknown*>(*out), desc->Width,
                      d12_flags_to_bind(desc->Flags), D3D11_RESOURCE_MISC_SHARED, 0,
                      arm);
        arm_fd_close(arm);
        return hr;
    }
    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        if (!share_shape_ok(desc->MipLevels, desc->DepthOrArraySize,
                            desc->SampleDesc.Count, "D3D12 SHARED texture create"))
            return E_INVALIDARG;
        DMN_INFO("hooks: D3D12 SHARED texture create %llux%u fmt=%u",
                 (unsigned long long)desc->Width, desc->Height, desc->Format);
        dmn_share_arm_producer(0);
        HRESULT hr = call();
        DmnShareArm arm{};
        bool captured = dmn_share_disarm(&arm);
        if (FAILED(hr) || !out || !*out) {
            arm_fd_close(arm);
            return hr;
        }
        if (!captured)
            return fail_unshared("D3D12 SHARED texture create", out);
        record_texture(reinterpret_cast<IUnknown*>(*out), d12_desc_to_11(*desc),
                       arm);
        arm_fd_close(arm);
        return hr;
    }
    DMN_WARN("hooks: D3D12 SHARED create with dimension %d has no shared "
             "backing; it is created UNSHARED and cannot be exported",
             (int)desc->Dimension);
    HRESULT hr = call();
    if (SUCCEEDED(hr))
        register_unshared(out);
    return hr;
}

/* Heap properties for a placed->committed translation.
 *
 * An imported heap is typically UPLOAD or READBACK (the import keeps the
 * caller's type, defaulting to UPLOAD), but D3D12 forbids
 * ALLOW_UNORDERED_ACCESS and the render-target/depth-stencil flags on UPLOAD
 * and READBACK heaps, so a placement carrying one of them
 * cannot be committed against those properties. Promote to DEFAULT: the Metal
 * allocation is substituted with a window into the heap's shm object either
 * way, so these properties only have to satisfy the framework's validation.
 * GetHeapProperties fidelity is already approximate here by design. */
D3D12_HEAP_PROPERTIES commit_props_for(const D3D12_HEAP_PROPERTIES& hp,
                                       D3D12_RESOURCE_FLAGS flags) {
    const bool cpu_heap = hp.Type == D3D12_HEAP_TYPE_UPLOAD ||
                          hp.Type == D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_FLAGS illegal_on_cpu_heap =
        (D3D12_RESOURCE_FLAGS)(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS |
                               D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                               D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    if (!cpu_heap || !(flags & illegal_on_cpu_heap))
        return hp;
    D3D12_HEAP_PROPERTIES out = hp;
    out.Type = D3D12_HEAP_TYPE_DEFAULT;
    out.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    out.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    /* Per-create path with a uniform outcome: one line, not one per resource. */
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed))
        DMN_INFO("hooks: placed create needs flags 0x%x illegal on heap type "
                 "%u; committing as DEFAULT. Logged once.",
                 (unsigned)flags, (unsigned)hp.Type);
    return out;
}

/* A placed create on a synthetic heap. The heap is ours, so the framework can
 * neither validate nor allocate for it — the create is translated into an
 * armed CreateCommittedResource (`commit`, built by the hook for its device
 * tier with the heap's Properties and kHeapCreateNotZeroed, since the
 * window bytes are the caller's and placed contents are undefined) whose Metal
 * allocation is substituted with a window into the heap's shm object:
 *
 *   BUFFER    -> buffer window at the placement offset (page-aligned by
 *                D3D12's own 64 KiB placement rule);
 *   TEXTURE2D -> linear texture window at the placement offset, layout derived
 *                from the descriptor;
 *   1D/3D     -> no window machinery exists: the resource is created UNSHARED
 *                (correct memory, correct lifetime, loud warning) and
 *                registered so exporting it FAILS instead of vending a handle
 *                to memory a peer cannot see. Imported heaps refuse instead —
 *                their pages are the caller's, buffers only.
 *
 * Overlapping placements alias because every window maps the same object. */
template <class DescT, class CommitFn>
HRESULT d12_place_on_synth(DmnSynthHeap* sh, UINT64 offset, const DescT* desc,
                           void** out, CommitFn&& commit) {
    if (!desc || offset >= sh->size)
        return E_INVALIDARG;

    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        UINT64 end = 0;
        if (__builtin_add_overflow(offset, desc->Width, &end) ||
            end > sh->size) {
            DMN_ERROR("hooks: placed buffer [%llu, +%llu) exceeds heap size "
                      "%llu", (unsigned long long)offset,
                      (unsigned long long)desc->Width,
                      (unsigned long long)sh->size);
            return E_INVALIDARG;
        }
        DMN_INFO("hooks: placed buffer on %s heap offset=%llu size=%llu",
                 sh->imported ? "imported" : "shared",
                 (unsigned long long)offset, (unsigned long long)desc->Width);
        dmn_share_arm_import_window(sh->fd, sh->fd_off + offset, desc->Width,
                                    sh->size - offset,
                                    /*create_not_zeroed=*/true);
        HRESULT hr = commit();
        DmnShareArm arm{};
        bool captured = dmn_share_disarm(&arm);
        if (FAILED(hr) || !out || !*out)
            return hr; /* validation call or failed create: nothing was made */
        if (!captured)
            return fail_unshared("placed buffer on synthetic heap", out);
        /* A resource placed in a SHARED heap must still be exportable; the POD
         * carries the window so a consumer maps the same bytes. Imported
         * heaps' placements are not registered: their pages belong to the
         * caller, who already has its own handle to them. */
        if (!sh->imported)
            record_buffer(reinterpret_cast<IUnknown*>(*out), desc->Width,
                          d12_flags_to_bind(desc->Flags),
                          D3D11_RESOURCE_MISC_SHARED, 0, arm,
                          sh->fd_off + offset);
        return hr;
    }

    if (sh->imported) {
        DMN_ERROR("hooks: non-buffer placed on an imported heap (dim=%d) — "
                  "imported heaps alias guest CPU pages and back buffers only",
                  (int)desc->Dimension);
        return E_INVALIDARG;
    }

    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        if (!share_shape_ok(desc->MipLevels, desc->DepthOrArraySize,
                            desc->SampleDesc.Count, "texture placed in a SHARED heap"))
            return E_INVALIDARG;
        DMN_INFO("hooks: placed texture on shared heap offset=%llu %llux%u "
                 "fmt=%u", (unsigned long long)offset,
                 (unsigned long long)desc->Width, desc->Height, desc->Format);
        dmn_share_arm_texture_window(sh->fd, sh->fd_off + offset,
                                     sh->size - offset);
        HRESULT hr = commit();
        DmnShareArm arm{};
        bool captured = dmn_share_disarm(&arm);
        if (FAILED(hr) || !out || !*out)
            return hr;
        if (!captured)
            return fail_unshared("placed texture on synthetic heap", out);
        record_texture(reinterpret_cast<IUnknown*>(*out), d12_desc_to_11(*desc),
                       arm, sh->fd_off + offset);
        return hr;
    }

    DMN_WARN("hooks: dimension-%d resource placed in a SHARED heap has no "
             "window backing; it is created UNSHARED (does not alias the heap, "
             "cannot be exported)", (int)desc->Dimension);
    HRESULT hr = commit(); /* unarmed: a plain private committed resource */
    if (SUCCEEDED(hr))
        register_unshared(out);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommittedResource(
        ID3D12Device* This, const D3D12_HEAP_PROPERTIES* hp,
        D3D12_HEAP_FLAGS flags, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear,
        REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, desc))
        return E_INVALIDARG;
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
    if (!dmn_desc_format_ok(This, desc))
        return E_INVALIDARG;
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
    if (!dmn_desc_format_ok(This, (const D3D12_RESOURCE_DESC*)desc))
        return E_INVALIDARG;
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
    if (!dmn_desc_format_ok(This, (const D3D12_RESOURCE_DESC*)desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreateCommittedResource3, This);
    if (!orig)
        return E_FAIL;
    auto call = [&] { return orig(This, hp, flags, desc, layout, clear, session,
                                  num_castable, castable, riid, out); };
    if (dmn_share_is_armed() || !desc || !(flags & D3D12_HEAP_FLAG_SHARED))
        return call();
    return d12_create_shared(desc, out, call);
}

/* CreateHeap(SHARED): mint a synthetic heap; the framework never sees it, so
 * its own heap allocation (whose memory a substituted placement would turn
 * into a live second copy) never exists. */
HRESULT d12_create_heap_shared(IUnknown* device, const D3D12_HEAP_DESC* desc,
                               REFIID riid, void** out) {
    if (!desc->SizeInBytes)
        return E_INVALIDARG;
    const int fd = dmn_share_anon_file((size_t)desc->SizeInBytes);
    if (fd < 0) {
        DMN_ERROR("hooks: SHARED heap backing (%llu bytes) could not be "
                  "allocated", (unsigned long long)desc->SizeInBytes);
        return E_OUTOFMEMORY;
    }
    HRESULT hr = synth_heap_new(device, *desc, fd, /*fd_off=*/0,
                                desc->SizeInBytes, /*imported=*/false, riid,
                                out);
    if (SUCCEEDED(hr))
        DMN_INFO("hooks: SHARED heap %p is synthetic, size=%llu fd=%d",
                 out ? *out : nullptr, (unsigned long long)desc->SizeInBytes,
                 fd);
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateHeap(
        ID3D12Device* This, const D3D12_HEAP_DESC* desc, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateHeap, This);
    if (!orig)
        return E_FAIL;
    /* A null `out` is a parameter-validation call; let the framework judge. */
    if (!desc || !out || !(desc->Flags & D3D12_HEAP_FLAG_SHARED))
        return orig(This, desc, riid, out);
    return d12_create_heap_shared(reinterpret_cast<IUnknown*>(This), desc, riid,
                                  out);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateHeap1(
        ID3D12Device4* This, const D3D12_HEAP_DESC* desc,
        DMN_ID3D12ProtectedResourceSession* session, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateHeap1, This);
    if (!orig)
        return E_FAIL;
    if (!desc || !out || !(desc->Flags & D3D12_HEAP_FLAG_SHARED))
        return orig(This, desc, session, riid, out);
    return d12_create_heap_shared(reinterpret_cast<IUnknown*>(This), desc, riid,
                                  out);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource(
        ID3D12Device* This, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES state,
        const D3D12_CLEAR_VALUE* clear, REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreatePlacedResource, This);
    if (!orig)
        return E_FAIL;
    /* A synthetic heap must never reach the framework — it is our COM object,
     * not one of D3DMetal's. Translate the placed create into a committed one
     * whose Metal backing becomes a window into the heap's shm object. The
     * committed call goes through the public (hooked) vtable slot: the arm set
     * by the translation makes that hook pass straight through. */
    if (DmnSynthHeap* sh = as_synth_heap(heap))
        return d12_place_on_synth(sh, offset, desc, out, [&] {
            D3D12_HEAP_PROPERTIES hp =
                commit_props_for(sh->desc.Properties, desc->Flags);
            return This->CreateCommittedResource(&hp, kHeapCreateNotZeroed,
                                                 desc, state, clear, riid, out);
        });
    return orig(This, heap, offset, desc, state, clear, riid, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource1(
        ID3D12Device8* This, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC1* desc, D3D12_RESOURCE_STATES state,
        const D3D12_CLEAR_VALUE* clear, REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, (const D3D12_RESOURCE_DESC*)desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreatePlacedResource1, This);
    if (!orig)
        return E_FAIL;
    /* Same synthetic-heap translation as hook_d3d12_CreatePlacedResource,
     * through this device tier's committed create. */
    if (DmnSynthHeap* sh = as_synth_heap(heap))
        return d12_place_on_synth(sh, offset, desc, out, [&] {
            D3D12_HEAP_PROPERTIES hp =
                commit_props_for(sh->desc.Properties, desc->Flags);
            return This->CreateCommittedResource2(&hp, kHeapCreateNotZeroed,
                                                  desc, state, clear, nullptr,
                                                  riid, out);
        });
    return orig(This, heap, offset, desc, state, clear, riid, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreatePlacedResource2(
        ID3D12Device10* This, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC1* desc, DMN_D3D12_BARRIER_LAYOUT layout,
        const D3D12_CLEAR_VALUE* clear, UINT32 num_castable,
        const DXGI_FORMAT* castable, REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, (const D3D12_RESOURCE_DESC*)desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreatePlacedResource2, This);
    if (!orig)
        return E_FAIL;
    /* Same synthetic-heap translation as hook_d3d12_CreatePlacedResource,
     * through this device tier's committed create. One repair on the way: a
     * BUFFER must carry D3D12_BARRIER_LAYOUT_UNDEFINED (~0u) into
     * CreateCommittedResource3 — the layout is meaningless for buffers and
     * UNDEFINED is the only value its validation accepts, while
     * CreatePlacedResource2 tolerated whatever the app passed. */
    if (DmnSynthHeap* sh = as_synth_heap(heap))
        return d12_place_on_synth(sh, offset, desc, out, [&] {
            D3D12_HEAP_PROPERTIES hp =
                commit_props_for(sh->desc.Properties, desc->Flags);
            const DMN_D3D12_BARRIER_LAYOUT lay =
                (desc && desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                    ? (DMN_D3D12_BARRIER_LAYOUT)~0u
                    : layout;
            return This->CreateCommittedResource3(&hp, kHeapCreateNotZeroed,
                                                  desc, lay, clear, nullptr,
                                                  num_castable, castable, riid,
                                                  out);
        });
    return orig(This, heap, offset, desc, layout, clear, num_castable,
                castable, riid, out);
}

/* == Residency calls ====================================================== */
/* A synthetic heap handed to the framework's MakeResident/Evict family would
 * be dereferenced as a D3DMetal object. Filter it out: its backing is plain
 * shared memory, always "resident" as far as D3D12 semantics can observe, so
 * dropping it from the array is the correct answer, not a shortcut. */

bool pageables_contain_synth(UINT count, ID3D12Pageable* const* objects) {
    for (UINT i = 0; i < count; i++)
        if (as_synth_heap(reinterpret_cast<ID3D12Heap*>(objects[i])))
            return true;
    return false;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_MakeResident(
        ID3D12Device* This, UINT count, ID3D12Pageable* const* objects) {
    auto orig = DMN_ORIG(d3d12_MakeResident, This);
    if (!orig)
        return E_FAIL;
    if (!objects || !pageables_contain_synth(count, objects))
        return orig(This, count, objects);
    std::vector<ID3D12Pageable*> kept;
    kept.reserve(count);
    for (UINT i = 0; i < count; i++)
        if (!as_synth_heap(reinterpret_cast<ID3D12Heap*>(objects[i])))
            kept.push_back(objects[i]);
    if (kept.empty())
        return S_OK;
    return orig(This, (UINT)kept.size(), kept.data());
}

HRESULT STDMETHODCALLTYPE hook_d3d12_Evict(
        ID3D12Device* This, UINT count, ID3D12Pageable* const* objects) {
    auto orig = DMN_ORIG(d3d12_Evict, This);
    if (!orig)
        return E_FAIL;
    if (!objects || !pageables_contain_synth(count, objects))
        return orig(This, count, objects);
    std::vector<ID3D12Pageable*> kept;
    kept.reserve(count);
    for (UINT i = 0; i < count; i++)
        if (!as_synth_heap(reinterpret_cast<ID3D12Heap*>(objects[i])))
            kept.push_back(objects[i]);
    if (kept.empty())
        return S_OK;
    return orig(This, (UINT)kept.size(), kept.data());
}

HRESULT STDMETHODCALLTYPE hook_d3d12_SetResidencyPriority(
        ID3D12Device1* This, UINT count, ID3D12Pageable* const* objects,
        const D3D12_RESIDENCY_PRIORITY* priorities) {
    auto orig = DMN_ORIG(d3d12_SetResidencyPriority, This);
    if (!orig)
        return E_FAIL;
    if (!objects || !priorities || !pageables_contain_synth(count, objects))
        return orig(This, count, objects, priorities);
    std::vector<ID3D12Pageable*> kept;
    std::vector<D3D12_RESIDENCY_PRIORITY> kept_prio;
    kept.reserve(count);
    kept_prio.reserve(count);
    for (UINT i = 0; i < count; i++) {
        if (as_synth_heap(reinterpret_cast<ID3D12Heap*>(objects[i])))
            continue;
        kept.push_back(objects[i]);
        kept_prio.push_back(priorities[i]);
    }
    if (kept.empty())
        return S_OK;
    return orig(This, (UINT)kept.size(), kept.data(), kept_prio.data());
}

HRESULT STDMETHODCALLTYPE hook_d3d12_EnqueueMakeResident(
        ID3D12Device3* This, UINT flags, UINT count,
        ID3D12Pageable* const* objects, ID3D12Fence* fence, UINT64 value) {
    auto orig = DMN_ORIG(d3d12_EnqueueMakeResident, This);
    if (!orig)
        return E_FAIL;
    if (!objects || !pageables_contain_synth(count, objects))
        return orig(This, flags, count, objects, fence, value);
    std::vector<ID3D12Pageable*> kept;
    kept.reserve(count);
    for (UINT i = 0; i < count; i++)
        if (!as_synth_heap(reinterpret_cast<ID3D12Heap*>(objects[i])))
            kept.push_back(objects[i]);
    if (kept.empty()) {
        /* Everything is already resident, but the API's contract — the fence
         * signals when the work completes — must still be honoured. */
        if (fence)
            fence->Signal(value);
        return S_OK;
    }
    return orig(This, flags, (UINT)kept.size(), kept.data(), fence, value);
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
    /* Same pin-before-registration as hook_f11_SetEventOnCompletion. */
    void* pin = ev ? dmn_event_duplicate(ev) : nullptr;
    HRESULT hr = orig ? orig(This, value, ev) : E_FAIL;
    if (SUCCEEDED(hr) && pin)
        dmn_fd3d_watch_slot(This, value, pin);
    else if (pin)
        dmn_event_close(pin);
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
    if (SUCCEEDED(hr)) {
        dmn_fd3d_keepalive_d3d12(fence, value);
        /* Producer: GPU-write the slot via the helper queue. Import signal-back:
         * store into the slot once the local fence completes the value. */
        dmn_fd3d_on_queue_signal(fence, value);
    }
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

/* D3D12 tile region -> the sparse module's POD; NULL parts take the D3D12
 * defaults (start at 0 / one tile). */
DmnSparseRegion sparse_region(const D3D12_TILED_RESOURCE_COORDINATE* c,
                              const D3D12_TILE_REGION_SIZE* sz) {
    DmnSparseRegion g{};
    if (c) { g.x = c->X; g.y = c->Y; g.z = c->Z; g.subresource = c->Subresource; }
    if (sz) {
        g.use_box = sz->UseBox; g.num_tiles = sz->NumTiles;
        g.width = sz->Width; g.height = sz->Height; g.depth = sz->Depth;
    } else {
        g.num_tiles = 1; g.width = g.height = g.depth = 1;
    }
    return g;
}

/* Tile mappings draw pages from a heap, so this is one of the entry points an
 * ID3D12Heap can arrive through (MSDN: pHeap, 5th parameter). A synthetic
 * heap must not reach the framework; there is also nothing it could do with
 * one — its tile pool is its own placement heap, not our shm object. Tiled
 * resources are exotic enough here that a loud drop is the whole story. */
void STDMETHODCALLTYPE hook_queue_UpdateTileMappings(
        ID3D12CommandQueue* This, ID3D12Resource* resource, UINT num_regions,
        const D3D12_TILED_RESOURCE_COORDINATE* coords,
        const D3D12_TILE_REGION_SIZE* sizes, ID3D12Heap* heap, UINT num_ranges,
        const D3D12_TILE_RANGE_FLAGS* range_flags, const UINT* range_offsets,
        const UINT* range_counts, D3D12_TILE_MAPPING_FLAGS flags) {
    if (resource && dmn_sparse_is_registered(dmn_com_identity(resource))) {
        /* The heap named here IS a tile pool, by definition of the call, and
         * the app can never map more tiles than the pools it owns -- so its
         * size is an authoritative ceiling on tile demand, which is what
         * sizes our sparse chunks.  The pages themselves still come from the
         * sparse heap, not from this one.  Translate and apply.  NULL
         * coordinates start at 0; NULL sizes are one tile per region, or --
         * with NULL coordinates too -- the whole resource, which the sparse
         * side sizes (regions NULL). */
        /* A synthetic heap is ours, not a framework object: it has no real
         * GetDesc to call, and it is not a tile pool. */
        if (heap && !as_synth_heap(heap)) {
            D3D12_HEAP_DESC hd = {};
            heap->GetDesc(&hd);
            dmn_sparse_note_tile_pool(heap, hd.SizeInBytes);
        }
        std::vector<DmnSparseRegion> regs;
        if (coords || sizes) {
            regs.resize(num_regions);
            for (UINT i = 0; i < num_regions; i++)
                regs[i] = sparse_region(coords ? &coords[i] : nullptr, sizes ? &sizes[i] : nullptr);
        }
        static_assert(sizeof(D3D12_TILE_RANGE_FLAGS) == sizeof(uint32_t), "flags size");
        dmn_sparse_update_mappings(dmn_com_identity(resource), num_regions,
                                   regs.empty() ? nullptr : regs.data(), num_ranges,
                                   reinterpret_cast<const uint32_t*>(range_flags),
                                   range_counts);
        return;
    }
    if (as_synth_heap(heap)) {
        DMN_ERROR("hooks: UpdateTileMappings with a SHARED/imported heap as "
                  "the tile pool is not supported — the mappings are DROPPED "
                  "(the heap is synthetic and has no framework tile pool)");
        return;
    }
    auto orig = DMN_ORIG(queue_UpdateTileMappings, This);
    if (orig)
        orig(This, resource, num_regions, coords, sizes, heap, num_ranges,
             range_flags, range_offsets, range_counts, flags);
}

/* Copying mappings makes the destination tiles share the source tiles'
 * pages.  Metal cannot alias pages between (or within) sparse textures, so
 * for a sparse-backed resource there is nothing faithful to do; say so
 * rather than let the framework's tier-0 no-op pass silently. */
void STDMETHODCALLTYPE hook_queue_CopyTileMappings(
        ID3D12CommandQueue* This, ID3D12Resource* dst,
        const D3D12_TILED_RESOURCE_COORDINATE* dst_start, ID3D12Resource* src,
        const D3D12_TILED_RESOURCE_COORDINATE* src_start,
        const D3D12_TILE_REGION_SIZE* size, D3D12_TILE_MAPPING_FLAGS flags) {
    if ((dst && dmn_sparse_is_registered(dmn_com_identity(dst))) ||
        (src && dmn_sparse_is_registered(dmn_com_identity(src)))) {
        static std::atomic<uint32_t> n{0};
        if (n++ < 4)
            DMN_ERROR("hooks: CopyTileMappings on a sparse-backed reserved resource is "
                      "not supported (Metal cannot alias sparse tiles) — the "
                      "destination's mappings are UNCHANGED");
        return;
    }
    auto orig = DMN_ORIG(queue_CopyTileMappings, This);
    if (orig)
        orig(This, dst, dst_start, src, src_start, size, flags);
}

/* CopyTiles on a sparse-backed resource: the framework has no tiled path, so
 * each standard tile becomes a CopyTextureRegion between the buffer and the
 * tile's texels -- CopyTiles' buffer layout is exactly a placed footprint per
 * tile (linear rows at the full tile pitch, 64 KiB apart).  Packed tiles are
 * skipped: the spec leaves CopyTiles on them undefined and sends apps to
 * CopyTextureRegion, which works on these textures as-is. */
void STDMETHODCALLTYPE hook_cmdlist_CopyTiles(
        ID3D12GraphicsCommandList* This, ID3D12Resource* tiled,
        const D3D12_TILED_RESOURCE_COORDINATE* start,
        const D3D12_TILE_REGION_SIZE* size, ID3D12Resource* buffer,
        UINT64 buffer_offset, D3D12_TILE_COPY_FLAGS flags) {
    if (!tiled || !dmn_sparse_is_registered(dmn_com_identity(tiled))) {
        auto orig = DMN_ORIG(cmdlist_CopyTiles, This);
        if (orig)
            orig(This, tiled, start, size, buffer, buffer_offset, flags);
        return;
    }
    static std::atomic<uint32_t> nerr{0};
    if (!buffer || !size || (buffer_offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) != 0) {
        if (nerr++ < 4)
            DMN_ERROR("hooks: CopyTiles on a sparse-backed reserved resource: buffer %p "
                      "offset %llu is not %u-aligned (or no size) — copy DROPPED",
                      (void*)buffer, (unsigned long long)buffer_offset,
                      (unsigned)D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        return;
    }
    const D3D12_RESOURCE_DESC rd = tiled->GetDesc();
    const bool to_buffer = flags & D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER;
    struct Ctx {
        ID3D12GraphicsCommandList* cl; ID3D12Resource* tex; ID3D12Resource* buf;
        UINT64 base; DXGI_FORMAT fmt; UINT16 mips; bool to_buffer;
    } ctx{This, tiled, buffer, buffer_offset, rd.Format, rd.MipLevels, to_buffer};
    auto emit = [](void* user, const DmnSparseTileCopy* c) {
        Ctx& x = *static_cast<Ctx*>(user);
        D3D12_TEXTURE_COPY_LOCATION tl{};
        tl.pResource = x.tex;
        tl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        tl.SubresourceIndex = c->slice * x.mips + c->mip;
        D3D12_TEXTURE_COPY_LOCATION bl{};
        bl.pResource = x.buf;
        bl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        bl.PlacedFootprint.Offset = x.base + c->buffer_offset;
        bl.PlacedFootprint.Footprint.Format = x.fmt;
        bl.PlacedFootprint.Footprint.Width = c->w;
        bl.PlacedFootprint.Footprint.Height = c->h;
        bl.PlacedFootprint.Footprint.Depth = 1;
        bl.PlacedFootprint.Footprint.RowPitch = c->row_pitch;
        if (x.to_buffer) {
            D3D12_BOX box{c->x, c->y, 0, c->x + c->w, c->y + c->h, 1};
            x.cl->CopyTextureRegion(&bl, 0, 0, 0, &tl, &box);
        } else {
            x.cl->CopyTextureRegion(&tl, c->x, c->y, 0, &bl, nullptr);
        }
    };
    DmnSparseRegion g = sparse_region(start, size);
    uint32_t skipped = 0;
    dmn_sparse_plan_tile_copy(dmn_com_identity(tiled), &g, emit, &ctx, &skipped);
    if (skipped && nerr++ < 4)
        DMN_WARN("hooks: CopyTiles touched %u packed-mip tile(s) of a sparse-backed "
                 "reserved resource — undefined per D3D12, skipped (use CopyTextureRegion "
                 "for packed mips)", skipped);
}

/* Whether D3DMetal itself supports textures of `format`.  For one it cannot
 * make, its FORMAT_SUPPORT answer fails or is empty, depending on version. */
static bool d3dmetal_has_format(ID3D12Device* dev, DXGI_FORMAT format) {
    auto orig = DMN_ORIG(d3d12_CheckFeatureSupport, dev);
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs = {format, D3D12_FORMAT_SUPPORT1_NONE,
                                            D3D12_FORMAT_SUPPORT2_NONE};
    return orig &&
           SUCCEEDED(orig(dev, D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))) &&
           fs.Support1 != D3D12_FORMAT_SUPPORT1_NONE;
}

/* For a texture in a format D3DMetal reports unsupported (the planar and
 * packed video formats: YUY2, NV12, AYUV, ...) it still builds an
 * MTLTextureDescriptor, which fails Metal's validation assertion and aborts
 * the process -- inside GetResourceAllocationInfo as much as inside a create.
 * Refuse such a description the way the runtime refuses a format the driver
 * does not support: an invalid-description answer from the size query and
 * E_INVALIDARG from the creates.  Buffers carry no format and pass. */
static bool dmn_desc_format_ok(ID3D12Device* dev, const D3D12_RESOURCE_DESC* d) {
    if (!d || d->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ||
        d->Format == DXGI_FORMAT_UNKNOWN || d3dmetal_has_format(dev, d->Format))
        return true;
    DMN_WARN("refusing a %llux%u texture in DXGI format %d: D3DMetal does not "
             "support it and aborts on it",
             (unsigned long long)d->Width, d->Height, (int)d->Format);
    return false;
}

/* The runtime's answer for a description it rejects (MSDN: SizeInBytes ==
 * UINT64_MAX).  The DESC1 variants start with a D3D12_RESOURCE_DESC. */
template <typename Desc>
static bool refuse_allocation_info(ID3D12Device* dev,
                                   D3D12_RESOURCE_ALLOCATION_INFO* ret, UINT n,
                                   const Desc* descs) {
    for (UINT i = 0; descs && i < n; i++) {
        if (!dmn_desc_format_ok(dev, (const D3D12_RESOURCE_DESC*)&descs[i])) {
            if (ret) {
                ret->SizeInBytes = UINT64_MAX;
                ret->Alignment = 0;
            }
            return true;
        }
    }
    return false;
}

D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo(
        ID3D12Device* This, D3D12_RESOURCE_ALLOCATION_INFO* ret, UINT visible_mask,
        UINT n, const D3D12_RESOURCE_DESC* descs) {
    if (refuse_allocation_info(This, ret, n, descs))
        return ret;
    auto orig = DMN_ORIG(d3d12_GetResourceAllocationInfo, This);
    return orig(This, ret, visible_mask, n, descs);
}

D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo1(
        ID3D12Device4* This, D3D12_RESOURCE_ALLOCATION_INFO* ret, UINT visible_mask,
        UINT n, const D3D12_RESOURCE_DESC* descs,
        D3D12_RESOURCE_ALLOCATION_INFO1* info1) {
    if (refuse_allocation_info(This, ret, n, descs))
        return ret;
    auto orig = DMN_ORIG(d3d12_GetResourceAllocationInfo1, This);
    return orig(This, ret, visible_mask, n, descs, info1);
}

D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo2(
        ID3D12Device8* This, D3D12_RESOURCE_ALLOCATION_INFO* ret, UINT visible_mask,
        UINT n, const D3D12_RESOURCE_DESC1* descs,
        D3D12_RESOURCE_ALLOCATION_INFO1* info1) {
    if (refuse_allocation_info(This, ret, n, descs))
        return ret;
    auto orig = DMN_ORIG(d3d12_GetResourceAllocationInfo2, This);
    return orig(This, ret, visible_mask, n, descs, info1);
}

D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE hook_d3d12_GetResourceAllocationInfo3(
        ID3D12Device12* This, D3D12_RESOURCE_ALLOCATION_INFO* ret, UINT visible_mask,
        UINT n, const D3D12_RESOURCE_DESC1* descs, const UINT32* num_castable,
        const DXGI_FORMAT* const* castable, D3D12_RESOURCE_ALLOCATION_INFO1* info1) {
    if (refuse_allocation_info(This, ret, n, descs))
        return ret;
    auto orig = DMN_ORIG(d3d12_GetResourceAllocationInfo3, This);
    return orig(This, ret, visible_mask, n, descs, num_castable, castable, info1);
}

/* GetResourceTiling for a sparse-backed resource comes from the tile model
 * dmn_sparse implements (the framework answers from its tier-0 world);
 * anything else passes through. */
void STDMETHODCALLTYPE hook_d3d12_GetResourceTiling(
        ID3D12Device* This, ID3D12Resource* res, UINT* total_tiles,
        D3D12_PACKED_MIP_INFO* packed, D3D12_TILE_SHAPE* shape, UINT* num_subs,
        UINT first_sub, D3D12_SUBRESOURCE_TILING* subs) {
    DmnSparseTiling t;
    std::vector<DmnSparseSubTiling> st;
    uint32_t n = num_subs ? *num_subs : 0;
    if (n && subs) st.resize(n);
    if (res && dmn_sparse_query_tiling(dmn_com_identity(res), &t, first_sub, &n,
                                       st.empty() ? nullptr : st.data())) {
        if (total_tiles) *total_tiles = t.total_tiles;
        if (packed) {
            packed->NumStandardMips = (UINT8)t.standard_mips;
            packed->NumPackedMips = (UINT8)t.packed_mips;
            packed->NumTilesForPackedMips = t.packed_tiles;
            /* D3D12_PACKED_MIP_INFO: 0 when nothing is packed. */
            packed->StartTileIndexInOverallResource =
                t.packed_mips ? t.packed_start_tile : 0;
        }
        if (shape) { shape->WidthInTexels = t.tile_w; shape->HeightInTexels = t.tile_h; shape->DepthInTexels = 1; }
        for (uint32_t i = 0; subs && i < n; i++) {
            subs[i].WidthInTiles = st[i].width_in_tiles;
            subs[i].HeightInTiles = (UINT16)st[i].height_in_tiles;
            subs[i].DepthInTiles = st[i].width_in_tiles ? 1 : 0;
            subs[i].StartTileIndexInOverallResource = st[i].start_tile;
        }
        if (num_subs) *num_subs = n;
        return;
    }
    auto orig = DMN_ORIG(d3d12_GetResourceTiling, This);
    if (orig)
        orig(This, res, total_tiles, packed, shape, num_subs, first_sub, subs);
}

/* D3D12_FORMAT_SUPPORT1/2 use the D3D11_FORMAT_SUPPORT/2 bit assignments, so
 * the D3D11 tables answer D3D12 once the D3D11-only bits are dropped. */
static_assert((UINT)D3D12_FORMAT_SUPPORT1_BUFFER == (UINT)D3D11_FORMAT_SUPPORT_BUFFER, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT1_TEXTURE2D == (UINT)D3D11_FORMAT_SUPPORT_TEXTURE2D, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT1_RENDER_TARGET == (UINT)D3D11_FORMAT_SUPPORT_RENDER_TARGET, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL == (UINT)D3D11_FORMAT_SUPPORT_DEPTH_STENCIL, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW ==
              (UINT)D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT1_SHADER_GATHER_COMPARISON ==
              (UINT)D3D11_FORMAT_SUPPORT_SHADER_GATHER_COMPARISON, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD == (UINT)D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_ADD, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE == (UINT)D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE, "");
static_assert((UINT)D3D12_FORMAT_SUPPORT2_TILED == (UINT)D3D11_FORMAT_SUPPORT2_TILED, "");
constexpr UINT kD3D11OnlySupport1 =
    D3D11_FORMAT_SUPPORT_MIP_AUTOGEN | D3D11_FORMAT_SUPPORT_CPU_LOCKABLE;
constexpr UINT kD3D11OnlySupport2 = D3D11_FORMAT_SUPPORT2_SHAREABLE;

/* D3D12_OPTIONS.TiledResourcesTier: the framework answers 0 (it has no tiled
 * support); with sparse backing available this library serves tier 2
 * through hook_d3d12_CreateReservedResource + hook_queue_UpdateTileMappings,
 * so report it.  The guest driver keys its reserved-resource path on this
 * (Triton: host tier >= 1 -> forward CreateReservedResource/UpdateTileMappings;
 * tier 0 -> its committed-backing shim), which keeps an old host safe: an
 * unhooked CreateReservedResource there returns an unbacked object. */
HRESULT STDMETHODCALLTYPE hook_d3d12_CheckFeatureSupport(
        ID3D12Device* This, D3D12_FEATURE feature, void* data, UINT size) {
    auto orig = DMN_ORIG(d3d12_CheckFeatureSupport, This);
    /* FORMAT_SUPPORT is D3DMetal's answer less the capabilities the format
     * tables know Metal lacks: D3DMetal claims some for every format (all
     * of them through GPTk 2.1; IA_INDEX_BUFFER, SO_BUFFER and
     * BACK_BUFFER_CAST on every format since), and a caller that trusts a
     * claim builds work Metal cannot run.  A format D3DMetal cannot make is
     * E_FAIL, as on D3D11 (MSDN CheckFormatSupport); D3DMetal answers it
     * S_OK with no bits.  TILED is dmn_sparse's to report when it is on. */
    if (feature == D3D12_FEATURE_FORMAT_SUPPORT) {
        auto* info = static_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT*>(data);
        if (!info || size != sizeof(*info))
            return E_INVALIDARG;
        HRESULT hr = orig ? orig(This, feature, data, size) : E_FAIL;
        if (FAILED(hr) || info->Format == DXGI_FORMAT_UNKNOWN)
            return hr;
        if (info->Support1 == D3D12_FORMAT_SUPPORT1_NONE) {
            info->Support2 = D3D12_FORMAT_SUPPORT2_NONE;
            return E_FAIL;
        }
        uint32_t s1 = 0, s2 = 0;
        if (dmn_format_support((uint32_t)info->Format, &s1)) {
            if (!dmn_format_support2((uint32_t)info->Format, &s2))
                s2 = 0;
            s1 &= ~kD3D11OnlySupport1;
            s2 &= ~kD3D11OnlySupport2;
            UINT tiled = dmn_sparse_enabled() ? (s2 & D3D12_FORMAT_SUPPORT2_TILED) : 0;
            info->Support1 = (D3D12_FORMAT_SUPPORT1)(info->Support1 & s1);
            info->Support2 = (D3D12_FORMAT_SUPPORT2)((info->Support2 & s2) | tiled);
        }
        return hr;
    }
    HRESULT hr = orig ? orig(This, feature, data, size) : E_FAIL;
    /* Same correction as the D3D11 CheckMultisampleQualityLevels hook. */
    if (SUCCEEDED(hr) && feature == D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS &&
        data && size == sizeof(D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS)) {
        auto* q = static_cast<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS*>(data);
        if (q->NumQualityLevels &&
            (!d3dmetal_has_format(This, q->Format) ||
             !dmn_format_msaa_ok((uint32_t)q->Format, q->SampleCount)))
            q->NumQualityLevels = 0;
    }
    if (SUCCEEDED(hr) && feature == D3D12_FEATURE_D3D12_OPTIONS && data &&
        size == sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS) && dmn_sparse_enabled()) {
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS*>(data);
        if (o->TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2)
            o->TiledResourcesTier = D3D12_TILED_RESOURCES_TIER_2;
    }
    return hr;
}

/* Reserved (tiled) resource: D3DMetal cannot back one (TiledResourcesTier
 * 0, its CreateReservedResource makes an unbacked object), so create it as
 * an ordinary framework resource whose Metal texture is substituted with one
 * from a sparse heap (dmn_sparse) -- no memory until UpdateTileMappings maps
 * tiles, unmapped tiles read zero.  D3DMetal sees a normal texture for
 * views, barriers and copies.  Without sparse support (or on a shape the
 * sparse heap refuses: buffers, 3D, MSAA) the committed create stands, fully
 * backed -- and fully resident, see dmn_sparse.h.
 *
 * The framework object is created PLACED at offset 0 of a dummy heap rather
 * than committed.  Measured (DMN_ALLOC_TRACE): a committed create -- with or
 * without HEAP_FLAG_SHARED -- makes D3DMetal reserve the texture's full size
 * in its 256 MiB pool heaps (heap_newtex on dev_newheap pools), and a pool
 * heap is resident wholesale, so the substituted sparse texture would sit on
 * top of a fully-backed pool reservation and save nothing.  Placed,
 * D3DMetal's bookkeeping allocation is the dummy heap, which every reserved
 * resource aliases at offset 0 (legal for placed resources); the swizzle
 * hands D3DMetal the sparse texture before Metal ever sees the placement.
 * D3DMetal validates offset + size <= heap size, so the dummy is sized to
 * the largest reserved resource seen and re-created bigger on demand (old
 * ones stay alive for the resources placed on them).  The committed create
 * is the fallback when the placed one fails.
 *
 * Shared body of the three CreateReservedResource variants: `create_placed`
 * (heap) and `create_committed` () perform this device tier's create of the
 * caller's copy of the desc into *out. */
template <class Placed, class Committed>
HRESULT d12_create_reserved(ID3D12Device* This, const D3D12_RESOURCE_DESC* desc,
                            void** out, Placed&& create_placed,
                            Committed&& create_committed) {
    const bool sparse = dmn_sparse_enabled() &&
                        desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                        desc->SampleDesc.Count <= 1;
    static ID3D12Device* dummy_dev = nullptr;
    static ID3D12Heap* dummy_heap = nullptr;
    static UINT64 dummy_bytes = 0;
    static std::mutex dummy_mtx;
    HRESULT hr = E_FAIL;
    bool did_placed = false;
    if (sparse) {
        D3D12_RESOURCE_DESC cd = *desc;
        cd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        D3D12_RESOURCE_ALLOCATION_INFO ai = This->GetResourceAllocationInfo(0, 1, &cd);
        if (ai.SizeInBytes == UINT64_MAX)
            return E_INVALIDARG;
        UINT64 need = ai.SizeInBytes ? ai.SizeInBytes : (64ull << 20);
        need = (need + 0xFFFFFull) & ~0xFFFFFull; /* 1 MiB granularity */
        std::lock_guard<std::mutex> lk(dummy_mtx);
        if (dummy_heap && dummy_dev != This) {
            /* A different device: its heaps are not this one's. */
            dummy_heap->Release();
            dummy_heap = nullptr;
            dummy_bytes = 0;
        }
        if (!dummy_heap || dummy_bytes < need) {
            UINT64 want = std::max<UINT64>(need, 64ull << 20);
            D3D12_HEAP_DESC hd{};
            hd.SizeInBytes = want;
            hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            hd.Alignment = 65536;
            hd.Flags = D3D12_HEAP_FLAG_NONE;
            ID3D12Heap* nh = nullptr;
            auto orig_ch = DMN_ORIG(d3d12_CreateHeap, This);
            HRESULT hh = orig_ch ? orig_ch(This, &hd, __uuidof(ID3D12Heap), (void**)&nh)
                                 : This->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&nh);
            if (FAILED(hh) || !nh) {
                DMN_WARN("hooks: sparse dummy heap (%llu MiB) create failed 0x%08x; this reserved "
                         "resource takes the committed path", (unsigned long long)(want >> 20), (unsigned)hh);
            } else {
                /* The previous dummy stays referenced by the resources placed on it
                 * (D3DMetal holds the heap); our reference can go. */
                if (dummy_heap) dummy_heap->Release();
                dummy_heap = nh; dummy_bytes = want; dummy_dev = This;
                DMN_INFO("hooks: sparse dummy heap now %llu MiB (largest reserved resource %llu MiB)",
                         (unsigned long long)(want >> 20), (unsigned long long)(need >> 20));
            }
        }
        if (dummy_heap && dummy_bytes >= need) {
            dmn_sparse_arm();
            hr = create_placed(dummy_heap);
            did_placed = true;
            if (FAILED(hr) || !*out) {
                void* t = nullptr;
                if (dmn_sparse_disarm(&t) && t)
                    dmn_sparse_release_texture(t);
                static std::atomic<uint32_t> w{0};
                if (w++ < 4)
                    DMN_WARN("hooks: reserved resource %llux%u fmt=%u mips=%u: placed-on-dummy-heap create "
                             "failed 0x%08x; retrying committed", (unsigned long long)desc->Width, desc->Height,
                             (unsigned)desc->Format, (unsigned)desc->MipLevels, (unsigned)hr);
                *out = nullptr;
                did_placed = false;
            }
        }
    }
    if (!did_placed) {
        if (sparse)
            dmn_sparse_arm();
        hr = create_committed();
    }
    void* tex = nullptr;
    bool captured = sparse && dmn_sparse_disarm(&tex);
    if (FAILED(hr) || !*out) {
        DMN_WARN("hooks: reserved resource %llux%u fmt=%u mips=%u: committed create failed 0x%08x",
                 (unsigned long long)desc->Width, desc->Height, (unsigned)desc->Format,
                 (unsigned)desc->MipLevels, (unsigned)hr);
        return hr;
    }
    auto* res = reinterpret_cast<IUnknown*>(*out);
    if (captured) {
        void* id = dmn_com_identity(res);
        dmn_sparse_register(id, tex, (uint32_t)desc->Dimension, (uint32_t)desc->Format,
                            desc->Width, desc->Height, desc->DepthOrArraySize,
                            desc->MipLevels);
        attach_evict_sentinel(res, id, EV_SPARSE);
    } else {
        static std::atomic<uint32_t> warned{0};
        if (warned++ < 8)
            DMN_WARN("hooks: reserved resource %llux%u fmt=%u mips=%u dim=%d is fully backed "
                     "(committed): sparse %s", (unsigned long long)desc->Width, desc->Height,
                     (unsigned)desc->Format, (unsigned)desc->MipLevels, (int)desc->Dimension,
                     sparse ? "capture missed" : "unavailable for this shape");
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateReservedResource(
        ID3D12Device* This, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear,
        REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreateReservedResource, This);
    if (!desc || !out)
        return orig ? orig(This, desc, state, clear, riid, out) : E_FAIL;
    D3D12_RESOURCE_DESC cd = *desc;
    cd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    auto orig_cp = DMN_ORIG(d3d12_CreatePlacedResource, This);
    auto orig_cc = DMN_ORIG(d3d12_CreateCommittedResource, This);
    return d12_create_reserved(This, desc, out,
        [&](ID3D12Heap* heap) {
            return orig_cp ? orig_cp(This, heap, 0, &cd, state, clear, riid, out)
                           : This->CreatePlacedResource(heap, 0, &cd, state, clear, riid, out);
        },
        [&] {
            return orig_cc ? orig_cc(This, &hp, D3D12_HEAP_FLAG_NONE, &cd, state, clear, riid, out)
                           : This->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &cd,
                                                           state, clear, riid, out);
        });
}

/* ID3D12Device4 variant: same, with a protected session -- which the placed
 * create has no room for, so a resource in a session keeps the framework's
 * own path (D3DMetal does not support protected sessions anyway). */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateReservedResource1(
        ID3D12Device4* This, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear,
        DMN_ID3D12ProtectedResourceSession* session, REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreateReservedResource1, This);
    if (!desc || !out || session)
        return orig ? orig(This, desc, state, clear, session, riid, out) : E_FAIL;
    D3D12_RESOURCE_DESC cd = *desc;
    cd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    auto orig_cp = DMN_ORIG(d3d12_CreatePlacedResource, This);
    auto orig_cc = DMN_ORIG(d3d12_CreateCommittedResource1, This);
    return d12_create_reserved(This, desc, out,
        [&](ID3D12Heap* heap) {
            return orig_cp ? orig_cp(This, heap, 0, &cd, state, clear, riid, out)
                           : This->CreatePlacedResource(heap, 0, &cd, state, clear, riid, out);
        },
        [&] {
            return orig_cc ? orig_cc(This, &hp, D3D12_HEAP_FLAG_NONE, &cd, state, clear,
                                     nullptr, riid, out)
                           : This->CreateCommittedResource1(&hp, D3D12_HEAP_FLAG_NONE, &cd,
                                                            state, clear, nullptr, riid, out);
        });
}

/* ID3D12Device10 variant: barrier layout + castable formats, forwarded to
 * this tier's placed/committed creates (DESC1 = DESC + a zero mip region). */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateReservedResource2(
        ID3D12Device10* This, const D3D12_RESOURCE_DESC* desc,
        DMN_D3D12_BARRIER_LAYOUT layout, const D3D12_CLEAR_VALUE* clear,
        DMN_ID3D12ProtectedResourceSession* session, UINT32 num_castable,
        const DXGI_FORMAT* castable, REFIID riid, void** out) {
    if (!dmn_desc_format_ok(This, desc))
        return E_INVALIDARG;
    auto orig = DMN_ORIG(d3d12_CreateReservedResource2, This);
    if (!desc || !out || session)
        return orig ? orig(This, desc, layout, clear, session, num_castable, castable, riid, out)
                    : E_FAIL;
    D3D12_RESOURCE_DESC1 cd{};
    memcpy(&cd, desc, sizeof(*desc));
    cd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    auto orig_cp = DMN_ORIG(d3d12_CreatePlacedResource2, This);
    auto orig_cc = DMN_ORIG(d3d12_CreateCommittedResource3, This);
    return d12_create_reserved(This, desc, out,
        [&](ID3D12Heap* heap) {
            return orig_cp ? orig_cp(This, heap, 0, &cd, layout, clear, num_castable, castable,
                                     riid, out)
                           : This->CreatePlacedResource2(heap, 0, &cd, layout, clear, num_castable,
                                                         castable, riid, out);
        },
        [&] {
            return orig_cc ? orig_cc(This, &hp, D3D12_HEAP_FLAG_NONE, &cd, layout, clear, nullptr,
                                     num_castable, castable, riid, out)
                           : This->CreateCommittedResource3(&hp, D3D12_HEAP_FLAG_NONE, &cd, layout,
                                                            clear, nullptr, num_castable, castable,
                                                            riid, out);
        });
}

/* Command lists are hooked for CopyTiles on sparse-backed reserved
 * resources; the framework has no tiled path (see hook_cmdlist_CopyTiles). */
void patch_d3d12_cmdlist(IUnknown* obj) {
    ID3D12GraphicsCommandList* cl = nullptr;
    if (SUCCEEDED(obj->QueryInterface(__uuidof(ID3D12GraphicsCommandList),
                                      reinterpret_cast<void**>(&cl))) && cl) {
        DMN_PATCH(cl, ID3D12GraphicsCommandList, CopyTiles, cmdlist_CopyTiles);
        cl->Release();
    }
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandList(
        ID3D12Device* This, UINT node_mask, D3D12_COMMAND_LIST_TYPE type,
        ID3D12CommandAllocator* allocator, ID3D12PipelineState* pso, REFIID riid,
        void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommandList, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, node_mask, type, allocator, pso, riid, out);
    if (SUCCEEDED(hr) && out && *out)
        patch_d3d12_cmdlist(reinterpret_cast<IUnknown*>(*out));
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateCommandList1(
        ID3D12Device4* This, UINT node_mask, D3D12_COMMAND_LIST_TYPE type,
        UINT flags, REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateCommandList1, This);
    if (!orig)
        return E_FAIL;
    HRESULT hr = orig(This, node_mask, type, flags, riid, out);
    if (SUCCEEDED(hr) && out && *out)
        patch_d3d12_cmdlist(reinterpret_cast<IUnknown*>(*out));
    return hr;
}

void patch_d3d12_queue(IUnknown* obj) {
    ID3D12CommandQueue* q = nullptr;
    if (SUCCEEDED(obj->QueryInterface(__uuidof(ID3D12CommandQueue),
                                      reinterpret_cast<void**>(&q))) && q) {
        DMN_PATCH(q, ID3D12CommandQueue, Signal, queue_Signal);
        DMN_PATCH(q, ID3D12CommandQueue, Wait, queue_Wait);
        /* The vendored declaration's slot index is right even though its
         * parameter list is not (see the forward declaration). */
        DMN_PATCH(q, ID3D12CommandQueue, UpdateTileMappings,
                  queue_UpdateTileMappings);
        DMN_PATCH(q, ID3D12CommandQueue, CopyTileMappings, queue_CopyTileMappings);
        q->Release();
    }
}

/* == DXIL subroutine inlining ============================================= */

/* A shader stage's bytecode after dmn_dxil_inline_subroutines, owned for
 * the duration of one pipeline create. */
struct DmnNormalizedStage {
    void* code = nullptr;
    ~DmnNormalizedStage() { free(code); }
    bool apply(D3D12_SHADER_BYTECODE& bc) {
        size_t n = 0;
        if (!bc.pShaderBytecode ||
            !dmn_dxil_inline_subroutines(bc.pShaderBytecode, bc.BytecodeLength,
                                         &code, &n))
            return false;
        bc.pShaderBytecode = code;
        bc.BytecodeLength = n;
        return true;
    }
};

/* In-driver-converted DXIL reaches D3DMetal with its subroutines inlined
 * (see dmn_dxil_inline_subroutines).  The runtime converts DXBC of every
 * stage that way -- D3D12 titles shipping FXC bytecode, D3D11On12,
 * D3D9On12 -- so every stage of both pipeline kinds goes through it. */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateGraphicsPipelineState(
        ID3D12Device* This, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc,
        REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateGraphicsPipelineState, This);
    if (!orig)
        return E_FAIL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC norm;
    DmnNormalizedStage stages[5];
    if (desc) {
        norm = *desc;
        bool changed = false;
        D3D12_SHADER_BYTECODE* bcs[5] = {&norm.VS, &norm.PS, &norm.DS,
                                         &norm.HS, &norm.GS};
        for (int i = 0; i < 5; i++)
            changed = stages[i].apply(*bcs[i]) || changed;
        if (changed)
            desc = &norm;
    }
    return orig(This, desc, riid, out);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_CreateComputePipelineState(
        ID3D12Device* This, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc,
        REFIID riid, void** out) {
    auto orig = DMN_ORIG(d3d12_CreateComputePipelineState, This);
    if (!orig)
        return E_FAIL;
    D3D12_COMPUTE_PIPELINE_STATE_DESC norm;
    DmnNormalizedStage cs;
    if (desc) {
        norm = *desc;
        if (cs.apply(norm.CS))
            desc = &norm;
    }
    return orig(This, desc, riid, out);
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
    auto orig = DMN_ORIG(d3d12_SetEventOnMultipleFenceCompletion, This);
    HRESULT hr = orig ? orig(This, fences, values, count, flags, ev) : E_NOTIMPL;
    /* Imports are real fences, so the native multi-wait works as-is; each
     * imported entry just needs its wait watcher armed so the local fence
     * reaches the awaited value. Only after the wait was actually registered
     * though: a rejected call (GPTk 2.1 and earlier do not implement this one)
     * would otherwise leave watcher threads waiting on values nobody is going
     * to deliver, outliving the state they were armed against. */
    if (SUCCEEDED(hr))
        for (UINT i = 0; fences && values && i < count; i++)
            dmn_fd3d_before_queue_wait(fences[i], values[i]);
    return hr;
}

/* == D3D12 share entry points ============================================== */
HRESULT STDMETHODCALLTYPE hook_d3d12_CreateSharedHandle(
        ID3D12Device* This, ID3D12DeviceChild* object,
        const SECURITY_ATTRIBUTES* attr, DWORD access, const WCHAR* name,
        HANDLE* out) {
    /* A heap is a legal argument here (MSDN: heap, resource, or fence) and
     * SHARED heaps are synthetic — D3DMetal must never receive one, so this
     * check runs before every fall-through, including the null-out one. The
     * handle is a buffer-magic POD carrying the heap's whole window;
     * OpenSharedHandle with IID_ID3D12Heap turns it back into a synthetic
     * heap. */
    if (DmnSynthHeap* sh = as_synth_heap(reinterpret_cast<ID3D12Heap*>(object))) {
        if (!out)
            return E_INVALIDARG;
        dxn_shared_buffer pod{};
        pod.magic = DXN_SHARED_BUFFER_MAGIC;
        pod.version = DXN_SHARED_VERSION;
        pod.struct_size = sizeof(pod);
        pod.fd = sh->fd; /* vend_pod_owned dups it */
        pod.size = sh->size;
        pod.offset = sh->fd_off;
        HRESULT hr = vend_pod_owned(pod, out);
        if (SUCCEEDED(hr))
            DMN_INFO("hooks: exported NT heap handle %p (size=%llu)", *out,
                     (unsigned long long)sh->size);
        return hr;
    }
    if (object && out) {
        auto* unk = reinterpret_cast<IUnknown*>(object);
        if (SUCCEEDED(return_fence_pod(unk, out)))
            return S_OK;
        if (SUCCEEDED(return_texture_pod(unk, out, /*nt=*/true)))
            return S_OK;
        if (SUCCEEDED(return_buffer_pod(unk, out, /*nt=*/true)))
            return S_OK;
        /* A SHARED 1D/3D resource has no shared backing (see g_unshared).
         * Fail here, where the app can see it, instead of at first use of a
         * blank surface. */
        void* id = dmn_com_identity(unk);
        bool unshared;
        {
            std::lock_guard<std::mutex> lk(g_reg_mtx);
            unshared = g_unshared.count(id) != 0;
        }
        if (unshared) {
            DMN_ERROR("hooks: CreateSharedHandle on a SHARED 1D/3D resource — "
                      "no shared backing exists for that dimension, refusing "
                      "to vend a handle");
            *out = nullptr;
            return E_INVALIDARG;
        }
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
        if (magic == DXN_SHARED_FENCE_MAGIC) {
            ID3D12Fence* f = nullptr;
            HRESULT hr = dmn_fd3d_import_d3d12(
                This, reinterpret_cast<dxn_shared_fence*>(handle), &f);
            if (FAILED(hr) || !f)
                return hr;
            track_fence_d3d12(f); /* a consumer-only process patches only here */
            hr = f->QueryInterface(riid, out);
            f->Release();
            return hr;
        }
        if (magic == DXN_SHARED_TEXTURE_MAGIC)
            return import_texture_d3d12(
                This, reinterpret_cast<dxn_shared_texture*>(handle), riid, out);
        if (magic == DXN_SHARED_BUFFER_MAGIC) {
            auto* pod = reinterpret_cast<dxn_shared_buffer*>(handle);
            /* The riid says what the handle referred to: a heap handle is
             * opened as a (synthetic) heap, anything else as a buffer. */
            if (dmn_iid_eq(riid, __uuidof(ID3D12Heap)))
                return import_heap_d3d12(This, pod, riid, out);
            return import_buffer_d3d12(This, pod, riid, out);
        }
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

/* == Public API: import a shared-memory window as an ID3D12Heap ========== */
/* Contract documented on the declaration in d3dmetal_native.h. Plain
 * extern "C" (SysV, no STDMETHODCALLTYPE): the Neptune render server resolves
 * this with dlsym and calls it with the host convention — the same deliberate
 * choice as virglrenderer's npt_library.h function-pointer typedefs. */
extern "C" int32_t dmn_open_existing_heap_from_fd(void* device, int fd,
                                                  uint64_t offset,
                                                  uint64_t size,
                                                  uint32_t heap_type,
                                                  uint32_t heap_flags,
                                                  const void* iid,
                                                  void** out_heap) {
    if (!device || !iid || !out_heap)
        return E_INVALIDARG;
    *out_heap = nullptr;
    /* Mirrors dmn_share_metal.mm's kMaxSharedBytes. */
    const uint64_t kMaxImportBytes = 1ull << 32;
    if (fd < 0 || !size || size > kMaxImportBytes) {
        DMN_ERROR("hooks: heap import rejected (fd=%d size=%llu)", fd,
                  (unsigned long long)size);
        return E_INVALIDARG;
    }
    const uint64_t page = (uint64_t)dmn_share_page_align(1);
    if (offset & (page - 1)) {
        DMN_ERROR("hooks: heap import offset %llu is not page-aligned",
                  (unsigned long long)offset);
        return E_INVALIDARG;
    }
    struct stat st;
    uint64_t end = 0;
    if (fstat(fd, &st) != 0 || st.st_size < 0 ||
        __builtin_add_overflow(offset, size, &end) ||
        end > (uint64_t)st.st_size) {
        DMN_ERROR("hooks: heap import window [%llu, +%llu) does not fit "
                  "fd=%d (size %lld)", (unsigned long long)offset,
                  (unsigned long long)size, fd, (long long)st.st_size);
        return E_INVALIDARG;
    }

    ID3D12Device* dev = nullptr;
    auto* unk = reinterpret_cast<IUnknown*>(device);
    if (FAILED(unk->QueryInterface(__uuidof(ID3D12Device),
                                   reinterpret_cast<void**>(&dev))) || !dev) {
        DMN_ERROR("hooks: heap import device %p is not an ID3D12Device",
                  device);
        return E_NOINTERFACE;
    }

    /* A synthetic heap backs the import (see DmnSynthHeap): a framework heap
     * would allocate heap-size memory that a placed buffer's substituted
     * backing can only duplicate. The synthetic heap holds a dup of the fd;
     * placed buffers get their Metal backing substituted with windows into
     * the shm object, and the residency hooks keep the pointer away from
     * D3DMetal. The guest's flags are deliberately dropped (only buffers may
     * alias the window); heap_type is advisory (the guest answers GetDesc
     * locally). */
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = size;
    hd.Properties.Type = (heap_type >= (uint32_t)D3D12_HEAP_TYPE_DEFAULT &&
                          heap_type <= (uint32_t)D3D12_HEAP_TYPE_READBACK)
                             ? (D3D12_HEAP_TYPE)heap_type
                             : D3D12_HEAP_TYPE_UPLOAD;
    hd.Alignment = 0;
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

    const int owned = fcntl(fd, F_DUPFD_CLOEXEC, 0); /* fd is borrowed */
    if (owned < 0) {
        DMN_ERROR("hooks: heap import dup(fd=%d) failed", fd);
        dev->Release();
        return E_FAIL;
    }
    HRESULT hr = synth_heap_new(reinterpret_cast<IUnknown*>(dev), hd, owned,
                                offset, size, /*imported=*/true,
                                *reinterpret_cast<const IID*>(iid), out_heap);
    dev->Release();
    if (FAILED(hr) || !*out_heap) {
        DMN_ERROR("hooks: heap import (size=%llu type=%u) failed 0x%08x",
                  (unsigned long long)size, (unsigned)hd.Properties.Type,
                  (unsigned)hr);
        return FAILED(hr) ? hr : E_FAIL;
    }
    DMN_INFO("hooks: imported shm window fd=%d off=%llu size=%llu app_type=%u "
             "app_flags=0x%x as synthetic heap %p", fd,
             (unsigned long long)offset, (unsigned long long)size, heap_type,
             heap_flags, *out_heap);
    return S_OK;
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

UINT64 dmn_hooks_f11_completed_orig(ID3D11Fence* fence) {
    auto orig = DMN_ORIG(f11_GetCompletedValue, fence);
    return orig ? orig(fence) : fence->GetCompletedValue();
}

UINT64 dmn_hooks_f12_completed_orig(ID3D12Fence* fence) {
    auto orig = DMN_ORIG(f12_GetCompletedValue, fence);
    return orig ? orig(fence) : fence->GetCompletedValue();
}

HRESULT dmn_hooks_f11_seoc_orig(ID3D11Fence* fence, UINT64 value, HANDLE ev) {
    auto orig = DMN_ORIG(f11_SetEventOnCompletion, fence);
    return orig ? orig(fence, value, ev) : fence->SetEventOnCompletion(value, ev);
}

HRESULT dmn_hooks_f12_seoc_orig(ID3D12Fence* fence, UINT64 value, HANDLE ev) {
    auto orig = DMN_ORIG(f12_SetEventOnCompletion, fence);
    return orig ? orig(fence, value, ev) : fence->SetEventOnCompletion(value, ev);
}

bool dmn_res_lookup_buffer_pod(IUnknown* res, dxn_shared_buffer* out) {
    void* id = dmn_com_identity(res);
    std::lock_guard<std::mutex> lk(g_reg_mtx);
    auto it = g_buf_reg.find(id);
    if (it == g_buf_reg.end())
        return false;
    *out = it->second;
    return true;
}

/* == Post-create patch entry points (called from dmn_init.cpp) =========== */

/* dmn_hook.h's identity resolution asks these (see the DXGI view identity
 * comment above); they live here because the view vtables and the probe do. */
extern "C" DmnQiRefTarget dmn_com_qi_ref_target(void* obj) {
    /* Only a DXGI view can deviate; a resource's own vtable is COM-correct on
     * every version. A view is exactly an object whose vtable we patched
     * GetSharedHandle onto. */
    if (!obj || !h_GetSharedHandle.original_for(obj))
        return DMN_QI_REF_RESULT;
    const int probed = g_view_qi_ref_target.load(std::memory_order_acquire);
    /* Not probed yet (the view was minted before any patch landed): assume the
     * COM-correct answer, which is what every version but 4.0b1 does. */
    return probed < 0 ? DMN_QI_REF_RESULT : (DmnQiRefTarget)probed;
}

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
        DMN_PATCH(d0, ID3D11Device, CheckMultisampleQualityLevels,
                  d3d11_CheckMultisampleQualityLevels);
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
        /* Repairs a 4.0b1 CreateBlendState1 that answers S_OK with no state. */
        DMN_PATCH(d1, ID3D11Device1, CreateBlendState1, d3d11_CreateBlendState1);
        d1->Release();
    }
    ID3D11Device2* d2 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device2),
                                      reinterpret_cast<void**>(&d2))) && d2) {
        DMN_PATCH(d2, ID3D11Device2, CheckMultisampleQualityLevels1,
                  d3d11_CheckMultisampleQualityLevels1);
        d2->Release();
    }
    ID3D11Device3* d3 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device3),
                                      reinterpret_cast<void**>(&d3))) && d3) {
        DMN_PATCH(d3, ID3D11Device3, CreateTexture2D1, d3d11_CreateTexture2D1);
        /* Repairs the E_NOTIMPL ...View1 creates (D3D11.3 DDI-tier UMDs). */
        DMN_PATCH(d3, ID3D11Device3, CreateShaderResourceView1, d3d11_CreateShaderResourceView1);
        DMN_PATCH(d3, ID3D11Device3, CreateRenderTargetView1, d3d11_CreateRenderTargetView1);
        DMN_PATCH(d3, ID3D11Device3, CreateUnorderedAccessView1, d3d11_CreateUnorderedAccessView1);
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
    /* The D3D12 shared-fence value store rides WriteBufferImmediate, which the
     * D3DMDevice constructor may have just gated off (GPTk 4.0b1). */
    dmn_force_write_buffer_immediate();
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
        /* Synthetic heaps must never reach the framework's residency paths. */
        DMN_PATCH(d, ID3D12Device, MakeResident, d3d12_MakeResident);
        DMN_PATCH(d, ID3D12Device, Evict, d3d12_Evict);
        DMN_PATCH(d, ID3D12Device, CreateCommandQueue, d3d12_CreateCommandQueue);
        DMN_PATCH(d, ID3D12Device, CreateReservedResource, d3d12_CreateReservedResource);
        DMN_PATCH(d, ID3D12Device, GetResourceTiling, d3d12_GetResourceTiling);
        {
            /* Overloaded (explicit-return slot + by-value wrapper): spell the
             * virtual's exact member type to pick the slot. */
            using Grai = D3D12_RESOURCE_ALLOCATION_INFO* (STDMETHODCALLTYPE ID3D12Device::*)(
                D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT, const D3D12_RESOURCE_DESC*);
            dmn_patch_slot(d, dmn_vindex(static_cast<Grai>(&ID3D12Device::GetResourceAllocationInfo)),
                           (void*)hook_d3d12_GetResourceAllocationInfo,
                           h_d3d12_GetResourceAllocationInfo);
        }
        DMN_PATCH(d, ID3D12Device, CheckFeatureSupport, d3d12_CheckFeatureSupport);
        DMN_PATCH(d, ID3D12Device, CreateCommandList, d3d12_CreateCommandList);
        DMN_PATCH(d, ID3D12Device, CreateGraphicsPipelineState,
                  d3d12_CreateGraphicsPipelineState);
        DMN_PATCH(d, ID3D12Device, CreateComputePipelineState,
                  d3d12_CreateComputePipelineState);
        d->Release();
    }
    /* Newer device interfaces (declared in dmn_d3d12_up.h) that add shared
     * create variants; D3DMetal exposes all of them via QI. */
    ID3D12Device1* d1 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device1),
                                      reinterpret_cast<void**>(&d1))) && d1) {
        DMN_PATCH(d1, ID3D12Device1, SetEventOnMultipleFenceCompletion,
                  d3d12_SetEventOnMultipleFenceCompletion);
        DMN_PATCH(d1, ID3D12Device1, SetResidencyPriority,
                  d3d12_SetResidencyPriority);
        d1->Release();
    }
    ID3D12Device3* dv3 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device3),
                                      reinterpret_cast<void**>(&dv3))) && dv3) {
        DMN_PATCH(dv3, ID3D12Device3, EnqueueMakeResident,
                  d3d12_EnqueueMakeResident);
        dv3->Release();
    }
    ID3D12Device4* d4 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device4),
                                      reinterpret_cast<void**>(&d4))) && d4) {
        DMN_PATCH(d4, ID3D12Device4, CreateCommittedResource1,
                  d3d12_CreateCommittedResource1);
        DMN_PATCH(d4, ID3D12Device4, CreateHeap1, d3d12_CreateHeap1);
        DMN_PATCH(d4, ID3D12Device4, CreateReservedResource1,
                  d3d12_CreateReservedResource1);
        DMN_PATCH(d4, ID3D12Device4, CreateCommandList1, d3d12_CreateCommandList1);
        DMN_PATCH(d4, ID3D12Device4, GetResourceAllocationInfo1,
                  d3d12_GetResourceAllocationInfo1);
        d4->Release();
    }
    ID3D12Device8* d8 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device8),
                                      reinterpret_cast<void**>(&d8))) && d8) {
        DMN_PATCH(d8, ID3D12Device8, CreateCommittedResource2,
                  d3d12_CreateCommittedResource2);
        DMN_PATCH(d8, ID3D12Device8, CreatePlacedResource1,
                  d3d12_CreatePlacedResource1);
        DMN_PATCH(d8, ID3D12Device8, GetResourceAllocationInfo2,
                  d3d12_GetResourceAllocationInfo2);
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
        DMN_PATCH(d10, ID3D12Device10, CreateReservedResource2,
                  d3d12_CreateReservedResource2);
        d10->Release();
    }
    ID3D12Device12* d12 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device12),
                                      reinterpret_cast<void**>(&d12))) && d12) {
        DMN_PATCH(d12, ID3D12Device12, GetResourceAllocationInfo3,
                  d3d12_GetResourceAllocationInfo3);
        d12->Release();
    }
    DMN_INFO("hooks: patched D3D12 device %p", device);
}
