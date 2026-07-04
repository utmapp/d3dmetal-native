/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * D3D-facing shared-fence implementation. See dmn_fence_d3d.h for the API and
 * the module boundaries.
 *
 * Producer mechanism (GPU writes the reached value into the companion slot):
 *   D3D12: a helper command queue does Wait(realFence,V) then a command list
 *          of two WriteBufferImmediate stores (low word, then high).
 *   D3D11: right before the app's own immediate-context Signal, the two slot
 *          words are written via ClearUnorderedAccessViewUint on that same
 *          context (same GPU timeline, so the store is ordered after the work
 *          the value represents; a compute-shader store would not reach the
 *          StorageModeShared backing under D3DMetal, a clear/blit does).
 *
 * Consumer mechanism: an import is a REAL fence created on the opening device,
 * plus a view of the shared slot (dmn_fence.cpp). Reads merge the slot into
 * the local value (hooked GetCompletedValue / SetEventOnCompletion). A GPU
 * wait stays native; the hooked wait entry point arms a watcher thread that
 * raises the local fence when the slot arrives (D3D12: CPU Signal; D3D11:
 * context Signal under ID3D11Multithread protection — there is no CPU fence
 * signal in D3D11), releasing the wait. GPU signal-back also stays native; a
 * watcher stores the value into the slot once the local fence completes it.
 * Watchers hold a reference on the imported fence, so import state (whose
 * lifetime is destruction-driven, like everything else) outlives them.
 */

#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>

#include <d3d11_4.h>
#include <d3d12.h>

#include "d3dmetal_native.h"
#include "dmn_fence.h"
#include "dmn_fence_d3d.h"
#include "dmn_hook.h"
#include "dmn_log.h"

/* GFXT event handles (dmn_gfxt_event.cpp): the HANDLEs apps pass into
 * SetEventOnCompletion are these, and the watcher threads use them too. */
extern "C" void dmn_event_signal(void* handle);
extern "C" void* dmn_event_create(int manual_reset, int initial_state);
extern "C" void dmn_event_close(void* handle);

namespace {

/* == Producer state ======================================================= */
struct GpuFence {
    bool      is_d3d12 = false;
    IUnknown* realFence = nullptr;         /* borrowed — an AddRef here would
                                              keep the fence alive forever and
                                              destruction could never evict us */
    dmn_shared_buffer_handle bufPod{};     /* companion buffer (fd/size); the
                                              value slot is at offset 0 */
    uint64_t  initial = 0;
    uint32_t  flags = 0;
    dmn_shared_fence_t view = nullptr;     /* our own mapping of the slot */

    /* D3D12 timed-write path. A ring of allocators is cycled per Signal so an
     * allocator is only ever Reset after its (tiny) prior write has surely
     * drained, without blocking the app thread on the signal path. */
    static constexpr int kRing = 8;
    ID3D12Resource*             d12buf = nullptr;
    UINT64                      gpuVA = 0;
    ID3D12CommandQueue*         helperQ = nullptr;
    ID3D12CommandAllocator*     helperAlloc[kRing] = {};
    ID3D12GraphicsCommandList2* helperList = nullptr;
    uint64_t                    seq = 0;

    /* D3D11 GPU-store path: two single-element R32_UINT UAVs (low word slot,
     * high word slot) cleared per Signal via ClearUnorderedAccessViewUint. */
    ID3D11Buffer*               d11buf = nullptr;
    ID3D11UnorderedAccessView*  d11uavLo = nullptr;
    ID3D11UnorderedAccessView*  d11uavHi = nullptr;
    ID3D11DeviceContext*        d11ctx = nullptr; /* immediate; AddRef'd */

    std::mutex mtx;                        /* serialize signals on this fence */
};

std::mutex g_fence_mtx;
std::unordered_map<void*, GpuFence*> g_fence_reg; /* com identity -> producer */

GpuFence* lookup_producer(IUnknown* fence) {
    if (!fence)
        return nullptr;
    void* id = dmn_com_identity(fence);
    std::lock_guard<std::mutex> lk(g_fence_mtx);
    auto it = g_fence_reg.find(id);
    return it == g_fence_reg.end() ? nullptr : it->second;
}

/* Open the producer's own view of its slot (for completed-merge/watch). */
void open_producer_view(GpuFence* gf) {
    dmn_shared_fence_handle pod{};
    pod.magic = DMN_SHARED_FENCE_MAGIC;
    pod.version = DMN_SHARED_HANDLE_VERSION;
    pod.fd = gf->bufPod.fd;
    pod.flags = gf->flags;
    pod.initial_value = gf->initial;
    gf->view = dmn_shared_fence_open(&pod);
    if (!gf->view)
        DMN_WARN("fence: producer slot view unavailable (signal-back from "
                 "consumers will not be visible)");
}

void register_producer(IUnknown* fence, GpuFence* gf) {
    void* id = dmn_com_identity(fence);
    {
        std::lock_guard<std::mutex> lk(g_fence_mtx);
        g_fence_reg[id] = gf;
    }
    DMN_INFO("fence: %s producer identity=%p buf fd=%d initial=%llu",
             gf->is_d3d12 ? "D3D12" : "D3D11", id, gf->bufPod.fd,
             (unsigned long long)gf->initial);
}

/* Free everything a GpuFence owns. D3D11 objects can be released with work in
 * flight (the runtime defers destruction); the D3D12 helper queue is drained
 * first — bounded, since a pending Wait whose value never arrives (app
 * destroyed the fence early) must not hang the app's Release call. */
void gpufence_teardown(GpuFence* gf) {
    if (gf->is_d3d12 && gf->helperQ) {
        ID3D12Device* dev = nullptr;
        if (SUCCEEDED(gf->helperQ->GetDevice(__uuidof(ID3D12Device),
                                             reinterpret_cast<void**>(&dev))) && dev) {
            ID3D12Fence* drain = nullptr;
            if (SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                           __uuidof(ID3D12Fence),
                                           reinterpret_cast<void**>(&drain))) && drain) {
                std::lock_guard<std::mutex> lk(gf->mtx);
                gf->helperQ->Signal(drain, 1);
                void* ev = dmn_event_create(1, 0);
                if (ev && SUCCEEDED(drain->SetEventOnCompletion(1, ev))) {
                    if (dmn_event_wait(ev, 250ull * 1000 * 1000) != DMN_WAIT_SIGNALED)
                        DMN_WARN("fence: helper-queue drain timed out on "
                                 "teardown (pending Wait never satisfied?)");
                }
                if (ev)
                    dmn_event_close(ev);
                drain->Release();
            }
            dev->Release();
        }
    }
    if (gf->helperList) gf->helperList->Release();
    for (int i = 0; i < GpuFence::kRing; i++)
        if (gf->helperAlloc[i]) gf->helperAlloc[i]->Release();
    if (gf->helperQ) gf->helperQ->Release();
    if (gf->d12buf) gf->d12buf->Release();
    if (gf->d11uavLo) gf->d11uavLo->Release();
    if (gf->d11uavHi) gf->d11uavHi->Release();
    if (gf->d11buf) gf->d11buf->Release();
    if (gf->d11ctx) gf->d11ctx->Release();
    if (gf->view) dmn_shared_fence_close(gf->view);
    delete gf;
}

bool gpufence_setup_d3d12(ID3D12Device* dev, GpuFence* gf) {
    /* Companion shared buffer via the standard hooked path. */
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 4096;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = dev->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
        __uuidof(ID3D12Resource), reinterpret_cast<void**>(&gf->d12buf));
    if (FAILED(hr) || !gf->d12buf) {
        DMN_ERROR("fence: companion buffer CreateCommittedResource 0x%08x",
                  (unsigned)hr);
        return false;
    }
    if (!dmn_res_lookup_buffer_pod(gf->d12buf, &gf->bufPod)) {
        DMN_ERROR("fence: companion buffer not routed through the swizzle");
        return false;
    }
    gf->gpuVA = gf->d12buf->GetGPUVirtualAddress();

    /* Helper queue + allocator ring + list for the Wait + timed write. */
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                 reinterpret_cast<void**>(&gf->helperQ));
    if (FAILED(hr)) { DMN_ERROR("fence: helper queue 0x%08x", (unsigned)hr); return false; }
    for (int i = 0; i < GpuFence::kRing; i++) {
        hr = dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                         __uuidof(ID3D12CommandAllocator),
                                         reinterpret_cast<void**>(&gf->helperAlloc[i]));
        if (FAILED(hr)) { DMN_ERROR("fence: helper alloc 0x%08x", (unsigned)hr); return false; }
    }
    ID3D12GraphicsCommandList* l0 = nullptr;
    hr = dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, gf->helperAlloc[0],
                                nullptr, __uuidof(ID3D12GraphicsCommandList),
                                reinterpret_cast<void**>(&l0));
    if (FAILED(hr) || !l0) { DMN_ERROR("fence: helper list 0x%08x", (unsigned)hr); return false; }
    hr = l0->QueryInterface(__uuidof(ID3D12GraphicsCommandList2),
                            reinterpret_cast<void**>(&gf->helperList));
    l0->Release();
    if (FAILED(hr) || !gf->helperList) {
        DMN_ERROR("fence: GraphicsCommandList2 unavailable 0x%08x", (unsigned)hr);
        return false;
    }
    gf->helperList->Close(); /* created open; each signal resets it */
    return true;
}

/* One single-element R32_UINT UAV over `buf` at element `elem`, so a subsequent
 * ClearUnorderedAccessViewUint writes exactly that 32-bit slot (nothing else). */
bool make_slot_uav(ID3D11Device* dev, ID3D11Buffer* buf, UINT elem,
                   ID3D11UnorderedAccessView** out) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format = DXGI_FORMAT_R32_UINT;
    ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.FirstElement = elem;
    ud.Buffer.NumElements = 1;
    HRESULT hr = dev->CreateUnorderedAccessView(buf, &ud, out);
    if (FAILED(hr)) { DMN_ERROR("fence: slot UAV 0x%08x", (unsigned)hr); return false; }
    return true;
}

bool gpufence_setup_d3d11(ID3D11Device* dev, GpuFence* gf) {
    /* Companion shared buffer via the standard hooked path. */
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 4096;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    HRESULT hr = dev->CreateBuffer(&bd, nullptr, &gf->d11buf);
    if (FAILED(hr) || !gf->d11buf) {
        DMN_ERROR("fence: companion CreateBuffer 0x%08x", (unsigned)hr);
        return false;
    }
    if (!dmn_res_lookup_buffer_pod(gf->d11buf, &gf->bufPod)) {
        DMN_ERROR("fence: companion buffer not routed through the swizzle");
        return false;
    }
    if (!make_slot_uav(dev, gf->d11buf, 0, &gf->d11uavLo) ||
        !make_slot_uav(dev, gf->d11buf, 1, &gf->d11uavHi))
        return false;

    dev->GetImmediateContext(&gf->d11ctx); /* AddRef'd */
    return true;
}

/* D3D12: helper queue waits for the real fence to reach V, then the GPU writes
 * V into the companion buffer (low word first so a torn consumer read of a
 * value crossing a 32-bit boundary under-reports rather than over-reports). */
void gpufence_signal_d3d12(GpuFence* gf, UINT64 value) {
    std::lock_guard<std::mutex> lk(gf->mtx);
    auto* rf = reinterpret_cast<ID3D12Fence*>(gf->realFence);
    ID3D12CommandAllocator* alloc = gf->helperAlloc[gf->seq++ % GpuFence::kRing];
    gf->helperQ->Wait(rf, value);
    alloc->Reset();
    gf->helperList->Reset(alloc, nullptr);
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER lo{gf->gpuVA,
                                            (UINT32)(value & 0xffffffffu)};
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER hi{gf->gpuVA + 4,
                                            (UINT32)(value >> 32)};
    gf->helperList->WriteBufferImmediate(1, &lo, nullptr);
    gf->helperList->WriteBufferImmediate(1, &hi, nullptr);
    gf->helperList->Close();
    ID3D12CommandList* lists[] = {gf->helperList};
    gf->helperQ->ExecuteCommandLists(1, lists);
}

/* D3D11: on the immediate context (the one the app signals on, so this rides
 * the same GPU timeline as the work the value represents), clear the two slot
 * UAVs to the value's low and high words. Low word first: see above. */
void gpufence_signal_d3d11(GpuFence* gf, UINT64 value) {
    std::lock_guard<std::mutex> lk(gf->mtx);
    ID3D11DeviceContext* c = gf->d11ctx;
    UINT lo[4] = {(UINT)(value & 0xffffffffu), 0, 0, 0};
    UINT hi[4] = {(UINT)(value >> 32), 0, 0, 0};
    c->ClearUnorderedAccessViewUint(gf->d11uavLo, lo);
    c->ClearUnorderedAccessViewUint(gf->d11uavHi, hi);
    c->Flush();
}

/* == Consumer (import) state =============================================== */
/* Watchers never touch ImportedFence after the fence could die: every watcher
 * holds its own AddRef on the imported fence for its whole lifetime, and the
 * import state is only torn down by destruction-driven eviction — which
 * cannot run while a watcher's reference is outstanding. Watchers also open
 * their own slot views (same reason as dmn_fd3d_watch_slot). */
struct ImportedFence {
    bool is_d3d12 = false;
    IUnknown* realFence = nullptr;      /* borrowed — an AddRef here would make
                                           destruction unobservable */
    dmn_shared_fence_t view = nullptr;  /* our own mapping of the slot */
    dmn_shared_fence_handle pod{};      /* fd = registry-owned dup (re-export +
                                           watcher views) */
    std::mutex mtx;                     /* guards raised + d11mt */
    uint64_t raised = 0;                /* highest value watcher-signaled into
                                           the local fence (Signal can rewind a
                                           fence; watchers must never) */
    ID3D11Multithread* d11mt = nullptr; /* AddRef'd; a watcher signaling on the
                                           immediate context races the app's
                                           thread without it */
};

std::unordered_map<void*, ImportedFence*> g_import_reg; /* identity -> import;
                                                           under g_fence_mtx */

ImportedFence* lookup_import(IUnknown* fence) {
    if (!fence)
        return nullptr;
    void* id = dmn_com_identity(fence);
    std::lock_guard<std::mutex> lk(g_fence_mtx);
    auto it = g_import_reg.find(id);
    return it == g_import_reg.end() ? nullptr : it->second;
}

/* Common import setup: slot view + owned fd dup. */
ImportedFence* import_open(const dmn_shared_fence_handle* pod, bool is_d3d12) {
    auto* imp = new (std::nothrow) ImportedFence();
    if (!imp)
        return nullptr;
    imp->is_d3d12 = is_d3d12;
    imp->view = dmn_shared_fence_open(pod);
    if (!imp->view) {
        delete imp;
        return nullptr;
    }
    imp->pod = *pod;
    imp->pod.fd = fcntl(pod->fd, F_DUPFD_CLOEXEC, 0);
    if (imp->pod.fd < 0) {
        DMN_ERROR("fence: import dup(fd=%d) failed", pod->fd);
        dmn_shared_fence_close(imp->view);
        delete imp;
        return nullptr;
    }
    return imp;
}

void register_import(IUnknown* fence, ImportedFence* imp) {
    imp->realFence = fence; /* borrowed, see ImportedFence */
    void* id = dmn_com_identity(fence);
    {
        std::lock_guard<std::mutex> lk(g_fence_mtx);
        g_import_reg[id] = imp;
    }
    DMN_INFO("fence: %s import identity=%p slot fd=%d",
             imp->is_d3d12 ? "D3D12" : "D3D11", id, imp->pod.fd);
}

/* Raise the local fence to `value` from a watcher, monotonically. Raises go
 * through the ORIGINAL Signal methods (dmn_hooks_*_signal_orig): a raise is
 * not an app signal — the value came FROM the slot, so mirroring it back is
 * pointless, and re-entering the hooks under imp->mtx would deadlock. */
void import_raise_d3d12(ImportedFence* imp, ID3D12Fence* f, UINT64 value) {
    std::lock_guard<std::mutex> lk(imp->mtx);
    if (value > imp->raised) {
        dmn_hooks_f12_signal_orig(f, value);
        imp->raised = value;
    }
}

void import_raise_d3d11(ImportedFence* imp, ID3D11DeviceContext4* c,
                        ID3D11Fence* f, UINT64 value) {
    std::lock_guard<std::mutex> lk(imp->mtx);
    if (value <= imp->raised || !imp->d11mt)
        return;
    imp->d11mt->Enter();
    dmn_hooks_ctx_signal_orig(c, f, value);
    c->Flush();
    imp->d11mt->Leave();
    imp->raised = value;
}

/* Local-completion wait for the signal-back watchers: poll the (hooked,
 * slot-merged) GetCompletedValue. The merge can only exit the loop early when
 * the slot already holds >= value — in which case the pending store is a
 * no-op anyway. */
template <class FenceT>
void wait_local_completion(FenceT* f, UINT64 value) {
    void* ev = dmn_event_create(1, 0);
    if (ev && SUCCEEDED(f->SetEventOnCompletion(value, ev))) {
        dmn_event_wait(ev, DMN_WAIT_INFINITE);
    } else {
        while (f->GetCompletedValue() < value) {
            struct timespec ns = {0, 200 * 1000};
            nanosleep(&ns, nullptr);
        }
    }
    if (ev)
        dmn_event_close(ev);
}

/* GPU signal-back: once the local fence completes `value`, store it into the
 * shared slot so the producer (and other consumers) observe it. */
template <class FenceT>
void arm_signal_store(ImportedFence* imp, FenceT* f, UINT64 value) {
    f->AddRef();
    std::thread([imp, f, value]() {
        wait_local_completion(f, value);
        dmn_shared_fence_signal(imp->view, value);
        f->Release();
    }).detach();
}

} // namespace

/* == Producer API ========================================================== */

bool dmn_fd3d_producer_create_d3d11(ID3D11Device5* dev, ID3D11Fence* fence,
                                    UINT64 initial, UINT flags) {
    auto* gf = new GpuFence();
    gf->is_d3d12 = false;
    gf->initial = initial;
    gf->flags = flags;
    if (!gpufence_setup_d3d11(dev, gf)) {
        gpufence_teardown(gf); /* releases whatever setup got to */
        return false;
    }
    gf->realFence = fence; /* borrowed, see GpuFence */
    open_producer_view(gf);
    register_producer(fence, gf);
    return true;
}

bool dmn_fd3d_producer_create_d3d12(ID3D12Device* dev, ID3D12Fence* fence,
                                    UINT64 initial, UINT flags) {
    auto* gf = new GpuFence();
    gf->is_d3d12 = true;
    gf->initial = initial;
    gf->flags = flags;
    if (!gpufence_setup_d3d12(dev, gf)) {
        gpufence_teardown(gf); /* releases whatever setup got to */
        return false;
    }
    gf->realFence = fence; /* borrowed, see GpuFence */
    open_producer_view(gf);
    register_producer(fence, gf);
    return true;
}

void dmn_fd3d_fence_destroy(void* identity) {
    GpuFence* gf = nullptr;
    ImportedFence* imp = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_fence_mtx);
        auto it = g_fence_reg.find(identity);
        if (it != g_fence_reg.end()) {
            gf = it->second;
            g_fence_reg.erase(it);
        }
        auto ii = g_import_reg.find(identity);
        if (ii != g_import_reg.end()) {
            imp = ii->second;
            g_import_reg.erase(ii);
        }
    }
    if (gf) {
        gpufence_teardown(gf);
        DMN_INFO("fence: destroyed producer identity=%p", identity);
    }
    if (imp) {
        if (imp->d11mt)
            imp->d11mt->Release();
        dmn_shared_fence_close(imp->view);
        if (imp->pod.fd >= 0)
            close(imp->pod.fd);
        delete imp;
        DMN_INFO("fence: destroyed import identity=%p", identity);
    }
}

ID3D11DeviceContext* dmn_fd3d_producer_d3d11_ctx(ID3D11Fence* fence) {
    GpuFence* gf = lookup_producer(fence);
    return (gf && !gf->is_d3d12) ? gf->d11ctx : nullptr;
}

/* The slot pod + live view of a fence that is a producer or an import; false
 * for foreign fences. One identity resolution and one lock — this sits on the
 * hooked GetCompletedValue path, which apps spin-poll. */
static bool fence_slot(IUnknown* fence, dmn_shared_fence_handle* pod,
                       dmn_shared_fence_t* view) {
    if (!fence)
        return false;
    void* id = dmn_com_identity(fence);
    GpuFence* gf = nullptr;
    ImportedFence* imp = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_fence_mtx);
        auto it = g_fence_reg.find(id);
        if (it != g_fence_reg.end()) {
            gf = it->second;
        } else {
            auto ii = g_import_reg.find(id);
            if (ii != g_import_reg.end())
                imp = ii->second;
        }
    }
    if (gf) {
        pod->magic = DMN_SHARED_FENCE_MAGIC;
        pod->version = DMN_SHARED_HANDLE_VERSION;
        pod->fd = gf->bufPod.fd;
        pod->flags = gf->flags;
        pod->initial_value = gf->initial;
        *view = gf->view;
        return true;
    }
    if (imp) {
        *pod = imp->pod;
        *view = imp->view;
        return true;
    }
    return false;
}

bool dmn_fd3d_export(IUnknown* fence, dmn_shared_fence_handle* out) {
    dmn_shared_fence_t view = nullptr;
    return out && fence_slot(fence, out, &view);
}

void dmn_fd3d_on_ctx_signal(ID3D11Fence* fence, UINT64 value) {
    GpuFence* gf = lookup_producer(fence);
    if (gf && !gf->is_d3d12) {
        DMN_INFO("fence: D3D11 Signal value=%llu -> GPU store",
                 (unsigned long long)value);
        gpufence_signal_d3d11(gf, value);
    }
}

void dmn_fd3d_after_ctx_signal(ID3D11Fence* fence, UINT64 value) {
    ImportedFence* imp = lookup_import(reinterpret_cast<IUnknown*>(fence));
    if (imp && !imp->is_d3d12)
        arm_signal_store(imp, fence, value);
}

void dmn_fd3d_on_queue_signal(ID3D12Fence* fence, UINT64 value) {
    if (GpuFence* gf = lookup_producer(fence)) {
        if (gf->is_d3d12)
            gpufence_signal_d3d12(gf, value);
        return;
    }
    ImportedFence* imp = lookup_import(reinterpret_cast<IUnknown*>(fence));
    if (imp && imp->is_d3d12)
        arm_signal_store(imp, fence, value);
}

void dmn_fd3d_on_cpu_signal(IUnknown* fence, UINT64 value) {
    dmn_shared_fence_handle pod{};
    dmn_shared_fence_t view = nullptr;
    if (!fence_slot(fence, &pod, &view) || !view)
        return;
    dmn_shared_fence_signal(view, value);
    /* The local fence now holds `value`; watchers must not signal it lower. */
    if (ImportedFence* imp = lookup_import(fence)) {
        std::lock_guard<std::mutex> lk(imp->mtx);
        if (value > imp->raised)
            imp->raised = value;
    }
}

UINT64 dmn_fd3d_completed_merge(IUnknown* fence, UINT64 from_fence) {
    dmn_shared_fence_handle pod{};
    dmn_shared_fence_t view = nullptr;
    if (!fence_slot(fence, &pod, &view) || !view)
        return from_fence;
    UINT64 slot = dmn_shared_fence_get_completed(view);
    return slot > from_fence ? slot : from_fence;
}

bool dmn_fd3d_watch_slot(IUnknown* fence, UINT64 value, HANDLE event) {
    dmn_shared_fence_handle pod{};
    dmn_shared_fence_t view = nullptr;
    if (!event || !fence_slot(fence, &pod, &view) || !view)
        return false;
    if (dmn_shared_fence_get_completed(view) >= value) {
        dmn_event_signal(event);
        return true;
    }
    /* The watcher gets its own mapping of the slot: the registered view is
     * torn down when the fence is destroyed, which must not yank the mapping
     * from under a thread that legitimately outlives it. */
    dmn_shared_fence_t watch = dmn_shared_fence_open(&pod);
    if (!watch)
        return false;
    std::thread([watch, value, event]() {
        dmn_shared_fence_wait(watch, value, DMN_WAIT_INFINITE);
        dmn_event_signal(event);
        dmn_shared_fence_close(watch);
    }).detach();
    return true;
}


namespace {

/* Failed import setup before registration: release what import_open built. */
void import_abandon(ImportedFence* imp) {
    dmn_shared_fence_close(imp->view);
    if (imp->pod.fd >= 0)
        close(imp->pod.fd);
    delete imp;
}

} // namespace

HRESULT dmn_fd3d_import_d3d12(ID3D12Device* dev,
                              const dmn_shared_fence_handle* pod,
                              ID3D12Fence** out) {
    if (!dev || !pod || !out)
        return E_INVALIDARG;
    ImportedFence* imp = import_open(pod, /*is_d3d12=*/true);
    if (!imp)
        return E_FAIL;
    /* Local value starts at the slot's current merged value; later slot
     * growth arrives via the completed-merge hook and the wait watchers. */
    UINT64 init = dmn_shared_fence_get_completed(imp->view);
    ID3D12Fence* f = nullptr;
    HRESULT hr = dev->CreateFence(init, D3D12_FENCE_FLAG_NONE,
                                  __uuidof(ID3D12Fence),
                                  reinterpret_cast<void**>(&f));
    if (FAILED(hr) || !f) {
        DMN_ERROR("fence: import CreateFence 0x%08x", (unsigned)hr);
        import_abandon(imp);
        return FAILED(hr) ? hr : E_FAIL;
    }
    imp->raised = init;
    register_import(reinterpret_cast<IUnknown*>(f), imp);
    *out = f; /* +1 from CreateFence; caller owns */
    return S_OK;
}

HRESULT dmn_fd3d_import_d3d11(ID3D11Device5* dev,
                              const dmn_shared_fence_handle* pod,
                              ID3D11Fence** out) {
    if (!dev || !pod || !out)
        return E_INVALIDARG;
    ImportedFence* imp = import_open(pod, /*is_d3d12=*/false);
    if (!imp)
        return E_FAIL;
    UINT64 init = dmn_shared_fence_get_completed(imp->view);
    ID3D11Fence* f = nullptr;
    HRESULT hr = dev->CreateFence(init, D3D11_FENCE_FLAG_NONE,
                                  __uuidof(ID3D11Fence),
                                  reinterpret_cast<void**>(&f));
    if (FAILED(hr) || !f) {
        DMN_ERROR("fence: D3D11 import CreateFence 0x%08x", (unsigned)hr);
        import_abandon(imp);
        return FAILED(hr) ? hr : E_FAIL;
    }
    imp->raised = init;
    register_import(reinterpret_cast<IUnknown*>(f), imp);
    *out = f; /* +1 from CreateFence; caller owns */
    return S_OK;
}

void dmn_fd3d_before_queue_wait(ID3D12Fence* fence, UINT64 value) {
    ImportedFence* imp = lookup_import(reinterpret_cast<IUnknown*>(fence));
    if (!imp || !imp->is_d3d12)
        return;
    if (dmn_shared_fence_get_completed(imp->view) >= value) {
        import_raise_d3d12(imp, fence, value);
        return;
    }
    dmn_shared_fence_t watch = dmn_shared_fence_open(&imp->pod);
    if (!watch) {
        DMN_WARN("fence: import wait watcher view failed; the wait may stall");
        return;
    }
    fence->AddRef();
    std::thread([imp, fence, value, watch]() {
        dmn_shared_fence_wait(watch, value, DMN_WAIT_INFINITE);
        dmn_shared_fence_close(watch);
        import_raise_d3d12(imp, fence, value);
        fence->Release();
    }).detach();
}

void dmn_fd3d_before_ctx_wait(ID3D11DeviceContext4* c, ID3D11Fence* fence,
                              UINT64 value) {
    ImportedFence* imp = lookup_import(reinterpret_cast<IUnknown*>(fence));
    if (!imp || imp->is_d3d12 || !c)
        return;
    {
        std::lock_guard<std::mutex> lk(imp->mtx);
        if (!imp->d11mt) {
            ID3D11Multithread* mt = nullptr;
            if (SUCCEEDED(c->QueryInterface(__uuidof(ID3D11Multithread),
                                            reinterpret_cast<void**>(&mt))) && mt) {
                mt->SetMultithreadProtected(TRUE);
                imp->d11mt = mt;
                DMN_INFO("fence: enabled ID3D11Multithread protection for "
                         "imported-fence waits");
            }
        }
        if (!imp->d11mt) {
            DMN_WARN("fence: no ID3D11Multithread; imported-fence wait may stall");
            return;
        }
    }
    if (dmn_shared_fence_get_completed(imp->view) >= value) {
        import_raise_d3d11(imp, c, fence, value);
        return;
    }
    dmn_shared_fence_t watch = dmn_shared_fence_open(&imp->pod);
    if (!watch) {
        DMN_WARN("fence: import wait watcher view failed; the wait may stall");
        return;
    }
    c->AddRef();
    fence->AddRef();
    std::thread([imp, c, fence, value, watch]() {
        dmn_shared_fence_wait(watch, value, DMN_WAIT_INFINITE);
        dmn_shared_fence_close(watch);
        import_raise_d3d11(imp, c, fence, value);
        c->Release();
        fence->Release();
    }).detach();
}
