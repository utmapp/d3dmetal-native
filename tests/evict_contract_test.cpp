/*
 * D3DMetal private-data eviction contract: the registry eviction in
 * dmn_com_hooks.cpp plants a sentinel IUnknown on every tracked object via
 * SetPrivateDataInterface and relies on D3D releasing it at object
 * destruction. This test proves the vendored D3DMetal.framework honors that
 * contract for both APIs (stores the interface — i.e. actually AddRefs it,
 * not a stubbed S_OK — and releases it when the object dies).
 *
 * There is no runtime re-probe; if a D3DMetal update ever breaks the
 * contract, this test fails — not the field.
 *
 * Prints "EVICT-CONTRACT: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>
#include <time.h>

#include <atomic>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>

#include "d3dmetal_native.h"
#include "common/com.h"

namespace {

/* {2B9C0D51-6E87-4F3A-A0C4-91D5B7E82F13} — test-only private-data slot. */
const GUID kProbeGuid = {0x2b9c0d51, 0x6e87, 0x4f3a,
                         {0xa0, 0xc4, 0x91, 0xd5, 0xb7, 0xe8, 0x2f, 0x13}};

struct Sentinel {
    void** vtbl;
    std::atomic<ULONG> refs;
    std::atomic<bool>* hit; /* raised on final release */
};

HRESULT STDMETHODCALLTYPE s_QueryInterface(Sentinel* self, REFIID riid,
                                           void** ppv) {
    if (!ppv)
        return E_POINTER;
    const GUID iid_unknown = __uuidof(IUnknown);
    if (!memcmp(&riid, &iid_unknown, sizeof(GUID))) {
        self->refs.fetch_add(1);
        *ppv = self;
        return S_OK;
    }
    *ppv = nullptr;
    return E_NOINTERFACE;
}
ULONG STDMETHODCALLTYPE s_AddRef(Sentinel* self) {
    return self->refs.fetch_add(1) + 1;
}
ULONG STDMETHODCALLTYPE s_Release(Sentinel* self) {
    ULONG r = self->refs.fetch_sub(1) - 1;
    if (r == 0) {
        self->hit->store(true, std::memory_order_release);
        delete self;
    }
    return r;
}

void* g_vtbl[3] = {(void*)s_QueryInterface, (void*)s_AddRef, (void*)s_Release};

/* Consume `victim`: plant a sentinel in its private-data slot, drop every
 * reference, and report whether the slot (a) actually held a reference and
 * (b) released it when the object was destroyed. */
template <class SetPDI>
bool probe(IUnknown* victim, SetPDI set_pdi, const char* what) {
    /* Heap-allocated so a sentinel surviving past a timeout (deferred
     * destruction that never lands) has nothing dangling to write to. */
    auto* hit = new std::atomic<bool>(false);
    auto* s = new Sentinel{g_vtbl, {1}, hit};
    HRESULT hr = set_pdi(victim, reinterpret_cast<IUnknown*>(s));
    if (FAILED(hr)) {
        fprintf(stderr, "EVICT-CONTRACT: %s SetPrivateDataInterface FAILED "
                "0x%08x\n", what, (unsigned)hr);
        s_Release(s);
        victim->Release();
        return false;
    }
    s_Release(s); /* the private-data slot must hold the surviving ref */
    if (hit->load(std::memory_order_acquire)) {
        fprintf(stderr, "EVICT-CONTRACT: %s slot returned S_OK but held no "
                "reference (stubbed)\n", what);
        victim->Release();
        return false;
    }
    victim->Release();
    /* Destruction may be deferred off-thread; a test can afford to wait. */
    for (int i = 0; i < 5000 && !hit->load(std::memory_order_acquire); i++) {
        struct timespec ns = {0, 1000 * 1000}; /* 1 ms */
        nanosleep(&ns, nullptr);
    }
    if (!hit->load(std::memory_order_acquire)) {
        fprintf(stderr, "EVICT-CONTRACT: %s sentinel not released within 5s "
                "of destruction\n", what);
        return false; /* hit intentionally leaked: a late release may still land */
    }
    delete hit;
    printf("EVICT-CONTRACT: %s release-on-destroy ok\n", what);
    return true;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "EVICT-CONTRACT: dmn_init FAILED\n");
        return 1;
    }

    bool ok11 = false, ok12 = false;

    {
        Com<ID3D11Device> dev;
        Com<ID3D11DeviceContext> ctx;
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
        if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                     0, &fl, 1, D3D11_SDK_VERSION, &dev, &flo,
                                     &ctx))) {
            fprintf(stderr, "EVICT-CONTRACT: D3D11CreateDevice FAILED\n");
            return 1;
        }
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = 16;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        ID3D11Buffer* victim = nullptr;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, &victim)) || !victim) {
            fprintf(stderr, "EVICT-CONTRACT: CreateBuffer FAILED\n");
            return 1;
        }
        ok11 = probe(victim, [](IUnknown* v, IUnknown* s) {
            ID3D11DeviceChild* c = nullptr;
            if (FAILED(v->QueryInterface(__uuidof(ID3D11DeviceChild),
                                         (void**)&c)) || !c)
                return E_NOINTERFACE;
            HRESULT hr = c->SetPrivateDataInterface(kProbeGuid, s);
            c->Release();
            return hr;
        }, "D3D11 buffer");
    }

    {
        Com<ID3D12Device> dev;
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                     __uuidof(ID3D12Device), (void**)&dev))) {
            fprintf(stderr, "EVICT-CONTRACT: D3D12CreateDevice FAILED\n");
            return 1;
        }
        ID3D12Fence* victim = nullptr;
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                    __uuidof(ID3D12Fence), (void**)&victim)) ||
            !victim) {
            fprintf(stderr, "EVICT-CONTRACT: CreateFence FAILED\n");
            return 1;
        }
        ok12 = probe(victim, [](IUnknown* v, IUnknown* s) {
            ID3D12Object* o = nullptr;
            if (FAILED(v->QueryInterface(__uuidof(ID3D12Object),
                                         (void**)&o)) || !o)
                return E_NOINTERFACE;
            HRESULT hr = o->SetPrivateDataInterface(kProbeGuid, s);
            o->Release();
            return hr;
        }, "D3D12 fence");
    }

    if (!ok11 || !ok12) {
        fprintf(stderr, "EVICT-CONTRACT: FAIL\n");
        return 1;
    }
    printf("EVICT-CONTRACT: PASS\n");
    return 0;
}
