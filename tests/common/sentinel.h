/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Destruction watch: an IUnknown planted in a D3D object's private-data slot
 * whose final Release is observable, so a test can tell exactly when the object
 * was destroyed rather than guessing from a side effect.
 *
 * This is the same mechanism dmn_com_hooks.cpp evicts registry entries with, so
 * a test that plants one is asking the question the library's own bookkeeping
 * asks: did this object actually die?
 *
 * A resource and its DXGI view share one private-data store, so plant on the
 * resource — planting on both would have the second overwrite the first.
 *
 * The watch outlives the sentinel deliberately. A destruction that lands after
 * a test gave up must have somewhere valid to write, so t_watch_new() heap-
 * allocates and nothing frees it on the failure path.
 */

#pragma once

#include <stdio.h>
#include <time.h>

#include <atomic>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>

/* {41D7E2A8-53BC-4E19-9F60-2C8A1B4D7E05} — test-only private-data slot,
 * distinct from the library's own eviction sentinel. */
static const GUID kTWatchGuid = {0x41d7e2a8, 0x53bc, 0x4e19,
                                 {0x9f, 0x60, 0x2c, 0x8a, 0x1b, 0x4d, 0x7e, 0x05}};

struct TWatch {
    std::atomic<bool> fired;
};

namespace t_watch_detail {

struct Sentinel {
    void** vtbl;
    std::atomic<ULONG> refs;
    TWatch* watch;
};

inline HRESULT STDMETHODCALLTYPE s_QueryInterface(Sentinel* self, REFIID riid,
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

inline ULONG STDMETHODCALLTYPE s_AddRef(Sentinel* self) {
    return self->refs.fetch_add(1) + 1;
}

inline ULONG STDMETHODCALLTYPE s_Release(Sentinel* self) {
    ULONG r = self->refs.fetch_sub(1) - 1;
    if (r == 0) {
        self->watch->fired.store(true, std::memory_order_release);
        delete self;
    }
    return r;
}

/* One shared vtable: the sentinel carries all its per-instance state. */
inline void** vtbl() {
    static void* v[3] = {(void*)s_QueryInterface, (void*)s_AddRef,
                         (void*)s_Release};
    return v;
}

} // namespace t_watch_detail

enum TPlantResult {
    T_PLANT_OK = 0,      /* stored, and the slot holds the only reference */
    T_PLANT_NO_SLOT,     /* no private-data interface, or the store failed */
    T_PLANT_STUBBED,     /* returned S_OK without retaining anything */
};

/* Never freed on the failure paths — see the header comment. */
inline TWatch* t_watch_new() {
    auto* w = new TWatch();
    w->fired.store(false, std::memory_order_relaxed);
    return w;
}

inline bool t_watch_fired(const TWatch* w) {
    return w->fired.load(std::memory_order_acquire);
}

/* Destruction can be deferred off-thread (the D3D runtime holds objects until
 * the GPU is done with them), so waiting is the only honest way to observe it. */
inline bool t_watch_wait(const TWatch* w, unsigned ms) {
    for (unsigned i = 0; i < ms && !t_watch_fired(w); i++) {
        struct timespec ns = {0, 1000 * 1000};
        nanosleep(&ns, nullptr);
    }
    return t_watch_fired(w);
}

/* `set_pdi(obj, sentinel)` stores the sentinel in obj's private-data slot; the
 * caller supplies it because ID3D11DeviceChild and ID3D12Object declare
 * SetPrivateDataInterface separately. */
template <class SetPDI>
inline TPlantResult t_watch_plant(TWatch* w, IUnknown* obj, SetPDI set_pdi) {
    auto* s = new t_watch_detail::Sentinel{t_watch_detail::vtbl(), {1}, w};
    HRESULT hr = set_pdi(obj, reinterpret_cast<IUnknown*>(s));
    if (FAILED(hr)) {
        t_watch_detail::s_Release(s);
        return T_PLANT_NO_SLOT;
    }
    /* Hand the slot sole ownership: from here the sentinel dies exactly when
     * the object does. */
    t_watch_detail::s_Release(s);
    return t_watch_fired(w) ? T_PLANT_STUBBED : T_PLANT_OK;
}

inline TPlantResult t_watch_plant_d3d11(TWatch* w, IUnknown* obj) {
    return t_watch_plant(w, obj, [](IUnknown* v, IUnknown* s) -> HRESULT {
        ID3D11DeviceChild* c = nullptr;
        if (FAILED(v->QueryInterface(__uuidof(ID3D11DeviceChild), (void**)&c)) ||
            !c)
            return E_NOINTERFACE;
        HRESULT hr = c->SetPrivateDataInterface(kTWatchGuid, s);
        c->Release();
        return hr;
    });
}

inline TPlantResult t_watch_plant_d3d12(TWatch* w, IUnknown* obj) {
    return t_watch_plant(w, obj, [](IUnknown* v, IUnknown* s) -> HRESULT {
        ID3D12Object* o = nullptr;
        if (FAILED(v->QueryInterface(__uuidof(ID3D12Object), (void**)&o)) || !o)
            return E_NOINTERFACE;
        HRESULT hr = o->SetPrivateDataInterface(kTWatchGuid, s);
        o->Release();
        return hr;
    });
}

inline const char* t_plant_str(TPlantResult r) {
    switch (r) {
    case T_PLANT_OK:      return "ok";
    case T_PLANT_NO_SLOT: return "no private-data slot";
    case T_PLANT_STUBBED: return "slot returned S_OK but retained nothing";
    }
    return "?";
}
