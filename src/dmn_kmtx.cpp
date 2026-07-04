/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Cross-process keyed mutex implementation. See dmn_kmtx.h.
 */

#include <sched.h>
#include <time.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <new>
#include <unordered_map>

#include <dxgi1_2.h>

#include "dmn_hook.h"
#include "dmn_kmtx.h"
#include "dmn_log.h"
#include "dmn_share.h"

namespace {

constexpr uint32_t kKmtxMagic = 0x584d4b44u; /* 'DKMX' */

/* The shared state page. Fields are only touched under `lock` (a cross-process
 * spinlock word taken with atomic exchange); waiting is poll+backoff. */
struct KmtxShared {
    uint32_t magic;
    volatile uint32_t lock;     /* 0 free, 1 held */
    volatile uint64_t key;      /* key it was last released with */
    volatile uint32_t acquired; /* 0 released, 1 acquired */
};

void kmtx_lock(KmtxShared* s) {
    while (__atomic_exchange_n(&s->lock, 1u, __ATOMIC_ACQUIRE) != 0)
        sched_yield();
}

void kmtx_unlock(KmtxShared* s) {
    __atomic_store_n(&s->lock, 0u, __ATOMIC_RELEASE);
}

uint64_t now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

/* == The vended COM object ================================================ */
/* IDXGIKeyedMutex : IDXGIDeviceSubObject : IDXGIObject : IUnknown. */
struct VendedKmtx {
    void** vtbl;
    std::atomic<ULONG> refs;
    KmtxShared* state;   /* inside `map` */
    void* map;
    size_t map_size;
};

HRESULT STDMETHODCALLTYPE km_QueryInterface(VendedKmtx* self, REFIID riid, void** ppv) {
    if (!ppv)
        return E_POINTER;
    if (dmn_iid_eq(riid, __uuidof(IUnknown)) ||
        dmn_iid_eq(riid, __uuidof(IDXGIObject)) ||
        dmn_iid_eq(riid, __uuidof(IDXGIDeviceSubObject)) ||
        dmn_iid_eq(riid, __uuidof(IDXGIKeyedMutex))) {
        self->refs.fetch_add(1, std::memory_order_relaxed);
        *ppv = self;
        return S_OK;
    }
    *ppv = nullptr;
    return E_NOINTERFACE;
}
ULONG STDMETHODCALLTYPE km_AddRef(VendedKmtx* self) {
    return self->refs.fetch_add(1, std::memory_order_relaxed) + 1;
}
ULONG STDMETHODCALLTYPE km_Release(VendedKmtx* self) {
    /* The registry holds one reference until dmn_kmtx_unregister (texture
     * destruction); destruction happens when the app's refs also drain. */
    ULONG r = self->refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (r == 0) {
        dmn_share_unmap(self->map, self->map_size);
        delete self;
    }
    return r;
}
HRESULT STDMETHODCALLTYPE km_SetPrivateData(VendedKmtx*, REFGUID, UINT, const void*) { return S_OK; }
HRESULT STDMETHODCALLTYPE km_SetPrivateDataInterface(VendedKmtx*, REFGUID, const IUnknown*) { return S_OK; }
HRESULT STDMETHODCALLTYPE km_GetPrivateData(VendedKmtx*, REFGUID, UINT*, void*) { return E_NOTIMPL; }
HRESULT STDMETHODCALLTYPE km_GetParent(VendedKmtx*, REFIID, void** ppv) {
    if (ppv) *ppv = nullptr;
    return E_NOTIMPL;
}
HRESULT STDMETHODCALLTYPE km_GetDevice(VendedKmtx*, REFIID, void** ppv) {
    if (ppv) *ppv = nullptr;
    return E_NOTIMPL;
}

HRESULT STDMETHODCALLTYPE km_AcquireSync(VendedKmtx* self, UINT64 key, DWORD ms) {
    KmtxShared* s = self->state;
    const bool infinite = (ms == 0xffffffffu);
    const uint64_t deadline = infinite ? 0 : now_ms() + ms;
    uint32_t spins = 0;
    for (;;) {
        kmtx_lock(s);
        if (!s->acquired && s->key == key) {
            s->acquired = 1;
            kmtx_unlock(s);
            return S_OK;
        }
        kmtx_unlock(s);
        if (!infinite && now_ms() >= deadline)
            return (HRESULT)WAIT_TIMEOUT;
        if (spins < 128) {
            sched_yield();
            spins++;
        } else {
            struct timespec ns = {0, 200 * 1000}; /* 0.2 ms */
            nanosleep(&ns, nullptr);
        }
    }
}

HRESULT STDMETHODCALLTYPE km_ReleaseSync(VendedKmtx* self, UINT64 key) {
    KmtxShared* s = self->state;
    kmtx_lock(s);
    if (!s->acquired) {
        kmtx_unlock(s);
        return DXGI_ERROR_INVALID_CALL;
    }
    s->acquired = 0;
    s->key = key;
    kmtx_unlock(s);
    return S_OK;
}

void* g_km_vtbl[10] = {
    (void*)km_QueryInterface, (void*)km_AddRef, (void*)km_Release,
    (void*)km_SetPrivateData, (void*)km_SetPrivateDataInterface,
    (void*)km_GetPrivateData, (void*)km_GetParent, (void*)km_GetDevice,
    (void*)km_AcquireSync, (void*)km_ReleaseSync,
};

std::mutex g_km_mtx;
std::unordered_map<void*, VendedKmtx*> g_km_reg; /* texture identity -> mutex */

} // namespace

bool dmn_kmtx_register(void* texture_identity, int fd, uint64_t offset, bool init) {
    if (!texture_identity || fd < 0)
        return false;
    {
        std::lock_guard<std::mutex> lk(g_km_mtx);
        if (g_km_reg.count(texture_identity))
            return true;
    }
    size_t map_size = (size_t)offset + dmn_share_page_align(sizeof(KmtxShared));
    void* map = dmn_share_map_fd(fd, map_size);
    if (!map) {
        DMN_ERROR("kmtx: map(fd=%d, %zu) failed", fd, map_size);
        return false;
    }
    auto* s = reinterpret_cast<KmtxShared*>(static_cast<char*>(map) + offset);
    if (init) {
        s->lock = 0;
        s->key = 0;
        s->acquired = 0;
        __atomic_store_n(&s->magic, kKmtxMagic, __ATOMIC_RELEASE);
    } else if (__atomic_load_n(&s->magic, __ATOMIC_ACQUIRE) != kKmtxMagic) {
        DMN_ERROR("kmtx: state page missing magic (producer without keyed "
                  "mutex, or layout drift)");
        dmn_share_unmap(map, map_size);
        return false;
    }
    auto* km = new (std::nothrow) VendedKmtx();
    if (!km) {
        dmn_share_unmap(map, map_size);
        return false;
    }
    km->vtbl = g_km_vtbl;
    km->refs.store(1, std::memory_order_relaxed); /* the registry's reference */
    km->state = s;
    km->map = map;
    km->map_size = map_size;
    {
        std::lock_guard<std::mutex> lk(g_km_mtx);
        g_km_reg[texture_identity] = km;
    }
    DMN_INFO("kmtx: %s keyed mutex for texture %p (fd=%d off=%llu)",
             init ? "created" : "opened", texture_identity, fd,
             (unsigned long long)offset);
    return true;
}

IDXGIKeyedMutex* dmn_kmtx_lookup(void* texture_identity) {
    std::lock_guard<std::mutex> lk(g_km_mtx);
    auto it = g_km_reg.find(texture_identity);
    if (it == g_km_reg.end())
        return nullptr;
    km_AddRef(it->second);
    return reinterpret_cast<IDXGIKeyedMutex*>(it->second);
}

void dmn_kmtx_unregister(void* texture_identity) {
    VendedKmtx* km = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_km_mtx);
        auto it = g_km_reg.find(texture_identity);
        if (it == g_km_reg.end())
            return;
        km = it->second;
        g_km_reg.erase(it);
    }
    km_Release(km); /* the registry's reference from dmn_kmtx_register */
    DMN_INFO("kmtx: unregistered keyed mutex for texture %p", texture_identity);
}
