/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * com_contract_test: the COM invariants the whole sharing registry rests on,
 * asserted directly against dmn_com_identity() rather than inferred from a
 * downstream symptom.
 *
 * Every registry in dmn_com_hooks.cpp / dmn_fence_d3d.cpp / dmn_kmtx.cpp is
 * keyed on dmn_com_identity(), and a shared resource is reached through more
 * than one interface: the app holds an ID3D11Texture2D, but GetSharedHandle,
 * CreateSharedHandle and the keyed-mutex QueryInterface all arrive on a DXGI
 * *view* of it — a separate object, minted fresh by each QueryInterface. Three
 * things therefore have to hold, and a shipped D3DMetal has broken each:
 *
 *   1. Identity is canonical. The value resolved from a view must equal the one
 *      resolved from the resource, or the lookup misses and the export falls
 *      through to D3DMetal's own stub ("Unsupported: IDXGIResource::
 *      GetSharedHandle"). GPTk 4.0b1 returns the *view* from
 *      QueryInterface(IID_IUnknown), which is what broke this.
 *
 *   2. Identity resolution is reference-neutral. It has to give back the
 *      reference its QueryInterface took, to whichever object took it — GPTk
 *      4.0b1's views AddRef *themselves* while handing back the resource. Give
 *      it back to the wrong object and the resource is destroyed under the app
 *      (its next Release faults); give none back and the view lives forever and
 *      pins the resource, so it is never destroyed and its fd and shared
 *      mapping leak.
 *
 *   3. Destruction-driven eviction reclaims. A registry entry and its fd are
 *      released when the object dies, so anything that keeps the object alive
 *      silently disables reclamation.
 *
 * (3) is checked on shared BUFFERS and FENCES, not textures, and that is
 * deliberate: every shipped D3DMetal — 1.0 through 4.0b1 — pools textures, so a
 * released ID3D11Texture2D or ID3D12Resource of TEXTURE2D dimension is NOT
 * destroyed at the app's last Release (measured: a plain, entirely
 * un-intercepted texture is not destroyed within 5s on any of the four). Only
 * 4.0b1 destroys a *shared* one, presumably because the substituted linear
 * backing makes it unpoolable. Buffers and fences are destroyed promptly on all
 * four, so they are what this test holds the library to. Retention of pooled
 * textures is bounded by the pool rather than a leak, and lifecycle-churn is
 * where that is measured.
 *
 * Windowless, single process, no pixels. Prints "COM-CONTRACT: PASS".
 */

#define T_TAG "COM-CONTRACT"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include "d3dmetal_native.h"
#include "common/check.h"
#include "common/com.h"
#include "common/sentinel.h"
#include "common/util.h"

/* The library's own identity resolution, exactly as its registries call it. */
#include "dmn_hook.h"

namespace {

constexpr uint32_t kW = 256, kH = 256;

/* How many times the reference-neutrality check resolves an identity. One
 * leaked or double-dropped reference per resolution has to be visible, and a
 * single resolution cannot tell "balanced" from "off by one each way". */
constexpr int kResolveReps = 64;

/* Destruction can be deferred off-thread — the runtime holds an object until the
 * GPU is done with it — so observing it means waiting. Generous on purpose: the
 * wait ends as soon as the thing happens, so the length only costs anything when
 * the test is about to fail, and being stingy here made it fail under suite load
 * on a framework it passes on when run alone. */
constexpr unsigned kDeathWaitMs = 30000;

HRESULT make_shared_texture(ID3D11Device* dev, Com<ID3D11Texture2D>& out) {
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    return dev->CreateTexture2D(&td, nullptr, &out);
}

/* == 1 + 2: identity through a DXGI view ================================== */
int check_identity_through_views(ID3D11Device* dev) {
    /* A MISC_SHARED create is what registers the resource and, on the way,
     * teaches the library how this D3DMetal's views count references — so what
     * follows runs against a fully armed library, as the field does. */
    Com<ID3D11Texture2D> tex;
    CK(make_shared_texture(dev, tex), "CreateTexture2D(MISC_SHARED)");

    void* id_res = dmn_com_identity(tex.ptr());
    EXPECT(id_res != nullptr, "identity of a shared texture is null");
    EXPECT(dmn_com_identity(tex.ptr()) == id_res,
           "identity of a resource is not stable across queries");

    /* Two independently queried views: each QueryInterface mints a new object on
     * some versions, so both must still resolve to the resource. */
    Com<IDXGIResource> v1, v2;
    CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&v1),
       "QI(IDXGIResource) #1");
    CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&v2),
       "QI(IDXGIResource) #2");

    void* id_v1 = dmn_com_identity(v1.ptr());
    void* id_v2 = dmn_com_identity(v2.ptr());
    if (id_v1 != id_res || id_v2 != id_res) {
        fprintf(stderr, T_TAG ": identity differs by interface: resource=%p "
                "view1=%p view2=%p — every registry lookup arriving through a "
                "DXGI view will miss, and the export falls through to "
                "D3DMetal's stub\n", id_res, id_v1, id_v2);
        return 1;
    }
    /* Called out separately: a view resolving to *itself* is the exact 4.0b1
     * fault, and saying so beats "not equal". Only meaningful where the view is
     * a distinct object. */
    if ((void*)v1.ptr() != id_res)
        EXPECT(id_v1 != (void*)v1.ptr(),
               "a DXGI view resolved to itself instead of the resource it fronts");
    if ((void*)v2.ptr() != id_res)
        EXPECT(id_v2 != (void*)v2.ptr(),
               "a DXGI view resolved to itself instead of the resource it fronts");
    printf(T_TAG ": identity canonical through DXGI views: OK\n");

    /* Reference neutrality, watched on both objects a view's QueryInterface
     * might charge: the resource it hands back, and the view itself. */
    const unsigned long res_before = dmn_com_refs(tex.ptr());
    const unsigned long view_before = dmn_com_refs(v1.ptr());
    for (int i = 0; i < kResolveReps; i++) {
        EXPECT(dmn_com_identity(v1.ptr()) == id_res,
               "identity through a view changed under repetition");
        EXPECT(dmn_com_identity(tex.ptr()) == id_res,
               "identity of a resource changed under repetition");
    }
    const unsigned long res_after = dmn_com_refs(tex.ptr());
    const unsigned long view_after = dmn_com_refs(v1.ptr());
    if (res_after != res_before || view_after != view_before) {
        fprintf(stderr, T_TAG ": resolving identity %d times moved reference "
                "counts: resource %lu -> %lu, view %lu -> %lu (a leak pins the "
                "resource and its fd; an over-release destroys it under the "
                "app)\n", kResolveReps, res_before, res_after, view_before,
                view_after);
        return 1;
    }
    printf(T_TAG ": identity resolution reference-neutral over %d "
           "resolutions: OK\n", kResolveReps);

    /* The functional consequence of (1): a legacy export is cached per
     * resource, so asking either view must give the same answer. */
    HANDLE h1 = nullptr, h2 = nullptr;
    CK(v1->GetSharedHandle(&h1), "GetSharedHandle via a view");
    CK(v2->GetSharedHandle(&h2), "GetSharedHandle via a second view");
    EXPECT(h1 && h1 == h2,
           "two views of one resource vended different legacy handles");
    EXPECT(((const dmn_shared_texture_handle*)h1)->magic ==
               DMN_SHARED_TEXTURE_MAGIC,
           "GetSharedHandle through a view did not return our POD");
    printf(T_TAG ": legacy export agrees through either view: OK\n");
    return 0;
}

/* == 3: destruction reclaims, on the kinds that are destroyed ============== */

/* One create/watch/release cycle. `fds_taken` reports how many fds the resource
 * held while alive, `reclaimed` whether the count came back to where it started.
 * Returns false only when the object was not destroyed at all. */
template <class Create, class Plant>
bool destruction_cycle(const char* what, Create create, Plant plant,
                       int* fds_taken, bool* reclaimed) {
    sleep_ms(300); /* let earlier deferred destruction settle */
    const int fds_before = t_count_fds();

    IUnknown* obj = create();
    if (!obj) {
        fprintf(stderr, T_TAG ": %s: shared create failed\n", what);
        return false;
    }

    TWatch* w = t_watch_new();
    TPlantResult pr = plant(w, obj);
    if (pr != T_PLANT_OK) {
        /* evict-contract owns this contract; here it is only a precondition. */
        fprintf(stderr, T_TAG ": %s: cannot watch destruction (%s)\n", what,
                t_plant_str(pr));
        obj->Release();
        return false;
    }

    *fds_taken = t_count_fds() - fds_before;
    obj->Release();
    if (!t_watch_wait(w, kDeathWaitMs)) {
        fprintf(stderr, T_TAG ": %s: outlived the app's last Release — "
                "something kept a reference, so its registry entry is never "
                "evicted and its fd and mapping leak\n", what);
        return false;
    }
    /* Eviction runs from the destructor, and the backing's deallocator can run a
     * moment later still (Metal drops the last reference off-thread). */
    for (int i = 0; i < (int)kDeathWaitMs / 10 && t_count_fds() > fds_before; i++)
        sleep_ms(10);
    *reclaimed = t_count_fds() <= fds_before;
    return true;
}

/* Destroying a shared resource must give its fd back.
 *
 * Measured on the SECOND cycle. The first one also pays whatever the framework
 * opens on first use of a resource kind — a real fd that never comes back and
 * has nothing to do with sharing — so a single cycle cannot tell that apart
 * from a leak. Steady state can. (lifecycle-churn warms up for the same
 * reason.) */
template <class Create, class Plant>
int check_destruction_reclaims(const char* what, Create create, Plant plant) {
    int taken = 0;
    bool reclaimed = false;
    if (!destruction_cycle(what, create, plant, &taken, &reclaimed))
        return 1;
    if (taken <= 0) {
        fprintf(stderr, T_TAG ": %s: a shared resource took no fd — it was not "
                "backed by shared memory\n", what);
        return 1;
    }
    if (!destruction_cycle(what, create, plant, &taken, &reclaimed))
        return 1;
    if (!reclaimed) {
        fprintf(stderr, T_TAG ": %s: destroyed, but the %d fd(s) it held were "
                "not reclaimed\n", what, taken);
        return 1;
    }
    printf(T_TAG ": %s destroyed on the app's last Release, its %d fd(s) "
           "reclaimed: OK\n", what, taken);
    return 0;
}

/* == A released identity must not be recycled onto another resource ======== */
/* D3DMetal pools textures, so a released texture object can come back from a
 * later create. If that object were handed out again while a stale shared
 * registration still keyed on it, a plain texture would answer GetSharedHandle
 * with somebody else's surface. No shipped version does this — assert it, since
 * the pooling that would make it possible is real. */
int check_no_identity_reuse(ID3D11Device* dev) {
    void* released_id = nullptr;
    {
        Com<ID3D11Texture2D> tex;
        CK(make_shared_texture(dev, tex), "CreateTexture2D(MISC_SHARED)");
        released_id = dmn_com_identity(tex.ptr());
        Com<IDXGIResource> v;
        CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&v), "QI view");
        HANDLE h = nullptr;
        CK(v->GetSharedHandle(&h), "GetSharedHandle");
    }

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = 0; /* plain: nothing about it is shared */

    int reused = 0;
    for (int i = 0; i < 64; i++) {
        Com<ID3D11Texture2D> plain;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &plain)) || !plain.ptr())
            break;
        if (dmn_com_identity(plain.ptr()) != released_id)
            continue;
        reused++;
        Com<IDXGIResource> v;
        if (FAILED(plain->QueryInterface(__uuidof(IDXGIResource), (void**)&v)) ||
            !v.ptr())
            continue;
        HANDLE h = nullptr;
        if (SUCCEEDED(v->GetSharedHandle(&h))) {
            fprintf(stderr, T_TAG ": a PLAIN texture reused a released shared "
                    "resource's identity (%p) and exported handle %p — a stale "
                    "registry entry answered for it\n", released_id, h);
            return 1;
        }
    }
    printf(T_TAG ": no stale registration answered for a later plain texture "
           "(%d identity reuse(s)): OK\n", reused);
    return 0;
}

int run() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
                         D3D11_SDK_VERSION, &dev, &flo, &ctx),
       "D3D11CreateDevice");

    if (check_identity_through_views(dev.ptr()) != 0)
        return 1;
    if (check_no_identity_reuse(dev.ptr()) != 0)
        return 1;

    ID3D11Device* d11 = dev.ptr();
    if (check_destruction_reclaims(
            "D3D11 shared buffer",
            [d11]() -> IUnknown* {
                D3D11_BUFFER_DESC bd = {};
                bd.ByteWidth = 65536;
                bd.Usage = D3D11_USAGE_DEFAULT;
                bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                bd.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
                ID3D11Buffer* b = nullptr;
                if (FAILED(d11->CreateBuffer(&bd, nullptr, &b)))
                    return nullptr;
                return b;
            },
            [](TWatch* w, IUnknown* o) { return t_watch_plant_d3d11(w, o); }) != 0)
        return 1;

    Com<ID3D12Device> d12c;
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                         __uuidof(ID3D12Device), (void**)&d12c),
       "D3D12CreateDevice");
    ID3D12Device* d12 = d12c.ptr();

    if (check_destruction_reclaims(
            "D3D12 shared buffer",
            [d12]() -> IUnknown* {
                D3D12_HEAP_PROPERTIES hp = {};
                hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC rd = {};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                rd.Width = 65536;
                rd.Height = 1;
                rd.DepthOrArraySize = 1;
                rd.MipLevels = 1;
                rd.Format = DXGI_FORMAT_UNKNOWN;
                rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                ID3D12Resource* r = nullptr;
                if (FAILED(d12->CreateCommittedResource(
                        &hp, D3D12_HEAP_FLAG_SHARED, &rd,
                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                        __uuidof(ID3D12Resource), (void**)&r)))
                    return nullptr;
                return r;
            },
            [](TWatch* w, IUnknown* o) { return t_watch_plant_d3d12(w, o); }) != 0)
        return 1;

    if (check_destruction_reclaims(
            "D3D12 shared fence",
            [d12]() -> IUnknown* {
                ID3D12Fence* f = nullptr;
                if (FAILED(d12->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                            __uuidof(ID3D12Fence), (void**)&f)))
                    return nullptr;
                return f;
            },
            [](TWatch* w, IUnknown* o) { return t_watch_plant_d3d12(w, o); }) != 0)
        return 1;

    T_PASS();
    return 0;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    return run();
}
