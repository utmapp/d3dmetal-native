/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * COM vtable hooking infrastructure. Header-only; used by the TUs that patch
 * or implement D3D interfaces (dmn_com_hooks.cpp, dmn_fence_d3d.cpp,
 * dmn_kmtx.cpp). Slot indices are resolved from the header's own method
 * declarations via the Itanium member-pointer representation — no hand-counted
 * indices. D3DMetal's vtables live in a plain RW __DATA segment (the framework
 * has no __DATA_CONST), so slots are written directly.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include <windows.h>
/* dmn_com_identity() canonicalizes through ID3D11DeviceChild / ID3D12Object,
 * so both hierarchies must be declared here. */
#include <d3d11.h>
#include <d3d12.h>

/* == Member-pointer -> vtable slot (Itanium C++ ABI) ======================= */
template <class F>
size_t dmn_vindex(F m) {
    static_assert(sizeof(F) == 2 * sizeof(void*),
                  "unexpected pointer-to-member-function representation");
    union { F fn; uintptr_t w[2]; } u;
    u.fn = m;
    /* For a virtual function, w[0] == vtable byte offset + 1. */
    return (u.w[0] - 1) / sizeof(void*);
}

/* == One hooked method ===================================================== */
/* Maps each concrete vtable to the original function it displaced, so a single
 * shared thunk can call through regardless of how many device classes exist. */
struct DmnMethodHook {
    std::mutex mtx;
    std::unordered_map<void*, void*> orig; /* vtable ptr -> original fn */

    void* original_for(void* obj) {
        void* vt = *reinterpret_cast<void**>(obj);
        std::lock_guard<std::mutex> lk(mtx);
        auto it = orig.find(vt);
        return it == orig.end() ? nullptr : it->second;
    }
};

/* Typed variant: DmnMethodHookT<decltype(&hook_Foo)> vends the displaced
 * pointer with the thunk's own signature (thunk and original always match). */
template <class Fn>
struct DmnMethodHookT : DmnMethodHook {
    Fn original(void* obj) {
        return reinterpret_cast<Fn>(original_for(obj));
    }
};

/* Patch (vtable,slot) to `thunk`, remembering the displaced pointer.
 * Idempotent per (vtable,slot). The lock is held across check-read-write:
 * device creation is free-threaded, so two threads can patch the same vtable
 * concurrently, and a check/write gap would let the loser read the winner's
 * thunk back as the "original" (infinite recursion on the first hooked call). */
inline void dmn_patch_slot(void* obj, size_t slot, void* thunk,
                           DmnMethodHook& hook) {
    if (!obj)
        return;
    void** vt = *reinterpret_cast<void***>(obj);
    std::lock_guard<std::mutex> lk(hook.mtx);
    if (hook.orig.count((void*)vt))
        return; /* this vtable already patched for this method */
    hook.orig[(void*)vt] = vt[slot];
    vt[slot] = thunk;
}

/* Declare the typed hook-state variable for thunk hook_<name>. */
#define DMN_HOOK_STATE(name) DmnMethodHookT<decltype(&hook_##name)> h_##name

/* Patch iface::method on `obj`'s vtable with thunk hook_<name>. */
#define DMN_PATCH(obj, iface, method, name) \
    dmn_patch_slot((obj), dmn_vindex(&iface::method), (void*)hook_##name, h_##name)

/* QueryInterface is overloaded (virtual + the templated unknwn.h helper), so
 * its slot needs the exact member type spelled out. */
inline size_t dmn_vindex_qi() {
    using QIType = HRESULT (STDMETHODCALLTYPE IUnknown::*)(REFIID, void**);
    return dmn_vindex(static_cast<QIType>(&IUnknown::QueryInterface));
}
#define DMN_PATCH_QI(obj, name) \
    dmn_patch_slot((obj), dmn_vindex_qi(), (void*)hook_##name, h_##name)

/* Fetch the displaced original inside thunk hook_<name>; evaluates to a typed
 * function pointer (null if the vtable is somehow unknown). */
#define DMN_ORIG(name, This) h_##name.original((void*)(This))

/* == COM helpers =========================================================== */

/* Which object a QueryInterface issued through a given interface pointer takes
 * its reference on. COM says the one it hands back, and every D3DMetal does
 * that from a resource's own vtable — but GPTk 4.0b1's DXGI resource views
 * AddRef *themselves* while handing back the resource they front. Releasing the
 * returned pointer there destroys a resource the app still holds; releasing
 * nothing pins it forever (the view keeps a reference on the resource). Either
 * way the resource's lifetime breaks, so identity resolution has to give back
 * whichever object actually got the count.
 *
 * Answered in dmn_com_hooks.cpp, where the views are minted and probed. */
enum DmnQiRefTarget {
    DMN_QI_REF_RESULT = 0, /* the returned interface (correct COM) */
    DMN_QI_REF_SOURCE,     /* the interface QueryInterface was called on */
    DMN_QI_REF_NONE,       /* no reference taken at all */
};
/* extern "C" so it lands in the dylib's exported dmn_* set (the tests assert
 * these invariants against the very function the registries call). */
extern "C" DmnQiRefTarget dmn_com_qi_ref_target(void* obj);

/* Current reference count of `p`, without changing it. */
inline unsigned long dmn_com_refs(IUnknown* p) {
    p->AddRef();
    return p->Release();
}

namespace dmn_detail {
/* QueryInterface purely for identity: the returned pointer is used as a map key
 * and never dereferenced, so the reference it took is given straight back — to
 * whichever object holds it (see DmnQiRefTarget). */
inline void* qi_ptr(IUnknown* p, REFIID iid, DmnQiRefTarget target) {
    IUnknown* o = nullptr;
    if (FAILED(p->QueryInterface(iid, reinterpret_cast<void**>(&o))) || !o)
        return nullptr;
    if (target == DMN_QI_REF_RESULT)
        o->Release();
    else if (target == DMN_QI_REF_SOURCE)
        p->Release();
    return o;
}
} // namespace dmn_detail

/* Canonical identity of a COM object, usable as a map key: the same value from
 * every interface view of the same underlying object, and stable across
 * repeated queries.
 *
 * QI(IID_IUnknown) is the textbook way to get that, but D3DMetal 4.0b1 broke
 * it: for a D3D11 resource, QueryInterface(IDXGIResource) mints a *fresh*
 * wrapper object per call, and IUnknown on that wrapper returns the wrapper
 * itself rather than the resource it fronts. Keying on IUnknown then makes
 * every registry lookup arriving through a DXGI view (GetSharedHandle,
 * CreateSharedHandle, the keyed-mutex QI) miss, and the export falls through to
 * D3DMetal's own "Unsupported: IDXGIResource::GetSharedHandle" stub.
 *
 * ID3D11DeviceChild / ID3D12Object *are* forwarded to the owning object on
 * every GPTk version tested (1.0, 2.1, 3.0, 4.0b1), from the object and from a
 * DXGI view alike, so ask those first. Every object the sharing registry keys
 * on (textures, buffers, fences, heaps, resources, queues) exposes one of them;
 * IUnknown remains the fallback for anything that exposes neither. */
inline void* dmn_com_identity(IUnknown* p) {
    if (!p)
        return nullptr;
    const DmnQiRefTarget t = dmn_com_qi_ref_target(p);
    if (void* c11 = dmn_detail::qi_ptr(p, __uuidof(ID3D11DeviceChild), t))
        return c11;
    if (void* o12 = dmn_detail::qi_ptr(p, __uuidof(ID3D12Object), t))
        return o12;
    if (void* unk = dmn_detail::qi_ptr(p, __uuidof(IUnknown), t))
        return unk;
    return p;
}

inline bool dmn_iid_eq(REFIID a, REFIID b) {
    return std::memcmp(&a, &b, sizeof(GUID)) == 0;
}
