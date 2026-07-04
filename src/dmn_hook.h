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
/* Canonical identity of a COM object (QI to IUnknown), usable as a map key. */
inline void* dmn_com_identity(IUnknown* p) {
    if (!p)
        return nullptr;
    IUnknown* unk = nullptr;
    if (SUCCEEDED(p->QueryInterface(__uuidof(IUnknown),
                                    reinterpret_cast<void**>(&unk))) && unk) {
        unk->Release();
        return unk;
    }
    return p;
}

inline bool dmn_iid_eq(REFIID a, REFIID b) {
    return std::memcmp(&a, &b, sizeof(GUID)) == 0;
}
