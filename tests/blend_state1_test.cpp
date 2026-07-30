/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * blend_state1_test: ID3D11Device1::CreateBlendState1 must hand back a usable
 * blend state.
 *
 * GPTk 4.0b1 and 4.0b2 return S_OK from it and write nothing, which no caller
 * testing FAILED(hr) can detect; dmn_com_hooks repairs that by satisfying the
 * call through CreateBlendState. The assertions here hold on any framework —
 * one that never had the bug, one being repaired, and one Apple has fixed — so
 * they are a regression test for the repair rather than a description of the
 * bug.
 *
 * They also cover what the repair assumes rather than only that it returns
 * non-NULL: that the pointer CreateBlendState vends really carries the
 * ID3D11BlendState1 vtable (GetDesc1 at slot 8 is the fidelity question), and
 * that routing through the older entry point still dedupes to one state per
 * desc instead of minting a second.
 *
 * The framework's own behaviour is measured first, through a device created by
 * its D3D11CreateDevice rather than the library's, so the vtable patching has
 * not happened yet — D3DMetal's vtables are per class, so once any device has
 * been created through the library every device in the process sees the
 * repaired method. That measurement is what reports the bug being fixed
 * upstream: when the framework starts answering correctly on its own, the
 * repair is dead code and this test says so.
 */

#include <dlfcn.h>

#include <cstring>

#include <d3d11_1.h>

#include "d3dmetal_native.h"

#define T_TAG "BS1"
#include "common/check.h"

namespace {

typedef HRESULT (STDMETHODCALLTYPE *PFN_D3D11CreateDevice)(
    IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT, const D3D_FEATURE_LEVEL*,
    UINT, UINT, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext**);

/* LogicOp is the one field the repair cannot carry through CreateBlendState, so
 * the desc under test leaves it at the value D3DMetal widens a DESC to
 * (disabled, NOOP): the round trip is then exact on the repaired path as well
 * as the native one, and GetDesc1 can be compared field for field. */
D3D11_BLEND_DESC1 test_desc(D3D11_BLEND src_blend) {
    D3D11_BLEND_DESC1 d{};
    d.RenderTarget[0].BlendEnable = TRUE;
    d.RenderTarget[0].LogicOpEnable = FALSE;
    d.RenderTarget[0].SrcBlend = src_blend;
    d.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    d.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    d.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    d.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    d.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    d.RenderTarget[0].LogicOp = D3D11_LOGIC_OP_NOOP;
    d.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    return d;
}

bool rt_desc_equal(const D3D11_RENDER_TARGET_BLEND_DESC1& a,
                   const D3D11_RENDER_TARGET_BLEND_DESC1& b) {
    return a.BlendEnable == b.BlendEnable &&
           a.LogicOpEnable == b.LogicOpEnable &&
           a.SrcBlend == b.SrcBlend && a.DestBlend == b.DestBlend &&
           a.BlendOp == b.BlendOp && a.SrcBlendAlpha == b.SrcBlendAlpha &&
           a.DestBlendAlpha == b.DestBlendAlpha &&
           a.BlendOpAlpha == b.BlendOpAlpha && a.LogicOp == b.LogicOp &&
           a.RenderTargetWriteMask == b.RenderTargetWriteMask;
}

enum NativeVerdict { NATIVE_CORRECT, NATIVE_SILENT_NULL, NATIVE_UNKNOWN };

/* What the loaded framework does with CreateBlendState1 unaided. */
NativeVerdict probe_native() {
    if (dmn_init(nullptr) != DMN_SUCCESS || !dmn_framework_path())
        return NATIVE_UNKNOWN;
    /* Already loaded, so this only takes a reference on the same image. */
    void* h = dlopen(dmn_framework_path(), RTLD_NOW | RTLD_LOCAL);
    if (!h)
        return NATIVE_UNKNOWN;
    auto create = (PFN_D3D11CreateDevice)dlsym(h, "D3D11CreateDevice");
    NativeVerdict verdict = NATIVE_UNKNOWN;
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    if (create && SUCCEEDED(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                   nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr,
                                   &ctx)) && dev) {
        ID3D11Device1* d1 = nullptr;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device1),
                                          reinterpret_cast<void**>(&d1))) && d1) {
            const D3D11_BLEND_DESC1 d = test_desc(D3D11_BLEND_SRC_ALPHA);
            ID3D11BlendState1* bs = nullptr;
            HRESULT hr = d1->CreateBlendState1(&d, &bs);
            if (SUCCEEDED(hr))
                verdict = bs ? NATIVE_CORRECT : NATIVE_SILENT_NULL;
            if (bs) bs->Release();
            d1->Release();
        }
    }
    if (ctx) ctx->Release();
    if (dev) dev->Release();
    dlclose(h);
    return verdict;
}

int run(void) {
    const NativeVerdict native = probe_native();
    switch (native) {
    case NATIVE_SILENT_NULL:
        printf(T_TAG ": framework CreateBlendState1 answers S_OK with no blend "
                     "state; the dmn_com_hooks repair is carrying it\n");
        break;
    case NATIVE_CORRECT:
        /* True of every framework before 4.0 as well as of one that has been
         * fixed, so this cannot say on its own which it is looking at: on a
         * build newer than 4.0b2 it means the bug is gone and the repair in
         * dmn_com_hooks.cpp can go with it. */
        printf(T_TAG ": framework CreateBlendState1 works unaided; the repair is "
                     "inert here — on anything newer than 4.0b2 that means it is "
                     "fixed upstream and the hook can be retired\n");
        break;
    case NATIVE_UNKNOWN:
        printf(T_TAG ": could not measure the framework's own CreateBlendState1; "
                     "checking the contract only\n");
        break;
    }

    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr,
                         0, D3D11_SDK_VERSION, &dev, nullptr, &ctx),
       "D3D11CreateDevice");
    ID3D11Device1* d1 = nullptr;
    CK(dev->QueryInterface(__uuidof(ID3D11Device1),
                           reinterpret_cast<void**>(&d1)),
       "QueryInterface(ID3D11Device1)");

    const D3D11_BLEND_DESC1 desc = test_desc(D3D11_BLEND_SRC_ALPHA);
    ID3D11BlendState1* bs = nullptr;
    CK(d1->CreateBlendState1(&desc, &bs), "CreateBlendState1");
    EXPECT(bs != nullptr, "CreateBlendState1 returned success without a blend "
                          "state (the out-parameter is still NULL)");
    printf(T_TAG ": CreateBlendState1: OK\n");

    /* The repair hands back the pointer CreateBlendState vends, so the vtable
     * really has to be ID3D11BlendState1's rather than its base. */
    D3D11_BLEND_DESC1 got1{};
    bs->GetDesc1(&got1);
    EXPECT(rt_desc_equal(got1.RenderTarget[0], desc.RenderTarget[0]),
           "GetDesc1 did not return the requested DESC1");
    EXPECT(got1.AlphaToCoverageEnable == desc.AlphaToCoverageEnable &&
           got1.IndependentBlendEnable == desc.IndependentBlendEnable,
           "GetDesc1 disagrees on the desc-wide flags");
    printf(T_TAG ": GetDesc1 round-trip: OK\n");

    D3D11_BLEND_DESC got{};
    bs->GetDesc(&got);
    EXPECT(got.RenderTarget[0].SrcBlend == desc.RenderTarget[0].SrcBlend &&
           got.RenderTarget[0].DestBlend == desc.RenderTarget[0].DestBlend &&
           got.RenderTarget[0].BlendEnable == desc.RenderTarget[0].BlendEnable,
           "GetDesc disagrees with the requested desc");

    ID3D11Device* owner = nullptr;
    bs->GetDevice(&owner);
    EXPECT(owner == dev, "the blend state is not owned by the device that "
                         "created it");
    if (owner) owner->Release();

    /* Usable as a blend state, not merely non-NULL. */
    const FLOAT factor[4] = {0, 0, 0, 0};
    ctx->OMSetBlendState(bs, factor, 0xffffffffu);
    ID3D11BlendState* back = nullptr;
    FLOAT got_factor[4] = {0, 0, 0, 0};
    UINT got_mask = 0;
    ctx->OMGetBlendState(&back, got_factor, &got_mask);
    EXPECT(back == static_cast<ID3D11BlendState*>(bs),
           "OMGetBlendState did not return the state OMSetBlendState was given");
    EXPECT(got_mask == 0xffffffffu, "the sample mask did not survive "
                                    "OMSetBlendState");
    if (back) back->Release();
    ctx->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
    printf(T_TAG ": OMSetBlendState round-trip: OK\n");

    /* D3DMetal keys blend states on the desc and vends one interface per
     * object; the repair must not defeat that by minting a second state for a
     * desc that already has one. */
    ID3D11BlendState1* same = nullptr;
    CK(d1->CreateBlendState1(&desc, &same), "CreateBlendState1 (repeat)");
    EXPECT(same == bs, "the same desc produced a different blend state object");
    if (same) same->Release();

    const D3D11_BLEND_DESC1 other_desc = test_desc(D3D11_BLEND_ONE);
    ID3D11BlendState1* other = nullptr;
    CK(d1->CreateBlendState1(&other_desc, &other), "CreateBlendState1 (other)");
    EXPECT(other != nullptr && other != bs,
           "a different desc did not produce a different blend state object");
    D3D11_BLEND_DESC1 other_got{};
    other->GetDesc1(&other_got);
    EXPECT(other_got.RenderTarget[0].SrcBlend == D3D11_BLEND_ONE,
           "the second state did not keep its own desc");
    if (other) other->Release();
    printf(T_TAG ": desc dedupe: OK\n");

    /* A NULL out-parameter is a validation call, not a create: every framework
     * answers S_FALSE, which is a success code, so the interesting property is
     * that it stays distinguishable from a create that produced a state. */
    EXPECT(d1->CreateBlendState1(&desc, nullptr) != S_OK,
           "CreateBlendState1 reported S_OK for a NULL out-parameter");

    bs->Release();
    d1->Release();
    ctx->Release();
    dev->Release();
    T_PASS();
    return 0;
}

} // namespace

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    return run();
}
