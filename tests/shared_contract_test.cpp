/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * shared_contract_test: the invariants a MISC_SHARED create/open must hold,
 * with no pixels involved.
 *
 *   1. A shared create that cannot be backed by shared memory FAILS. It must
 *      never return S_OK with an ordinary process-private texture: the handle
 *      it then vends opens to nothing in the peer, and the mistake surfaces
 *      much later as a blank surface with no failing call to point at.
 *   2. An import validates the geometry it is handed. Stride and size in a
 *      shared handle crossed a process boundary, so a stride shorter than one
 *      row of pixels, or a region smaller than stride*height, has to be
 *      refused rather than turned into a texture that reads past its mapping.
 *   3. A rejected import reports its own failure. "Not one of our handles" and
 *      "our handle, but bad" must not collapse into the same HRESULT: the
 *      first falls through to D3DMetal, and taking that path with a POD
 *      pointer hands it something that is not a Windows HANDLE at all.
 *
 * Same-process throughout: opening a handle this process exported exercises
 * exactly the consumer path, and none of these assertions are about data
 * actually crossing a process boundary.
 *
 * Prints "CONTRACT: PASS" and exits 0 on success.
 */

#define T_TAG "CONTRACT"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/check.h"
#include "common/com.h"

namespace {

D3D11_TEXTURE2D_DESC shared_desc(UINT w, UINT h, DXGI_FORMAT fmt, UINT bind) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = fmt;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = bind;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    return td;
}

/* Export a shared texture's POD. The HANDLE our IDXGIResource::GetSharedHandle
 * vends is the POD bytes themselves. */
bool export_pod(ID3D11Texture2D* tex, dmn_shared_texture_handle* out) {
    Com<IDXGIResource> res;
    HANDLE h = nullptr;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
        FAILED(res->GetSharedHandle(&h)) || !h)
        return false;
    memcpy(out, h, sizeof(*out));
    return out->magic == DMN_SHARED_TEXTURE_MAGIC;
}

/* == 1. Unshareable formats fail the create ============================== */
/* A format with no linear element size cannot back a buffer-backed texture, so
 * the substitution declines it. What this pins down is what happens NEXT: the
 * create must report failure instead of quietly handing back a private
 * texture. (D3DMetal may reject some of these on its own, which is equally
 * fine — the assertion is on the outcome, not on who produced it.) */
int test_reject_unshareable(ID3D11Device* dev) {
    const struct {
        DXGI_FORMAT fmt;
        UINT        bind;
        const char* name;
    } cases[] = {
        { DXGI_FORMAT_BC1_UNORM, D3D11_BIND_SHADER_RESOURCE, "BC1_UNORM" },
        { DXGI_FORMAT_BC3_UNORM, D3D11_BIND_SHADER_RESOURCE, "BC3_UNORM" },
        { DXGI_FORMAT_D32_FLOAT, D3D11_BIND_DEPTH_STENCIL,   "D32_FLOAT" },
    };
    for (const auto& c : cases) {
        D3D11_TEXTURE2D_DESC td = shared_desc(64, 64, c.fmt, c.bind);
        Com<ID3D11Texture2D> tex;
        HRESULT hr = dev->CreateTexture2D(&td, nullptr, &tex);
        if (SUCCEEDED(hr)) {
            fprintf(stderr,
                    T_TAG ": MISC_SHARED %s create SUCCEEDED; a format with no "
                    "linear layout must not silently produce an unshared "
                    "texture\n", c.name);
            return 1;
        }
        printf(T_TAG ": MISC_SHARED %s rejected (0x%08x): OK\n", c.name,
               (unsigned)hr);
    }
    return 0;
}

/* == 2. Imports validate the geometry they are handed ==================== */
int test_import_validation(ID3D11Device* dev) {
    D3D11_TEXTURE2D_DESC td =
        shared_desc(256, 256, DXGI_FORMAT_B8G8R8A8_UNORM,
                    D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
    Com<ID3D11Texture2D> src;
    CK(dev->CreateTexture2D(&td, nullptr, &src), "control shared create");

    dmn_shared_texture_handle pod{};
    EXPECT(export_pod(src.ptr(), &pod), "control export");
    EXPECT(pod.stride >= 256u * 4u, "exported stride covers a row");
    EXPECT(pod.size >= pod.stride * 256u, "exported size covers the surface");

    /* Control: the untouched handle must open, or the negatives below prove
     * nothing. */
    {
        Com<ID3D11Texture2D> ok;
        CK(dev->OpenSharedResource((HANDLE)&pod, __uuidof(ID3D11Texture2D),
                                   (void**)&ok),
           "control import");
        EXPECT(ok.ptr() != nullptr, "control import returned an object");
    }

    const struct {
        const char* name;
        uint64_t    stride;
        uint64_t    size;
    } bad[] = {
        { "stride 0",              0,             pod.size },
        { "stride under one row",  16,            pod.size },
        { "size 0",                pod.stride,    0 },
        { "size under the surface",pod.stride,    pod.stride * 4 },
        { "size past any surface", pod.stride,    (uint64_t)1 << 40 },
    };
    for (const auto& b : bad) {
        dmn_shared_texture_handle corrupt = pod;
        corrupt.stride = b.stride;
        corrupt.size = b.size;
        Com<ID3D11Texture2D> opened;
        HRESULT hr = dev->OpenSharedResource(
            (HANDLE)&corrupt, __uuidof(ID3D11Texture2D), (void**)&opened);
        if (SUCCEEDED(hr)) {
            fprintf(stderr,
                    T_TAG ": import with %s SUCCEEDED; wire geometry must be "
                    "validated, not trusted\n", b.name);
            return 1;
        }
        /* E_NOTIMPL is what D3DMetal answers for a HANDLE it cannot parse, so
         * seeing it here means our rejection was mistaken for "not one of our
         * handles" and fell through with a POD pointer in hand. */
        if (hr == E_NOTIMPL) {
            fprintf(stderr,
                    T_TAG ": import with %s fell through to D3DMetal instead of "
                    "reporting its own rejection\n", b.name);
            return 1;
        }
        printf(T_TAG ": import rejected %s (0x%08x): OK\n", b.name,
               (unsigned)hr);
    }

    /* And a handle that is genuinely not ours must still reach D3DMetal rather
     * than being claimed and failed by us. */
    {
        uint32_t not_ours[16] = { 0xdeadbeefu };
        Com<ID3D11Texture2D> opened;
        HRESULT hr = dev->OpenSharedResource(
            (HANDLE)not_ours, __uuidof(ID3D11Texture2D), (void**)&opened);
        EXPECT(FAILED(hr), "a foreign handle must not open");
        printf(T_TAG ": foreign handle passed through (0x%08x): OK\n",
               (unsigned)hr);
    }
    return 0;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flOut;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                         D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                         D3D11_SDK_VERSION, &dev, &flOut, &ctx),
       "D3D11CreateDevice");

    if (test_reject_unshareable(dev.ptr()) != 0)
        return 1;
    if (test_import_validation(dev.ptr()) != 0)
        return 1;

    T_PASS();
    return 0;
}
