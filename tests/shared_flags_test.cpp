/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * shared_flags_test: a shared texture's view rights cross the POD in both
 * directions, whichever API made it and whichever opens it.
 *
 *   1. A D3D12 export records the producer's resource flags as D3D11 bind
 *      flags: ALLOW_UNORDERED_ACCESS without ALLOW_RENDER_TARGET is
 *      SHADER_RESOURCE | UNORDERED_ACCESS.
 *   2. A D3D11 open of it carries those binds, so a UAV can be made on it
 *      and no RT bind is invented.
 *   3. A D3D12 open -- of that export, and of a D3D11 SRV|UAV export --
 *      asks for exactly the producer's flags: ALLOW_UNORDERED_ACCESS, no
 *      ALLOW_RENDER_TARGET.
 *
 * Same-process: opening a handle this process exported is the consumer path.
 *
 * Prints "FLAGS: PASS" and exits 0 on success.
 */

#define T_TAG "FLAGS"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/check.h"
#include "common/com.h"
#include "common/dx11.h"
#include "common/dx12.h"

namespace {

const UINT kW = 64, kH = 64;

int check_d3d12_open(ID3D12Device* d12, dmn_shared_texture_handle* pod,
                     const char* what) {
    Com<ID3D12Resource> opened;
    CK(d12->OpenSharedHandle((HANDLE)pod, __uuidof(ID3D12Resource),
                             (void**)&opened),
       what);
    const D3D12_RESOURCE_FLAGS f = opened->GetDesc().Flags;
    printf(T_TAG ": %s: Flags 0x%x\n", what, (unsigned)f);
    EXPECT(f == D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
           "D3D12 open does not carry exactly the producer's UAV flag");
    return 0;
}

int test_d3d12_producer(ID3D12Device* d12, ID3D11Device* d11) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = kW;
    rd.Height = kH;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Com<ID3D12Resource> tex;
    CK(d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                    __uuidof(ID3D12Resource), (void**)&tex),
       "D3D12 shared UAV create");
    HANDLE h = nullptr;
    CK(d12->CreateSharedHandle(tex.ptr(), nullptr, 0, nullptr, &h),
       "D3D12 CreateSharedHandle");
    dmn_shared_texture_handle pod;
    memcpy(&pod, h, sizeof(pod));
    printf(T_TAG ": D3D12 export: bind_flags 0x%x misc_flags 0x%x\n",
           pod.bind_flags, pod.misc_flags);
    EXPECT(pod.bind_flags ==
               (D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS),
           "D3D12 export does not record SRV|UAV bind flags");

    {
        Com<ID3D11Texture2D> opened;
        CK(d11->OpenSharedResource((HANDLE)&pod, __uuidof(ID3D11Texture2D),
                                   (void**)&opened),
           "D3D11 OpenSharedResource(D3D12 export)");
        D3D11_TEXTURE2D_DESC od{};
        opened->GetDesc(&od);
        printf(T_TAG ": D3D11 open: BindFlags 0x%x\n", od.BindFlags);
        EXPECT((od.BindFlags & D3D11_BIND_UNORDERED_ACCESS) &&
                   !(od.BindFlags & D3D11_BIND_RENDER_TARGET),
               "D3D11 open does not carry the producer's binds");
        Com<ID3D11UnorderedAccessView> uav;
        CK(d11->CreateUnorderedAccessView(opened.ptr(), nullptr, &uav),
           "D3D11 UAV on the opened D3D12 export");
    }

    if (check_d3d12_open(d12, &pod, "D3D12 OpenSharedHandle(D3D12 export)"))
        return 1;
    CK(dmn_shared_handle_close(h) == DMN_SUCCESS ? S_OK : E_FAIL, "close h");
    return 0;
}

int test_d3d11_producer(ID3D12Device* d12, ID3D11Device* d11) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Texture2D> tex;
    CK(d11->CreateTexture2D(&td, nullptr, &tex), "D3D11 shared UAV create");
    Com<IDXGIResource> res;
    HANDLE h = nullptr;
    CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res),
       "IDXGIResource");
    CK(res->GetSharedHandle(&h), "GetSharedHandle");
    dmn_shared_texture_handle pod;
    memcpy(&pod, h, sizeof(pod));
    return check_d3d12_open(d12, &pod, "D3D12 OpenSharedHandle(D3D11 export)");
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> d12;
    CK(make_d3d12_device(d12), "D3D12CreateDevice");
    Com<ID3D11Device> d11;
    Com<ID3D11DeviceContext> ctx;
    CK(make_d3d11_device(d11, ctx), "D3D11CreateDevice");

    if (test_d3d12_producer(d12.ptr(), d11.ptr()) != 0)
        return 1;
    if (test_d3d11_producer(d12.ptr(), d11.ptr()) != 0)
        return 1;

    T_PASS();
    return 0;
}
