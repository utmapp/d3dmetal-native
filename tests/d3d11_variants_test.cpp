/*
 * D3D11 variant shared entry points, windowless, single process — the hooked
 * paths the main tests do not touch:
 *
 *  1. ID3D11Device3::CreateTexture2D1 with MISC_SHARED -> substituted backing,
 *     legacy export via GetSharedHandle, POD dims verified.
 *  2. ID3D11Device1::OpenSharedResource1 import of that handle.
 *
 * Prints "VARIANTS: PASS" and exits 0 on success.
 */

#include <cstdio>

#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "VARIANTS"
#include "common/check.h"
#include "common/dx11.h"

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "VARIANTS: dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    CK(make_d3d11_device(dev, ctx), "D3D11CreateDevice");

    Com<ID3D11Device3> dev3;
    CK(dev->QueryInterface(__uuidof(ID3D11Device3), (void**)&dev3),
       "ID3D11Device3");
    Com<ID3D11Device1> dev1;
    CK(dev->QueryInterface(__uuidof(ID3D11Device1), (void**)&dev1),
       "ID3D11Device1");

    /* 1) CreateTexture2D1 producer. */
    const UINT kW = 160, kH = 120;
    D3D11_TEXTURE2D_DESC1 td{};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    td.TextureLayout = D3D11_TEXTURE_LAYOUT_UNDEFINED;
    Com<ID3D11Texture2D1> tex;
    CK(dev3->CreateTexture2D1(&td, nullptr, &tex), "CreateTexture2D1(SHARED)");

    Com<IDXGIResource> res;
    HANDLE h = nullptr;
    CK(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res), "IDXGIResource");
    CK(res->GetSharedHandle(&h), "GetSharedHandle");
    auto* pod = (dmn_shared_texture_handle*)h;
    EXPECT(pod->magic == DMN_SHARED_TEXTURE_MAGIC, "bad POD magic");
    EXPECT(pod->width == kW && pod->height == kH, "POD dims mismatch");

    /* 2) OpenSharedResource1 consumer. */
    Com<ID3D11Texture2D> opened;
    CK(dev1->OpenSharedResource1(h, __uuidof(ID3D11Texture2D), (void**)&opened),
       "OpenSharedResource1");
    D3D11_TEXTURE2D_DESC od{};
    opened->GetDesc(&od);
    EXPECT(od.Width == kW && od.Height == kH, "opened desc mismatch");

    T_PASS();
    return 0;
}
