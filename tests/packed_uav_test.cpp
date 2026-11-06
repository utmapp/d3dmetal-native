/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Packed 16-bit textures take typed UAV loads and stores in DXGI packing:
 * a compute shader loads texel (0,0) of one texture into a structured
 * buffer and stores two known colours into another; both are read back.
 * dmn_formats reports these formats UAV-capable on the strength of this.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <d3dcompiler.h>
#include <d3d11_1.h>
#include "d3dmetal_native.h"
#include "common/com.h"
#define T_TAG "PACKEDUAV"
#include "common/check.h"
#include "common/dx11.h"

namespace {
const char kCS[] =
    "RWTexture2D<float4> tin : register(u0);\n"
    "RWTexture2D<float4> tout : register(u1);\n"
    "RWStructuredBuffer<float4> o : register(u2);\n"
    "[numthreads(1,1,1)] void main() {\n"
    "  o[0] = tin[uint2(0,0)];\n"
    "  tout[uint2(0,0)] = float4(1, 0, 0, 1);\n"
    "  tout[uint2(1,0)] = float4(0, 0, 1, 0);\n"
    "}\n";

struct Fmt {
    DXGI_FORMAT f;
    const char* name;
    uint16_t red;        /* DXGI packing of (1, 0, 0, 0) */
    uint16_t store0;     /* DXGI packing of (1, 0, 0, 1) */
    uint16_t store1;     /* DXGI packing of (0, 0, 1, 0) */
    float load_a;        /* alpha read back from `red` */
};

const Fmt kFmts[] = {
    {DXGI_FORMAT_B5G6R5_UNORM, "B5G6R5", 0xf800, 0xf800, 0x001f, 1.0f},
    {DXGI_FORMAT_B5G5R5A1_UNORM, "B5G5R5A1", 0x7c00, 0xfc00, 0x001f, 0.0f},
    {DXGI_FORMAT_B4G4R4A4_UNORM, "B4G4R4A4", 0x0f00, 0xff00, 0x000f, 0.0f},
};
} // namespace

int main() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    CK_OK(make_d3d11_device(dev, ctx), "D3D11CreateDevice");

    Com<ID3DBlob> csb, err;
    CK_OK(D3DCompile(kCS, sizeof(kCS) - 1, "cs", nullptr, nullptr, "main", "cs_5_0", 0, 0,
                     &csb, &err), "D3DCompile cs");
    Com<ID3D11ComputeShader> cs;
    CK_OK(dev->CreateComputeShader(csb->GetBufferPointer(), csb->GetBufferSize(), nullptr, &cs),
          "CreateComputeShader");

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 16;
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = 16;
    Com<ID3D11Buffer> ob;
    CK_OK(dev->CreateBuffer(&bd, nullptr, &ob), "CreateBuffer");
    Com<ID3D11UnorderedAccessView> ouav;
    CK_OK(dev->CreateUnorderedAccessView(ob.ptr(), nullptr, &ouav), "CreateUnorderedAccessView buffer");
    bd.BindFlags = 0;
    bd.Usage = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Buffer> ost;
    CK_OK(dev->CreateBuffer(&bd, nullptr, &ost), "CreateBuffer staging");

    bool all = true;
    for (const Fmt& fmt : kFmts) {
        D3D11_TEXTURE2D_DESC td{};
        td.Width = 2;
        td.Height = 1;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = fmt.f;
        td.SampleDesc.Count = 1;
        td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        const uint16_t texels[2] = {fmt.red, fmt.red};
        D3D11_SUBRESOURCE_DATA init{texels, 4, 0};
        Com<ID3D11Texture2D> tin, tout;
        Com<ID3D11UnorderedAccessView> uin, uout;
        if (FAILED(dev->CreateTexture2D(&td, &init, &tin)) ||
            FAILED(dev->CreateTexture2D(&td, nullptr, &tout)) ||
            FAILED(dev->CreateUnorderedAccessView(tin.ptr(), nullptr, &uin)) ||
            FAILED(dev->CreateUnorderedAccessView(tout.ptr(), nullptr, &uout))) {
            printf("PACKEDUAV: %-9s create FAILED\n", fmt.name);
            all = false;
            continue;
        }
        D3D11_TEXTURE2D_DESC sd = td;
        sd.BindFlags = 0;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Com<ID3D11Texture2D> st;
        CK_OK(dev->CreateTexture2D(&sd, nullptr, &st), "CreateTexture2D staging");

        ID3D11UnorderedAccessView* uavs[] = {uin.ptr(), uout.ptr(), ouav.ptr()};
        ctx->CSSetShader(cs.ptr(), nullptr, 0);
        ctx->CSSetUnorderedAccessViews(0, 3, uavs, nullptr);
        ctx->Dispatch(1, 1, 1);
        ctx->CopyResource(st.ptr(), tout.ptr());
        ctx->CopyResource(ost.ptr(), ob.ptr());

        D3D11_MAPPED_SUBRESOURCE m{};
        CK_OK(ctx->Map(ost.ptr(), 0, D3D11_MAP_READ, 0, &m), "Map buffer");
        float v[4];
        memcpy(v, m.pData, sizeof(v));
        ctx->Unmap(ost.ptr(), 0);
        CK_OK(ctx->Map(st.ptr(), 0, D3D11_MAP_READ, 0, &m), "Map texture");
        uint16_t s[2];
        memcpy(s, m.pData, sizeof(s));
        ctx->Unmap(st.ptr(), 0);

        const bool ok = v[0] == 1.0f && v[1] == 0.0f && v[2] == 0.0f && v[3] == fmt.load_a &&
                        s[0] == fmt.store0 && s[1] == fmt.store1;
        printf("PACKEDUAV: %-9s load 0x%04x -> (%g %g %g %g), store -> 0x%04x 0x%04x "
               "(want 0x%04x 0x%04x) %s\n", fmt.name, fmt.red, v[0], v[1], v[2], v[3], s[0], s[1],
               fmt.store0, fmt.store1, ok ? "OK" : "MISMATCH");
        all = all && ok;
    }
    return all ? 0 : 1;
}
