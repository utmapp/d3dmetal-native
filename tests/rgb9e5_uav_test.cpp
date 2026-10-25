/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * R9G9B9E5_SHAREDEXP takes typed UAV loads and stores (D3D11.3 revised
 * 999E5 format grouping), on textures and on typed buffers: a compute shader
 * loads element 0 of one texture and one buffer into a structured buffer and
 * stores two known colours into another of each; all are read back and
 * decoded.  CheckFormatSupport(2) must report the typed
 * UAV bits the HLK's RevisedFormat999E5 RTV/UAV grouping requires.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <d3dcompiler.h>
#include <d3d11_1.h>
#include "d3dmetal_native.h"
#include "common/com.h"
#define T_TAG "RGB9E5UAV"
#include "common/check.h"
#include "common/dx11.h"

namespace {
const char kCS[] =
    "RWTexture2D<float4> tin : register(u0);\n"
    "RWTexture2D<float4> tout : register(u1);\n"
    "RWStructuredBuffer<float4> o : register(u2);\n"
    "RWBuffer<float4> bin : register(u3);\n"
    "RWBuffer<float4> bout : register(u4);\n"
    "[numthreads(1,1,1)] void main() {\n"
    "  o[0] = tin[uint2(0,0)];\n"
    "  o[1] = bin[0];\n"
    "  tout[uint2(0,0)] = float4(1, 0, 0, 1);\n"
    "  tout[uint2(1,0)] = float4(0, 0.25, 4, 0);\n"
    "  bout[0] = float4(1, 0, 0, 1);\n"
    "  bout[1] = float4(0, 0.25, 4, 0);\n"
    "}\n";

void decode(uint32_t v, float out[3]) {
    const float scale = std::ldexp(1.0f, (int)(v >> 27) - 15 - 9);
    out[0] = (float)(v & 0x1ff) * scale;
    out[1] = (float)((v >> 9) & 0x1ff) * scale;
    out[2] = (float)((v >> 18) & 0x1ff) * scale;
}
} // namespace

int main() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    CK_OK(make_d3d11_device(dev, ctx), "D3D11CreateDevice");

    /* The RTV/UAV grouping: render target, blend, MS target/resolve/load,
     * typed UAV; and in FORMAT_SUPPORT2 typed UAV load and store. */
    const UINT want1 = D3D11_FORMAT_SUPPORT_RENDER_TARGET | D3D11_FORMAT_SUPPORT_BLENDABLE |
                       D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET |
                       D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE |
                       D3D11_FORMAT_SUPPORT_MULTISAMPLE_LOAD |
                       D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW;
    const UINT want2 = D3D11_FORMAT_SUPPORT2_UAV_TYPED_LOAD | D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE;
    UINT sup = 0;
    CK_OK(dev->CheckFormatSupport(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, &sup), "CheckFormatSupport");
    D3D11_FEATURE_DATA_FORMAT_SUPPORT2 fs2{DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 0};
    CK_OK(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &fs2, sizeof(fs2)),
          "CheckFeatureSupport FORMAT_SUPPORT2");
    const bool caps_ok = (sup & want1) == want1 && (fs2.OutFormatSupport2 & want2) == want2;
    printf("RGB9E5UAV: support 0x%08x (want 0x%08x) support2 0x%08x (want 0x%08x) %s\n", sup,
           want1, fs2.OutFormatSupport2, want2, caps_ok ? "OK" : "MISMATCH");

    Com<ID3DBlob> csb, err;
    CK_OK(D3DCompile(kCS, sizeof(kCS) - 1, "cs", nullptr, nullptr, "main", "cs_5_0", 0, 0,
                     &csb, &err), "D3DCompile cs");
    Com<ID3D11ComputeShader> cs;
    CK_OK(dev->CreateComputeShader(csb->GetBufferPointer(), csb->GetBufferSize(), nullptr, &cs),
          "CreateComputeShader");

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = 32;
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = 16;
    Com<ID3D11Buffer> ob;
    CK_OK(dev->CreateBuffer(&bd, nullptr, &ob), "CreateBuffer");
    Com<ID3D11UnorderedAccessView> ouav;
    CK_OK(dev->CreateUnorderedAccessView(ob.ptr(), nullptr, &ouav), "CreateUnorderedAccessView buffer");
    bd.BindFlags = 0;
    bd.MiscFlags = 0;
    bd.Usage = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Buffer> ost;
    CK_OK(dev->CreateBuffer(&bd, nullptr, &ost), "CreateBuffer staging");

    D3D11_TEXTURE2D_DESC td{};
    td.Width = 2;
    td.Height = 1;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    /* (0.5, 2, 0): exponent 17, mantissas 64 / 256 / 0. */
    const uint32_t texels[2] = {64u | (256u << 9) | (17u << 27), 0};
    D3D11_SUBRESOURCE_DATA init{texels, 8, 0};
    Com<ID3D11Texture2D> tin, tout;
    CK_OK(dev->CreateTexture2D(&td, &init, &tin), "CreateTexture2D in");
    CK_OK(dev->CreateTexture2D(&td, nullptr, &tout), "CreateTexture2D out");
    Com<ID3D11UnorderedAccessView> uin, uout;
    CK_OK(dev->CreateUnorderedAccessView(tin.ptr(), nullptr, &uin), "CreateUnorderedAccessView in");
    CK_OK(dev->CreateUnorderedAccessView(tout.ptr(), nullptr, &uout), "CreateUnorderedAccessView out");
    D3D11_TEXTURE2D_DESC sd = td;
    sd.BindFlags = 0;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Texture2D> st;
    CK_OK(dev->CreateTexture2D(&sd, nullptr, &st), "CreateTexture2D staging");

    D3D11_BUFFER_DESC tbd{};
    tbd.ByteWidth = 8;
    tbd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    Com<ID3D11Buffer> bin, bout;
    CK_OK(dev->CreateBuffer(&tbd, &init, &bin), "CreateBuffer typed in");
    CK_OK(dev->CreateBuffer(&tbd, nullptr, &bout), "CreateBuffer typed out");
    D3D11_UNORDERED_ACCESS_VIEW_DESC bud{};
    bud.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    bud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    bud.Buffer.NumElements = 2;
    Com<ID3D11UnorderedAccessView> ubin, ubout;
    CK_OK(dev->CreateUnorderedAccessView(bin.ptr(), &bud, &ubin), "CreateUnorderedAccessView typed in");
    CK_OK(dev->CreateUnorderedAccessView(bout.ptr(), &bud, &ubout), "CreateUnorderedAccessView typed out");
    tbd.BindFlags = 0;
    tbd.Usage = D3D11_USAGE_STAGING;
    tbd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Buffer> bst;
    CK_OK(dev->CreateBuffer(&tbd, nullptr, &bst), "CreateBuffer typed staging");

    ID3D11UnorderedAccessView* uavs[] = {uin.ptr(), uout.ptr(), ouav.ptr(), ubin.ptr(), ubout.ptr()};
    ctx->CSSetShader(cs.ptr(), nullptr, 0);
    ctx->CSSetUnorderedAccessViews(0, 5, uavs, nullptr);
    ctx->Dispatch(1, 1, 1);
    ctx->CopyResource(st.ptr(), tout.ptr());
    ctx->CopyResource(ost.ptr(), ob.ptr());
    ctx->CopyResource(bst.ptr(), bout.ptr());

    D3D11_MAPPED_SUBRESOURCE m{};
    CK_OK(ctx->Map(ost.ptr(), 0, D3D11_MAP_READ, 0, &m), "Map buffer");
    float v[8];
    memcpy(v, m.pData, sizeof(v));
    ctx->Unmap(ost.ptr(), 0);
    CK_OK(ctx->Map(st.ptr(), 0, D3D11_MAP_READ, 0, &m), "Map texture");
    uint32_t s[2];
    memcpy(s, m.pData, sizeof(s));
    ctx->Unmap(st.ptr(), 0);

    CK_OK(ctx->Map(bst.ptr(), 0, D3D11_MAP_READ, 0, &m), "Map typed buffer");
    uint32_t b[2];
    memcpy(b, m.pData, sizeof(b));
    ctx->Unmap(bst.ptr(), 0);

    bool all = caps_ok;
    const char* kind[2] = {"texture", "buffer"};
    const uint32_t* stored[2] = {s, b};
    for (int k = 0; k < 2; k++) {
        const float* l = v + 4 * k;
        float s0[3], s1[3];
        decode(stored[k][0], s0);
        decode(stored[k][1], s1);
        const bool load_ok = l[0] == 0.5f && l[1] == 2.0f && l[2] == 0.0f && l[3] == 1.0f;
        const bool store_ok = s0[0] == 1.0f && s0[1] == 0.0f && s0[2] == 0.0f &&
                              s1[0] == 0.0f && s1[1] == 0.25f && s1[2] == 4.0f;
        printf("RGB9E5UAV: %-7s load -> (%g %g %g %g) %s\n", kind[k], l[0], l[1], l[2], l[3],
               load_ok ? "OK" : "MISMATCH");
        printf("RGB9E5UAV: %-7s store -> 0x%08x (%g %g %g) 0x%08x (%g %g %g) %s\n", kind[k],
               stored[k][0], s0[0], s0[1], s0[2], stored[k][1], s1[0], s1[1], s1[2],
               store_ok ? "OK" : "MISMATCH");
        all = all && load_ok && store_ok;
    }
    return all ? 0 : 1;
}
