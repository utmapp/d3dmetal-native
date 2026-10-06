/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Packed 16-bit textures stay sampleable in DXGI packing: upload texels,
 * Load them in a pixel shader into a BGRA8 target and read back.  The
 * render side lives in packed-rt-test; B4G4R4A4 must keep passing here even
 * though it is not a render target (D3DMetal swizzles its reads).
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <d3dcompiler.h>
#include <d3d11_1.h>
#include "d3dmetal_native.h"
#include "common/com.h"
#define T_TAG "PACKEDSAMPLE"
#include "common/check.h"
#include "common/dx11.h"

namespace {
const char kVS[] =
    "float4 main(uint id : SV_VertexID) : SV_Position {\n"
    "  float2 p = float2((id << 1) & 2, id & 2);\n"
    "  return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "}\n";
const char kPS[] =
    "Texture2D<float4> t : register(t0);\n"
    "float4 main(float4 pos : SV_Position) : SV_Target {\n"
    "  return t.Load(int3(pos.xy, 0));\n"
    "}\n";

struct Fmt {
    DXGI_FORMAT f;
    const char* name;
    uint16_t texel;      /* DXGI packing: first-named channel in the low bits */
    uint8_t want[4];     /* expected r,g,b,a after UNORM8 conversion */
};

/* r=11/15 g=4/15 b=8/15 a=1 in B4G4R4A4 -> 0xfb48; the 5-bit formats use
 * r=24/31 g=8/31(5b) or 16/63(6b) b=16/31. */
const Fmt kFmts[] = {
    {DXGI_FORMAT_B4G4R4A4_UNORM, "B4G4R4A4", 0xfb48, {0xbb, 0x44, 0x88, 0xff}},
    {DXGI_FORMAT_B5G6R5_UNORM, "B5G6R5", (uint16_t)((24u << 11) | (16u << 5) | 16u),
     {0xc6, 0x41, 0x84, 0xff}},
    {DXGI_FORMAT_B5G5R5A1_UNORM, "B5G5R5A1",
     (uint16_t)((1u << 15) | (24u << 10) | (8u << 5) | 16u), {0xc6, 0x42, 0x84, 0xff}},
};

bool near(unsigned a, unsigned b) { return (a > b ? a - b : b - a) <= 2; }
} // namespace

int main() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    CK_OK(make_d3d11_device(dev, ctx), "D3D11CreateDevice");

    Com<ID3DBlob> vsb, psb, err;
    CK_OK(D3DCompile(kVS, sizeof(kVS) - 1, "vs", nullptr, nullptr, "main", "vs_5_0", 0, 0,
                     &vsb, &err), "D3DCompile vs");
    CK_OK(D3DCompile(kPS, sizeof(kPS) - 1, "ps", nullptr, nullptr, "main", "ps_5_0", 0, 0,
                     &psb, &err), "D3DCompile ps");
    Com<ID3D11VertexShader> vs;
    Com<ID3D11PixelShader> ps;
    CK_OK(dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs),
          "CreateVertexShader");
    CK_OK(dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps),
          "CreatePixelShader");

    enum { kW = 4, kH = 4 };
    D3D11_TEXTURE2D_DESC rd{};
    rd.Width = kW;
    rd.Height = kH;
    rd.MipLevels = 1;
    rd.ArraySize = 1;
    rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.BindFlags = D3D11_BIND_RENDER_TARGET;
    Com<ID3D11Texture2D> rt;
    CK_OK(dev->CreateTexture2D(&rd, nullptr, &rt), "CreateTexture2D rt");
    Com<ID3D11RenderTargetView> rtv;
    CK_OK(dev->CreateRenderTargetView(rt.ptr(), nullptr, &rtv), "CreateRenderTargetView");
    D3D11_TEXTURE2D_DESC sd = rd;
    sd.BindFlags = 0;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Texture2D> staging;
    CK_OK(dev->CreateTexture2D(&sd, nullptr, &staging), "CreateTexture2D staging");

    bool all = true;
    for (const Fmt& fmt : kFmts) {
        uint16_t texels[kW * kH];
        for (auto& t : texels)
            t = fmt.texel;
        D3D11_TEXTURE2D_DESC td = rd;
        td.Format = fmt.f;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA init{texels, kW * 2, 0};
        Com<ID3D11Texture2D> tex;
        if (FAILED(dev->CreateTexture2D(&td, &init, &tex))) {
            printf("PACKEDSAMPLE: %-9s create FAILED\n", fmt.name);
            all = false;
            continue;
        }
        Com<ID3D11ShaderResourceView> srv;
        CK_OK(dev->CreateShaderResourceView(tex.ptr(), nullptr, &srv), "CreateShaderResourceView");

        const float black[4] = {0, 0, 0, 0};
        ctx->ClearRenderTargetView(rtv.ptr(), black);
        ID3D11RenderTargetView* rtvs[] = {rtv.ptr()};
        ctx->OMSetRenderTargets(1, rtvs, nullptr);
        D3D11_VIEWPORT vp{0, 0, kW, kH, 0, 1};
        ctx->RSSetViewports(1, &vp);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs.ptr(), nullptr, 0);
        ctx->PSSetShader(ps.ptr(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = {srv.ptr()};
        ctx->PSSetShaderResources(0, 1, srvs);
        ctx->Draw(3, 0);
        ctx->CopyResource(staging.ptr(), rt.ptr());

        D3D11_MAPPED_SUBRESOURCE m{};
        CK_OK(ctx->Map(staging.ptr(), 0, D3D11_MAP_READ, 0, &m), "Map staging");
        const uint8_t* p = (const uint8_t*)m.pData + (kH / 2) * m.RowPitch + (kW / 2) * 4;
        const unsigned b = p[0], g = p[1], r = p[2], a = p[3];
        ctx->Unmap(staging.ptr(), 0);
        const bool ok = near(r, fmt.want[0]) && near(g, fmt.want[1]) &&
                        near(b, fmt.want[2]) && near(a, fmt.want[3]);
        printf("PACKEDSAMPLE: %-9s texel 0x%04x -> rgba %02x %02x %02x %02x (want %02x %02x %02x %02x) %s\n",
               fmt.name, fmt.texel, r, g, b, a, fmt.want[0], fmt.want[1], fmt.want[2],
               fmt.want[3], ok ? "OK" : "MISMATCH");
        all = all && ok;
    }
    return all ? 0 : 1;
}
