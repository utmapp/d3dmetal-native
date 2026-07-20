/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Regression: opening a shared surface must not discard what is already in it.
 *
 * OpenSharedResource reconstructs the surface by calling CreateTexture2D on the
 * consumer's device. Creating a texture with no initial data leaves D3DMetal
 * holding it as "undefined", and the first time it is attached as a render
 * target the attachment loads with a clear rather than with the existing
 * contents — so the producer's pixels vanish the moment the consumer renders
 * to the surface, everywhere outside whatever the consumer actually drew.
 *
 * D3D says the opposite: an opened shared resource aliases the producer's
 * storage, and a render pass that does not cover a texel must leave it alone.
 *
 * The check is deliberately made against the shared mapping itself rather than
 * through a D3D readback: that mapping is the ground truth a peer process sees,
 * and it takes CopyResource, staging textures and format conversion out of the
 * loop. The producer's fill is verified there first, so a failure after the
 * consumer's draw can only be the attach.
 *
 * The draw is confined to a small corner by the viewport, leaving the rest of
 * the surface as the untouched region the assertion reads. One device
 * throughout: this is about the create-with-no-initial-data path, not about
 * cross-device or cross-process propagation.
 *
 * Prints "OPENPRES: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <sys/mman.h>

#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "OPENPRES"
#include "common/check.h"

namespace {

constexpr uint32_t kW = 64, kH = 64;
/* The consumer draws only into this corner; everything outside it is what the
 * assertion reads back. */
constexpr uint32_t kDrawW = 8, kDrawH = 8;

/* Per-texel gradient (BGRA8), so a wipe cannot be mistaken for a stale-but-
 * plausible value the way a solid fill could. */
void gradient_texel(uint32_t x, uint32_t y, uint8_t out[4]) {
    out[0] = (uint8_t)(x * 255 / (kW - 1));        /* B */
    out[1] = (uint8_t)(y * 255 / (kH - 1));        /* G */
    out[2] = (uint8_t)((x + y) * 255 / (kW + kH)); /* R */
    out[3] = 0xff;                                 /* A */
}

const char* kVS =
    "float4 main(uint vid : SV_VertexID) : SV_Position {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    return float4(p * 2.0 - 1.0, 0, 1);\n"
    "}\n";

const char* kPS =
    "float4 main(float4 pos : SV_Position) : SV_Target {\n"
    "    return float4(0, 1, 0, 1);\n"
    "}\n";

HRESULT compile(const char* src, const char* target, Com<ID3DBlob>& out) {
    Com<ID3DBlob> err;
    HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr, nullptr, "main",
                            target, 0, 0, &out, &err);
    if (FAILED(hr) && err.ptr())
        fprintf(stderr, T_TAG ": shader compile: %.*s\n",
                (int)err->GetBufferSize(), (const char*)err->GetBufferPointer());
    return hr;
}

/* Force every submitted command to retire. Map on a staging copy blocks until
 * the GPU is done, which is what makes the mapping below safe to read. */
int drain(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* src) {
    D3D11_TEXTURE2D_DESC d = {};
    src->GetDesc(&d);
    d.BindFlags = 0;
    d.MiscFlags = 0;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Texture2D> staging;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &staging)))
        return 1;
    ctx->CopyResource(staging.ptr(), src);
    ctx->Flush();
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(staging.ptr(), 0, D3D11_MAP_READ, 0, &m)))
        return 1;
    ctx->Unmap(staging.ptr(), 0);
    return 0;
}

/* Compare the shared mapping against the gradient, skipping the drawn corner.
 * Returns the number of mismatching texels and reports the first one. */
size_t check_untouched(const uint8_t* base, uint64_t stride) {
    size_t bad = 0;
    for (uint32_t y = 0; y < kH; y++) {
        for (uint32_t x = 0; x < kW; x++) {
            if (x < kDrawW && y < kDrawH)
                continue; /* the consumer legitimately wrote here */
            uint8_t want[4];
            gradient_texel(x, y, want);
            const uint8_t* got = base + (uint64_t)y * stride + (uint64_t)x * 4;
            if (memcmp(got, want, 3) != 0) {
                if (bad == 0)
                    fprintf(stderr,
                            T_TAG ": (%u,%u) BGRA=%02x%02x%02x%02x expected "
                            "%02x%02x%02x%02x\n", x, y, got[0], got[1], got[2],
                            got[3], want[0], want[1], want[2], want[3]);
                bad++;
            }
        }
    }
    return bad;
}

int run() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                         D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                         D3D11_SDK_VERSION, &dev, &flo, &ctx),
       "D3D11CreateDevice");

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
    Com<ID3D11Texture2D> shared;
    CK(dev->CreateTexture2D(&td, nullptr, &shared), "CreateTexture2D(shared)");

    /* Producer fill, via a staging upload + CopyResource. */
    std::vector<uint8_t> fill((size_t)kW * kH * 4);
    for (uint32_t y = 0; y < kH; y++)
        for (uint32_t x = 0; x < kW; x++)
            gradient_texel(x, y, &fill[((size_t)y * kW + x) * 4]);

    D3D11_TEXTURE2D_DESC ud = td;
    ud.BindFlags = 0;
    ud.MiscFlags = 0;
    ud.Usage = D3D11_USAGE_STAGING;
    ud.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    Com<ID3D11Texture2D> upload;
    CK(dev->CreateTexture2D(&ud, nullptr, &upload), "CreateTexture2D(upload)");
    D3D11_MAPPED_SUBRESOURCE mm = {};
    CK(ctx->Map(upload.ptr(), 0, D3D11_MAP_WRITE, 0, &mm), "Map(upload)");
    for (uint32_t y = 0; y < kH; y++)
        memcpy((uint8_t*)mm.pData + (size_t)y * mm.RowPitch,
               fill.data() + (size_t)y * kW * 4, (size_t)kW * 4);
    ctx->Unmap(upload.ptr(), 0);
    ctx->CopyResource(shared.ptr(), upload.ptr());
    ctx->Flush();
    EXPECT(drain(dev.ptr(), ctx.ptr(), shared.ptr()) == 0, "drain after fill");

    /* Map the backing the way a peer would, and confirm the fill is there
     * before anything is opened. */
    Com<IDXGIResource> dxgiRes;
    CK(shared->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgiRes),
       "QI(IDXGIResource)");
    HANDLE texH = nullptr;
    CK(dxgiRes->GetSharedHandle(&texH), "GetSharedHandle");
    EXPECT(texH != nullptr, "GetSharedHandle returned null");
    dmn_shared_texture_handle wire = {};
    memcpy(&wire, texH, sizeof(wire));

    uint8_t* view = (uint8_t*)mmap(nullptr, (size_t)wire.size, PROT_READ,
                                   MAP_SHARED, wire.fd, 0);
    EXPECT(view != MAP_FAILED, "mmap of the shared backing failed");

    size_t bad = check_untouched(view, wire.stride);
    if (bad) {
        fprintf(stderr, T_TAG ": producer fill did not reach the shared "
                        "backing (%zu texels wrong) — repro is invalid\n", bad);
        return 1;
    }
    printf(T_TAG ": producer fill present in the shared backing: OK\n");

    /* Consumer reconstruct. */
    Com<ID3D11Texture2D> opened;
    CK(dev->OpenSharedResource((HANDLE)&wire, __uuidof(ID3D11Texture2D),
                               (void**)&opened),
       "OpenSharedResource");

    /* Attach the opened surface and draw into one corner only. */
    Com<ID3DBlob> vsb, psb;
    CK(compile(kVS, "vs_5_0", vsb), "compile VS");
    CK(compile(kPS, "ps_5_0", psb), "compile PS");
    Com<ID3D11VertexShader> vs;
    Com<ID3D11PixelShader> ps;
    CK(dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(),
                               nullptr, &vs), "CreateVertexShader");
    CK(dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(),
                              nullptr, &ps), "CreatePixelShader");

    Com<ID3D11RenderTargetView> rtv;
    CK(dev->CreateRenderTargetView(opened.ptr(), nullptr, &rtv),
       "CreateRenderTargetView(opened)");

    D3D11_VIEWPORT vp = {0.0f, 0.0f, (float)kDrawW, (float)kDrawH, 0.0f, 1.0f};
    ctx->RSSetViewports(1, &vp);
    ID3D11RenderTargetView* rtvs[] = {rtv.ptr()};
    ctx->OMSetRenderTargets(1, rtvs, nullptr);
    ctx->VSSetShader(vs.ptr(), nullptr, 0);
    ctx->PSSetShader(ps.ptr(), nullptr, 0);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->Draw(3, 0);
    ctx->Flush();
    EXPECT(drain(dev.ptr(), ctx.ptr(), opened.ptr()) == 0, "drain after draw");

    /* Everything outside the drawn corner must still hold the producer's
     * gradient. A whole-surface wipe here is the attach clearing a texture
     * D3DMetal considers undefined because it was created with no data. */
    bad = check_untouched(view, wire.stride);
    munmap(view, (size_t)wire.size);
    if (bad) {
        fprintf(stderr,
                T_TAG ": %zu of %u texels outside the drawn %ux%u corner were "
                "discarded by rendering to the opened surface\n",
                bad, kW * kH - kDrawW * kDrawH, kDrawW, kDrawH);
        return 1;
    }
    printf(T_TAG ": contents survived the consumer's render pass: OK\n");

    T_PASS();
    return 0;
}

} // namespace

int main() { return run(); }
