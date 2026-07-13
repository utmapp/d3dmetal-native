/*
 * Regression: shader-sampling an opened (consumer-side) shared texture's SRV.
 *
 * D3DMetal reconstructs a MISC_SHARED surface opened with OpenSharedResource as
 * an MTLTextureType2DArray and emits texture2d_array sample code for its SRV,
 * but the substituted shared-memory backing is a buffer-backed *linear* Metal
 * texture, which Metal forces to plain MTLTextureType2D. Sampling a 2D texture
 * through the array-typed binding reads a garbage array dimension and returns
 * zero for the colour channels — a flat box where a sampled shared surface
 * (e.g. every DWM-composited XAML glyph) should be. A CopyResource/blit of the
 * same surface reads back correctly, so the fault is specific to shader
 * sampling.
 *
 * A MISC_SHARED texture is filled with a per-texel gradient, then reopened with
 * OpenSharedResource (the consumer-reconstruct path DWM uses) and sampled
 * through an SRV in a pixel shader; the shaded result is read back. The sampled
 * gradient must match the source; the bug flattens the colour channels to zero.
 * (Filling, opening and sampling share one device so the readback is not gated
 * on cross-device propagation of the fill.)
 *
 * Prints "SRVSAMPLE: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "SRVSAMPLE"
#include "common/check.h"

namespace {

constexpr uint32_t kW = 64, kH = 64;

/* Per-texel gradient (BGRA8): B = x scaled, G = y scaled, R = (x+y) scaled.
 * A flat/box sample zeroes the colour channels, which this catches. */
void gradient_texel(uint32_t x, uint32_t y, uint8_t out[4]) {
    out[0] = (uint8_t)(x * 255 / (kW - 1));        /* B */
    out[1] = (uint8_t)(y * 255 / (kH - 1));        /* G */
    out[2] = (uint8_t)((x + y) * 255 / (kW + kH)); /* R */
    out[3] = 0xff;                                 /* A */
}

const char* kVS =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut main(uint vid : SV_VertexID) {\n"
    "    VSOut o;\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    o.uv = p;\n"
    "    o.pos = float4(p * 2.0 - 1.0, 0, 1);\n"
    "    o.pos.y = -o.pos.y;\n"
    "    return o;\n"
    "}\n";

/* Normalised-coordinate sample (the natural D2D/DWM path); this is the access
 * that flattens for a 2D-as-array binding. */
const char* kPS =
    "Texture2D tex : register(t0);\n"
    "SamplerState smp : register(s0);\n"
    "float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n"
    "    return tex.Sample(smp, uv);\n"
    "}\n";

HRESULT make_device(Com<ID3D11Device>& dev, Com<ID3D11DeviceContext>& ctx) {
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    return D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                             D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                             D3D11_SDK_VERSION, &dev, &flo, &ctx);
}

/* CopyResource a texture into a fresh CPU-readable staging copy and read texel
 * (x,y). Also forces the source GPU work to complete (Map blocks). */
int read_texel(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11Texture2D* src,
               uint32_t x, uint32_t y, uint8_t out[4]) {
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
    const uint8_t* p = (const uint8_t*)m.pData + (size_t)y * m.RowPitch + (size_t)x * 4;
    memcpy(out, p, 4);
    ctx->Unmap(staging.ptr(), 0);
    return 0;
}

int run() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }

    /* Create + fill a MISC_SHARED gradient texture. */
    Com<ID3D11Device> pdev;
    Com<ID3D11DeviceContext> pctx;
    CK(make_device(pdev, pctx), "D3D11CreateDevice");

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
    CK(pdev->CreateTexture2D(&td, nullptr, &shared), "CreateTexture2D(shared)");

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
    CK(pdev->CreateTexture2D(&ud, nullptr, &upload), "CreateTexture2D(upload)");
    D3D11_MAPPED_SUBRESOURCE mm = {};
    CK(pctx->Map(upload.ptr(), 0, D3D11_MAP_WRITE, 0, &mm), "Map(upload)");
    for (uint32_t y = 0; y < kH; y++)
        memcpy((uint8_t*)mm.pData + (size_t)y * mm.RowPitch,
               fill.data() + (size_t)y * kW * 4, (size_t)kW * 4);
    pctx->Unmap(upload.ptr(), 0);
    pctx->CopyResource(shared.ptr(), upload.ptr());
    pctx->Flush();

    /* Export the shared handle (intercepted GetSharedHandle -> POD). */
    Com<IDXGIResource> dxgiRes;
    CK(shared->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgiRes),
       "QI(IDXGIResource)");
    HANDLE texH = nullptr;
    CK(dxgiRes->GetSharedHandle(&texH), "GetSharedHandle");
    EXPECT(texH != nullptr, "GetSharedHandle returned null (shared tex not intercepted)");
    dmn_shared_texture_handle wire = {};
    memcpy(&wire, texH, sizeof(wire));

    /* Reopen the surface (consumer-reconstruct path) and sample its SRV. Same
     * device, so the readback is not gated on cross-device fill propagation. */
    Com<ID3D11Device>& cdev = pdev;
    Com<ID3D11DeviceContext>& cctx = pctx;
    Com<ID3D11Texture2D> opened;
    CK(cdev->OpenSharedResource((HANDLE)&wire, __uuidof(ID3D11Texture2D),
                                (void**)&opened),
       "OpenSharedResource(texture)");

    /* Control: blit readback of the opened surface — the always-correct path. */
    uint8_t bc[4];
    EXPECT(read_texel(cdev.ptr(), cctx.ptr(), opened.ptr(), 48, 16, bc) == 0,
           "blit readback failed");
    uint8_t want[4];
    gradient_texel(48, 16, want);
    printf(T_TAG ": blit  readback BGRA=%02x%02x%02x%02x (expect %02x%02x%02x%02x)\n",
           bc[0], bc[1], bc[2], bc[3], want[0], want[1], want[2], want[3]);
    EXPECT(bc[0] == want[0] && bc[1] == want[1] && bc[2] == want[2],
           "blit readback did not match gradient (producer/copy path broken)");

    /* Sample the opened SRV through a pixel shader into a private RT. */
    Com<ID3D11ShaderResourceView> srv;
    CK(cdev->CreateShaderResourceView(opened.ptr(), nullptr, &srv), "CreateSRV");

    D3D11_TEXTURE2D_DESC rd = td;
    rd.BindFlags = D3D11_BIND_RENDER_TARGET;
    rd.MiscFlags = 0;
    Com<ID3D11Texture2D> rt;
    CK(cdev->CreateTexture2D(&rd, nullptr, &rt), "CreateTexture2D(rt)");
    Com<ID3D11RenderTargetView> rtv;
    CK(cdev->CreateRenderTargetView(rt.ptr(), nullptr, &rtv), "CreateRTV");

    Com<ID3DBlob> vsb, psb, errb;
    CK(D3DCompile(kVS, strlen(kVS), "vs", nullptr, nullptr, "main", "vs_5_0", 0, 0,
                  &vsb, &errb), "VS compile");
    CK(D3DCompile(kPS, strlen(kPS), "ps", nullptr, nullptr, "main", "ps_5_0", 0, 0,
                  &psb, &errb), "PS compile");
    Com<ID3D11VertexShader> vs;
    Com<ID3D11PixelShader> ps;
    CK(cdev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr,
                                &vs), "CreateVS");
    CK(cdev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr,
                               &ps), "CreatePS");
    D3D11_SAMPLER_DESC smpd = {};
    smpd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    smpd.AddressU = smpd.AddressV = smpd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    Com<ID3D11SamplerState> samp;
    CK(cdev->CreateSamplerState(&smpd, &samp), "CreateSampler");

    FLOAT clear[4] = {0, 0, 0, 1};
    cctx->ClearRenderTargetView(rtv.ptr(), clear);
    cctx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp = {0, 0, (float)kW, (float)kH, 0, 1};
    cctx->RSSetViewports(1, &vp);
    cctx->IASetInputLayout(nullptr);
    cctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cctx->VSSetShader(vs.ptr(), nullptr, 0);
    cctx->PSSetShader(ps.ptr(), nullptr, 0);
    ID3D11ShaderResourceView* srvs[] = {srv.ptr()};
    cctx->PSSetShaderResources(0, 1, srvs);
    ID3D11SamplerState* samps[] = {samp.ptr()};
    cctx->PSSetSamplers(0, 1, samps);
    cctx->Draw(3, 0);
    cctx->Flush();

    /* Read the shaded texel that sampled source (48,16). */
    uint8_t sc[4];
    EXPECT(read_texel(cdev.ptr(), cctx.ptr(), rt.ptr(), 48, 16, sc) == 0,
           "sampled readback failed");
    printf(T_TAG ": shader sample BGRA=%02x%02x%02x%02x (expect %02x%02x%02x%02x)\n",
           sc[0], sc[1], sc[2], sc[3], want[0], want[1], want[2], want[3]);

    /* Tolerate filter/sRGB rounding; the bug returns flat zero colour (box). */
    auto near = [](uint8_t a, uint8_t b) { return (a > b ? a - b : b - a) <= 8; };
    EXPECT(near(sc[0], want[0]) && near(sc[1], want[1]) && near(sc[2], want[2]),
           "shader-sampled shared SRV colour flattened (glyph-box regression)");

    T_PASS();
    return 0;
}

}  // namespace

int main() { return run(); }
