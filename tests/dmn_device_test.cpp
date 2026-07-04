/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Device test (windowless):
 *   phase 1 — factory/adapter enumeration, D3D11 device creation,
 *             headless clear+readback, D3DCompile.
 *   phase 2 — swapchain on an offscreen CAMetalLayer: CreateSwapChainForHwnd
 *             with a dmn pseudo-HWND, GetBuffer, clear, Present x10.
 *
 * Prints "DEVICE: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>

#include <d3dcompiler.h>
#include <d3d11_1.h>
#include <dxgi1_6.h>

#include "d3dmetal_native.h"
#include "cocoa_window.h"
#include "common/com.h"

#define T_TAG "DEVICE"
#include "common/check.h"


static const char g_vertexShaderCode[] =
    "float4 main(float4 v_pos : IN_POSITION) : SV_POSITION {\n"
    "  return v_pos;\n"
    "}\n";

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "DEVICE: dmn_init FAILED\n");
        return 1;
    }
    printf("DEVICE: dmn_init: OK\n");

    /* == Phase 1: factory, adapter, device, headless render ============= */

    Com<IDXGIFactory1> factory;
    CK_OK(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory),
          "CreateDXGIFactory1");

    Com<IDXGIAdapter1> adapter;
    CK_OK(factory->EnumAdapters1(0, &adapter), "EnumAdapters1(0)");

    DXGI_ADAPTER_DESC1 adapterDesc = {};
    CK_OK(adapter->GetDesc1(&adapterDesc), "GetDesc1");
    {
        char name[129] = {};
        for (int i = 0; i < 128 && adapterDesc.Description[i]; i++) {
            WCHAR c = adapterDesc.Description[i];
            name[i] = (c >= 0x20 && c < 0x7f) ? (char)c : '?';
        }
        printf("DEVICE: adapter: \"%s\" vendor=0x%04x device=0x%04x "
               "luid=%08x:%08x vram=%zuMB\n",
               name, adapterDesc.VendorId, adapterDesc.DeviceId,
               (unsigned)adapterDesc.AdapterLuid.HighPart,
               (unsigned)adapterDesc.AdapterLuid.LowPart,
               (size_t)(adapterDesc.DedicatedVideoMemory >> 20));
    }

    Com<ID3D11Device> device;
    Com<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL flOut = (D3D_FEATURE_LEVEL)0;
    {
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
        CK_OK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                &fl, 1, D3D11_SDK_VERSION, &device, &flOut,
                                &context),
              "D3D11CreateDevice");
        printf("DEVICE: feature level: 0x%04x\n", (unsigned)flOut);
    }

    /* Headless render: clear a small RT red, copy to staging, verify. */
    {
        D3D11_TEXTURE2D_DESC texDesc = {};
        texDesc.Width = 64;
        texDesc.Height = 64;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc = {1, 0};
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = D3D11_BIND_RENDER_TARGET;

        Com<ID3D11Texture2D> rt;
        CK_OK(device->CreateTexture2D(&texDesc, nullptr, &rt),
              "CreateTexture2D(rt)");

        Com<ID3D11RenderTargetView> rtv;
        CK_OK(device->CreateRenderTargetView(rt.ptr(), nullptr, &rtv),
              "CreateRenderTargetView");

        FLOAT red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        context->ClearRenderTargetView(rtv.ptr(), red);

        texDesc.Usage = D3D11_USAGE_STAGING;
        texDesc.BindFlags = 0;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        Com<ID3D11Texture2D> staging;
        CK_OK(device->CreateTexture2D(&texDesc, nullptr, &staging),
              "CreateTexture2D(staging)");

        context->CopyResource(staging.ptr(), rt.ptr());

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        CK_OK(context->Map(staging.ptr(), 0, D3D11_MAP_READ, 0, &mapped),
              "Map(staging)");

        const uint8_t* px = (const uint8_t*)mapped.pData;
        bool ok = px[0] == 0xff && px[1] == 0x00 && px[2] == 0x00 &&
                  px[3] == 0xff;
        printf("DEVICE: readback pixel: %02x %02x %02x %02x (%s)\n", px[0],
               px[1], px[2], px[3], ok ? "OK" : "MISMATCH");
        context->Unmap(staging.ptr(), 0);
        if (!ok)
            return 1;
    }

    /* D3DCompile — exercises the dxcompiler load path through the GFXT
     * library interface before any windowing exists. */
    {
        Com<ID3DBlob> code;
        Com<ID3DBlob> errors;
        HRESULT hr = D3DCompile(g_vertexShaderCode, sizeof(g_vertexShaderCode) - 1,
                                "vs", nullptr, nullptr, "main", "vs_5_0", 0, 0,
                                &code, &errors);
        if (FAILED(hr)) {
            fprintf(stderr, "DEVICE: D3DCompile FAILED hr=0x%08x: %s\n",
                    (unsigned)hr,
                    errors ? (const char*)errors->GetBufferPointer() : "?");
            return 1;
        }
        printf("DEVICE: D3DCompile: OK (%zu bytes of DXBC)\n",
               code->GetBufferSize());
    }

    /* == Phase 2: offscreen-layer swapchain ============================== */

    void* layer = cocoa_create_offscreen_layer(640, 480);
    dmn_window_t window = dmn_window_create_for_layer(layer);
    if (!window) {
        fprintf(stderr, "DEVICE: dmn_window_create_for_layer FAILED\n");
        return 1;
    }
    void* hwnd = dmn_window_get_hwnd(window);
    printf("DEVICE: pseudo-hwnd: %p\n", hwnd);

    Com<IDXGIFactory2> factory2;
    CK_OK(factory->QueryInterface(__uuidof(IDXGIFactory2), (void**)&factory2),
          "QueryInterface(IDXGIFactory2)");

    DXGI_SWAP_CHAIN_DESC1 scDesc = {};
    scDesc.Width = 640;
    scDesc.Height = 480;
    scDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scDesc.SampleDesc = {1, 0};
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = 2;
    scDesc.Scaling = DXGI_SCALING_STRETCH;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    scDesc.Flags = 0;

    Com<IDXGISwapChain1> swapchain;
    CK_OK(factory2->CreateSwapChainForHwnd(device.ptr(), (HWND)hwnd, &scDesc,
                                           nullptr, nullptr, &swapchain),
          "CreateSwapChainForHwnd");

    Com<ID3D11Texture2D> backbuffer;
    CK_OK(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D),
                               (void**)&backbuffer),
          "GetBuffer(0)");

    {
        D3D11_TEXTURE2D_DESC bbDesc = {};
        backbuffer->GetDesc(&bbDesc);
        printf("DEVICE: backbuffer: %ux%u fmt=%u bind=0x%x\n", bbDesc.Width,
               bbDesc.Height, (unsigned)bbDesc.Format, bbDesc.BindFlags);
    }

    Com<ID3D11RenderTargetView> rtv;
    CK_OK(device->CreateRenderTargetView(backbuffer.ptr(), nullptr, &rtv),
          "CreateRenderTargetView(backbuffer)");

    for (int frame = 0; frame < 10; frame++) {
        FLOAT color[4] = {0.0f, (frame % 2) ? 1.0f : 0.5f, 0.25f, 1.0f};
        context->ClearRenderTargetView(rtv.ptr(), color);
        HRESULT hr = swapchain->Present(0, 0);
        if (FAILED(hr)) {
            fprintf(stderr, "DEVICE: Present #%d FAILED hr=0x%08x\n", frame,
                    (unsigned)hr);
            return 1;
        }
    }
    printf("DEVICE: Present x10: OK\n");

    /* Teardown order: views/buffers, swapchain, then the dmn window. */
    rtv = nullptr;
    backbuffer = nullptr;
    swapchain = nullptr;
    dmn_window_destroy(window);
    cocoa_release_layer(layer);

    printf("DEVICE: PASS\n");
    return 0;
}
