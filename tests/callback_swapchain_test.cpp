/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Callback-backed swapchain (dmn_window_create_with_callbacks): the embedder
 * supplies MTLTextures and receives configure/acquire/present callbacks — no
 * CAMetalLayer, no window, no main thread. End to end:
 *
 *   1. CreateSwapChainForHwnd on a callbacks window fires configure() with
 *      the swapchain's dims.
 *   2. Each frame D3DMetal acquires an embedder texture, renders the cleared
 *      backbuffer into it, and present() fires with that texture's slot.
 *   3. The final frame's clear color is verified IN the embedder's texture
 *      (StorageModeShared readback) — the whole point of the backend.
 *
 * The Metal side lives in callback_embedder.mm (the vendored Windows headers
 * cannot be included from ObjC++). Prints "CBSWAP: PASS" and exits 0.
 */

#include <cstdint>
#include <cstdio>

#include <time.h>

#include <d3d11_1.h>
#include <dxgi1_2.h>

#include "d3dmetal_native.h"
#include "callback_embedder.h"
#include "common/com.h"

#define T_TAG "CBSWAP"
#include "common/check.h"
#include "common/dx11.h"

static const uint32_t kW = 320, kH = 200, kFrames = 8;

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "CBSWAP: dmn_init FAILED\n");
        return 1;
    }

    void* emb = cbemb_create();
    EXPECT(emb, "no Metal device");
    dmn_window_callbacks cb;
    cbemb_fill_callbacks(emb, &cb);
    dmn_window_t window = dmn_window_create_with_callbacks(&cb, kW, kH);
    EXPECT(window, "dmn_window_create_with_callbacks failed");
    void* hwnd = dmn_window_get_hwnd(window);

    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    CK(make_d3d11_device(dev, ctx), "D3D11CreateDevice");

    Com<IDXGIDevice> dxgiDev;
    Com<IDXGIAdapter> adapter;
    Com<IDXGIFactory2> factory2;
    CK(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev), "IDXGIDevice");
    CK(dxgiDev->GetAdapter(&adapter), "GetAdapter");
    CK(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory2),
       "GetParent(IDXGIFactory2)");

    DXGI_SWAP_CHAIN_DESC1 scDesc{};
    scDesc.Width = kW;
    scDesc.Height = kH;
    scDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scDesc.SampleDesc = {1, 0};
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = 2;
    scDesc.Scaling = DXGI_SCALING_STRETCH;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    Com<IDXGISwapChain1> swapchain;
    CK(factory2->CreateSwapChainForHwnd(dev.ptr(), (HWND)hwnd, &scDesc, nullptr,
                                        nullptr, &swapchain),
       "CreateSwapChainForHwnd(callbacks window)");
    EXPECT(cbemb_configures(emb) >= 1, "configure() never fired");
    EXPECT(cbemb_config_w(emb) == kW && cbemb_config_h(emb) == kH,
           "configure() dims mismatch");

    /* Render kFrames clears; the last one is the color we verify. */
    const float kLast[4] = {0.2f, 0.4f, 0.6f, 1.0f};
    for (uint32_t i = 0; i < kFrames; i++) {
        Com<ID3D11Texture2D> bb;
        CK(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb),
           "GetBuffer");
        Com<ID3D11RenderTargetView> rtv;
        CK(dev->CreateRenderTargetView(bb.ptr(), nullptr, &rtv), "RTV");
        float c[4] = {0.05f * i, 0.1f, 0.3f, 1.0f};
        const float* color = (i == kFrames - 1) ? kLast : c;
        ctx->ClearRenderTargetView(rtv.ptr(), color);
        CK(swapchain->Present(0, 0), "Present");
    }
    ctx->Flush();

    /* D3DMetal presents from an async scheduled handler; give the last
     * frames a moment to land. */
    for (int i = 0; i < 100 && cbemb_presents(emb) < kFrames; i++) {
        struct timespec ns = {0, 50 * 1000 * 1000};
        nanosleep(&ns, nullptr);
    }
    EXPECT(cbemb_acquires(emb) >= kFrames, "acquire_texture underfired");
    EXPECT(cbemb_presents(emb) >= kFrames, "present underfired");
    printf("CBSWAP: %u configures, %u acquires, %u presents, last slot %u\n",
           cbemb_configures(emb), cbemb_acquires(emb), cbemb_presents(emb),
           cbemb_last_slot(emb));

    /* Give the GPU blit into the embedder texture a moment, then read the
     * center pixel of the last-presented slot. BGRA8: B=0.6 G=0.4 R=0.2. */
    struct timespec ns = {0, 300 * 1000 * 1000};
    nanosleep(&ns, nullptr);
    uint8_t px[4] = {};
    EXPECT(cbemb_read_center(emb, px), "no presented texture to read");
    auto near8 = [](uint8_t got, float want) {
        int w = (int)(want * 255.0f + 0.5f);
        return got >= w - 2 && got <= w + 2;
    };
    EXPECT(near8(px[0], kLast[2]) && near8(px[1], kLast[1]) &&
           near8(px[2], kLast[0]) && near8(px[3], kLast[3]),
           "final clear color not found in the embedder texture");
    printf("CBSWAP: pixel check ok (B=%u G=%u R=%u A=%u)\n", px[0], px[1],
           px[2], px[3]);

    dmn_window_destroy(window);
    cbemb_destroy(emb);
    T_PASS();
    return 0;
}
