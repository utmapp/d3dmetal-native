/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Exported swapchain (dmn_window_create_exported): the library allocates
 * shared-memory-backed images and drives the three embedder callbacks — no
 * CAMetalLayer, no window, no Metal objects on the embedder side. End to
 * end:
 *
 *   1. CreateSwapChainForHwnd on an exported window fires on_images_changed
 *      with the image set (fds, stride, size) before it returns.
 *   2. Each frame D3DMetal acquires a slot (on_acquire), renders the cleared
 *      backbuffer into its image, and on_present fires with the slot and a
 *      GPU-done fence fd.
 *   3. The final frame's clear color is verified by reading the presented
 *      pixels straight out of the mmap'd shared memory, gated on the fence
 *      fd — the whole point of the backend.
 *   4. ResizeBuffers republishes a new image set (on_images_changed again,
 *      new dimensions) and frames keep flowing; frame_id stays monotonic
 *      across the resize.
 *   5. A second swapchain with a non-presentable desc format exercises
 *      preferred_dxgi_format (allocation falls back to it) and
 *      preferred_image_count (overrides BufferCount).
 *
 * Prints "EXPSWAP: PASS" and exits 0.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <poll.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <d3d11_1.h>
#include <dxgi1_2.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "EXPSWAP"
#include "common/check.h"
#include "common/dx11.h"

static const uint32_t kW = 320, kH = 200, kFrames = 8;

namespace {

size_t page_align(size_t n) {
    size_t pg = (size_t)getpagesize();
    return (n + pg - 1) & ~(pg - 1);
}

struct Embedder {
    void*    map[DMN_EXPORTED_MAX_IMAGES] = {};
    size_t   map_size[DMN_EXPORTED_MAX_IMAGES] = {};
    uint32_t stride = 0, w = 0, h = 0, count = 0;
    uint32_t dxgi_format = 0;
    uint32_t configures = 0, acquires = 0, presents = 0;
    uint32_t last_slot = UINT32_MAX;
    uint64_t last_frame_id = 0;
    bool     frame_id_monotonic = true;
    bool     slot_in_range = true;
    int      last_gpu_fd = -1;

    void unmap_all() {
        for (uint32_t i = 0; i < DMN_EXPORTED_MAX_IMAGES; i++) {
            if (map[i])
                munmap(map[i], map_size[i]);
            map[i] = nullptr;
            map_size[i] = 0;
        }
    }
};

void emb_images_changed(void* ctx, const dmn_exported_image_set* set) {
    auto* e = static_cast<Embedder*>(ctx);
    e->unmap_all();
    e->w = set->width;
    e->h = set->height;
    e->count = set->num_images;
    e->dxgi_format = set->dxgi_format;
    e->stride = set->num_images ? set->images[0].stride : 0;
    for (uint32_t i = 0; i < set->num_images; i++) {
        size_t sz = page_align((size_t)set->images[i].size);
        void* p = mmap(nullptr, sz, PROT_READ, MAP_SHARED,
                       set->images[i].fd, 0);
        e->map[i] = p == MAP_FAILED ? nullptr : p;
        e->map_size[i] = sz;
    }
    e->configures++;
}

void emb_acquire(void* ctx, uint32_t slot) {
    (void)slot;
    static_cast<Embedder*>(ctx)->acquires++;
}

void emb_present(void* ctx, uint32_t slot, uint64_t frame_id, int gpu_fd) {
    auto* e = static_cast<Embedder*>(ctx);
    if (frame_id <= e->last_frame_id)
        e->frame_id_monotonic = false;
    e->last_frame_id = frame_id;
    if (slot >= e->count)
        e->slot_in_range = false;
    e->last_slot = slot;
    e->presents++;
    if (e->last_gpu_fd >= 0)
        close(e->last_gpu_fd);
    e->last_gpu_fd = gpu_fd;
}

/* Render `frames` clears ending in `last`, wait for the embedder's
 * cumulative present count to reach `expect_presents`, gate on the final
 * GPU-done fence fd, and verify `last` at the center of the presented
 * image. `rgba` picks the byte order to check. */
int render_and_verify(Embedder& emb, ID3D11Device* dev,
                      ID3D11DeviceContext* ctx, IDXGISwapChain1* swapchain,
                      uint32_t frames, uint32_t expect_presents,
                      const float last[4], bool rgba) {
    for (uint32_t i = 0; i < frames; i++) {
        Com<ID3D11Texture2D> bb;
        CK(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb),
           "GetBuffer");
        Com<ID3D11RenderTargetView> rtv;
        CK(dev->CreateRenderTargetView(bb.ptr(), nullptr, &rtv), "RTV");
        float c[4] = {0.05f * i, 0.1f, 0.3f, 1.0f};
        const float* color = (i == frames - 1) ? last : c;
        ctx->ClearRenderTargetView(rtv.ptr(), color);
        CK(swapchain->Present(0, 0), "Present");
    }
    ctx->Flush();

    /* D3DMetal presents from an async scheduled handler; give the last
     * frames a moment to land. */
    for (int i = 0; i < 100 && emb.presents < expect_presents; i++) {
        struct timespec ns = {0, 50 * 1000 * 1000};
        nanosleep(&ns, nullptr);
    }
    EXPECT(emb.acquires >= expect_presents, "on_acquire underfired");
    EXPECT(emb.presents >= expect_presents, "on_present underfired");
    EXPECT(emb.slot_in_range, "presented slot out of image-set range");
    EXPECT(emb.frame_id_monotonic, "frame_id not monotonic");
    printf("EXPSWAP: %u configures, %u acquires, %u presents, last slot %u, "
           "frame %llu\n",
           emb.configures, emb.acquires, emb.presents, emb.last_slot,
           (unsigned long long)emb.last_frame_id);

    EXPECT(emb.last_slot < emb.count && emb.map[emb.last_slot],
           "no presented image to read");
    if (emb.last_gpu_fd >= 0) {
        struct pollfd pfd = {emb.last_gpu_fd, POLLIN, 0};
        EXPECT(poll(&pfd, 1, 5000) == 1, "GPU-done fence fd never signaled");
        close(emb.last_gpu_fd);
        emb.last_gpu_fd = -1;
    }
    /* The fence covers a tracking command buffer, not D3DMetal's own present
     * blit; give that a beat too. */
    struct timespec ns = {0, 300 * 1000 * 1000};
    nanosleep(&ns, nullptr);

    const uint8_t* px = (const uint8_t*)emb.map[emb.last_slot] +
                        (size_t)(emb.h / 2) * emb.stride +
                        (size_t)(emb.w / 2) * 4;
    auto near8 = [](uint8_t got, float want) {
        int w = (int)(want * 255.0f + 0.5f);
        return got >= w - 2 && got <= w + 2;
    };
    const int ri = rgba ? 0 : 2, bi = rgba ? 2 : 0;
    EXPECT(near8(px[ri], last[0]) && near8(px[1], last[1]) &&
           near8(px[bi], last[2]) && near8(px[3], last[3]),
           "final clear color not found in the exported image");
    printf("EXPSWAP: pixel check ok (R=%u G=%u B=%u A=%u)\n", px[ri], px[1],
           px[bi], px[3]);
    return 0;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "EXPSWAP: dmn_init FAILED\n");
        return 1;
    }

    Embedder emb;
    dmn_exported_swapchain_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.ctx = &emb;
    cfg.on_images_changed = emb_images_changed;
    cfg.on_acquire = emb_acquire;
    cfg.on_present = emb_present;
    dmn_window_t window = dmn_window_create_exported(&cfg, kW, kH);
    EXPECT(window, "dmn_window_create_exported failed");
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
       "CreateSwapChainForHwnd(exported window)");
    EXPECT(emb.configures >= 1, "on_images_changed never fired");
    EXPECT(emb.w == kW && emb.h == kH, "image set dims mismatch");
    EXPECT(emb.count >= 2 && emb.stride >= kW * 4, "bad image set");
    for (uint32_t i = 0; i < emb.count; i++)
        EXPECT(emb.map[i], "image fd did not mmap");

    /* Render kFrames clears; the last one is the color we verify. */
    const float kLast[4] = {0.2f, 0.4f, 0.6f, 1.0f};
    if (render_and_verify(emb, dev.ptr(), ctx.ptr(), swapchain.ptr(), kFrames,
                          kFrames, kLast, /*rgba=*/false))
        return 1;

    /* ResizeBuffers must republish the image set at the new size and keep
     * frames flowing (frame_id monotonic across the resize). */
    const uint32_t kW2 = 512, kH2 = 384;
    const uint32_t configures_before = emb.configures;
    CK(swapchain->ResizeBuffers(0, kW2, kH2, DXGI_FORMAT_UNKNOWN, 0),
       "ResizeBuffers");
    EXPECT(emb.configures > configures_before,
           "on_images_changed did not refire on resize");
    EXPECT(emb.w == kW2 && emb.h == kH2, "resized image set dims mismatch");
    EXPECT(emb.count >= 2 && emb.stride >= kW2 * 4, "bad resized image set");
    for (uint32_t i = 0; i < emb.count; i++)
        EXPECT(emb.map[i], "resized image fd did not mmap");
    const float kLast2[4] = {0.8f, 0.2f, 0.4f, 1.0f};
    if (render_and_verify(emb, dev.ptr(), ctx.ptr(), swapchain.ptr(), kFrames,
                          2 * kFrames, kLast2, /*rgba=*/false))
        return 1;
    printf("EXPSWAP: resize ok (%ux%u -> %ux%u)\n", kW, kH, kW2, kH2);

    swapchain = nullptr; /* release before the window goes away */
    dmn_window_destroy(window);
    emb.unmap_all();

    /* Second swapchain: a non-presentable desc format (R10G10B10A2) must
     * fall back to preferred_dxgi_format for the exported allocation, and
     * preferred_image_count must override BufferCount. */
    Embedder emb2;
    dmn_exported_swapchain_config cfg2{};
    cfg2.struct_size = sizeof(cfg2);
    cfg2.ctx = &emb2;
    cfg2.preferred_dxgi_format = DXGI_FORMAT_R8G8B8A8_UNORM;
    cfg2.preferred_image_count = 3;
    cfg2.on_images_changed = emb_images_changed;
    cfg2.on_acquire = emb_acquire;
    cfg2.on_present = emb_present;
    dmn_window_t window2 = dmn_window_create_exported(&cfg2, kW, kH);
    EXPECT(window2, "dmn_window_create_exported (2nd) failed");

    scDesc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    Com<IDXGISwapChain1> swapchain2;
    CK(factory2->CreateSwapChainForHwnd(dev.ptr(),
                                        (HWND)dmn_window_get_hwnd(window2),
                                        &scDesc, nullptr, nullptr,
                                        &swapchain2),
       "CreateSwapChainForHwnd(R10G10B10A2)");
    EXPECT(emb2.configures >= 1, "2nd on_images_changed never fired");
    EXPECT(emb2.dxgi_format == (uint32_t)DXGI_FORMAT_R8G8B8A8_UNORM,
           "allocation did not fall back to preferred_dxgi_format");
    EXPECT(emb2.count == 3, "preferred_image_count not honored");
    for (uint32_t i = 0; i < emb2.count; i++)
        EXPECT(emb2.map[i], "2nd image fd did not mmap");
    const float kLast3[4] = {0.6f, 0.2f, 0.8f, 1.0f};
    if (render_and_verify(emb2, dev.ptr(), ctx.ptr(), swapchain2.ptr(),
                          kFrames, kFrames, kLast3, /*rgba=*/true))
        return 1;
    printf("EXPSWAP: format fallback + image count ok (dxgi=%u, %u images)\n",
           emb2.dxgi_format, emb2.count);

    swapchain2 = nullptr;
    dmn_window_destroy(window2);
    emb2.unmap_all();
    T_PASS();
    return 0;
}
