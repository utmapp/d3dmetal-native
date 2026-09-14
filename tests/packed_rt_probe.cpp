/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic probe (not a test): packed 16-bit render targets in isolation.
 * Clear a B4G4R4A4, B5G6R5 and B5G5R5A1 render target to a known colour,
 * copy it into a readback buffer and decode the pixel; BGRA8 is the control.
 * On D3DMetal the B4G4R4A4 pixel comes back with its channels rotated
 * (0xb48f where DXGI packing puts the clear at 0xfb48), which is every
 * A4R4G4B4/X4R4G4B4 failure in the HLK D3D9 alphaprecision, mrt and
 * Colorfill tests; the other three formats round-trip.
 *
 * Prints one line per format; exit status 1 if any format decodes wrong.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <d3d12.h>
#include <windows.h>
#include "d3dmetal_native.h"
#include "common/com.h"
#define T_TAG "PACKEDRT"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

namespace {
enum { kW = 64, kH = 64 };

struct Submit {
    Com<ID3D12CommandQueue> queue;
    Com<ID3D12CommandAllocator> alloc;
    Com<ID3D12GraphicsCommandList> list;
    Com<ID3D12Fence> fence;
    UINT64 fv = 0;
    bool init(ID3D12Device* dev) {
        if (FAILED(make_d3d12_queue(dev, queue)))
            return false;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               __uuidof(ID3D12CommandAllocator),
                                               (void**)&alloc)) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          alloc.ptr(), nullptr,
                                          __uuidof(ID3D12GraphicsCommandList),
                                          (void**)&list)))
            return false;
        list->Close();
        return SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                          __uuidof(ID3D12Fence), (void**)&fence));
    }
    bool open() {
        return SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.ptr(), nullptr));
    }
    bool run() {
        if (FAILED(list->Close()))
            return false;
        ID3D12CommandList* ls[] = {list.ptr()};
        queue->ExecuteCommandLists(1, ls);
        const UINT64 want = ++fv;
        queue->Signal(fence.ptr(), want);
        const uint64_t t0 = now_ms();
        while (fence->GetCompletedValue() < want) {
            if (now_ms() - t0 > 10000)
                return false;
            sleep_ms(1);
        }
        return true;
    }
};

struct Fmt {
    DXGI_FORMAT f;
    const char* name;
    unsigned bpp;
    /* channel bit widths b,g,r,a from the low bit up (DXGI little-endian packing) */
    unsigned bits[4];
};

static const Fmt kFmts[] = {
    {DXGI_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8", 4, {8, 8, 8, 8}},
    {DXGI_FORMAT_B4G4R4A4_UNORM, "B4G4R4A4", 2, {4, 4, 4, 4}},
    {DXGI_FORMAT_B5G6R5_UNORM,   "B5G6R5",   2, {5, 6, 5, 0}},
    {DXGI_FORMAT_B5G5R5A1_UNORM, "B5G5R5A1", 2, {5, 5, 5, 1}},
};

/* Cleared colour as RGBA floats. */
static const float kClear[4] = {0.75f, 0.25f, 0.5f, 1.0f};

static unsigned decode_channel(uint32_t px, unsigned shift, unsigned bits) {
    return (px >> shift) & ((1u << bits) - 1u);
}

static bool probe(ID3D12Device* dev, Submit& sub, const Fmt& fmt) {
    D3D12_HEAP_PROPERTIES hpDef{};
    hpDef.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = kW;
    rd.Height = kH;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = fmt.f;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv{};
    cv.Format = fmt.f;
    memcpy(cv.Color, kClear, sizeof(kClear));
    Com<ID3D12Resource> rt;
    HRESULT hr = dev->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &rd,
                                              D3D12_RESOURCE_STATE_RENDER_TARGET, &cv,
                                              __uuidof(ID3D12Resource), (void**)&rt);
    if (FAILED(hr)) {
        printf("PACKEDRT: %-9s create FAILED 0x%08lx\n", fmt.name, (unsigned long)hr);
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 1;
    Com<ID3D12DescriptorHeap> heap;
    if (FAILED(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&heap))) {
        printf("PACKEDRT: %-9s RTV heap FAILED\n", fmt.name);
        return false;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(rt.ptr(), nullptr, rtv);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES hpRb{};
    hpRb.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Com<ID3D12Resource> rb;
    if (FAILED(dev->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd,
                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                            __uuidof(ID3D12Resource), (void**)&rb))) {
        printf("PACKEDRT: %-9s readback create FAILED\n", fmt.name);
        return false;
    }

    if (!sub.open())
        return false;
    sub.list->ClearRenderTargetView(rtv, kClear, 0, nullptr);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = rt.ptr();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    sub.list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = rb.ptr();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = rt.ptr();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    sub.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (!sub.run()) {
        printf("PACKEDRT: %-9s submit FAILED\n", fmt.name);
        return false;
    }

    void* m = nullptr;
    if (FAILED(rb->Map(0, nullptr, &m))) {
        printf("PACKEDRT: %-9s Map FAILED\n", fmt.name);
        return false;
    }
    const uint8_t* base = (const uint8_t*)m + fp.Offset;
    /* Sample three pixels: origin, centre, far corner. */
    const unsigned xs[] = {0, kW / 2, kW - 1}, ys[] = {0, kH / 2, kH - 1};
    bool ok = true;
    uint32_t first = 0;
    for (unsigned i = 0; i < 3; i++) {
        const uint8_t* p = base + (size_t)ys[i] * fp.Footprint.RowPitch + (size_t)xs[i] * fmt.bpp;
        uint32_t px = 0;
        memcpy(&px, p, fmt.bpp);
        if (i == 0)
            first = px;
        /* DXGI packs the first-named channel in the LOW bits: B4G4R4A4 =
         * b[3:0] g[7:4] r[11:8] a[15:12]; B5G6R5 = b[4:0] g[10:5] r[15:11]. */
        unsigned shift = 0;
        const float rgba_expect[4] = {kClear[2], kClear[1], kClear[0], kClear[3]}; /* b,g,r,a */
        for (unsigned c = 0; c < 4; c++) {
            const unsigned bits = fmt.bits[c];
            if (!bits)
                continue;
            const unsigned got = decode_channel(px, shift, bits);
            const unsigned maxv = (1u << bits) - 1u;
            const unsigned want = (unsigned)lroundf(rgba_expect[c] * (float)maxv);
            if ((got > want ? got - want : want - got) > 1)
                ok = false;
            shift += bits;
        }
    }
    rb->Unmap(0, nullptr);
    printf("PACKEDRT: %-9s pixel=0x%0*x %s\n", fmt.name, fmt.bpp * 2, first,
           ok ? "matches the clear colour" : "WRONG");
    return ok;
}
} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "PACKEDRT: dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");
    Submit sub;
    EXPECT(sub.init(dev.ptr()), "queue/list/fence init failed");
    unsigned bad = 0;
    for (const Fmt& f : kFmts)
        if (!probe(dev.ptr(), sub, f))
            bad++;
    EXPECT(bad == 0, "a packed render target decoded wrong");
    T_PASS();
    return 0;
}
