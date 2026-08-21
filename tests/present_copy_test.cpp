/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * The companion-present copy, in isolation: CopyResource where the DEST is a
 * substituted shared texture (linear, buffer-backed) and the SOURCE is a
 * native optimally-tiled texture, both sitting in COMMON so the copy relies
 * on implicit state promotion.  This is byte-for-byte the GPU work Triton's
 * D3D12 present-time blit records (WI-1 in HANDOFF-forspoken-parity.md):
 * back buffers stay native, and each present copies into a per-resource
 * shared companion whose KM allocation the swapchain machinery holds.
 *
 * Every other test writes into shared textures through draws (the convert-blt
 * shape) or through buffer copies; none exercises texture CopyResource with a
 * substituted dest, which is exactly the operation the driver change leans on.
 *
 *  1. Native RT texture is filled with a deterministic pattern via an upload
 *     buffer, then returned to COMMON.
 *  2. Companion: same desc + D3D12_HEAP_FLAG_SHARED (the substitution
 *     trigger), created in COMMON, never written by anything else.
 *  3. CopyResource(companion, native) in its own command list, no explicit
 *     barriers -- the driver's present path has only COMMON/PRESENT states to
 *     work with.
 *  4. The bytes are verified through the exported POD's fd + stride: what a
 *     consumer (DWM, scanout) would actually sample.
 *
 * Prints "PRESENTCOPY: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "PRESENTCOPY"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

namespace {

enum { kW = 1280, kH = 720, kBpp = 4 };
enum { kPitch = kW * kBpp }; /* 5120: already 256-aligned for the footprint */

static uint32_t pattern_at(unsigned x, unsigned y) {
    /* BGRA8: unique per pixel, includes both coordinates and a constant. */
    return 0xFF000000u | ((x & 0xFFFu) << 12) | (y & 0xFFFu);
}

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
                                          __uuidof(ID3D12Fence),
                                          (void**)&fence));
    }
    bool open() {
        return SUCCEEDED(alloc->Reset()) &&
               SUCCEEDED(list->Reset(alloc.ptr(), nullptr));
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

static D3D12_RESOURCE_DESC rt_desc(void) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = kW;
    rd.Height = kH;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    return rd;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "PRESENTCOPY: dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");

    Submit sub;
    EXPECT(sub.init(dev.ptr()), "queue/list/fence init failed");

    /* 1) Native back-buffer stand-in, patterned via an upload buffer. */
    D3D12_HEAP_PROPERTIES hpDef{};
    hpDef.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = rt_desc();
    Com<ID3D12Resource> native;
    CK(dev->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &rd,
                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                    __uuidof(ID3D12Resource), (void**)&native),
       "CreateCommittedResource(native RT)");

    D3D12_HEAP_PROPERTIES hpUp{};
    hpUp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = (uint64_t)kPitch * kH;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Com<ID3D12Resource> up;
    CK(dev->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                    __uuidof(ID3D12Resource), (void**)&up),
       "CreateCommittedResource(upload)");
    {
        void* m = nullptr;
        CK(up->Map(0, nullptr, &m), "Map(upload)");
        for (unsigned y = 0; y < kH; y++) {
            uint32_t* row = (uint32_t*)((uint8_t*)m + (size_t)y * kPitch);
            for (unsigned x = 0; x < kW; x++)
                row[x] = pattern_at(x, y);
        }
        up->Unmap(0, nullptr);
    }

    EXPECT(sub.open(), "list open (upload) failed");
    {
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = native.ptr();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = up.ptr();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = 0;
        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        src.PlacedFootprint.Footprint.Width = kW;
        src.PlacedFootprint.Footprint.Height = kH;
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = kPitch;
        sub.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        /* Back to COMMON: at present time the driver only ever sees
         * PRESENT (== COMMON). */
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = native.ptr();
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        sub.list->ResourceBarrier(1, &b);
    }
    EXPECT(sub.run(), "upload submit failed");

    /* 2) Companion: identical desc, SHARED so the allocation is substituted
     *    with the linear shm surface, COMMON, never otherwise written. */
    Com<ID3D12Resource> comp;
    CK(dev->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_SHARED, &rd,
                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                    __uuidof(ID3D12Resource), (void**)&comp),
       "CreateCommittedResource(SHARED companion)");

    /* 3) The present copy: both COMMON, no barriers, own submission. */
    EXPECT(sub.open(), "list open (present copy) failed");
    sub.list->CopyResource(comp.ptr(), native.ptr());
    EXPECT(sub.run(), "present-copy submit failed");

    /* 4) Verify through the exported mapping -- the consumer's view. */
    HANDLE h = nullptr;
    CK(dev->CreateSharedHandle(comp.ptr(), nullptr, 0, nullptr, &h),
       "CreateSharedHandle(companion)");
    dmn_shared_texture_handle* pod = (dmn_shared_texture_handle*)h;
    EXPECT(pod && pod->magic == DMN_SHARED_TEXTURE_MAGIC,
           "exported handle is not a texture POD");
    EXPECT(pod->width == kW && pod->height == kH,
           "POD dims mismatch");
    EXPECT(pod->stride >= kPitch, "POD stride smaller than a packed row");
    const size_t mapLen = (size_t)(pod->offset + pod->stride * kH);
    void* m = mmap(nullptr, mapLen, PROT_READ, MAP_SHARED, pod->fd, 0);
    EXPECT(m != MAP_FAILED, "mmap(companion fd) failed");
    const uint8_t* base = (const uint8_t*)m + pod->offset;
    unsigned bad = 0;
    for (unsigned y = 0; y < kH && bad < 8; y += 37) {
        const uint32_t* row = (const uint32_t*)(base + (size_t)y * pod->stride);
        for (unsigned x = 0; x < kW && bad < 8; x += 41) {
            const uint32_t want = pattern_at(x, y);
            if (row[x] != want) {
                fprintf(stderr,
                        "PRESENTCOPY: pixel (%u,%u) = 0x%08x want 0x%08x\n",
                        x, y, row[x], want);
                bad++;
            }
        }
    }
    /* Corners exactly. */
    {
        const unsigned xs[] = {0, kW - 1};
        const unsigned ys[] = {0, kH - 1};
        for (unsigned iy = 0; iy < 2 && bad < 8; iy++)
            for (unsigned ix = 0; ix < 2 && bad < 8; ix++) {
                const uint32_t* row =
                    (const uint32_t*)(base + (size_t)ys[iy] * pod->stride);
                if (row[xs[ix]] != pattern_at(xs[ix], ys[iy])) {
                    fprintf(stderr,
                            "PRESENTCOPY: corner (%u,%u) = 0x%08x want 0x%08x\n",
                            xs[ix], ys[iy], row[xs[ix]],
                            pattern_at(xs[ix], ys[iy]));
                    bad++;
                }
            }
    }
    munmap(m, mapLen);
    EXPECT(bad == 0, "companion bytes do not match the native source");

    printf("PRESENTCOPY: copy native->substituted 1280x720 verified "
           "(stride=%llu)\n", (unsigned long long)pod->stride);
    dmn_shared_handle_close(h);
    T_PASS();
    return 0;
}
