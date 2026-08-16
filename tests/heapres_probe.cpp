/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic probe (not a test): is a standalone D3D12 heap that no command
 * ever references still made resident by the backend on GPU work?  And a
 * committed texture that is never touched?
 *
 *   heapres-probe [heap_MiB=1024] [flags=0]   flags: 0 none, 1 DENY_BUFFERS|DENY_RT_DS (tile-pool-like)
 */
#include "common/dx12.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
extern "C" unsigned long long dmn_test_metal_allocated_size(void);
extern "C" unsigned long long dmn_test_phys_footprint(void);
static double mib(unsigned long long b) { return b / 1048576.0; }
static void rep(const char* t) { printf("%-52s metal=%6.0f MiB footprint=%6.0f MiB\n", t, mib(dmn_test_metal_allocated_size()), mib(dmn_test_phys_footprint())); fflush(stdout); }
int main(int argc, char** argv) {
    unsigned heap_mb = argc > 1 ? atoi(argv[1]) : 1024; int fl = argc > 2 ? atoi(argv[2]) : 0;
    Com<ID3D12Device> dev; Com<ID3D12CommandQueue> q;
    if (FAILED(make_d3d12_device(dev)) || FAILED(make_d3d12_queue(dev.ptr(), q))) { printf("no device\n"); return 1; }
    /* small working set: one 4 MiB texture we blit into, so "GPU work" exists */
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td{}; td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; td.Width = 1024; td.Height = 1024; td.DepthOrArraySize = 1; td.MipLevels = 1; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    ID3D12Resource* small = nullptr; dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), (void**)&small);
    D3D12_HEAP_PROPERTIES uhp{}; uhp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC ub{}; ub.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; ub.Width = 4u << 20; ub.Height = 1; ub.DepthOrArraySize = 1; ub.MipLevels = 1; ub.SampleDesc.Count = 1; ub.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* up = nullptr; dev->CreateCommittedResource(&uhp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, __uuidof(ID3D12Resource), (void**)&up);
    ID3D12CommandAllocator* al = nullptr; ID3D12GraphicsCommandList* cl = nullptr; ID3D12Fence* fe = nullptr; UINT64 fv = 0;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&al);
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&cl);
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fe);
    auto work = [&]() {
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint.Format = td.Format; src.PlacedFootprint.Footprint.Width = 1024; src.PlacedFootprint.Footprint.Height = 1024; src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = 4096;
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = small; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr); cl->Close();
        ID3D12CommandList* l = cl; q->ExecuteCommandLists(1, &l); q->Signal(fe, ++fv);
        for (int i = 0; i < 20000 && fe->GetCompletedValue() < fv; i++) usleep(1000);
        al->Reset(); cl->Reset(al, nullptr);
    };
    work(); rep("baseline after first GPU work (small tex only)");
    D3D12_HEAP_DESC hd{}; hd.SizeInBytes = (UINT64)heap_mb << 20; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT; hd.Alignment = 65536;
    hd.Flags = fl ? (D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES) : D3D12_HEAP_FLAG_NONE;
    ID3D12Heap* heap = nullptr; HRESULT hr = dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&heap);
    char b[96]; snprintf(b, sizeof b, "CreateHeap %u MiB flags=0x%x -> 0x%08x", heap_mb, (unsigned)hd.Flags, (unsigned)hr); rep(b);
    work(); rep("after unrelated GPU work (heap unreferenced)");
    work(); rep("after more unrelated GPU work");
    /* now a big committed texture that is never touched */
    D3D12_RESOURCE_DESC bt = td; bt.Width = 4096; bt.Height = 4096; bt.MipLevels = 11; bt.Format = DXGI_FORMAT_BC7_UNORM;
    ID3D12Resource* big[16] = {}; for (int i = 0; i < 16; i++) dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bt, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), (void**)&big[i]);
    rep("16 x 21 MiB committed BC7 textures created (untouched)");
    work(); rep("after unrelated GPU work (textures untouched)");
    if (heap) heap->Release(); heap = nullptr; work(); rep("heap released + GPU work");
    sleep(2); rep("2 s later");
    return 0;
}
