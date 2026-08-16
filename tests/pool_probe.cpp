/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic probe (not a test): does D3DMetal return sub-allocator pool
 * memory when committed resources are released?  Reports
 * MTLDevice.currentAllocatedSize around create/release rounds.
 *
 * Measured (M4 Pro, D3DMetal 2026-08): 64 x 32 MiB DEFAULT buffers grow the
 * counter by 2048 MiB; releasing them all leaves it unchanged, and the next
 * round reuses the pool.  The framework's footprint is its high-water mark.
 *
 * pool-probe [n=64] [MiB=32] [textures=0]
 */
#include "common/dx12.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" unsigned long long dmn_test_metal_allocated_size(void);
static double mib(unsigned long long b) { return b / 1048576.0; }
int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 64;      /* resources per round */
    int mb = argc > 2 ? atoi(argv[2]) : 32;     /* MiB each */
    int tex = argc > 3 ? atoi(argv[3]) : 0;     /* 1 = textures instead of buffers */
    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev))) { printf("no device\n"); return 1; }
    printf("baseline metal=%.0f MiB\n", mib(dmn_test_metal_allocated_size()));
    for (int round = 0; round < 3; round++) {
        std::vector<Com<ID3D12Resource>> keep;
        for (int i = 0; i < n; i++) {
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd{};
            if (!tex) {
                rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = (UINT64)mb << 20;
                rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
                rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            } else {
                rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                rd.Width = 2048; rd.Height = (UINT)((mb << 20) / (2048 * 4));
                rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            }
            ID3D12Resource* r = nullptr;
            HRESULT hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource), (void**)&r);
            if (FAILED(hr)) { printf("create failed 0x%08x at %d\n", (unsigned)hr, i); break; }
            keep.push_back(Com<ID3D12Resource>(r)); r->Release();
        }
        printf("round %d: holding %zu x %d MiB (%.0f MiB) -> metal=%.0f MiB\n", round,
               keep.size(), mb, keep.size() * (double)mb, mib(dmn_test_metal_allocated_size()));
        keep.clear();
        printf("round %d: released -> metal=%.0f MiB\n", round, mib(dmn_test_metal_allocated_size()));
    }
    return 0;
}
