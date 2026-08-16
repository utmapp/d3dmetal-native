/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic probe (not a test): reproduce a streaming title's reserved-
 * resource pattern -- CREATE many tiled resources up front, then map a small
 * fraction of each much later -- and report whether any sparse chunk runs dry
 * (tiles that silently READ ZERO), how many chunks that took, and what it
 * cost in RAM.
 *
 * This is the shape that stresses chunk placement: dmn_sparse binds a texture
 * to the heap it is created on, so placement must act during the burst of
 * creates, when nothing has been mapped yet.  A virtual-texturing title
 * reserves tens of GiB of texture and maps a few GiB of it.
 *
 *   sparse-pool-probe [count=512] [dim=2048] [map_pct=4] [pools=drip]
 *
 * `pools` picks the shape of the app's own D3D12 tile pools, which the module
 * reads its bounds from:
 *
 *   drip  -- a new small pool every few resources, sized to the tiles actually
 *            mapped through it.  A streaming title's shape: many pools of a
 *            few MiB, each first seen at its first use, so the total is a
 *            lower bound that climbs all run and is near nothing when the
 *            first chunk is placed.
 *   big   -- one pool covering the whole run, created up front.  The textbook
 *            streaming shape, where the real size is known immediately.
 *   tiny  -- one 64 MiB pool regardless of demand, so the declared pool badly
 *            understates the tiles mapped: what a dishonest pool signal does.
 *
 * Watch the log at DMN_LOG=warn for "chunk N is DRY"; none is a pass.  At
 * DMN_LOG=info the module's own "sparse stats:" line reports the same numbers
 * it reports in any run.
 */
#include "common/dx12.h"
#include "dmn_sparse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" unsigned long long dmn_test_metal_allocated_size(void);
extern "C" unsigned long long dmn_test_phys_footprint(void);
static double mib(unsigned long long b) { return b / 1048576.0; }
static double mib_signed(int64_t b) { return b / 1048576.0; }

typedef void (STDMETHODCALLTYPE *UTM)(ID3D12CommandQueue*, ID3D12Resource*, UINT,
    const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*,
    UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS);

int main(int argc, char** argv) {
    const UINT count   = argc > 1 ? (UINT)atoi(argv[1]) : 512;
    const UINT dim     = argc > 2 ? (UINT)atoi(argv[2]) : 2048;
    const UINT map_pct = argc > 3 ? (UINT)atoi(argv[3]) : 4;
    const char* pools  = argc > 4 ? argv[4] : "drip";
    const bool drip = !strcmp(pools, "drip");
    const bool tiny = !strcmp(pools, "tiny");

    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev))) { printf("no device\n"); return 1; }
    Com<ID3D12CommandQueue> q;
    if (FAILED(make_d3d12_queue(dev.ptr(), q))) { printf("no queue\n"); return 1; }
    void** vt = *reinterpret_cast<void***>(q.ptr());
    UTM utm = reinterpret_cast<UTM>(vt[8]);

    const uint64_t f0 = dmn_test_phys_footprint();
    printf("baseline: metal=%.0f MiB footprint=%.0f MiB\n",
           mib(dmn_test_metal_allocated_size()), mib(f0));

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = dim; rd.Height = dim; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_BC7_UNORM; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;

    /* Phase 1: create everything, mapping nothing -- the level load. */
    std::vector<ID3D12Resource*> res;
    for (UINT i = 0; i < count; i++) {
        ID3D12Resource* r = nullptr;
        if (SUCCEEDED(dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  __uuidof(ID3D12Resource), (void**)&r)) && r)
            res.push_back(r);
    }
    UINT total_tiles = 0;
    {
        UINT n = 0; D3D12_PACKED_MIP_INFO pm{}; D3D12_TILE_SHAPE ts{}; UINT ns = 1;
        D3D12_SUBRESOURCE_TILING st[1]{};
        if (!res.empty()) dev->GetResourceTiling(res[0], &n, &pm, &ts, &ns, 0, st);
        total_tiles = n;
        printf("created %zu reserved %ux%u BC7 (%u tiles each = %.1f MiB virtual each, "
               "%.1f GiB virtual total)\n", res.size(), dim, dim, n, n * 64.0 / 1024.0,
               res.size() * n * 64.0 / 1048576.0);
    }
    printf("after creates: metal=%.0f MiB footprint=%+.0f MiB\n",
           mib(dmn_test_metal_allocated_size()), mib_signed((int64_t)dmn_test_phys_footprint() - (int64_t)f0));

    /* Phase 2: map a fraction of each, long after they were placed.  The tiles
     * come from the app's own pools -- the heaps named below -- and their
     * total is what the module is allowed to size itself from, so declaring
     * them honestly is the whole point of this probe. */
    const UINT per = total_tiles * map_pct / 100 ? total_tiles * map_pct / 100 : 1;
    const uint64_t need = (uint64_t)res.size() * per * 65536ull;
    const uint64_t drip_bytes = 16ull << 20;   /* a new pool every ~16 MiB of tiles */

    auto new_pool = [&](uint64_t bytes) -> ID3D12Heap* {
        D3D12_HEAP_DESC hd{}; hd.SizeInBytes = bytes;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        ID3D12Heap* h = nullptr;
        dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&h);
        return h;
    };

    std::vector<ID3D12Heap*> heaps;
    ID3D12Heap* pool = nullptr;
    if (!drip) {
        pool = new_pool(tiny ? (64ull << 20) : need);
        if (pool) heaps.push_back(pool);
        printf("pools: one %s heap of %.0f MiB up front for %.0f MiB of tiles\n",
               tiny ? "DELIBERATELY UNDERSIZED" : "honest", mib(tiny ? (64ull << 20) : need),
               mib(need));
    } else {
        printf("pools: a fresh %.0f MiB heap every %.0f MiB of tiles, created as they are "
               "needed (%.0f MiB of tiles in total)\n", mib(drip_bytes), mib(drip_bytes),
               mib(need));
    }

    uint64_t mapped_through_pool = 0;
    for (size_t i = 0; i < res.size(); i++) {
        if (drip && (!pool || mapped_through_pool >= drip_bytes)) {
            pool = new_pool(drip_bytes);
            if (pool) heaps.push_back(pool);
            mapped_through_pool = 0;
        }
        D3D12_TILED_RESOURCE_COORDINATE c{}; c.Subresource = 0;
        D3D12_TILE_REGION_SIZE sz{}; sz.NumTiles = per;
        D3D12_TILE_RANGE_FLAGS fl = D3D12_TILE_RANGE_FLAG_NONE; UINT off = 0, cnt = per;
        utm(q.ptr(), res[i], 1, &c, &sz, pool, 1, &fl, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
        mapped_through_pool += (uint64_t)per * 65536ull;
    }
    printf("mapped %u tiles each (%u%%, %.0f MiB total) through %zu pool(s): "
           "metal=%.0f MiB footprint=%+.0f MiB\n",
           per, map_pct, mib(need), heaps.size(),
           mib(dmn_test_metal_allocated_size()), mib_signed((int64_t)dmn_test_phys_footprint() - (int64_t)f0));
    printf("device removed reason=0x%08x\n", (unsigned)dev->GetDeviceRemovedReason());

    DmnSparseStats st{};
    dmn_sparse_get_stats(&st);
    printf("sparse: %u chunks (%.0f MiB, largest %.0f MiB) of %.0f MiB budget; pools %.0f MiB in "
           "%u; %.0f MiB reserved / %.0f MiB mapped; DRY %llu; refused %llu\n",
           st.chunks, mib(st.chunk_bytes), mib(st.largest_chunk), mib(st.budget_bytes),
           mib(st.app_pool_bytes), st.app_pools, mib(st.reserved_bytes), mib(st.mapped_bytes),
           (unsigned long long)st.dry_events, (unsigned long long)st.refused);

    for (auto* h : heaps) h->Release();
    for (auto r : res) r->Release();
    return 0;
}
