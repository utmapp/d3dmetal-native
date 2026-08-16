/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic probe (not a test): what does the host D3D12 backend itself do
 * with tiled (reserved) resources?  TiledResourcesTier from CheckFeature
 * Support, then CreateReservedResource + CreateHeap (tile pool) +
 * UpdateTileMappings, reporting each HRESULT and Metal's allocated size.
 *
 * Measured (D3DMetal 2026-08, M4 Pro): TiledResourcesTier=0;
 * CreateReservedResource logs "Unsupported: Sparse 2D textures" and returns
 * S_OK with an unbacked object; UpdateTileMappings logs "Unsupported" and
 * does nothing (no device removal).  Tiled resources are the guest UMD's
 * committed-backing shim, not something the framework can do.  Those are
 * the framework's own answers; through the hooked device (dmn_sparse) the
 * probe reports tier 2 and a real tiling.
 *
 * NOTE: the vendored d3d12.h declares ID3D12CommandQueue::UpdateTileMappings
 * WITHOUT the ID3D12Heap* parameter (wine header bug); anything calling it
 * through that header passes shifted arguments.  The probe calls the slot
 * with the real signature.
 */
#include "common/dx12.h"
#include <cstdio>
extern "C" unsigned long long dmn_test_metal_allocated_size(void);
static double mib(unsigned long long b) { return b / 1048576.0; }
int main() {
    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev))) { printf("no device\n"); return 1; }
    D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
    HRESULT hr = dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof o);
    printf("CheckFeatureSupport(OPTIONS) hr=0x%08x TiledResourcesTier=%d ResourceHeapTier=%d\n",
           (unsigned)hr, (int)o.TiledResourcesTier, (int)o.ResourceHeapTier);
    printf("baseline metal=%.0f MiB\n", mib(dmn_test_metal_allocated_size()));

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = 4096; rd.Height = 4096; rd.DepthOrArraySize = 1; rd.MipLevels = 11;
    rd.Format = DXGI_FORMAT_BC7_UNORM; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    ID3D12Resource* res = nullptr;
    hr = dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&res);
    printf("CreateReservedResource(4096^2 BC7, 11 mips, 64KB_UNDEFINED_SWIZZLE) hr=0x%08x res=%p metal=%.0f MiB\n",
           (unsigned)hr, (void*)res, mib(dmn_test_metal_allocated_size()));
    if (FAILED(hr) || !res) {
        rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        hr = dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                         __uuidof(ID3D12Resource), (void**)&res);
        printf("CreateReservedResource(layout UNKNOWN) hr=0x%08x res=%p\n", (unsigned)hr, (void*)res);
    }
    if (SUCCEEDED(hr) && res) {
        UINT numTiles = 0; D3D12_PACKED_MIP_INFO pm{}; D3D12_TILE_SHAPE ts{}; UINT numSub = 11;
        D3D12_SUBRESOURCE_TILING st[11]{};
        dev->GetResourceTiling(res, &numTiles, &pm, &ts, &numSub, 0, st);
        printf("GetResourceTiling: tiles=%u tileShape=%ux%ux%u standardMips=%u packedMips=%u tilesForPacked=%u\n",
               numTiles, ts.WidthInTexels, ts.HeightInTexels, ts.DepthInTexels,
               pm.NumStandardMips, pm.NumPackedMips, pm.NumTilesForPackedMips);
        D3D12_HEAP_DESC hd{}; hd.SizeInBytes = 64ull << 20; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        ID3D12Heap* pool = nullptr;
        hr = dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&pool);
        printf("CreateHeap(tile pool 64 MiB) hr=0x%08x metal=%.0f MiB\n", (unsigned)hr, mib(dmn_test_metal_allocated_size()));
        Com<ID3D12CommandQueue> q;
        if (SUCCEEDED(make_d3d12_queue(dev.ptr(), q)) && pool) {
            D3D12_TILED_RESOURCE_COORDINATE c{}; c.Subresource = 0;
            D3D12_TILE_REGION_SIZE sz{}; sz.NumTiles = 16;
            D3D12_TILE_RANGE_FLAGS f = D3D12_TILE_RANGE_FLAG_NONE; UINT off = 0; UINT cnt = 16;
            /* The vendored header drops the ID3D12Heap* parameter; call the
             * slot with the real signature. */
            typedef void (STDMETHODCALLTYPE *UTM)(ID3D12CommandQueue*, ID3D12Resource*, UINT,
                const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*,
                UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS);
            void** vt = *reinterpret_cast<void***>(q.ptr());
            UTM utm = reinterpret_cast<UTM>(vt[8]); /* IUnknown 0-2, ID3D12Object 3-6, DeviceChild 7, UpdateTileMappings 8 */
            utm(q.ptr(), res, 1, &c, &sz, pool, 1, &f, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
            printf("UpdateTileMappings(16 tiles of mip 0) issued; device removed reason=0x%08x metal=%.0f MiB\n",
                   (unsigned)dev->GetDeviceRemovedReason(), mib(dmn_test_metal_allocated_size()));
        }
        if (pool) pool->Release();
        res->Release();
    }
    printf("end metal=%.0f MiB removed=0x%08x\n", mib(dmn_test_metal_allocated_size()), (unsigned)dev->GetDeviceRemovedReason());
    return 0;
}
