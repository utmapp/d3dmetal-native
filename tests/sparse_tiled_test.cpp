/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Reserved (tiled) resource content semantics through the hooked device:
 * D3DMetal itself has TiledResourcesTier 0, so this exercises the sparse
 * backing in dmn_sparse.mm.  A 1024x1024 RGBA8 reserved texture (128x128
 * texel tiles, 64 tiles): map a 2x2 tile box at (0,0), upload a pattern into
 * a mapped and an unmapped area, read both back (mapped == pattern, unmapped
 * == 0), unmap, read the mapped area again (== 0).  Then the rest of the
 * tier-2 surface the hooks serve: GetResourceTiling (tile shape, counts,
 * packed-mip info, per-subresource tiling), CopyTiles both ways in its
 * linear-per-tile buffer layout, and the NULL-everything UpdateTileMappings
 * form that clears a whole resource.
 */
#include "common/dx12.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <cstdlib>

extern "C" unsigned long long dmn_test_metal_allocated_size(void);
extern "C" unsigned long long dmn_test_phys_footprint(void);
static double mib(unsigned long long b) { return b / 1048576.0; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

struct Ctx {
    Com<ID3D12Device> dev; Com<ID3D12CommandQueue> q;
    ID3D12CommandAllocator* alloc = nullptr; ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr; UINT64 fv = 0;
    bool init() {
        if (FAILED(make_d3d12_device(dev)) || FAILED(make_d3d12_queue(dev.ptr(), q))) return false;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&alloc))) return false;
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&list))) return false;
        list->Close();
        return SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence));
    }
    void begin() { alloc->Reset(); list->Reset(alloc, nullptr); }
    void submit_wait() {
        list->Close();
        ID3D12CommandList* l = list; q->ExecuteCommandLists(1, &l);
        q->Signal(fence, ++fv);
        for (int i = 0; i < 5000 && fence->GetCompletedValue() < fv; i++) usleep(1000);
        if (fence->GetCompletedValue() < fv) { fails++; printf("FAIL: GPU wait timed out\n"); }
    }
};

static ID3D12Resource* make_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = type;
    D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
    return r;
}

/* Copy a 256x256 RGBA8 block from `up` (tight, pitch 1024) into tex at (x,y). */
static void copy_up(Ctx& c, ID3D12Resource* up, ID3D12Resource* tex, UINT x, UINT y) {
    D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width = 256; src.PlacedFootprint.Footprint.Height = 256;
    src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = 1024;
    D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
    c.list->CopyTextureRegion(&dst, x, y, 0, &src, nullptr);
}
/* Read a 256x256 block at (x,y) into rb (pitch 1024). */
static void copy_down(Ctx& c, ID3D12Resource* tex, ID3D12Resource* rb, UINT x, UINT y) {
    D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = tex; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width = 256; dst.PlacedFootprint.Footprint.Height = 256;
    dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = 1024;
    D3D12_BOX box{x, y, 0, x + 256, y + 256, 1};
    c.list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
}
static unsigned count_nonzero(ID3D12Resource* rb) {
    void* p = nullptr; D3D12_RANGE rr{0, 256 * 1024};
    if (FAILED(rb->Map(0, &rr, &p)) || !p) return 0xffffffff;
    unsigned n = 0; const uint32_t* px = (const uint32_t*)p;
    for (unsigned i = 0; i < 256 * 256; i++) n += px[i] != 0;
    D3D12_RANGE wr{0, 0}; rb->Unmap(0, &wr);
    return n;
}
static uint32_t pixel_at(ID3D12Resource* rb, unsigned i) {
    void* p = nullptr; D3D12_RANGE rr{0, 256 * 1024}; rb->Map(0, &rr, &p);
    uint32_t v = ((const uint32_t*)p)[i]; D3D12_RANGE wr{0, 0}; rb->Unmap(0, &wr); return v;
}

typedef void (STDMETHODCALLTYPE *UTM)(ID3D12CommandQueue*, ID3D12Resource*, UINT,
    const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*,
    UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS);

int main() {
    /* The module is on by default; pin it on so an environment that disabled
     * it cannot turn this into a test of the committed path. */
    setenv("DMN_SPARSE", "1", 1);
    Ctx c;
    if (!c.init()) { printf("no device\n"); return 1; }
    printf("baseline metal=%.0f MiB phys=%.0f MiB\n", mib(dmn_test_metal_allocated_size()), mib(dmn_test_phys_footprint()));

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = 1024; rd.Height = 1024;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    ID3D12Resource* tex = nullptr;
    HRESULT hr = c.dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               __uuidof(ID3D12Resource), (void**)&tex);
    CHECK(SUCCEEDED(hr) && tex, "CreateReservedResource 1024^2 RGBA8 hr=0x%08x", (unsigned)hr);
    if (!tex) return 1;
    printf("after reserved create metal=%.0f MiB phys=%.0f MiB\n", mib(dmn_test_metal_allocated_size()), mib(dmn_test_phys_footprint()));

    D3D12_HEAP_DESC hd{}; hd.SizeInBytes = 16ull << 20; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    ID3D12Heap* pool = nullptr;
    c.dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&pool);

    void** vt = *reinterpret_cast<void***>(c.q.ptr());
    UTM utm = reinterpret_cast<UTM>(vt[8]);

    /* map 2x2 tiles at (0,0): texels [0,256)x[0,256) */
    D3D12_TILED_RESOURCE_COORDINATE co{0, 0, 0, 0};
    D3D12_TILE_REGION_SIZE sz{}; sz.NumTiles = 4; sz.UseBox = TRUE; sz.Width = 2; sz.Height = 2; sz.Depth = 1;
    D3D12_TILE_RANGE_FLAGS fl = D3D12_TILE_RANGE_FLAG_NONE; UINT off = 0, cnt = 4;
    utm(c.q.ptr(), tex, 1, &co, &sz, pool, 1, &fl, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
    printf("after map 4 tiles metal=%.0f MiB phys=%.0f MiB removed=0x%08x\n", mib(dmn_test_metal_allocated_size()),
           mib(dmn_test_phys_footprint()), (unsigned)c.dev->GetDeviceRemovedReason());

    ID3D12Resource* up = make_buffer(c.dev.ptr(), 256 * 1024, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* rb = make_buffer(c.dev.ptr(), 256 * 1024, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    { void* p = nullptr; D3D12_RANGE rr{0, 0}; up->Map(0, &rr, &p);
      uint32_t* px = (uint32_t*)p; for (unsigned i = 0; i < 256 * 256; i++) px[i] = 0xff0000ff | (i << 8);
      up->Unmap(0, nullptr); }

    c.begin(); copy_up(c, up, tex, 0, 0); copy_up(c, up, tex, 512, 512); c.submit_wait();
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = tex; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; b.Transition.Subresource = 0;

    c.begin(); c.list->ResourceBarrier(1, &b); copy_down(c, tex, rb, 0, 0); c.submit_wait();
    unsigned nz = count_nonzero(rb);
    CHECK(nz == 256 * 256, "mapped region reads back the pattern (%u/65536 nonzero, px[1]=0x%08x)", nz, pixel_at(rb, 1));

    c.begin(); copy_down(c, tex, rb, 512, 512); c.submit_wait();
    nz = count_nonzero(rb);
    CHECK(nz == 0, "unmapped region reads zero (%u nonzero)", nz);

    /* GetResourceTiling: 1024^2 RGBA8 = 8x8 tiles of 128x128, no packed mips. */
    {
        UINT total = 0, nsub = 1; D3D12_PACKED_MIP_INFO pm{}; D3D12_TILE_SHAPE ts{}; D3D12_SUBRESOURCE_TILING st{};
        c.dev->GetResourceTiling(tex, &total, &pm, &ts, &nsub, 0, &st);
        CHECK(total == 64 && ts.WidthInTexels == 128 && ts.HeightInTexels == 128 && ts.DepthInTexels == 1,
              "GetResourceTiling RGBA8: %u tiles of %ux%ux%u", total, ts.WidthInTexels, ts.HeightInTexels, ts.DepthInTexels);
        CHECK(pm.NumStandardMips == 1 && pm.NumPackedMips == 0 && pm.NumTilesForPackedMips == 0,
              "GetResourceTiling RGBA8: %u standard, %u packed mips in %u tiles",
              pm.NumStandardMips, pm.NumPackedMips, pm.NumTilesForPackedMips);
        CHECK(nsub == 1 && st.WidthInTiles == 8 && st.HeightInTiles == 8 && st.DepthInTiles == 1 && st.StartTileIndexInOverallResource == 0,
              "GetResourceTiling RGBA8 sub 0: %ux%ux%u tiles from %u", st.WidthInTiles, st.HeightInTiles, st.DepthInTiles,
              st.StartTileIndexInOverallResource);
    }

    /* CopyTiles buffer -> the mapped 2x2 tile box: each tile is 64 KiB of
     * linear 128x128 RGBA8 (pitch 512), tiles in x-then-y order.  Read the
     * 256^2 region back with a plain copy and check every pixel lands where
     * the tile layout says. */
    ID3D12Resource* tb = make_buffer(c.dev.ptr(), 256 * 1024, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    { void* p = nullptr; D3D12_RANGE rr{0, 0}; tb->Map(0, &rr, &p);
      uint32_t* px = (uint32_t*)p;
      for (unsigned k = 0; k < 4; k++) for (unsigned i = 0; i < 128 * 128; i++) px[k * 16384 + i] = 0x01000000u * (k + 1) | i;
      tb->Unmap(0, nullptr); }
    D3D12_RESOURCE_BARRIER b2 = b; b2.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; b2.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    c.begin(); c.list->ResourceBarrier(1, &b2);
    c.list->CopyTiles(tex, &co, &sz, tb, 0, D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
    c.list->ResourceBarrier(1, &b); copy_down(c, tex, rb, 0, 0); c.submit_wait();
    {
        void* p = nullptr; D3D12_RANGE rr{0, 256 * 1024}; rb->Map(0, &rr, &p);
        const uint32_t* px = (const uint32_t*)p; unsigned bad = 0; uint32_t first_bad = 0, first_want = 0;
        for (unsigned y = 0; y < 256 && bad < 100000; y++) for (unsigned x = 0; x < 256; x++) {
            unsigned k = (y / 128) * 2 + (x / 128), i = (y % 128) * 128 + (x % 128);
            uint32_t want = 0x01000000u * (k + 1) | i, got = px[y * 256 + x];
            if (got != want) { if (!bad) { first_bad = got; first_want = want; } bad++; }
        }
        D3D12_RANGE wr{0, 0}; rb->Unmap(0, &wr);
        CHECK(bad == 0, "CopyTiles buffer->tiles lands each tile linearly (%u wrong pixels; first got 0x%08x want 0x%08x)", bad, first_bad, first_want);
    }
    /* ... and back: CopyTiles tiles -> buffer must reproduce the upload bytes. */
    ID3D12Resource* rb2 = make_buffer(c.dev.ptr(), 256 * 1024, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    c.begin();
    c.list->CopyTiles(tex, &co, &sz, rb2, 0, D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER);
    c.submit_wait();
    {
        void* p = nullptr; D3D12_RANGE rr{0, 256 * 1024}; rb2->Map(0, &rr, &p);
        const uint32_t* px = (const uint32_t*)p; unsigned bad = 0;
        for (unsigned k = 0; k < 4; k++) for (unsigned i = 0; i < 128 * 128; i++) bad += px[k * 16384 + i] != (0x01000000u * (k + 1) | i);
        D3D12_RANGE wr{0, 0}; rb2->Unmap(0, &wr);
        CHECK(bad == 0, "CopyTiles tiles->buffer round-trips the tile layout (%u wrong texels)", bad);
    }
    tb->Release(); rb2->Release();

    /* unmap and re-read */
    fl = D3D12_TILE_RANGE_FLAG_NULL;
    utm(c.q.ptr(), tex, 1, &co, &sz, nullptr, 1, &fl, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
    c.begin(); copy_down(c, tex, rb, 0, 0); c.submit_wait();
    nz = count_nonzero(rb);
    CHECK(nz == 0, "after unmap the region reads zero (%u nonzero)", nz);

    /* Re-map, write, then the documented "clear a whole surface" form: every
     * region/range parameter NULL, one NULL range. */
    fl = D3D12_TILE_RANGE_FLAG_NONE;
    utm(c.q.ptr(), tex, 1, &co, &sz, pool, 1, &fl, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
    c.begin(); c.list->ResourceBarrier(1, &b2); copy_up(c, up, tex, 0, 0); c.list->ResourceBarrier(1, &b); copy_down(c, tex, rb, 0, 0); c.submit_wait();
    nz = count_nonzero(rb);
    CHECK(nz == 256 * 256, "re-mapped region holds the pattern again (%u/65536 nonzero)", nz);
    fl = D3D12_TILE_RANGE_FLAG_NULL;
    utm(c.q.ptr(), tex, 1, nullptr, nullptr, nullptr, 1, &fl, nullptr, nullptr, D3D12_TILE_MAPPING_FLAG_NONE);
    c.begin(); copy_down(c, tex, rb, 0, 0); c.submit_wait();
    nz = count_nonzero(rb);
    CHECK(nz == 0, "NULL-region NULL-range UpdateTileMappings clears the whole resource (%u nonzero)", nz);
    printf("end metal=%.0f MiB removed=0x%08x\n", mib(dmn_test_metal_allocated_size()),
           (unsigned)c.dev->GetDeviceRemovedReason());
    tex->Release(); if (pool) pool->Release(); up->Release(); rb->Release();

    /* BC7 4096^2 full mip chain, the shape the game uses: create + map a
     * couple of standard tiles and the packed tail, no device removal. */
    {
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; bd.Width = 4096; bd.Height = 4096;
        bd.DepthOrArraySize = 1; bd.MipLevels = 11; bd.Format = DXGI_FORMAT_BC7_UNORM;
        bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        ID3D12Resource* bt = nullptr;
        HRESULT bhr = c.dev->CreateReservedResource(&bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    __uuidof(ID3D12Resource), (void**)&bt);
        CHECK(SUCCEEDED(bhr) && bt, "CreateReservedResource 4096^2 BC7 11 mips hr=0x%08x", (unsigned)bhr);
        if (bt) {
            D3D12_TILED_RESOURCE_COORDINATE cs[2] = {{0, 0, 0, 0}, {0, 0, 0, 5}}; /* mip 0 tile + packed region */
            D3D12_TILE_REGION_SIZE ss[2]{}; ss[0].NumTiles = 2; ss[1].NumTiles = 1;
            D3D12_TILE_RANGE_FLAGS f2 = D3D12_TILE_RANGE_FLAG_NONE; UINT o2 = 0, n2 = 3;
            utm(c.q.ptr(), bt, 2, cs, ss, nullptr, 1, &f2, &o2, &n2, D3D12_TILE_MAPPING_FLAG_NONE);
            CHECK(c.dev->GetDeviceRemovedReason() == S_OK, "BC7 map (2 tiles + packed tail): device intact");
            /* Tiling of the chain: 256^2 tiles; mips 0-4 (4096..256) standard =
             * 256+64+16+4+1 = 341 tiles; mips 5-10 (128..4) packed, 21840 bytes
             * -> 1 tile at index 341; 342 in all. */
            {
                UINT total = 0, nsub = 11; D3D12_PACKED_MIP_INFO pm{}; D3D12_TILE_SHAPE ts{}; D3D12_SUBRESOURCE_TILING st[11]{};
                c.dev->GetResourceTiling(bt, &total, &pm, &ts, &nsub, 0, st);
                CHECK(total == 342 && ts.WidthInTexels == 256 && ts.HeightInTexels == 256,
                      "GetResourceTiling BC7: %u tiles of %ux%u", total, ts.WidthInTexels, ts.HeightInTexels);
                CHECK(pm.NumStandardMips == 5 && pm.NumPackedMips == 6 && pm.NumTilesForPackedMips == 1 && pm.StartTileIndexInOverallResource == 341,
                      "GetResourceTiling BC7: %u standard, %u packed mips in %u tile(s) from %u",
                      pm.NumStandardMips, pm.NumPackedMips, pm.NumTilesForPackedMips, pm.StartTileIndexInOverallResource);
                CHECK(nsub == 11 && st[0].WidthInTiles == 16 && st[0].HeightInTiles == 16 && st[0].StartTileIndexInOverallResource == 0 &&
                      st[4].WidthInTiles == 1 && st[4].StartTileIndexInOverallResource == 340 &&
                      st[5].WidthInTiles == 0 && st[5].HeightInTiles == 0 && st[5].StartTileIndexInOverallResource == 0xffffffffu,
                      "GetResourceTiling BC7 subresources: mip0 %ux%u@%u, mip4 %ux%u@%u, mip5 %ux%u@0x%08x",
                      st[0].WidthInTiles, st[0].HeightInTiles, st[0].StartTileIndexInOverallResource,
                      st[4].WidthInTiles, st[4].HeightInTiles, st[4].StartTileIndexInOverallResource,
                      st[5].WidthInTiles, st[5].HeightInTiles, st[5].StartTileIndexInOverallResource);
            }
            /* Content through the packed tail: CopyTextureRegion a pattern into
             * every mip from the packed start (5) down to 4x4, plus mip 0 tile
             * (0,0) that is mapped, and read each back. Mip 4 (256^2) is a
             * standard mip that was NOT mapped: it must read zero. */
            {
                ID3D12Resource* up = make_buffer(c.dev.ptr(), 1u << 20, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
                ID3D12Resource* rb = make_buffer(c.dev.ptr(), 1u << 20, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
                void* m = nullptr; D3D12_RANGE nr{0, 0};
                if (up && SUCCEEDED(up->Map(0, &nr, &m)) && m) { memset(m, 0x5A, 1u << 20); up->Unmap(0, nullptr); }
                for (UINT mip = 0; mip < 11; mip++) {
                    if (mip >= 1 && mip <= 3) continue; /* not mapped, not written; mip 4 tests unmapped-reads-zero */
                    UINT w = 4096 >> mip, h = 4096 >> mip; if (w < 4) w = 4; if (h < 4) h = 4;
                    UINT cw = mip == 0 ? 256 : w, ch = mip == 0 ? 256 : h;   /* mip 0: one 256^2 tile */
                    UINT pitch = ((cw / 4) * 16 + 255) & ~255u;
                    c.begin();
                    if (mip != 4) {
                        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_BC7_UNORM;
                        src.PlacedFootprint.Footprint.Width = cw; src.PlacedFootprint.Footprint.Height = ch;
                        src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = pitch;
                        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = bt; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = mip;
                        c.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                    }
                    D3D12_TEXTURE_COPY_LOCATION s2{}; s2.pResource = bt; s2.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; s2.SubresourceIndex = mip;
                    D3D12_TEXTURE_COPY_LOCATION d2{}; d2.pResource = rb; d2.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    d2.PlacedFootprint.Footprint.Format = DXGI_FORMAT_BC7_UNORM;
                    d2.PlacedFootprint.Footprint.Width = cw; d2.PlacedFootprint.Footprint.Height = ch;
                    d2.PlacedFootprint.Footprint.Depth = 1; d2.PlacedFootprint.Footprint.RowPitch = pitch;
                    D3D12_BOX box{0, 0, 0, cw, ch, 1};
                    if (m) { up->Map(0, &nr, &m); memset(m, 0x5A, 1u << 20); up->Unmap(0, nullptr); }
                    /* scrub the readback first so a dropped copy cannot look like content */
                    void* z = nullptr; if (SUCCEEDED(rb->Map(0, &nr, &z)) && z) { memset(z, 0, 1u << 20); D3D12_RANGE wr{0, 1u << 20}; rb->Unmap(0, &wr); }
                    c.list->CopyTextureRegion(&d2, 0, 0, 0, &s2, &box);
                    c.submit_wait();
                    unsigned nz = 0; void* p = nullptr; D3D12_RANGE rr{0, 1u << 20};
                    if (SUCCEEDED(rb->Map(0, &rr, &p)) && p) {
                        const uint8_t* b = (const uint8_t*)p; unsigned rows = ch / 4, rowb = (cw / 4) * 16;
                        for (unsigned r = 0; r < rows; r++) for (unsigned i = 0; i < rowb; i++) nz += b[r * pitch + i] != 0;
                        D3D12_RANGE wr{0, 0}; rb->Unmap(0, &wr);
                    }
                    unsigned want = mip == 4 ? 0 : (cw / 4) * (ch / 4) * 16;
                    CHECK(nz == want, "BC7 mip %u (%ux%u) %s: %u/%u nonzero bytes", mip, cw, ch,
                          mip == 4 ? "UNMAPPED reads zero" : "written via CopyTextureRegion reads back", nz, want);
                }
                if (up) up->Release(); if (rb) rb->Release();
            }
            bt->Release();
        }
    }
    printf("final phys=%.0f MiB\n", mib(dmn_test_phys_footprint()));
    printf(fails ? "SPARSE-TILED FAIL (%d)\n" : "SPARSE-TILED PASS\n", fails);
    return fails ? 1 : 0;
}
