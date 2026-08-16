/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * How dmn_sparse SIZES itself: when a sparse chunk is opened and how big it
 * is made, which chunk a reserved texture lands on, and how much the chunks
 * may total.  The content semantics of a tiled resource are
 * tests/sparse-tiled's job; this one looks at dmn_sparse_get_stats() -- except
 * the one case that has to prove a texture really can map more tiles than an
 * ordinary chunk holds, which reads the pixels back.
 *
 * Each case runs in a FRESH PROCESS.  The module reads every knob once and
 * its chunks are process-global, so cases cannot share one: the parent
 * re-execs this binary with DMN_SPARSE_CASE set (the same shape
 * tests/shared_footprint_test.cpp uses), and each child runs exactly one
 * case.  Run one directly with DMN_SPARSE_CASE=<name>.
 *
 * Sizes throughout: an RGBA8 D3D12 tile is 128x128 texels = 64 KiB, so a
 * 2048^2 texture is 16 MiB of virtual texture and a 4096^2 one is 64 MiB.
 */
#include "common/dx12.h"
#include "dmn_sparse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <mach-o/dyld.h>

extern char** environ;

static int fails;
#define CHECK(c, ...) do { \
        if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
        else { printf("ok: " __VA_ARGS__); printf("\n"); } \
    } while (0)
#define NOTE(...) do { printf("  .. " __VA_ARGS__); printf("\n"); } while (0)

static double mib(unsigned long long b) { return b / 1048576.0; }

/* The vendored d3d12.h declares ID3D12CommandQueue::UpdateTileMappings WITHOUT
 * the ID3D12Heap* parameter (wine header bug), so calling it through the header
 * passes shifted arguments.  Call the vtable slot with the real signature --
 * IUnknown 0-2, ID3D12Object 3-6, ID3D12DeviceChild 7, UpdateTileMappings 8. */
typedef void (STDMETHODCALLTYPE *UTM)(ID3D12CommandQueue*, ID3D12Resource*, UINT,
    const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*,
    UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS);

struct Ctx {
    Com<ID3D12Device> dev;
    Com<ID3D12CommandQueue> q;
    UTM utm = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 fv = 0;

    bool init() {
        if (FAILED(make_d3d12_device(dev)) || FAILED(make_d3d12_queue(dev.ptr(), q)))
            return false;
        void** vt = *reinterpret_cast<void***>(q.ptr());
        utm = reinterpret_cast<UTM>(vt[8]);
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               __uuidof(ID3D12CommandAllocator), (void**)&alloc)))
            return false;
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
                                          __uuidof(ID3D12GraphicsCommandList), (void**)&list)))
            return false;
        list->Close();
        return SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                          (void**)&fence));
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

static ID3D12Resource* make_reserved(ID3D12Device* dev, UINT dim,
                                     D3D12_RESOURCE_STATES st = D3D12_RESOURCE_STATE_COPY_DEST) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = dim; rd.Height = dim; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    ID3D12Resource* r = nullptr;
    dev->CreateReservedResource(&rd, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
    return r;
}

static ID3D12Heap* make_pool(ID3D12Device* dev, UINT64 bytes) {
    D3D12_HEAP_DESC hd{}; hd.SizeInBytes = bytes;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    ID3D12Heap* h = nullptr;
    dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&h);
    return h;
}

/* Map the first `tiles` tiles of `tex` (a box `w` wide when given, else a
 * linear run) from `pool`, which is also how the module learns the pool
 * exists and how big it is.  `flag` is NONE unless a case is testing
 * aliasing. */
static void map_tiles(Ctx& c, ID3D12Resource* tex, ID3D12Heap* pool, UINT tiles, UINT w = 0,
                      D3D12_TILE_RANGE_FLAGS flag = D3D12_TILE_RANGE_FLAG_NONE) {
    D3D12_TILED_RESOURCE_COORDINATE co{0, 0, 0, 0};
    D3D12_TILE_REGION_SIZE sz{}; sz.NumTiles = tiles;
    if (w) { sz.UseBox = TRUE; sz.Width = w; sz.Height = tiles / w; sz.Depth = 1; }
    UINT off = 0, cnt = tiles;
    c.utm(c.q.ptr(), tex, 1, &co, &sz, pool, 1, &flag, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
}

static DmnSparseStats stats() {
    DmnSparseStats s{};
    dmn_sparse_get_stats(&s);
    return s;
}
static void dump(const char* when) {
    DmnSparseStats s = stats();
    NOTE("%s: %u chunks (%.0f MiB, largest %.0f MiB) budget %.0f MiB, pools %.0f MiB/%u%s, "
         "live %u, reserved %.0f MiB, mapped %.0f MiB at %ux, refused %llu, dry %llu, "
         "chunk size %u MiB",
         when, s.chunks, mib(s.chunk_bytes), mib(s.largest_chunk), mib(s.budget_bytes),
         mib(s.app_pool_bytes), s.app_pools, s.app_pool_aliased ? " ALIASED" : "", s.live,
         mib(s.reserved_bytes), mib(s.mapped_bytes), s.oversubscribe,
         (unsigned long long)s.refused, (unsigned long long)s.dry_events, s.chunk_size_mb);
}

/* == cases =============================================================== */

/* Chunks are opened on demand, at DMN_SPARSE_HEAP_MB each, and DMN_SPARSE_MAX_MB
 * caps their total: past it no chunk opens and the module says so. */
static int case_budget_caps_chunks(Ctx& c) {
    /* 32 MiB chunks, 64 MiB budget: two chunks and not a third.  Each texture
     * maps from a 24 MiB pool of its own, small enough that no texture is
     * outsized for a 32 MiB chunk. */
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 24ull << 20);
    ID3D12Heap* pool2 = make_pool(c.dev.ptr(), 24ull << 20);
    ID3D12Resource* a = make_reserved(c.dev.ptr(), 4096);
    if (!pool || !pool2 || !a) { printf("FAIL: setup\n"); return 1; }
    dump("after the first create");
    DmnSparseStats s = stats();
    CHECK(s.chunks == 1 && s.chunk_bytes == (32ull << 20),
          "the first reserved resource opens one chunk of DMN_SPARSE_HEAP_MB (%u x %.0f MiB)",
          s.chunks, mib(s.chunk_bytes));
    CHECK(s.live == 1, "and is registered (%u live)", s.live);

    /* Fill it: 24 MiB of a 64 MiB texture is past the 75% mark that sends
     * the next texture to a new chunk. */
    map_tiles(c, a, pool, 384, 32);
    ID3D12Resource* b = make_reserved(c.dev.ptr(), 4096);
    dump("after filling it and creating again");
    s = stats();
    CHECK(s.mapped_bytes == (24ull << 20), "the tally follows the mapping (%.0f MiB)",
          mib(s.mapped_bytes));
    CHECK(s.chunks == 2, "a texture created against a full chunk gets a new one (%u chunks)",
          s.chunks);
    CHECK(s.dry_events == 0, "nothing ran dry (%llu)", (unsigned long long)s.dry_events);

    map_tiles(c, b, pool2, 384, 32);
    ID3D12Resource* d = make_reserved(c.dev.ptr(), 4096);
    dump("after filling that one and creating again");
    s = stats();
    CHECK(s.chunks == 2 && s.chunk_bytes == (64ull << 20),
          "the budget holds the chunks to 64 MiB (%u chunks, %.0f MiB)", s.chunks,
          mib(s.chunk_bytes));
    CHECK(s.refused == 1, "and the refusal is counted (%llu)", (unsigned long long)s.refused);
    CHECK(d != nullptr, "the resource itself is still created");

    if (d) d->Release();
    b->Release(); a->Release();
    s = stats();
    CHECK(s.live == 0 && s.mapped_bytes == 0,
          "releasing them unregisters and returns their tiles (%u live, %.0f MiB mapped)",
          s.live, mib(s.mapped_bytes));
    pool->Release(); pool2->Release();
    return fails;
}

/* A texture is charged its virtual footprint to the chunk it lands on, and a
 * chunk carries at most DMN_SPARSE_OVERSUBSCRIBE times its size, so a burst of
 * creates spreads over chunks before a single tile is mapped. */
static int case_spread_by_footprint(Ctx& c) {
    /* 32 MiB chunks at 2x -> 64 MiB of virtual texture per chunk, i.e. four
     * 2048^2 textures. */
    std::vector<ID3D12Resource*> res;
    for (int i = 0; i < 4; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("after 4 creates");
    DmnSparseStats a = stats();
    CHECK(a.chunks == 1, "four 16 MiB textures fit one chunk at 2x (%u chunks)", a.chunks);
    CHECK(a.reserved_bytes == (64ull << 20), "and are charged their footprint (%.0f MiB)",
          mib(a.reserved_bytes));
    CHECK(a.oversubscribe == 2, "the multiple in force is the knob's (%ux)", a.oversubscribe);

    for (int i = 0; i < 4; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("after 8 creates");
    DmnSparseStats b = stats();
    CHECK(b.chunks == 2, "the fifth opens a second chunk with nothing mapped (%u chunks)",
          b.chunks);
    CHECK(b.mapped_bytes == 0, "(%.0f MiB mapped)", mib(b.mapped_bytes));

    for (auto* r : res) r->Release();
    DmnSparseStats d = stats();
    CHECK(d.reserved_bytes == 0, "releasing them returns their reservations (%.0f MiB)",
          mib(d.reserved_bytes));
    return fails;
}

/* The app's own tile pools may RAISE the budget, because their total proves it
 * needs at least that much.  Without that, a large-pool app falls off the
 * fixed ceiling and every further reserved texture is fully backed. */
static int case_budget_grows(Ctx& c) {
    /* 32 MiB chunks, 2x oversubscription -> 64 MiB of virtual texture per
     * chunk, i.e. four 2048^2 textures; a 64 MiB floor allows two chunks. */
    std::vector<ID3D12Resource*> res;
    for (int i = 0; i < 12; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("after 12 creates at the floor");
    DmnSparseStats a = stats();
    CHECK(a.chunks == 2, "the floor holds the budget to 2 chunks (%u)", a.chunks);
    CHECK(a.refused > 0, "past it, reserved textures go to the fully-backed path (%llu refused)",
          (unsigned long long)a.refused);
    const uint64_t budget_before = a.budget_bytes;

    /* Now the app names a 256 MiB tile pool: it cannot map more tiles than
     * that, but it clearly intends to map more than the floor allows. */
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 256ull << 20);
    CHECK(pool != nullptr, "created a 256 MiB tile pool");
    if (!pool) return 1;
    map_tiles(c, res[0], pool, 1);
    dump("after the pool is named");

    DmnSparseStats b = stats();
    CHECK(b.app_pool_bytes == (256ull << 20), "the pool total is learned (%.0f MiB)",
          mib(b.app_pool_bytes));
    CHECK(b.budget_bytes > budget_before,
          "the budget rises above the floor (%.0f -> %.0f MiB)",
          mib(budget_before), mib(b.budget_bytes));
    CHECK(b.budget_bytes >= (320ull << 20),
          "it reaches the pool total plus the rounding margin (%.0f MiB)", mib(b.budget_bytes));

    for (int i = 0; i < 12; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("after 12 more creates");
    DmnSparseStats d = stats();
    CHECK(d.chunks > a.chunks, "chunks resume (%u -> %u)", a.chunks, d.chunks);
    CHECK(d.refused == a.refused, "and nothing more is refused (%llu)",
          (unsigned long long)d.refused);
    CHECK(d.budget_bytes >= b.budget_bytes, "the budget never went backwards (%.0f -> %.0f MiB)",
          mib(b.budget_bytes), mib(d.budget_bytes));

    for (auto* r : res) r->Release();
    pool->Release();
    return fails;
}

/* An explicitly set DMN_SPARSE_MAX_MB is a HARD cap: someone limiting our
 * memory gets the limit they asked for, whatever the app's pools say, and a
 * texture that would need a chunk past it is fully backed instead of piled
 * onto a chunk that is full by reservation. */
static int case_cap_is_hard(Ctx& c) {
    std::vector<ID3D12Resource*> res;
    for (int i = 0; i < 8; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 256ull << 20);
    if (!pool || res.empty()) { printf("FAIL: setup\n"); return 1; }
    map_tiles(c, res[0], pool, 1);
    for (int i = 0; i < 12; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("with a 256 MiB pool and an explicit 64 MiB cap");

    DmnSparseStats s = stats();
    CHECK(res.size() == 20, "every create succeeds (%zu)", res.size());
    CHECK(s.app_pool_bytes == (256ull << 20), "the pool is still learned (%.0f MiB)",
          mib(s.app_pool_bytes));
    CHECK(s.budget_bytes == (64ull << 20), "the explicit cap is not raised (%.0f MiB)",
          mib(s.budget_bytes));
    CHECK(s.chunks == 2 && s.chunk_bytes == (64ull << 20),
          "chunks stay inside it (%.0f MiB in %u)", mib(s.chunk_bytes), s.chunks);
    CHECK(s.refused > 0, "the excess is fully backed instead (%llu refused)",
          (unsigned long long)s.refused);
    CHECK(s.reserved_bytes == (128ull << 20),
          "and charged to no chunk (%.0f MiB reserved)", mib(s.reserved_bytes));

    for (auto* r : res) r->Release();
    pool->Release();
    return fails;
}

/* The app cannot map more tiles than the pools it owns, so once it names one
 * the multiple a chunk may carry is bounded by reserved / pools -- and only
 * ever downwards from the default: a pool total that is still growing reads
 * sparser than the app is, and acting on that would leave tiles unmapped. */
static int case_pool_tightens_bound(Ctx& c) {
    /* 32 MiB chunks at the default 8x -> 256 MiB of virtual per chunk, so
     * eight 2048^2 textures share one chunk ... */
    std::vector<ID3D12Resource*> res;
    for (int i = 0; i < 8; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("after 8 creates");
    DmnSparseStats a = stats();
    CHECK(a.chunks == 1 && a.oversubscribe == 8, "8 x 16 MiB share one chunk at the default 8x "
          "(%u chunks, %ux)", a.chunks, a.oversubscribe);

    /* ... until the app names a 32 MiB pool: it can map at most 32 MiB of the
     * 128 MiB reserved, so a chunk may carry 4x its size, and the next
     * texture no longer fits the first. */
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 32ull << 20);
    if (!pool) { printf("FAIL: setup\n"); return 1; }
    map_tiles(c, res[0], pool, 1);
    if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    dump("after a 32 MiB pool is named and one more create");
    DmnSparseStats b = stats();
    CHECK(b.app_pool_bytes == (32ull << 20) && b.app_pools == 1,
          "the pool is learned from UpdateTileMappings (%.0f MiB in %u)", mib(b.app_pool_bytes),
          b.app_pools);
    CHECK(b.oversubscribe == 4, "the multiple tightens to reserved / pool (%ux)",
          b.oversubscribe);
    CHECK(b.chunks == 2, "so the next texture opens a second chunk (%u chunks)", b.chunks);

    /* A further pool re-derives the bound from the new total (144 / 40 MiB),
     * still under the default. */
    ID3D12Heap* pool2 = make_pool(c.dev.ptr(), 8ull << 20);
    if (pool2) {
        map_tiles(c, res[1], pool2, 1);
        DmnSparseStats d = stats();
        CHECK(d.app_pool_bytes == (40ull << 20) && d.oversubscribe == 3,
              "a further pool is added to the total (%.0f MiB) and the multiple follows (%ux)",
              mib(d.app_pool_bytes), d.oversubscribe);
        pool2->Release();
    }

    for (auto* r : res) r->Release();
    pool->Release();
    return fails;
}

/* The density measured so far is the second signal, once enough is mapped to
 * mean anything: reserved / mapped bounds the multiple the same way. */
static int case_density_tightens_bound(Ctx& c) {
    /* Two 4096^2 textures (128 MiB reserved) on one 128 MiB chunk. */
    ID3D12Resource* a = make_reserved(c.dev.ptr(), 4096);
    ID3D12Resource* b = make_reserved(c.dev.ptr(), 4096);
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 256ull << 20);
    if (!a || !b || !pool) { printf("FAIL: setup\n"); return 1; }
    DmnSparseStats s0 = stats();
    CHECK(s0.chunks == 1 && s0.oversubscribe == 8, "two 64 MiB textures on one chunk at 8x "
          "(%u chunks, %ux)", s0.chunks, s0.oversubscribe);

    /* Map 40 MiB into each: 80 MiB of 128 MiB is 62.5% dense, so a chunk can
     * only carry 1.6x its size -- clamped to the floor of 2. */
    map_tiles(c, a, pool, 640, 32);
    map_tiles(c, b, pool, 640, 32);
    dump("after mapping 80 of the 128 MiB reserved");
    DmnSparseStats s1 = stats();
    CHECK(s1.mapped_bytes == (80ull << 20), "80 MiB mapped (%.0f MiB)", mib(s1.mapped_bytes));
    CHECK(s1.oversubscribe == 2, "the multiple follows the measured density (%ux)",
          s1.oversubscribe);
    CHECK(s1.dry_events == 0, "and nothing ran dry (%llu)", (unsigned long long)s1.dry_events);

    a->Release(); b->Release(); pool->Release();
    return fails;
}

/* REUSE_SINGLE_TILE points many resource tiles at ONE pool tile.  Metal sparse
 * textures cannot alias, so we spend a real tile for each and our demand runs
 * past the pool -- which is exactly the property that made the pool a bound.
 * It must stop counting, in both directions. */
static int case_aliasing_drops_the_pool(Ctx& c) {
    std::vector<ID3D12Resource*> res;
    for (int i = 0; i < 8; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) res.push_back(r);
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 32ull << 20);
    if (res.size() < 8 || !pool) { printf("FAIL: setup\n"); return 1; }
    map_tiles(c, res[0], pool, 1);
    DmnSparseStats before = stats();
    CHECK(before.oversubscribe == 4, "the pool bounds the multiple first (%ux)",
          before.oversubscribe);

    map_tiles(c, res[1], pool, 8, 0, D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE);
    dump("after a REUSE_SINGLE_TILE mapping");
    DmnSparseStats s = stats();
    CHECK(s.app_pool_aliased == 1, "the aliasing latch is set");
    CHECK(s.oversubscribe == 8, "and the pool no longer bounds the multiple (%ux)",
          s.oversubscribe);
    CHECK(s.budget_bytes == before.budget_bytes,
          "the budget is frozen where it stood (%.0f MiB), not raised by a pool that no longer "
          "bounds us", mib(s.budget_bytes));

    /* A larger pool named afterwards must not move either. */
    ID3D12Heap* pool2 = make_pool(c.dev.ptr(), 512ull << 20);
    if (pool2) {
        map_tiles(c, res[2], pool2, 1);
        DmnSparseStats t = stats();
        CHECK(t.oversubscribe == 8 && t.budget_bytes == before.budget_bytes,
              "a pool named after the latch is ignored too (%ux, %.0f MiB)", t.oversubscribe,
              mib(t.budget_bytes));
        pool2->Release();
    }

    for (auto* r : res) r->Release();
    pool->Release();
    return fails;
}

/* A destroyed resource returns its reservation to the chunk it lived on, so
 * that space must be usable again rather than stranded behind newer chunks. */
static int case_reuse_freed_space(Ctx& c) {
    /* 32 MiB chunks at 2x -> four 2048^2 textures each. */
    std::vector<ID3D12Resource*> first, second;
    for (int i = 0; i < 8; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) first.push_back(r);
    dump("after 8 creates");
    DmnSparseStats a = stats();
    CHECK(a.chunks == 2, "8 textures fill exactly 2 chunks (%u)", a.chunks);
    CHECK(a.live == 8, "all 8 are registered (%u)", a.live);

    /* Drop the four that landed on the first chunk. */
    for (int i = 0; i < 4; i++) first[(size_t)i]->Release();
    first.erase(first.begin(), first.begin() + 4);
    DmnSparseStats b = stats();
    CHECK(b.live == 4, "releasing 4 unregisters them (%u live)", b.live);
    CHECK(b.reserved_bytes < a.reserved_bytes,
          "and returns their reservation (%.0f -> %.0f MiB)",
          mib(a.reserved_bytes), mib(b.reserved_bytes));

    for (int i = 0; i < 4; i++)
        if (ID3D12Resource* r = make_reserved(c.dev.ptr(), 2048)) second.push_back(r);
    dump("after 4 more creates");
    DmnSparseStats d = stats();
    CHECK(d.chunks == 2,
          "the freed space is reused instead of opening a third chunk (%u chunks)", d.chunks);

    for (auto* r : first) r->Release();
    for (auto* r : second) r->Release();
    return fails;
}

/* A texture draws tiles only from its own heap, so a chunk smaller than what
 * one resource can map leaves that resource short however empty the rest of
 * the pool is.  Such a resource must get a chunk sized for it -- sized from
 * the app's pools, never from its virtual footprint. */
static int case_outsized_resource(Ctx& c) {
    /* 16 MiB chunks. First a small texture and a 128 MiB pool, so the module
     * knows what the app can map before the big texture arrives. */
    ID3D12Resource* small = make_reserved(c.dev.ptr(), 1024);   /* 4 MiB virtual */
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 128ull << 20);
    if (!small || !pool) { printf("FAIL: setup\n"); return 1; }
    map_tiles(c, small, pool, 1);
    dump("with a small texture and a 128 MiB pool");

    ID3D12Resource* big = make_reserved(c.dev.ptr(), 4096);     /* 64 MiB virtual */
    CHECK(big != nullptr, "created a 4096^2 reserved texture (64 MiB of virtual texture)");
    if (!big) { small->Release(); pool->Release(); return 1; }
    dump("after the outsized create");

    DmnSparseStats s = stats();
    CHECK(s.largest_chunk >= (64ull << 20),
          "it gets a chunk sized to hold its tiles, not the 16 MiB default (largest %.0f MiB)",
          mib(s.largest_chunk));
    CHECK(s.chunk_bytes <= (128ull << 20) + (32ull << 20),
          "sized from the app's pool rather than its virtual footprint (%.0f MiB total)",
          mib(s.chunk_bytes));

    /* The proof: map 24 MiB of tiles -- more than an ordinary chunk holds --
     * and read a pixel back from beyond that point.  A 4096^2 RGBA8 texture is
     * 32x32 tiles of 128x128 texels; rows 0..11 are 384 tiles = 24 MiB. */
    map_tiles(c, big, pool, 384, 32);
    dump("after mapping 24 MiB of its tiles");

    DmnSparseStats m = stats();
    CHECK(m.dry_events == 0, "no chunk ran dry (%llu dry events)",
          (unsigned long long)m.dry_events);

    /* Write a pattern into tile rows 10-11 (texel rows 1280..1536) and read it
     * back.  Those are tiles 320..383 of the resource, well past the 256 tiles
     * (16 MiB) an ordinary chunk would have held -- and the block must stay
     * INSIDE the mapped rows 0..11, or the unmapped half reads zero and the
     * case fails for the wrong reason. */
    D3D12_HEAP_PROPERTIES up{}; up.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES rbp{}; rbp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = 256 * 256 * 4; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* ub = nullptr; ID3D12Resource* rb = nullptr;
    c.dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd,
                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                   __uuidof(ID3D12Resource), (void**)&ub);
    c.dev->CreateCommittedResource(&rbp, D3D12_HEAP_FLAG_NONE, &bd,
                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                   __uuidof(ID3D12Resource), (void**)&rb);
    if (ub && rb) {
        { void* p = nullptr; D3D12_RANGE rr{0, 0}; ub->Map(0, &rr, &p);
          uint32_t* px = (uint32_t*)p;
          for (unsigned i = 0; i < 256 * 256; i++) px[i] = 0xff00ff00u | (i << 8);
          ub->Unmap(0, nullptr); }

        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = ub;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        src.PlacedFootprint.Footprint.Width = 256;
        src.PlacedFootprint.Footprint.Height = 256;
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = 1024;
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = big;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
        c.begin(); c.list->CopyTextureRegion(&dst, 0, 1280, 0, &src, nullptr); c.submit_wait();

        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = big;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.Subresource = 0;
        D3D12_TEXTURE_COPY_LOCATION rsrc{}; rsrc.pResource = big;
        rsrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; rsrc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION rdst{}; rdst.pResource = rb;
        rdst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        rdst.PlacedFootprint.Footprint = src.PlacedFootprint.Footprint;
        D3D12_BOX box{0, 1280, 0, 256, 1280 + 256, 1};
        c.begin(); c.list->ResourceBarrier(1, &b);
        c.list->CopyTextureRegion(&rdst, 0, 0, 0, &rsrc, &box); c.submit_wait();

        void* p = nullptr; D3D12_RANGE rr{0, 256 * 256 * 4}; rb->Map(0, &rr, &p);
        unsigned nz = 0;
        if (p) { const uint32_t* px = (const uint32_t*)p;
                 for (unsigned i = 0; i < 256 * 256; i++) if (px[i]) nz++; }
        rb->Unmap(0, nullptr);
        CHECK(nz == 256 * 256,
              "tiles past the 16 MiB an ordinary chunk holds are really mapped and read back "
              "(%u/65536 nonzero)", nz);
    }
    if (ub) ub->Release();
    if (rb) rb->Release();
    big->Release(); small->Release(); pool->Release();
    return fails;
}

/* When one resource does outrun its chunk -- which only happens when nothing
 * was known about the app in time -- the resource itself cannot be helped, but
 * every later one can: the chunks opened from then on are bigger.  A bigger
 * chunk is strictly safer for dryness, so this may be loosened on the spot. */
static int case_heals_chunk_size(Ctx& c) {
    /* 16 MiB chunks and NO pool named, so nothing sizes the chunk up front. */
    ID3D12Resource* big = make_reserved(c.dev.ptr(), 4096);
    ID3D12Heap* pool = make_pool(c.dev.ptr(), 256ull << 20);
    if (!big || !pool) { printf("FAIL: setup\n"); return 1; }
    DmnSparseStats a = stats();
    CHECK(a.chunk_size_mb == 16, "it starts on an ordinary 16 MiB chunk (%u MiB)",
          a.chunk_size_mb);

    /* Map 32 MiB of tiles into it: twice what its chunk holds. */
    map_tiles(c, big, pool, 512, 32);
    dump("after one resource outran its chunk");

    DmnSparseStats b = stats();
    CHECK(b.dry_events > 0, "the chunk is reported dry (%llu events)",
          (unsigned long long)b.dry_events);
    CHECK(b.chunk_size_mb > a.chunk_size_mb,
          "later chunks are made bigger (%u -> %u MiB)", a.chunk_size_mb, b.chunk_size_mb);

    /* And the next resource really does land on a bigger chunk. */
    ID3D12Resource* next = make_reserved(c.dev.ptr(), 4096);
    dump("after the next create");
    DmnSparseStats d = stats();
    CHECK(d.largest_chunk > (16ull << 20),
          "the next chunk opened is larger than the old default (%.0f MiB)",
          mib(d.largest_chunk));
    if (next) next->Release();
    big->Release(); pool->Release();
    return fails;
}

/* == case table and the per-case re-exec ================================= */

struct Case {
    const char* name;
    int (*fn)(Ctx&);
    const char* env[8];   /* NAME=VALUE, NULL-terminated */
};

static const Case kCases[] = {
    {"budget-caps-chunks", case_budget_caps_chunks,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_MAX_MB=64", nullptr}},
    {"spread-by-footprint", case_spread_by_footprint,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_MAX_MB=512", "DMN_SPARSE_OVERSUBSCRIBE=2", nullptr}},
    {"budget-grows", case_budget_grows,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_BUDGET_FLOOR_MB=64", "DMN_SPARSE_OVERSUBSCRIBE=2",
      nullptr}},
    {"cap-is-hard", case_cap_is_hard,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_MAX_MB=64", "DMN_SPARSE_OVERSUBSCRIBE=2", nullptr}},
    {"pool-tightens-bound", case_pool_tightens_bound,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_MAX_MB=512", nullptr}},
    {"density-tightens-bound", case_density_tightens_bound,
     {"DMN_SPARSE_HEAP_MB=128", "DMN_SPARSE_MAX_MB=512", nullptr}},
    {"aliasing-drops-pool", case_aliasing_drops_the_pool,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_BUDGET_FLOOR_MB=512", nullptr}},
    {"reuse-freed-space", case_reuse_freed_space,
     {"DMN_SPARSE_HEAP_MB=32", "DMN_SPARSE_BUDGET_FLOOR_MB=512", "DMN_SPARSE_OVERSUBSCRIBE=2",
      nullptr}},
    {"outsized-resource", case_outsized_resource,
     {"DMN_SPARSE_HEAP_MB=16", "DMN_SPARSE_BUDGET_FLOOR_MB=512", nullptr}},
    {"heals-chunk-size", case_heals_chunk_size,
     {"DMN_SPARSE_HEAP_MB=16", "DMN_SPARSE_BUDGET_FLOOR_MB=512", nullptr}},
};

static int run_case(const Case& c) {
    printf("== %s\n", c.name);
    Ctx ctx;
    if (!ctx.init()) { printf("no device\n"); return 1; }
    return c.fn(ctx);
}

/* Re-exec this binary for one case, inheriting stdio so its lines land in the
 * test log.  setenv() before spawning puts the case's knobs into `environ`,
 * which carries DYLD_LIBRARY_PATH and D3DMETAL_FRAMEWORK_PATH along with it. */
static bool spawn_case(const Case& c) {
    char exe[4096];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0) return false;

    setenv("DMN_SPARSE_CASE", c.name, 1);
    setenv("DMN_SPARSE", "1", 1);
    for (const char* e : c.env) {
        if (!e) break;
        std::string kv(e);
        const size_t eq = kv.find('=');
        setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
    }

    char* argv[] = {exe, nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, exe, nullptr, nullptr, argv, environ) != 0) return false;
    int status = 0;
    waitpid(pid, &status, 0);

    /* Leave nothing set for the next case: it gets its own process, but the
     * PARENT's environment is what seeds it. */
    unsetenv("DMN_SPARSE_CASE");
    for (const char* e : c.env) {
        if (!e) break;
        std::string kv(e);
        unsetenv(kv.substr(0, kv.find('=')).c_str());
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main(void) {
    if (const char* one = getenv("DMN_SPARSE_CASE")) {
        for (const Case& c : kCases)
            if (!strcmp(c.name, one))
                return run_case(c) == 0 ? 0 : 1;
        printf("unknown case %s\n", one);
        return 1;
    }

    int bad = 0;
    for (const Case& c : kCases) {
        if (!spawn_case(c)) { printf("CASE FAILED: %s\n", c.name); bad++; }
    }
    printf(bad ? "SPARSE-SIZING FAIL (%d case(s))\n" : "SPARSE-SIZING PASS\n", bad);
    return bad ? 1 : 0;
}
