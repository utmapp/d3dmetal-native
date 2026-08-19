/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * See dmn_sparse.h.  Metal side of reserved-resource backing.
 */

#import <Metal/Metal.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <stdlib.h>
#include <mach/mach_time.h>

#include "dmn_directx_types.h"
#include "dmn_private.h"
#include "dmn_sparse.h"

extern "C" void dmn_sub_resource_track(id res, int sparse);

namespace {

struct TileShape { uint32_t w, h; };

/* D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES */
constexpr uint64_t kD3DTileBytes = 65536;

/* Bits per format element: per texel, or per 4x4 block for the
 * block-compressed formats (*out_block says which).  Zero for every format
 * D3D12 forbids in a tiled resource -- the 96-bit group, the video and
 * palettized formats, R1_UNORM, the 4:2:2 pair and UNKNOWN -- which is also
 * every format with no standard tile shape. */
uint32_t dxgi_element_bits(uint32_t f, bool* out_block) {
    *out_block = false;
    switch ((DXGI_FORMAT)f) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return 128;

    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        return 64;

    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return 32;

    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
    case DXGI_FORMAT_B4G4R4A4_UNORM:
        return 16;

    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
        return 8;

    /* 8 bytes per 4x4 block. */
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        *out_block = true;
        return 64;

    /* 16 bytes per 4x4 block. */
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        *out_block = true;
        return 128;

    default:
        return 0;
    }
}

/* The D3D12 standard tile shape of a 2D subresource, in texels (D3D12 spec,
 * "Standard Tile Shapes": 64 KiB of elements laid out as squarely as the
 * element size allows).  A block-compressed format uses the same shape
 * counted in 4x4 blocks, hence four times the texels on each axis.
 *
 * A format with no standard shape cannot be a tiled resource in the first
 * place, so any shape would do; it gets the 32-bit one rather than a zero
 * that would divide by zero in the tile arithmetic. */
TileShape d3d_tile_shape(uint32_t f) {
    bool block = false;
    TileShape s;
    switch (dxgi_element_bits(f, &block)) {
    case 8:   s = {256, 256}; break;
    case 16:  s = {256, 128}; break;
    case 32:  s = {128, 128}; break;
    case 64:  s = {128, 64};  break;
    case 128: s = {64, 64};   break;
    default:  s = {128, 128}; break;
    }
    if (block) { s.w *= 4; s.h *= 4; }
    return s;
}

struct Chunk {
    id<MTLHeap> heap;
    uint64_t bytes;
    int64_t mapped_bytes;    /* our own tally of map - unmap, in Metal tiles */
    int64_t reserved_bytes;  /* sum of the placement reservations on it */
};

struct SparseRec {
    id<MTLTexture> tex;
    size_t chunk;
    uint32_t dimension;
    uint32_t format;
    uint64_t width; uint32_t height; uint16_t depth_or_array; uint16_t mips;
    TileShape tile;
    uint32_t elem_bits;       /* dxgi_element_bits() of the format */
    bool block;               /* ... per 4x4 block rather than per texel */
    uint32_t standard_mips;   /* mips >= tile in both dimensions */
    uint32_t packed_tiles;    /* D3D12 tiles the packed mips occupy, per slice */
    uint32_t array_size;
    MTLSize metal_tile;
    uint32_t first_tail_mip;  /* Metal mip tail: always mapped, never touch */
    int64_t mapped_bytes;
    int64_t reserved_bytes;   /* what its placement charged to the chunk */
};

/* The D3D12 tile grid of a record (see dmn_sparse.h, DmnSparseTiling). */
uint64_t mip_extent(uint64_t full, uint32_t mip) { return full >> mip ? full >> mip : 1; }
uint32_t tiles_w(const SparseRec& r, uint32_t mip) {
    return (uint32_t)((mip_extent(r.width, mip) + r.tile.w - 1) / r.tile.w);
}
uint32_t tiles_h(const SparseRec& r, uint32_t mip) {
    return (uint32_t)((mip_extent(r.height, mip) + r.tile.h - 1) / r.tile.h);
}
/* Tile index of a standard mip's first tile within its slice; for the packed
 * mips, the packed region's first tile. */
uint32_t mip_start_tile(const SparseRec& r, uint32_t mip) {
    uint32_t n = 0;
    for (uint32_t m = 0; m < mip && m < r.standard_mips; m++) n += tiles_w(r, m) * tiles_h(r, m);
    return n;
}
uint32_t slice_tiles(const SparseRec& r) { return mip_start_tile(r, r.mips) + r.packed_tiles; }
uint64_t mip_bytes(const SparseRec& r, uint32_t mip) {
    uint64_t w = mip_extent(r.width, mip), h = mip_extent(r.height, mip);
    if (r.block) { w = (w + 3) / 4; h = (h + 3) / 4; }
    return w * h * r.elem_bits / 8;
}

std::mutex g_mtx;
std::unordered_map<void*, SparseRec> g_regs;
std::vector<Chunk> g_chunks;
id<MTLDevice> g_dev;
id<MTLCommandQueue> g_queue;
uint64_t g_chunk_bytes;
uint64_t g_max_bytes;          /* the budget floor; the app's pools may raise it */
uint64_t g_max_explicit;       /* DMN_SPARSE_MAX_MB, when set: a hard cap */
uint64_t g_heap_max_bytes;     /* largest single chunk we will ever open */
uint64_t g_budget_high_water;  /* the budget only ever grows (see budget_locked) */
uint64_t g_metal_tile_bytes;
bool g_init_done;
bool g_supported;
std::atomic<uint64_t> g_calls{0}, g_ops{0}, g_wait_us{0}, g_dry{0}, g_refused{0};

/* Tile pools the app itself owns, by identity, and their total: the app
 * cannot map more tiles than the pools it owns, so this bounds its tile
 * demand.  Only in aggregate -- D3D12 lets one reserved resource hold tiles
 * from several heaps at once ("Reserved resources may be mapped to pages from
 * multiple heaps at the same time", D3D12_TILED_RESOURCES_TIER remarks) --
 * which costs nothing here, because the pages handed out come from our own
 * sparse chunk and never from the app's heap; the heap is a measurement, not
 * a supply. */
std::unordered_map<void*, uint64_t> g_app_pools;
uint64_t g_app_pool_bytes;
/* Set once the app maps several resource tiles onto ONE heap tile
 * (D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE).  Metal sparse textures cannot
 * alias tiles, so from then on a real tile is spent where the app spent a
 * shared one and our demand can exceed the pool it owns -- exactly the
 * property that made its pool size a ceiling. */
bool g_app_pool_aliased;

thread_local bool t_armed;
thread_local id<MTLTexture> t_captured;

/* DMN_SPARSE_STATS=<n> emits the sizing summary every n UpdateTileMappings
 * calls at WARN, the level a host is likely to run at; unset, the summary
 * still goes out at INFO on the ordinary cadence. */
uint32_t stats_every(void) {
    static int cached;
    static uint32_t v;
    if (!cached) {
        cached = 1;
        const char* e = getenv("DMN_SPARSE_STATS");
        v = (e && *e) ? (uint32_t)strtoul(e, nullptr, 10) : 0;
    }
    return v;
}

/* ON by default; DMN_SPARSE=0 opts out (reserved resources fully backed). */
bool env_enabled(void) {
    const char* e = getenv("DMN_SPARSE");
    return !(e && *e == '0');
}
uint64_t env_mb(const char* name, uint64_t def_mb) {
    const char* e = getenv(name);
    if (!e || !*e) return def_mb << 20;
    unsigned long long v = strtoull(e, nullptr, 10);
    return (v ? v : def_mb) << 20;
}

void init_locked(void) {
    if (g_init_done) return;
    g_init_done = true;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev || !env_enabled()) {
        if (g_dev) DMN_INFO("sparse: disabled by DMN_SPARSE=0; reserved resources are fully backed");
        return;
    }
    if (![g_dev respondsToSelector:@selector(sparseTileSizeWithTextureType:pixelFormat:sampleCount:)]) {
        DMN_WARN("sparse: device has no sparse texture support; reserved resources stay committed");
        return;
    }
    g_chunk_bytes = env_mb("DMN_SPARSE_HEAP_MB", 1024);
    /* An explicit DMN_SPARSE_MAX_MB is a HARD cap -- someone limiting our RAM
     * gets exactly what they asked for.  Unset, the budget starts at
     * DMN_SPARSE_BUDGET_FLOOR_MB and the app's own tile pools may raise it
     * (budget_locked). */
    g_max_explicit = env_mb("DMN_SPARSE_MAX_MB", 0);
    g_max_bytes = g_max_explicit ? g_max_explicit : env_mb("DMN_SPARSE_BUDGET_FLOOR_MB", 6144);
    if (g_max_bytes < g_chunk_bytes) g_max_bytes = g_chunk_bytes;
    /* A chunk sized for one outsized resource may exceed DMN_SPARSE_HEAP_MB;
     * Metal documents no maximum heap size, but a sparse heap commits its whole
     * size once touched, so cap how far that can go. */
    g_heap_max_bytes = env_mb("DMN_SPARSE_HEAP_MAX_MB", 4096);
    if (g_heap_max_bytes < g_chunk_bytes) g_heap_max_bytes = g_chunk_bytes;
    g_budget_high_water = g_max_bytes;
    g_metal_tile_bytes = [g_dev sparseTileSizeInBytes];
    if (!g_metal_tile_bytes) g_metal_tile_bytes = 16384;
    g_queue = [g_dev newCommandQueue];
    g_supported = g_queue != nil;
    DMN_INFO("sparse: reserved resources get sparse backing (chunk %llu MiB, budget %llu MiB%s, "
             "chunk max %llu MiB, Metal tile %llu B; DMN_SPARSE_HEAP_MB / DMN_SPARSE_MAX_MB / "
             "DMN_SPARSE_HEAP_MAX_MB / DMN_SPARSE=0)",
             (unsigned long long)(g_chunk_bytes >> 20), (unsigned long long)(g_max_bytes >> 20),
             g_max_explicit ? " (hard cap)" : " floor",
             (unsigned long long)(g_heap_max_bytes >> 20),
             (unsigned long long)g_metal_tile_bytes);
}

/* Virtual texture placed so far, across every chunk.  Caller holds g_mtx. */
uint64_t total_reserved_locked(void) {
    uint64_t v = 0;
    for (const Chunk& c : g_chunks)
        v += (uint64_t)std::max<int64_t>(0, c.reserved_bytes);
    return v;
}

/* Tiles mapped so far, across every chunk.  Caller holds g_mtx. */
uint64_t total_mapped_locked(void) {
    uint64_t v = 0;
    for (const Chunk& c : g_chunks)
        v += (uint64_t)std::max<int64_t>(0, c.mapped_bytes);
    return v;
}

/* Bytes of sparse heap currently open.  Caller holds g_mtx. */
uint64_t open_bytes_locked(void) {
    uint64_t have = 0;
    for (const Chunk& c : g_chunks) have += c.bytes;
    return have;
}

/* How many bytes of sparse heap we are allowed to hold.
 *
 * The app's own tile pools bound this from BELOW, and only from below: the
 * total is a running sum of the pools it has named so far, which starts near
 * nothing (a title opens chunk 0 before naming a single pool and keeps
 * naming them all run).  A number that only grows can prove "the app needs
 * at least this much" and nothing else, so it may RAISE a ceiling and must
 * never lower one.
 *
 * The direction is the OPPOSITE of oversubscribe_locked's, deliberately.
 * There, reading the app as sparser than it is loosens a per-chunk limit and
 * silently leaves tiles unmapped, so estimates may only tighten.  Here, being
 * wrong high costs nothing until a chunk is actually demanded, and the thing
 * a permitted chunk replaces is a fully-backed texture that costs its entire
 * virtual size.  A budget may only loosen.
 *
 * Two known imprecisions, both absorbed by the margin and neither able to
 * starve a chunk -- they only shift how much we are PERMITTED to hold:
 * g_app_pools is keyed by heap pointer with no destruction hook, so address
 * reuse undercounts while a dead heap's bytes linger and overcount.
 *
 * Caller holds g_mtx. */
uint64_t budget_locked(void) {
    if (g_max_explicit)
        return g_max_explicit;   /* someone capping our RAM gets their cap */
    uint64_t b = g_budget_high_water;
    /* Once the app aliases tiles our demand legitimately runs past the pools
     * it owns, so their total stops bounding anything and the budget freezes
     * where it stands. */
    if (!g_app_pool_aliased && g_app_pool_bytes) {
        /* +25% for our outward tile rounding (measured at 1.1%: 2210 MiB of
         * pools declared against 2234 MiB of tiles mapped) and for rounding
         * up to a whole chunk. */
        uint64_t want = g_app_pool_bytes + g_app_pool_bytes / 4;
        const uint64_t unit = g_chunk_bytes ? g_chunk_bytes : (1ull << 20);
        want = ((want + unit - 1) / unit) * unit;
        if (want > b) {
            b = want;
            DMN_INFO("sparse: budget raised to %llu MiB -- the app has named %llu MiB of tile "
                     "pools, so it needs at least that much and the %llu MiB floor would have "
                     "sent the excess to the fully-backed path",
                     (unsigned long long)(b >> 20), (unsigned long long)(g_app_pool_bytes >> 20),
                     (unsigned long long)(g_max_bytes >> 20));
        }
    }
    /* Never below what is already open: those chunks exist. */
    const uint64_t have = open_bytes_locked();
    if (b < have) b = have;
    g_budget_high_water = b;
    return b;
}

/* A new sparse chunk of at least `want_bytes`, or -1 past the budget.  Caller
 * holds g_mtx.  The heap's whole size becomes resident at first use (measured;
 * Apple documents only that "you allocate memory when you create the heap"),
 * so this is the real cost of the module: chunks are opened only as mapping
 * demand grows.  `want_bytes` is 0 for an ordinary chunk and non-zero only for
 * a resource that cannot fit one -- a texture draws tiles solely from the heap
 * it was created on (Metal's updateTextureMapping: takes no heap), so the
 * chunk size is a hard per-resource ceiling. */
long open_chunk_locked(uint64_t want_bytes = 0) {
    uint64_t size = g_chunk_bytes;
    if (want_bytes > size) size = want_bytes;
    if (size > g_heap_max_bytes) size = g_heap_max_bytes;
    /* A heap size Metal is happy with, and one our own tile tally lines up
     * with: whole sparse tiles. */
    if (g_metal_tile_bytes)
        size = ((size + g_metal_tile_bytes - 1) / g_metal_tile_bytes) * g_metal_tile_bytes;

    const uint64_t have = open_bytes_locked();
    const uint64_t budget = budget_locked();
    if (have + size > budget) {
        if (g_refused++ == 0)
            DMN_WARN("sparse: budget of %llu MiB reached (%zu chunks holding %llu MiB of virtual "
                     "texture); further reserved resources are fully backed, at their whole "
                     "virtual size. The app has named %llu MiB of tile pools. %s",
                     (unsigned long long)(budget >> 20), g_chunks.size(),
                     (unsigned long long)(total_reserved_locked() >> 20),
                     (unsigned long long)(g_app_pool_bytes >> 20),
                     g_max_explicit
                         ? "DMN_SPARSE_MAX_MB is set, so this cap is deliberate."
                         : "The budget follows those pools, so a budget short of them means they "
                           "are still growing; a budget well past them means the app reserves "
                           "texture far faster than it maps it, which DMN_SPARSE_OVERSUBSCRIBE "
                           "spreads over fewer chunks.");
        return -1;
    }
    MTLHeapDescriptor* hd = [[MTLHeapDescriptor alloc] init];
    hd.type = MTLHeapTypeSparse;
    hd.storageMode = MTLStorageModePrivate;
    hd.hazardTrackingMode = MTLHazardTrackingModeTracked;
    hd.size = size;
    id<MTLHeap> heap = [g_dev newHeapWithDescriptor:hd];
    [hd release];
    if (!heap) {
        DMN_ERROR("sparse: newHeapWithDescriptor(sparse, %llu MiB) failed",
                  (unsigned long long)(size >> 20));
        return -1;
    }
    Chunk c{}; c.heap = heap; c.bytes = size;
    g_chunks.push_back(c);
    DMN_INFO("sparse: opened chunk %zu (%llu MiB%s; %llu MiB total of %llu budget)%s",
             g_chunks.size() - 1, (unsigned long long)(size >> 20),
             size > g_chunk_bytes ? " -- sized for one outsized resource" : "",
             (unsigned long long)((have + size) >> 20),
             (unsigned long long)(budget >> 20),
             g_chunks.size() > 1 ? "" : " -- first reserved resource");
    return (long)g_chunks.size() - 1;
}

/* Bytes a texture would occupy if every one of its tiles were mapped: the only
 * predictor of its tile demand available at creation, and creation is the only
 * moment placement can act, because Metal binds a texture to the heap it is
 * created on while a reserved resource's tiles arrive long afterwards. */
uint64_t virtual_footprint(MTLTextureDescriptor* d) {
    if (!d || !g_dev) return 0;
    /* Count Metal sparse tiles, not texels: the tile is what the heap
     * allocates, and its dimensions already account for the pixel format,
     * including the compressed ones a bits-per-pixel helper returns 0 for. */
    const MTLSize t = [g_dev sparseTileSizeWithTextureType:d.textureType
                                              pixelFormat:d.pixelFormat
                                              sampleCount:1];
    if (!t.width || !t.height) return 0;
    const uint64_t w = d.width ? d.width : 1, h = d.height ? d.height : 1;
    const uint32_t mips = d.mipmapLevelCount ? (uint32_t)d.mipmapLevelCount : 1;
    uint64_t tiles = 0;
    for (uint32_t m = 0; m < mips; m++) {
        const uint64_t mw = (w >> m) ? (w >> m) : 1;
        const uint64_t mh = (h >> m) ? (h >> m) : 1;
        tiles += ((mw + t.width - 1) / t.width) * ((mh + t.height - 1) / t.height);
    }
    const uint64_t slices = d.textureType == MTLTextureType3D
                                ? (d.depth ? d.depth : 1)
                                : (d.arrayLength ? d.arrayLength : 1);
    /* Slightly over: Metal packs the small mips into one tail rather than
     * giving each its own tiles.  Over-reserving costs a chunk boundary,
     * under-reserving costs unmapped tiles that read zero. */
    return tiles * slices * (g_metal_tile_bytes ? g_metal_tile_bytes : 16384);
}

/* The explicit DMN_SPARSE_OVERSUBSCRIBE, or 0 when it is not set.  An
 * explicit value is a ceiling the user chose and is never exceeded; the
 * built-in default is only a starting point. */
uint64_t oversubscribe_explicit(void) {
    static int cached;
    static uint64_t v;
    if (!cached) {
        cached = 1;
        const char* e = getenv("DMN_SPARSE_OVERSUBSCRIBE");
        v = e && *e ? strtoull(e, nullptr, 10) : 0;
    }
    return v;
}

/* Where to start before anything is known about the app: the densest tile
 * usage seen in practice (a virtual-texturing title mapping 11.3% of what it
 * reserves, so a chunk may carry at most ~8.8x its size there). */
const uint64_t kOversubscribeDefault = 8;

/* How much virtual texture one chunk may carry, as a multiple of its size.  A
 * reserved resource exists so that only a fraction of it is ever mapped, so a
 * chunk must be oversubscribed or nothing fits; the bound is 1 / (tile
 * density), since a chunk carrying V bytes of virtual texture at density d
 * must hold V*d bytes of tiles.  Three sources for d, best first:
 *
 *   1. The app's own tile pools.  It cannot map more tiles than it owns, so
 *      d = (sum of its pools) / (virtual it has reserved) is a real bound and
 *      not an estimate.  Known once it names a pool in UpdateTileMappings.
 *   2. The density measured so far, once enough tiles are mapped to mean
 *      anything.  Applies to apps that map before reserving much more.
 *   3. DMN_SPARSE_OVERSUBSCRIBE, for the burst of creates at a level load,
 *      where neither of the above exists yet.
 *
 * Both signals accumulate as the app runs -- its tile pools are created over
 * time, and mapping lags creation -- so early on both ratios read far sparser
 * than the app will end up being.  Reading "sparser" loosens the bound, which
 * is the direction that silently leaves tiles unmapped, so neither may do
 * that: they only ever TIGHTEN it, and the ceiling is the default (or the
 * explicit setting).  Sizing a run that is genuinely sparser than the default
 * assumes is the budget's job (DMN_SPARSE_MAX_MB), not this bound's.
 *
 * Clamped either way: a runaway value would either starve chunks (too high)
 * or open a chunk per texture (too low).  Caller holds g_mtx. */
uint64_t oversubscribe_locked(void) {
    const uint64_t explicit_v = oversubscribe_explicit();
    const uint64_t reserved = total_reserved_locked();
    uint64_t v = explicit_v ? explicit_v : kOversubscribeDefault;
    /* The pool bound rests on the app being unable to map more tiles than it
     * owns, which REUSE_SINGLE_TILE breaks (see g_app_pool_aliased): once it
     * has aliased, a bound derived from the pool is too LOOSE. */
    if (!g_app_pool_aliased && g_app_pool_bytes && reserved > g_app_pool_bytes) {
        const uint64_t est = reserved / g_app_pool_bytes;
        if (est < v) v = est;
    }
    const uint64_t mapped = total_mapped_locked();
    if (mapped > (64ull << 20) && reserved > mapped) {
        /* Per-chunk density runs to about twice the global figure (measured:
         * 21.9% on one chunk against an 11.3% average), so a reading of
         * "denser than assumed" is worth acting on. */
        const uint64_t est = reserved / mapped;
        if (est < v) v = est;
    }
    if (v < 2) v = 2;
    if (v > 64) v = 64;
    /* An explicit setting is an absolute ceiling: someone lowering it is
     * working around a starving chunk, and nothing derived may undo that. */
    if (explicit_v && v > explicit_v)
        v = explicit_v;
    return v;
}

/* The most tiles ONE resource could ever have mapped at once, or 0 when that
 * is not knowable yet.
 *
 * This matters because a texture draws tiles only from the heap it was created
 * on, so a chunk smaller than this number can leave that one resource short no
 * matter how empty the rest of the pool is -- the failure the "SINGLE resource"
 * error reports, which no oversubscription setting can fix.
 *
 * Its own virtual footprint is the obvious bound and a useless one: a reserved
 * resource exists precisely so that most of it is never mapped, so sizing a
 * heap that way would commit 20 GiB for a texture that touches 200 MiB.  The
 * app's own tile pools are the real ceiling -- it cannot map more tiles than
 * it owns -- so the answer is whichever is smaller.  Before any pool has been
 * named there is nothing to go on, and 0 leaves placement to the ordinary
 * chunk size.  Caller holds g_mtx. */
uint64_t resource_need_locked(uint64_t reserve) {
    if (!g_app_pool_bytes || g_app_pool_aliased)
        return 0;
    const uint64_t pool_cap = g_app_pool_bytes + g_app_pool_bytes / 4;
    return pool_cap < reserve ? pool_cap : reserve;
}

/* The chunk a new texture goes on, charging `reserve` (its virtual footprint)
 * to that chunk so textures spread in proportion to the tile demand they could
 * generate.  A mapped-bytes watermark alone cannot do this: a level load is a
 * burst of creates with nothing mapped yet, so every texture would land on one
 * chunk and its excess tiles would stay unmapped and read zero. */
long place_chunk_locked(uint64_t reserve) {
    const uint64_t need = resource_need_locked(reserve);
    if (g_chunks.empty()) return open_chunk_locked(need);
    const uint64_t over = oversubscribe_locked();

    /* First fit, oldest chunk first: a resource that dies returns its
     * reservation to whichever chunk it lived on, so a streaming app frees
     * space in older chunks continuously, and that space must be usable
     * again.  This never RAISES a chunk's limit -- a texture just lands
     * somewhere that has room under the same rule -- so it cannot starve a
     * chunk that a newest-only policy would have spared. */
    long fallback = -1;
    for (size_t i = 0; i < g_chunks.size(); i++) {
        const Chunk& c = g_chunks[i];
        /* Too small to ever hold this one resource's tiles: passing it over
         * costs a chunk, placing here would cost correct pixels. */
        if (need && c.bytes < need)
            continue;
        const uint64_t cap = c.bytes * over;
        const bool full_by_reservation =
            (uint64_t)std::max<int64_t>(0, c.reserved_bytes) + reserve > cap;
        /* A chunk this far into its tiles has little left to give, whatever
         * its reservation says. */
        const bool full_by_mapping =
            (uint64_t)std::max<int64_t>(0, c.mapped_bytes) * 4 >= c.bytes * 3;
        if (!full_by_reservation && !full_by_mapping)
            return (long)i;
        /* Over the MAPPED watermark alone is a soft signal -- that chunk may
         * still have room -- so remember it in case nothing better exists and
         * the budget refuses a new chunk. */
        if (!full_by_reservation && fallback < 0)
            fallback = (long)i;
    }

    /* Nothing has room: open one, sized for this resource if it needs it. */
    long n = open_chunk_locked(need);
    if (n >= 0) return n;
    /* Past the budget.  Piling onto a chunk that is already full by
     * reservation is how tiles end up unmapped and reading zero, so hand the
     * texture back to the committed path instead -- it costs its whole virtual
     * size in RAM but renders correctly, and open_chunk_locked has already
     * reported the budget.  A chunk that was only over the mapped watermark
     * still gets the texture. */
    return fallback;
}

thread_local size_t t_captured_chunk;
thread_local uint64_t t_captured_reserve;

} // namespace

bool dmn_sparse_enabled(void) {
    std::lock_guard<std::mutex> lk(g_mtx);
    init_locked();
    return g_supported;
}

void dmn_sparse_arm(void) {
    if (t_captured) { [t_captured release]; t_captured = nil; }
    t_armed = true;
}

bool dmn_sparse_disarm(void** out) {
    t_armed = false;
    if (out) *out = (void*)t_captured;
    bool got = t_captured != nil;
    t_captured = nil; /* ownership to caller */
    return got;
}

/* Called from the texture-creation swizzles (dmn_share_metal.mm) before their
 * own share arm check.  Returns a +0 texture the swizzle must return to
 * D3DMetal (it takes over the caller's +1 by returning it), or nil. */
extern "C" id dmn_sparse_try_substitute(id device, MTLTextureDescriptor* desc) {
    if (!t_armed) return nil;
    t_armed = false; /* one shot */
    std::lock_guard<std::mutex> lk(g_mtx);
    init_locked();
    if (!g_supported) return nil;
    (void)device;
    if (desc.textureType != MTLTextureType2D && desc.textureType != MTLTextureType2DArray)
        return nil;
    const uint64_t reserve = virtual_footprint(desc);
    long ci = place_chunk_locked(reserve);
    if (ci < 0) return nil;
    MTLTextureDescriptor* d = [desc copy];
    d.storageMode = MTLStorageModePrivate;
    d.resourceOptions = (d.resourceOptions & ~MTLResourceStorageModeMask) | MTLResourceStorageModePrivate;
    d.hazardTrackingMode = MTLHazardTrackingModeTracked;
    id<MTLTexture> tex = [g_chunks[(size_t)ci].heap newTextureWithDescriptor:d];
    if (!tex) {
        /* That chunk cannot even hold the texture's mip tail: open a fresh one
         * (sized for this resource if it needs it) and try there. */
        long n = open_chunk_locked(resource_need_locked(reserve));
        if (n >= 0) { ci = n; tex = [g_chunks[(size_t)ci].heap newTextureWithDescriptor:d]; }
    }
    [d release];
    if (!tex) {
        DMN_WARN("sparse: heap newTextureWithDescriptor failed (%lux%lu fmt=%lu mips=%lu usage=0x%lx); "
                 "falling back to committed", (unsigned long)desc.width,
                 (unsigned long)desc.height, (unsigned long)desc.pixelFormat,
                 (unsigned long)desc.mipmapLevelCount, (unsigned long)desc.usage);
        return nil;
    }
    dmn_sub_resource_track(tex, /*sparse=*/1);
    g_chunks[(size_t)ci].reserved_bytes += (int64_t)reserve;
    t_captured_chunk = (size_t)ci;
    t_captured_reserve = reserve;
    t_captured = [tex retain]; /* +1 for the registry */
    return tex;                /* +1 for D3DMetal (it releases its texture) */
}

void dmn_sparse_register(void* identity, void* metal_texture, uint32_t dimension,
                         uint32_t dxgi_format, uint64_t width, uint32_t height,
                         uint16_t depth_or_array, uint16_t mips) {
    id<MTLTexture> tex = (id<MTLTexture>)metal_texture;
    if (!identity || !tex) return;
    SparseRec r{};
    r.tex = tex; r.chunk = t_captured_chunk; r.reserved_bytes = (int64_t)t_captured_reserve;
    r.dimension = dimension; r.format = dxgi_format;
    r.width = width; r.height = height; r.depth_or_array = depth_or_array;
    /* D3D12_RESOURCE_DESC::MipLevels 0 means "the full chain" (what
     * CD3DX12_RESOURCE_DESC::Tex2D defaults to), and the resolved count is
     * what the framework built the texture with -- ask the texture rather
     * than believe the 0, or every subresource would address mip 0. */
    r.mips = mips ? mips : (uint32_t)tex.mipmapLevelCount;
    if (!r.mips) r.mips = 1;
    r.tile = d3d_tile_shape(dxgi_format);
    r.elem_bits = dxgi_element_bits(dxgi_format, &r.block);
    if (!r.elem_bits) r.elem_bits = 32; /* no standard shape; d3d_tile_shape gave the 32-bit one */
    r.standard_mips = 0;
    for (uint32_t m = 0; m < r.mips; m++) {
        if (mip_extent(width, m) >= r.tile.w && mip_extent(height, m) >= r.tile.h)
            r.standard_mips = m + 1;
        else
            break;
    }
    /* Packed mips: as many 64 KiB tiles as their bytes need, at least one. */
    r.packed_tiles = 0;
    if (r.standard_mips < r.mips) {
        uint64_t b = 0;
        for (uint32_t m = r.standard_mips; m < r.mips; m++) b += mip_bytes(r, m);
        r.packed_tiles = (uint32_t)std::max<uint64_t>(1, (b + kD3DTileBytes - 1) / kD3DTileBytes);
    }
    r.array_size = (dimension == 4 /* D3D12_RESOURCE_DIMENSION_TEXTURE3D */ || !depth_or_array)
                       ? 1 : depth_or_array;
    r.metal_tile = [g_dev sparseTileSizeWithTextureType:tex.textureType pixelFormat:tex.pixelFormat sampleCount:1];
    r.first_tail_mip = (uint32_t)tex.firstMipmapInTail;
    std::lock_guard<std::mutex> lk(g_mtx);
    g_regs[identity] = r;
    static std::atomic<uint32_t> n{0};
    uint32_t k = ++n;
    if (k <= 16 || (k % 256) == 0)
        DMN_INFO("sparse: reserved #%u %llux%u fmt=%u mips=%u arrays=%u: D3D tile %ux%u (%u standard mips, "
                 "%u packed tiles, %u tiles/slice), Metal tile %lux%lu, Metal tail from mip %u, chunk %zu; "
                 "%zu live",
                 k, (unsigned long long)width, height, dxgi_format, r.mips, r.array_size,
                 r.tile.w, r.tile.h, r.standard_mips, r.packed_tiles, slice_tiles(r),
                 (unsigned long)r.metal_tile.width, (unsigned long)r.metal_tile.height,
                 r.first_tail_mip, r.chunk, g_regs.size());
}

void dmn_sparse_unregister(void* identity) {
    id<MTLTexture> tex = nil;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_regs.find(identity);
        if (it == g_regs.end()) return;
        tex = it->second.tex;
        if (it->second.chunk < g_chunks.size()) {
            Chunk& c = g_chunks[it->second.chunk];
            c.mapped_bytes -= it->second.mapped_bytes;
            c.reserved_bytes -= it->second.reserved_bytes;
            if (c.reserved_bytes < 0) c.reserved_bytes = 0;
        }
        g_regs.erase(it);
    }
    /* Releasing the texture returns its mapped tiles to the heap. */
    [tex release];
}

void dmn_sparse_release_texture(void* metal_texture) {
    id<MTLTexture> tex = (id<MTLTexture>)metal_texture;
    if (!tex) return;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (t_captured_chunk < g_chunks.size()) {
            Chunk& c = g_chunks[t_captured_chunk];
            c.reserved_bytes -= (int64_t)t_captured_reserve;
            if (c.reserved_bytes < 0) c.reserved_bytes = 0;
        }
    }
    [tex release];
}

void dmn_sparse_note_tile_pool(void* pool_identity, uint64_t bytes) {
    if (!pool_identity || !bytes)
        return;
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_app_pools.find(pool_identity);
    if (it != g_app_pools.end()) {
        if (it->second >= bytes)
            return;
        g_app_pool_bytes -= it->second;
        it->second = bytes;
    } else {
        g_app_pools[pool_identity] = bytes;
    }
    g_app_pool_bytes += bytes;
    static uint64_t last_logged;
    if (g_app_pool_bytes != last_logged) {
        last_logged = g_app_pool_bytes;
        DMN_INFO("sparse: app tile pools now %llu MiB across %zu heap(s); "
                 "chunk oversubscription is bounded by that rather than by "
                 "DMN_SPARSE_OVERSUBSCRIBE",
                 (unsigned long long)(g_app_pool_bytes >> 20), g_app_pools.size());
    }
}

bool dmn_sparse_is_registered(void* identity) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_regs.count(identity) != 0;
}

extern "C" void dmn_sparse_get_stats(struct DmnSparseStats* out) {
    if (!out) return;
    DmnSparseStats st{};
    std::lock_guard<std::mutex> lk(g_mtx);
    st.chunks = (uint32_t)g_chunks.size();
    st.live = (uint32_t)g_regs.size();
    for (const Chunk& c : g_chunks) {
        st.chunk_bytes += c.bytes;
        if (c.bytes > st.largest_chunk) st.largest_chunk = c.bytes;
    }
    st.chunk_size_mb = (uint32_t)(g_chunk_bytes >> 20);
    st.reserved_bytes = total_reserved_locked();
    st.mapped_bytes = total_mapped_locked();
    st.app_pool_bytes = g_app_pool_bytes;
    st.app_pools = (uint32_t)g_app_pools.size();
    st.app_pool_aliased = g_app_pool_aliased ? 1 : 0;
    /* Both of these can move the high-water mark, so take them last. */
    st.oversubscribe = (uint32_t)oversubscribe_locked();
    st.budget_bytes = g_supported ? budget_locked() : 0;
    st.dry_events = g_dry.load();
    st.refused = g_refused.load();
    *out = st;
}

namespace {

struct Op { MTLRegion pixels; uint32_t mip, slice; bool map; };

/* Walk a region's tiles in D3D12 tile order -- x, then y, then subresource
 * for a linear count; x, y, z within a box -- calling f(x, y, sub) for each
 * until it returns false.  In the linear order the packed region is
 * `packed_tiles` wide and one high and is followed by the next slice's
 * mip 0. */
template <class F>
void for_each_tile(const SparseRec& r, const DmnSparseRegion& g, F&& f) {
    if (g.use_box) {
        for (uint32_t z = 0; z < (g.depth ? g.depth : 1); z++)
            for (uint32_t y = 0; y < g.height; y++)
                for (uint32_t x = 0; x < g.width; x++)
                    if (!f(g.x + x, g.y + y, g.subresource)) return;
        return;
    }
    uint32_t x = g.x, y = g.y, sub = g.subresource;
    for (uint32_t n = 0; n < g.num_tiles; n++) {
        if (!f(x, y, sub)) return;
        uint32_t mip = sub % r.mips;
        bool packed = mip >= r.standard_mips;
        uint32_t tw = packed ? r.packed_tiles : tiles_w(r, mip);
        uint32_t th = packed ? 1 : tiles_h(r, mip);
        if (++x >= tw) {
            x = 0;
            if (++y >= th) {
                y = 0;
                sub = packed ? (sub / r.mips + 1) * r.mips : sub + 1;
            }
        }
    }
}

/* Emit the ops for one D3D12 tile at (x,y,sub).  The packed mips are one
 * region: their first tile maps or unmaps every packed mip (from the
 * addressed one down -- D3D12 names the region by its first mip, but be
 * lenient) and their other tiles are the region's size, not separate
 * mappings.  Metal keeps its own mip tail (mips >= firstMipmapInTail) in
 * one tile that is NOT implicitly mapped on this driver
 * (tests/sparse-tiled-test: writes to an unmapped tail read back zero), so
 * the tail level is mapped explicitly, once, whenever any mip inside it is
 * addressed. */
void tile_ops(const SparseRec& r, uint32_t x, uint32_t y, uint32_t sub, bool map,
              std::vector<Op>& ops) {
    uint32_t mip = sub % r.mips, slice = sub / r.mips;
    if (slice >= r.array_size) return;
    auto tail_op = [&]() {
        uint32_t t = r.first_tail_mip < r.mips ? r.first_tail_mip : r.mips - 1;
        ops.push_back({MTLRegionMake2D(0, 0, mip_extent(r.width, t), mip_extent(r.height, t)),
                       t, slice, map});
    };
    if (mip >= r.standard_mips) {
        if (x != 0 || y != 0) return;
        for (uint32_t m = mip; m < r.mips; m++) {
            if (m >= r.first_tail_mip) { tail_op(); break; }
            ops.push_back({MTLRegionMake2D(0, 0, mip_extent(r.width, m), mip_extent(r.height, m)),
                           m, slice, map});
        }
        return;
    }
    if (mip >= r.first_tail_mip) { tail_op(); return; }
    uint64_t mw = mip_extent(r.width, mip), mh = mip_extent(r.height, mip);
    uint64_t px = (uint64_t)x * r.tile.w, py = (uint64_t)y * r.tile.h;
    if (px >= mw || py >= mh) return;
    uint64_t pw = std::min<uint64_t>(r.tile.w, mw - px), ph = std::min<uint64_t>(r.tile.h, mh - py);
    ops.push_back({MTLRegionMake2D(px, py, pw, ph), mip, slice, map});
}

/* The one line a run's log needs: everything the sizing decisions rest on. */
void log_stats_summary(dmn_log_level level) {
    DmnSparseStats st{};
    dmn_sparse_get_stats(&st);
    dmn_log_impl(level,
        "sparse stats: %u live on %u chunks (%llu MiB, largest %llu MiB of %llu budget), "
        "%llu MiB reserved / %llu MiB mapped (%.1f%% density), app pools %llu MiB in %u%s, "
        "oversubscribe %u, dry %llu, refused %llu",
        st.live, st.chunks, (unsigned long long)(st.chunk_bytes >> 20),
        (unsigned long long)(st.largest_chunk >> 20),
        (unsigned long long)(st.budget_bytes >> 20), (unsigned long long)(st.reserved_bytes >> 20),
        (unsigned long long)(st.mapped_bytes >> 20),
        st.reserved_bytes ? 100.0 * (double)st.mapped_bytes / (double)st.reserved_bytes : 0.0,
        (unsigned long long)(st.app_pool_bytes >> 20), st.app_pools,
        st.app_pool_aliased ? " (ALIASED, dropped as a bound)" : "",
        st.oversubscribe, (unsigned long long)st.dry_events, (unsigned long long)st.refused);
}

bool find_rec(void* identity, SparseRec& out) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_regs.find(identity);
    if (it == g_regs.end()) return false;
    out = it->second;
    return true;
}

} // namespace

bool dmn_sparse_query_tiling(void* identity, DmnSparseTiling* out,
                             uint32_t first, uint32_t* count, DmnSparseSubTiling* subs) {
    SparseRec r;
    if (!find_rec(identity, r)) return false;
    uint32_t per_slice = slice_tiles(r);
    if (out) {
        out->tile_w = r.tile.w; out->tile_h = r.tile.h;
        out->mips = r.mips; out->array_size = r.array_size;
        out->standard_mips = r.standard_mips;
        out->packed_mips = r.mips - r.standard_mips;
        out->packed_tiles = r.packed_tiles;
        out->packed_start_tile = mip_start_tile(r, r.mips);
        out->total_tiles = per_slice * r.array_size;
    }
    if (count) {
        uint32_t total_subs = r.mips * r.array_size;
        uint32_t n = first < total_subs ? std::min(*count, total_subs - first) : 0;
        for (uint32_t i = 0; subs && i < n; i++) {
            uint32_t sub = first + i, mip = sub % r.mips, slice = sub / r.mips;
            DmnSparseSubTiling& t = subs[i];
            if (mip < r.standard_mips) {
                t.width_in_tiles = tiles_w(r, mip);
                t.height_in_tiles = tiles_h(r, mip);
                t.start_tile = slice * per_slice + mip_start_tile(r, mip);
            } else {
                t.width_in_tiles = t.height_in_tiles = 0;
                t.start_tile = DMN_SPARSE_PACKED_TILE;
            }
        }
        *count = n;
    }
    return true;
}

bool dmn_sparse_plan_tile_copy(void* identity, const DmnSparseRegion* region,
                               void (*emit)(void* user, const DmnSparseTileCopy* c),
                               void* user, uint32_t* skipped_packed) {
    SparseRec r;
    if (!find_rec(identity, r) || !region) return false;
    uint32_t rows = r.block ? r.tile.h / 4 : r.tile.h;
    uint32_t pitch = (uint32_t)(kD3DTileBytes / rows);
    uint32_t skipped = 0;
    uint64_t k = 0;
    for_each_tile(r, *region, [&](uint32_t x, uint32_t y, uint32_t sub) {
        uint64_t off = k++ * kD3DTileBytes;
        uint32_t mip = sub % r.mips, slice = sub / r.mips;
        if (slice >= r.array_size) return false;
        if (mip >= r.standard_mips) { skipped++; return true; }
        uint64_t mw = mip_extent(r.width, mip), mh = mip_extent(r.height, mip);
        uint64_t px = (uint64_t)x * r.tile.w, py = (uint64_t)y * r.tile.h;
        if (px >= mw || py >= mh) return true;
        DmnSparseTileCopy c;
        c.mip = mip; c.slice = slice;
        c.x = (uint32_t)px; c.y = (uint32_t)py;
        c.w = (uint32_t)std::min<uint64_t>(r.tile.w, mw - px);
        c.h = (uint32_t)std::min<uint64_t>(r.tile.h, mh - py);
        c.buffer_offset = off;
        c.row_pitch = pitch;
        emit(user, &c);
        return true;
    });
    if (skipped_packed) *skipped_packed = skipped;
    return true;
}

void dmn_sparse_update_mappings(void* identity, uint32_t num_regions,
                                const DmnSparseRegion* regions,
                                uint32_t num_ranges, const uint32_t* range_flags,
                                const uint32_t* range_counts) {
    SparseRec r;
    if (!find_rec(identity, r)) return;
    DmnSparseRegion whole{};
    if (!regions) {
        /* NULL coordinates and sizes: the entire resource. */
        whole.num_tiles = slice_tiles(r) * r.array_size;
        regions = &whole;
        num_regions = 1;
    }
    std::vector<Op> ops;
    /* Walk regions tile by tile, consuming ranges in parallel. */
    uint32_t ri = 0, ri_left = 0;
    auto next_range = [&](uint32_t& flag) -> bool {
        while (ri_left == 0) {
            if (num_ranges == 0) { flag = DMN_TILE_RANGE_NONE; ri_left = UINT32_MAX; break; }
            if (ri >= num_ranges) return false;
            /* NULL counts: a single range covers every tile, several are one
             * tile each (ID3D12CommandQueue::UpdateTileMappings). */
            ri_left = range_counts ? range_counts[ri] : (num_ranges == 1 ? UINT32_MAX : 1);
            ri++;
        }
        flag = range_flags ? range_flags[ri - 1] : DMN_TILE_RANGE_NONE;
        if (num_ranges == 0) flag = DMN_TILE_RANGE_NONE;
        if (flag == DMN_TILE_RANGE_REUSE_SINGLE_TILE) {
            static std::atomic<uint64_t> seen{0};
            const uint64_t n = ++seen;
            if (n <= 4 || (n % 4096) == 0)
                DMN_ERROR("sparse: UpdateTileMappings REUSE_SINGLE_TILE on a %llux%u "
                          "reserved resource (%llu tiles so far): D3D12 points all of these "
                          "tiles at ONE pool tile so they share its contents, and Metal "
                          "sparse textures cannot alias -- each gets storage of its own and "
                          "reads its own data, so expect wrong pixels wherever the app "
                          "relied on that sharing. Tile demand also runs past the app's own "
                          "pool, which is therefore dropped as a bound on chunk "
                          "oversubscription.",
                          (unsigned long long)r.width, r.height, (unsigned long long)n);
            std::lock_guard<std::mutex> lk2(g_mtx);
            g_app_pool_aliased = true;
        }
        ri_left--;
        return true;
    };
    for (uint32_t i = 0; i < num_regions; i++) {
        uint32_t flag = DMN_TILE_RANGE_NONE;
        bool more = true;
        for_each_tile(r, regions[i], [&](uint32_t x, uint32_t y, uint32_t sub) {
            if (!next_range(flag)) { more = false; return false; }
            if (flag != DMN_TILE_RANGE_SKIP)
                tile_ops(r, x, y, sub, flag != DMN_TILE_RANGE_NULL, ops);
            return true;
        });
        if (!more) break;
    }
    g_calls++;
    if (ops.empty()) return;

    uint64_t t0 = mach_absolute_time();
    int64_t delta = 0;
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLResourceStateCommandEncoder> rse = [cb resourceStateCommandEncoder];
    for (const Op& op : ops) {
        MTLRegion tiles;
        /* Map outward (covers every Metal tile the D3D tile touches); unmap
         * inward so a neighbouring still-mapped D3D tile keeps its pages. */
        [g_dev convertSparsePixelRegions:&op.pixels toTileRegions:&tiles
                            withTileSize:r.metal_tile
                           alignmentMode:op.map ? MTLSparseTextureRegionAlignmentModeOutward
                                                : MTLSparseTextureRegionAlignmentModeInward
                              numRegions:1];
        if (tiles.size.width == 0 || tiles.size.height == 0) continue;
        [rse updateTextureMapping:r.tex
                             mode:op.map ? MTLSparseTextureMappingModeMap
                                         : MTLSparseTextureMappingModeUnmap
                           region:tiles
                         mipLevel:op.mip
                            slice:op.slice];
        int64_t b = (int64_t)(tiles.size.width * tiles.size.height * (tiles.size.depth ? tiles.size.depth : 1))
                    * (int64_t)g_metal_tile_bytes;
        delta += op.map ? b : -b;
    }
    [rse endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    uint64_t us = (mach_absolute_time() - t0) * tb.numer / tb.denom / 1000;
    g_wait_us += us;
    g_ops += ops.size();

    uint64_t used = 0, size = 0, tally = 0, reserved = 0, own = 0, over = 0; bool dry = false;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        over = oversubscribe_locked();
        auto it = g_regs.find(identity);
        if (it != g_regs.end()) it->second.mapped_bytes += delta;
        if (r.chunk < g_chunks.size()) {
            Chunk& c = g_chunks[r.chunk];
            c.mapped_bytes += delta;
            if (c.mapped_bytes < 0) c.mapped_bytes = 0;
            used = [c.heap usedSize]; size = c.bytes; tally = (uint64_t)c.mapped_bytes;
            reserved = (uint64_t)std::max<int64_t>(0, c.reserved_bytes);
            if (it != g_regs.end())
                own = (uint64_t)std::max<int64_t>(0, it->second.mapped_bytes);
            /* Our tally says more is mapped than the chunk can hold: Metal
             * has silently left tiles unmapped (they read zero). */
            dry = delta > 0 && tally > c.bytes;
        }
    }
    /* One resource wanting more tiles than a whole chunk holds is a different
     * failure from a chunk being oversubscribed, and no oversubscription
     * setting can fix it: a texture is bound to the heap it was created on and
     * cannot draw tiles from a second one, so the chunk size is a hard ceiling
     * on any single resource. Name it, because the remedy is different. */
    if (dry && size && own > size) {
        static std::atomic<uint64_t> n{0};
        const uint64_t k = ++n;
        /* Nothing can be done for THIS resource -- it is bound to its heap for
         * life -- but every later one can be spared: grow the size of the
         * chunks we open from now on.  A bigger chunk is strictly safer for
         * dryness (it only costs RAM), so unlike the oversubscription bound
         * this may be loosened on the spot. */
        uint64_t grown = 0;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            uint64_t want = own * 2;
            if (want > g_heap_max_bytes) want = g_heap_max_bytes;
            if (want > g_chunk_bytes) { g_chunk_bytes = want; grown = want; }
        }
        if (k <= 4 || (k % 256) == 0)
            DMN_ERROR("sparse: a SINGLE %llux%u reserved resource has %llu MiB of tiles "
                      "mapped, more than the %llu MiB chunk it lives on (%llu occurrences). "
                      "A resource cannot span chunks, so its excess tiles stay unmapped and "
                      "read zero whatever DMN_SPARSE_OVERSUBSCRIBE is set to. This resource "
                      "cannot be helped; %s",
                      (unsigned long long)r.width, r.height,
                      (unsigned long long)(own >> 20), (unsigned long long)(size >> 20),
                      (unsigned long long)k,
                      grown ? "later chunks are now larger (see the next 'opened chunk' line)"
                            : "raise DMN_SPARSE_HEAP_MB (and DMN_SPARSE_HEAP_MAX_MB with it)");
        if (grown && k <= 4)
            DMN_WARN("sparse: chunk size raised to %llu MiB so later reserved resources of "
                     "this size fit", (unsigned long long)(grown >> 20));
    }
    if (dry) {
        uint64_t k = ++g_dry;
        /* A chunk carrying V bytes of virtual texture at tile density d must
         * hold V*d bytes of tiles, so 1/d is the bound DMN_SPARSE_OVERSUBSCRIBE
         * has to respect. Report both: the number is the whole answer. */
        const double density = reserved ? (double)tally / (double)reserved : 0.0;
        if (k <= 8 || (k % 256) == 0)
            DMN_WARN("sparse: chunk %zu is DRY (tally %llu MiB > %llu MiB): tiles of a %llux%u "
                     "reserved resource stay unmapped and read zero (%llu occurrences). It "
                     "carries %llu MiB of virtual texture at %.1f%% density, so "
                     "DMN_SPARSE_OVERSUBSCRIBE must be at most %llu (it is %llu); lowering it "
                     "spreads textures over more chunks, at one chunk of RAM each.",
                     r.chunk, (unsigned long long)(tally >> 20),
                     (unsigned long long)(size >> 20), (unsigned long long)r.width, r.height,
                     (unsigned long long)k, (unsigned long long)(reserved >> 20),
                     density * 100.0,
                     (unsigned long long)(density > 0.0 ? (uint64_t)(1.0 / density) : 0),
                     (unsigned long long)over);
    }
    uint64_t k = g_calls.load();
    const uint32_t every = stats_every();
    if (every && (k % every) == 0)
        log_stats_summary(DMN_LOG_WARN);
    if (k <= 8 || (k % 1024) == 0) {
        /* usedSize is decorative: Metal documents it as "the size of all
         * resources allocated from the heap" and says nothing about what that
         * means for a sparse heap, where a placed texture costs no bytes until
         * its tiles are mapped.  Our own tally is the number to trust. */
        DMN_INFO("sparse: UpdateTileMappings #%llu: %zu tile ops (%s%lld KiB) on %llux%u fmt=%u; chunk %zu "
                 "usedSize %llu MiB, tally %llu / %llu MiB; cumulative ops=%llu wait=%llu ms",
                 (unsigned long long)k, ops.size(), delta < 0 ? "-" : "+",
                 (long long)((delta < 0 ? -delta : delta) >> 10), (unsigned long long)r.width, r.height,
                 r.format, r.chunk, (unsigned long long)(used >> 20), (unsigned long long)(tally >> 20),
                 (unsigned long long)(size >> 20), (unsigned long long)g_ops.load(),
                 (unsigned long long)(g_wait_us.load() / 1000));
        if (!every)
            log_stats_summary(DMN_LOG_INFO);
    }
}
