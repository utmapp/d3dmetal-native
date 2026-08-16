/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Sparse backing for D3D12 reserved (tiled) resources.
 *
 * D3DMetal has no tiled-resource support (TiledResourcesTier 0: its
 * CreateReservedResource makes an unbacked object and UpdateTileMappings is
 * a no-op), so the COM hooks turn CreateReservedResource into an ordinary
 * committed create.  Without this module that committed create is FULLY
 * BACKED: measured on this driver (tests/resident-probe, 8 GiB scale, host
 * page balance), every Metal allocation D3DMetal makes is made resident
 * wholesale at the next GPU submission -- whether or not anything touches
 * it -- and D3DMetal's sub-allocator pools never shrink.  A virtual-texturing
 * game that reserves 12-14 GiB and maps a few hundred MiB of tiles therefore
 * costs 12-14 GiB of RAM.
 *
 * This module substitutes the committed create's Metal texture with one
 * placed on an MTLHeapTypeSparse heap and translates UpdateTileMappings into
 * Metal tile mappings, so RAM is the sparse heap(s) -- the app's tile-pool
 * budget, exactly what real hardware pays -- instead of the textures' virtual
 * size.  A sparse heap commits its whole size at first use (measured), so the
 * heaps are created in chunks on demand rather than one big guess:
 *
 *   DMN_SPARSE=0             disable (reserved resources fully backed again)
 *   DMN_SPARSE_HEAP_MB=<n>   chunk size of each sparse heap (default 1024)
 *   DMN_SPARSE_MAX_MB=<n>    a HARD cap on the total across chunks; unset, the
 *                            budget starts at 6144 and the app's own tile
 *                            pools may raise it (see below)
 *   DMN_SPARSE_HEAP_MAX_MB=<n>  largest single chunk (default 4096), for a
 *                            resource that needs more than one ordinary chunk
 *   DMN_SPARSE_OVERSUBSCRIBE=<n>  virtual texture a chunk may carry, as a
 *                            multiple of its size (default 8)
 *   DMN_SPARSE_BUDGET_FLOOR_MB=<n>  the budget's starting point when no hard
 *                            cap is set (default 6144)
 *   DMN_SPARSE_STATS=<n>     log the sizing summary every n UpdateTileMappings
 *                            calls at WARN (it goes out at INFO regardless)
 *
 * A texture is bound to the sparse heap it was created on (Metal maps a
 * texture's tiles only from its own heap) and its tiles arrive much later, so
 * creation is the only moment placement can act and the only predictor it has
 * is the texture's own virtual footprint.  Each chunk therefore carries at
 * most DMN_SPARSE_OVERSUBSCRIBE times its size in virtual texture, and a new
 * chunk is opened when no existing one has room by that measure (a chunk over
 * 75% mapped is passed over too -- a watermark that says nothing during a
 * level load, which is a burst of creates with nothing mapped yet, hence the
 * reservation test alongside it).  Oversubscription is the whole point of a
 * reserved resource; the bound on it is 1 / (tile density), since a chunk
 * carrying V bytes of virtual texture at density d must hold V*d bytes of
 * tiles.  A virtual-texturing title measured at 19854 MiB reserved against
 * 2234 MiB mapped (11.3%) can therefore carry at most ~8.8x a chunk's size
 * on it, which is where the default comes from.
 *
 * The two numbers the app itself tells us, and the opposite directions they
 * may move (get this backwards and tiles silently read zero):
 *
 *   - Its tile pools bound how many tiles it can EVER map, so their total may
 *     RAISE the budget -- a fixed ceiling otherwise sends a large-pool app's
 *     textures to the fully-backed path at their whole virtual size.  The
 *     total only grows and starts near nothing, so it can only ever prove "at
 *     least this much"; being wrong high costs nothing until a chunk is really
 *     demanded.  See budget_locked().
 *   - Density estimates may only TIGHTEN the oversubscription bound, never
 *     loosen it, for the mirror-image reason: reading the app as sparser than
 *     it is lets a chunk carry more texture than its tiles will fit.  See
 *     oversubscribe_locked().
 *
 * If a texture's own chunk still runs dry, a map silently yields unmapped
 * tiles (reads zero) -- Metal's documented behaviour, no fault -- and the
 * module logs it with the knob to turn.  Unmapped tiles read zero as D3D12
 * tier 2 requires.
 *
 * With it, dmn_com_hooks.cpp serves the tier-2 surface for these resources:
 * CreateReservedResource (all three device variants), UpdateTileMappings,
 * GetResourceTiling (from the tile model below), CopyTiles (as per-tile
 * texture copies), and CheckFeatureSupport reports TiledResourcesTier 2.
 * Two things Metal cannot express are logged instead: CopyTileMappings and
 * the REUSE_SINGLE_TILE range flag both need pages shared between tiles.
 *
 * Same TU split as dmn_share.h: this header carries plain-C entry points and
 * PODs only, so dmn_com_hooks.cpp (DirectX headers) and dmn_sparse.mm (Metal)
 * never include each other's world.
 */

#pragma once

#include <cstdint>

/* One UpdateTileMappings region in D3D12 tile units (mirrors
 * D3D12_TILED_RESOURCE_COORDINATE + D3D12_TILE_REGION_SIZE). */
struct DmnSparseRegion {
    uint32_t x, y, z, subresource;
    uint32_t use_box;
    uint32_t num_tiles, width, height, depth;
};

/* D3D12_TILE_RANGE_FLAGS values. */
enum { DMN_TILE_RANGE_NONE = 0, DMN_TILE_RANGE_NULL = 1,
       DMN_TILE_RANGE_SKIP = 2, DMN_TILE_RANGE_REUSE_SINGLE_TILE = 4 };

/* Whether sparse backing is available at all (device supports sparse
 * textures and DMN_SPARSE=0 has not disabled it). */
bool dmn_sparse_enabled(void);

/* Arm the calling thread: the next Metal texture creation D3DMetal performs
 * (device or heap path) is allocated from a sparse heap instead.  disarm
 * returns whether a texture was captured; the capture is +1 retained and
 * handed to the caller. */
void dmn_sparse_arm(void);
bool dmn_sparse_disarm(void** out_metal_texture);

/* Registry keyed by the D3D12 resource's COM identity. `dimension` is the
 * D3D12_RESOURCE_DIMENSION value, `dxgi_format` the DXGI_FORMAT value; the
 * D3D12 tile shape and packed-mip layout are derived from them exactly as
 * the runtime does for tiled tier 2. Takes ownership of the +1 texture. */
void dmn_sparse_register(void* identity, void* metal_texture, uint32_t dimension,
                         uint32_t dxgi_format, uint64_t width, uint32_t height,
                         uint16_t depth_or_array, uint16_t mips);
void dmn_sparse_unregister(void* identity);
/* Drop a captured (+1) texture that will not be registered. */
void dmn_sparse_release_texture(void* metal_texture);
bool dmn_sparse_is_registered(void* identity);

/* A snapshot of the module's sizing state: what the tests assert on, and
 * what a run's log needs to say about the module.  Byte counts are bytes. */
struct DmnSparseStats {
    uint32_t chunks;           /* sparse heaps open */
    uint32_t live;             /* reserved resources registered */
    uint64_t chunk_bytes;      /* total across chunks -- the module's RAM cost */
    uint64_t largest_chunk;    /* biggest single chunk (the per-resource ceiling) */
    uint64_t budget_bytes;     /* what the chunks may total right now */
    uint64_t reserved_bytes;   /* virtual texture placed on chunks */
    uint64_t mapped_bytes;     /* our tally of tiles mapped (Metal has no query) */
    uint64_t app_pool_bytes;   /* tile pools the app has named so far */
    uint32_t app_pools;        /* how many */
    uint32_t app_pool_aliased; /* it used REUSE_SINGLE_TILE; pool no longer bounds us */
    uint32_t oversubscribe;    /* virtual-per-chunk multiple in force */
    uint32_t chunk_size_mb;    /* size of the next ordinary chunk */
    uint64_t dry_events;       /* mappings that found the chunk full (tiles read zero) */
    uint64_t refused;          /* chunks not opened because the budget was reached */
};
/* extern "C" so it survives src/exports.symbols' `_dmn_*` rule: the tests
 * link the dylib and call it from outside. */
extern "C" void dmn_sparse_get_stats(struct DmnSparseStats* out);

/* Report the size of a heap the app is actually using as a D3D12 tile pool
 * (learned when it names one in UpdateTileMappings).  The app cannot map more
 * tiles than the pools it owns, so the total of these is an authoritative
 * ceiling on tile demand -- far better than inferring one from how much
 * virtual texture was reserved.  Idempotent per heap: pass a stable
 * identity. */
void dmn_sparse_note_tile_pool(void* pool_identity, uint64_t bytes);

/* Apply an UpdateTileMappings call to the registered resource, with the
 * documented NULL defaults: `regions` NULL is the one region covering the
 * whole resource; `range_flags` NULL is all NONE; `range_counts` NULL is
 * "every tile" for a single range and one tile per range otherwise.
 * REUSE_SINGLE_TILE is honoured as a mapping (the tiles get pages) but not
 * as aliasing -- Metal has no shared-page mapping within a sparse texture,
 * so the tiles do not share content; logged as an error, with a running
 * count. */
void dmn_sparse_update_mappings(void* identity, uint32_t num_regions,
                                const DmnSparseRegion* regions,
                                uint32_t num_ranges, const uint32_t* range_flags,
                                const uint32_t* range_counts);

/* GetResourceTiling for a registered resource, in the D3D12 tile model this
 * module implements: every mip at least a tile in both dimensions is a
 * standard mip; the rest are the packed mips, which occupy `packed_tiles`
 * tiles per array slice (sized from their bytes, at least one), addressed by
 * X in the first packed mip's subresource, and mapped or unmapped as a unit
 * through their first tile.  Tile order is x, then y, then mip, then slice
 * (the packed tiles follow the slice's last standard mip). */
struct DmnSparseTiling {
    uint32_t tile_w, tile_h;      /* standard tile shape, texels */
    uint32_t mips, array_size;
    uint32_t standard_mips, packed_mips;
    uint32_t packed_tiles;        /* per slice; 0 when nothing is packed */
    uint32_t packed_start_tile;   /* slice 0's first packed tile */
    uint32_t total_tiles;
};
/* One D3D12_SUBRESOURCE_TILING; a packed subresource reports 0 x 0 tiles and
 * DMN_SPARSE_PACKED_TILE (D3D12_PACKED_TILE) as its start. */
struct DmnSparseSubTiling { uint32_t width_in_tiles, height_in_tiles, start_tile; };
enum : uint32_t { DMN_SPARSE_PACKED_TILE = 0xffffffffu };
/* Fills *out and, when `subs` is non-NULL, up to *count entries from
 * subresource `first` (clamped; *count becomes the number written).  Returns
 * false when the identity is not registered. */
bool dmn_sparse_query_tiling(void* identity, DmnSparseTiling* out,
                             uint32_t first, uint32_t* count, DmnSparseSubTiling* subs);

/* One CopyTiles tile as a texture copy: `x`,`y`,`w`,`h` in texels on the
 * subresource (a partial edge tile is clipped), the tile's `buffer_offset`
 * relative to the call's start, and `row_pitch` -- the full tile row, since
 * CopyTiles lays each tile out linearly at its full width whether or not it
 * is partially filled. */
struct DmnSparseTileCopy {
    uint32_t mip, slice, x, y, w, h;
    uint64_t buffer_offset;
    uint32_t row_pitch;
};
/* Enumerate a CopyTiles region's standard tiles in tile order.  Packed tiles
 * are skipped and counted in *skipped_packed -- CopyTiles on them is
 * undefined by the D3D12 spec, which sends apps to CopyTextureRegion for
 * those mips.  Returns false when the identity is not registered. */
bool dmn_sparse_plan_tile_copy(void* identity, const DmnSparseRegion* region,
                               void (*emit)(void* user, const DmnSparseTileCopy* c),
                               void* user, uint32_t* skipped_packed);
