/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic probe (not a test): how much RAM does the machine actually HOLD
 * for committed DEFAULT textures allocated through D3DMetal, as a function of
 * how much of them the GPU touches?  Answers "is a fully-backed committed
 * texture demand-paged, or does the backing become resident wholesale?"
 *
 *   resident-probe [n=64] [tiles_written=4] [bc=1] [hold_sec=0] [read=0] [ntex_written=n] [reserved=0]
 *
 * reserved=1 creates the textures with CreateReservedResource instead (the
 * sparse-backed path of dmn_sparse) and maps `tiles_written` tiles of each
 * through UpdateTileMappings before writing them -- the A/B for what a
 * reserved texture costs under each backing.
 *
 * Reports host-wide wired/active/compressed page deltas (host_statistics64,
 * the counters `vm_stat` prints), the task phys_footprint and Metal's
 * currentAllocatedSize after: create, GPU write of `tiles_written` 64 KiB
 * tiles per texture, (optional) GPU read of every texture, and release.
 * `hold_sec` pauses after the write so `vmmap`/`footprint` can be run on the
 * pid from outside.
 */
#include "common/dx12.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <mach/mach.h>

extern "C" unsigned long long dmn_test_metal_allocated_size(void);
extern "C" unsigned long long dmn_test_phys_footprint(void);
static double mib(unsigned long long b) { return b / 1048576.0; }

struct HostPages { uint64_t wired, active, inactive, compressed, speculative, free; };
static HostPages hostpages(void) {
    vm_statistics64_data_t vs; mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
    HostPages h{};
    kern_return_t kr = host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &c);
    if (kr != KERN_SUCCESS) { fprintf(stderr, "host_statistics64: %d\n", kr); return h; }
    h.wired = vs.wire_count; h.active = vs.active_count; h.inactive = vs.inactive_count;
    h.compressed = vs.compressor_page_count; h.speculative = vs.speculative_count; h.free = vs.free_count;
    return h;
}
static uint64_t footprint(void) {
    task_vm_info_data_t ti; mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&ti, &c);
    return ti.phys_footprint;
}
static HostPages g_base;
static void report(const char* tag) {
    HostPages h = hostpages();
    const double pg = 16384.0 / 1048576.0;
    printf("%-44s metal=%7.0f MiB  footprint=%7.0f MiB  | host delta: wired %+7.0f  active %+7.0f  inactive %+7.0f  compressed %+7.0f  (MiB)\n",
           tag, mib(dmn_test_metal_allocated_size()), mib(footprint()),
           ((double)h.wired - (double)g_base.wired) * pg,
           ((double)h.active - (double)g_base.active) * pg,
           ((double)h.inactive - (double)g_base.inactive) * pg,
           ((double)h.compressed - (double)g_base.compressed) * pg);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 64;
    int tiles = argc > 2 ? atoi(argv[2]) : 4;
    bool bc = argc > 3 ? atoi(argv[3]) != 0 : true;
    int hold = argc > 4 ? atoi(argv[4]) : 0;
    bool doread = argc > 5 ? atoi(argv[5]) != 0 : false;
    int nwrite = argc > 6 ? atoi(argv[6]) : n;
    bool reserved = argc > 7 ? atoi(argv[7]) != 0 : false;
    Com<ID3D12Device> dev; Com<ID3D12CommandQueue> q;
    if (FAILED(make_d3d12_device(dev)) || FAILED(make_d3d12_queue(dev.ptr(), q))) { printf("no device\n"); return 1; }
    printf("pid %d\n", getpid());
    g_base = hostpages();
    report("baseline");

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = 4096; rd.Height = 4096;
    rd.DepthOrArraySize = 1; rd.MipLevels = 11;
    rd.Format = bc ? DXGI_FORMAT_BC7_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    D3D12_RESOURCE_ALLOCATION_INFO ai = dev->GetResourceAllocationInfo(0, 1, &rd);
    printf("each 4096^2 %s 11-mip committed texture: D3D12 says %.1f MiB (x%d = %.0f MiB)\n",
           bc ? "BC7" : "RGBA8", mib(ai.SizeInBytes), n, mib(ai.SizeInBytes) * n);

    std::vector<ID3D12Resource*> keep;
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    if (reserved) rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    for (int i = 0; i < n; i++) {
        ID3D12Resource* r = nullptr;
        HRESULT hr = reserved
            ? dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), (void**)&r)
            : dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                           nullptr, __uuidof(ID3D12Resource), (void**)&r);
        if (FAILED(hr) || !r) { printf("create %d failed 0x%08x\n", i, (unsigned)hr); break; }
        keep.push_back(r);
    }
    char tag[128];
    snprintf(tag, sizeof tag, "created %zu (%.0f MiB nominal)", keep.size(), mib(ai.SizeInBytes) * keep.size());
    report(tag);

    D3D12_HEAP_PROPERTIES uhp{}; uhp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC ub{}; ub.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ub.Width = 1u << 20; ub.Height = 1; ub.DepthOrArraySize = 1; ub.MipLevels = 1;
    ub.SampleDesc.Count = 1; ub.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* up = nullptr;
    dev->CreateCommittedResource(&uhp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_GENERIC_READ,
                                 nullptr, __uuidof(ID3D12Resource), (void**)&up);
    void* m = nullptr; D3D12_RANGE nr{0, 0};
    if (up && SUCCEEDED(up->Map(0, &nr, &m)) && m) { memset(m, 0xA5, 1u << 20); up->Unmap(0, nullptr); }
    ID3D12CommandAllocator* al = nullptr; ID3D12GraphicsCommandList* cl = nullptr; ID3D12Fence* fe = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&al);
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&cl);
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fe);
    UINT64 fv = 0;
    auto submit = [&]() {
        cl->Close();
        ID3D12CommandList* l = cl; q->ExecuteCommandLists(1, &l);
        q->Signal(fe, ++fv);
        for (int i = 0; i < 20000 && fe->GetCompletedValue() < fv; i++) usleep(1000);
        al->Reset(); cl->Reset(al, nullptr);
    };
    report("upload buffer + command list ready");

    if (reserved && tiles > 0) {
        /* Map the tiles that will be written: one linear run of `tiles` tiles at mip 0. */
        typedef void (STDMETHODCALLTYPE *UTM)(ID3D12CommandQueue*, ID3D12Resource*, UINT,
            const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*,
            UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS);
        void** vt = *reinterpret_cast<void***>(q.ptr());
        UTM utm = reinterpret_cast<UTM>(vt[8]);
        for (size_t i = 0; i < keep.size() && (int)i < nwrite; i++) {
            D3D12_TILED_RESOURCE_COORDINATE c{0, 0, 0, 0};
            D3D12_TILE_REGION_SIZE s{}; s.NumTiles = (UINT)tiles;
            D3D12_TILE_RANGE_FLAGS f = D3D12_TILE_RANGE_FLAG_NONE; UINT off = 0, cnt = (UINT)tiles;
            utm(q.ptr(), keep[i], 1, &c, &s, nullptr, 1, &f, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
        }
        report("tiles mapped (UpdateTileMappings)");
    }
    if (tiles > 0) {
        /* Write `tiles` 64 KiB tiles into mip 0 of every texture (256x256 BC7 / 128x128 RGBA8 per tile). */
        for (size_t i = 0; i < keep.size() && (int)i < nwrite; i++) {
            for (int t = 0; t < tiles; t++) {
                D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = up;
                src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint.Footprint.Format = rd.Format;
                src.PlacedFootprint.Footprint.Width = bc ? 256 : 128; src.PlacedFootprint.Footprint.Height = bc ? 256 : 128;
                src.PlacedFootprint.Footprint.Depth = 1;
                src.PlacedFootprint.Footprint.RowPitch = bc ? 4096 : 512;
                D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = keep[i];
                dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
                UINT tx = (UINT)(t % 16), ty = (UINT)(t / 16);
                cl->CopyTextureRegion(&dst, tx * (bc ? 256 : 128), ty * (bc ? 256 : 128), 0, &src, nullptr);
            }
            if ((i % 8) == 7) submit();
        }
        submit();
        snprintf(tag, sizeof tag, "GPU wrote %d tiles into %d/%zu tex (%.1f MiB of pixels) removed=0x%08x", tiles, nwrite, keep.size(),
                 (double)nwrite * tiles * 65536 / 1048576.0, (unsigned)dev->GetDeviceRemovedReason());
        report(tag);
    }
    if (doread) {
        /* GPU reads one tile of each texture back into a buffer (touch by read, not write). */
        D3D12_HEAP_PROPERTIES rhp{}; rhp.Type = D3D12_HEAP_TYPE_READBACK;
        ID3D12Resource* rb = nullptr;
        dev->CreateCommittedResource(&rhp, D3D12_HEAP_FLAG_NONE, &ub, D3D12_RESOURCE_STATE_COPY_DEST,
                                     nullptr, __uuidof(ID3D12Resource), (void**)&rb);
        for (size_t i = 0; i < keep.size(); i++) {
            D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = keep[i];
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 5; /* untouched small mip */
            D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = rb;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Footprint.Format = rd.Format;
            dst.PlacedFootprint.Footprint.Width = 128; dst.PlacedFootprint.Footprint.Height = 128;
            dst.PlacedFootprint.Footprint.Depth = 1; dst.PlacedFootprint.Footprint.RowPitch = bc ? 2048 : 512;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            if ((i % 8) == 7) submit();
        }
        submit();
        snprintf(tag, sizeof tag, "GPU read mip5 of every tex removed=0x%08x", (unsigned)dev->GetDeviceRemovedReason());
        report(tag);
        if (rb) rb->Release();
    }
    if (hold > 0) {
        printf("HOLDING %d s (pid %d) -- run vmmap/footprint/vm_stat now\n", hold, getpid()); fflush(stdout);
        sleep((unsigned)hold);
        report("after hold");
    }
    for (auto* r : keep) r->Release();
    keep.clear();
    report("all textures released");
    if (hold > 0) { sleep(3); report("3 s after release"); }
    if (up) up->Release(); if (cl) cl->Release(); if (al) al->Release(); if (fe) fe->Release();
    return 0;
}
