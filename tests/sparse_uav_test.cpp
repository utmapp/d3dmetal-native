/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Shader writes into a reserved texture through a UAV.  A 256x256 R32_UINT
 * reserved texture with ALLOW_UNORDERED_ACCESS, all four 128x128 tiles
 * mapped, is filled by a compute kernel -- once by plain stores, once by
 * InterlockedMax onto a cleared texture (the virtual-shadow-map pattern) --
 * and read back texel by texel.  A committed texture runs the same kernels
 * as the control.
 *
 * Metal does not support MTLTextureUsageShaderWrite on a texture placed in a
 * MTLHeapTypeSparse heap on Apple GPUs, so such a texture must not get the
 * sparse backing (dmn_sparse.mm).  Subnautica 2 rendered garbage, and hung
 * or page-faulted the GPU, while it did.
 */
#include "common/dx12.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <utility>

#include "shaders/uavtexsh/atomic.h"
#include "shaders/uavtexsh/store.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static const UINT kSize = 256;

typedef void (STDMETHODCALLTYPE *UTM)(ID3D12CommandQueue*, ID3D12Resource*, UINT,
    const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*,
    UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS);

struct Ctx {
    Com<ID3D12Device> dev; Com<ID3D12CommandQueue> q;
    ID3D12CommandAllocator* alloc = nullptr; ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr; UINT64 fv = 0;
    ID3D12RootSignature* rs = nullptr;
    ID3D12PipelineState* store = nullptr; ID3D12PipelineState* atomic = nullptr;
    ID3D12DescriptorHeap* gpu_heap = nullptr; ID3D12DescriptorHeap* cpu_heap = nullptr;
    ID3D12Resource* rb = nullptr;
    bool init();
    void begin() { alloc->Reset(); list->Reset(alloc, nullptr); }
    void submit_wait() {
        list->Close();
        ID3D12CommandList* l = list; q->ExecuteCommandLists(1, &l);
        q->Signal(fence, ++fv);
        for (int i = 0; i < 5000 && fence->GetCompletedValue() < fv; i++) usleep(1000);
        if (fence->GetCompletedValue() < fv) { fails++; printf("FAIL: GPU wait timed out\n"); }
    }
};

static ID3D12PipelineState* make_pso(ID3D12Device* dev, ID3D12RootSignature* rs,
                                     const unsigned char* code, size_t len) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd{};
    cd.pRootSignature = rs; cd.CS.pShaderBytecode = code; cd.CS.BytecodeLength = len;
    ID3D12PipelineState* pso = nullptr;
    dev->CreateComputePipelineState(&cd, __uuidof(ID3D12PipelineState), (void**)&pso);
    return pso;
}

bool Ctx::init() {
    if (FAILED(make_d3d12_device(dev)) || FAILED(make_d3d12_queue(dev.ptr(), q))) return false;
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&alloc))) return false;
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&list))) return false;
    list->Close();
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence))) return false;

    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER rp{}; rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 1; rp.DescriptorTable.pDescriptorRanges = &range;
    D3D12_ROOT_SIGNATURE_DESC rsd{1, &rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) return false;
    HRESULT hr = dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          __uuidof(ID3D12RootSignature), (void**)&rs);
    blob->Release();
    if (FAILED(hr)) return false;
    store = make_pso(dev.ptr(), rs, kUavTexStore, sizeof(kUavTexStore));
    atomic = make_pso(dev.ptr(), rs, kUavTexAtomic, sizeof(kUavTexAtomic));
    if (!store || !atomic) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&gpu_heap))) return false;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&cpu_heap))) return false;

    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = kSize * kSize * 4;
    bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, __uuidof(ID3D12Resource), (void**)&rb));
}

static D3D12_RESOURCE_DESC tex_desc() {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = kSize; rd.Height = kSize;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = DXGI_FORMAT_R32_UINT;
    rd.SampleDesc.Count = 1; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return rd;
}

/* Run `pso` over the whole texture (cleared first when `clear`), copy it to
 * the readback buffer and count texels that differ from the kernel's value. */
static unsigned fill_and_check(Ctx& c, ID3D12Resource* tex, ID3D12PipelineState* pso, bool clear,
                               uint32_t* first_bad, unsigned* first_at) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.Format = DXGI_FORMAT_R32_UINT;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    c.dev->CreateUnorderedAccessView(tex, nullptr, &ud, c.gpu_heap->GetCPUDescriptorHandleForHeapStart());
    c.dev->CreateUnorderedAccessView(tex, nullptr, &ud, c.cpu_heap->GetCPUDescriptorHandleForHeapStart());

    c.begin();
    c.list->SetDescriptorHeaps(1, &c.gpu_heap);
    if (clear) {
        const UINT zero[4] = {0, 0, 0, 0};
        c.list->ClearUnorderedAccessViewUint(c.gpu_heap->GetGPUDescriptorHandleForHeapStart(),
                                             c.cpu_heap->GetCPUDescriptorHandleForHeapStart(), tex, zero, 0, nullptr);
        D3D12_RESOURCE_BARRIER u{}; u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; u.UAV.pResource = tex;
        c.list->ResourceBarrier(1, &u);
    }
    c.list->SetPipelineState(pso);
    c.list->SetComputeRootSignature(c.rs);
    c.list->SetComputeRootDescriptorTable(0, c.gpu_heap->GetGPUDescriptorHandleForHeapStart());
    c.list->Dispatch(kSize / 8, kSize / 8, 1);
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = tex;
    b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    c.list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = tex; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = c.rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_UINT, kSize, kSize, 1, kSize * 4};
    c.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    c.list->ResourceBarrier(1, &b);
    c.submit_wait();

    void* p = nullptr; D3D12_RANGE rr{0, kSize * kSize * 4};
    if (FAILED(c.rb->Map(0, &rr, &p)) || !p) return kSize * kSize;
    const uint32_t* px = (const uint32_t*)p;
    unsigned bad = 0;
    for (unsigned y = 0; y < kSize; y++)
        for (unsigned x = 0; x < kSize; x++) {
            uint32_t want = (x << 16) | y | 0x80000000u, got = px[y * kSize + x];
            if (got != want) { if (!bad) { *first_bad = got; *first_at = y * kSize + x; } bad++; }
        }
    D3D12_RANGE wr{0, 0}; c.rb->Unmap(0, &wr);
    return bad;
}

static void run_kernels(Ctx& c, ID3D12Resource* tex, const char* what) {
    uint32_t got = 0; unsigned at = 0;
    unsigned bad = fill_and_check(c, tex, c.store, false, &got, &at);
    CHECK(bad == 0, "%s: UAV stores land (%u/%u wrong; first texel %u = 0x%08x)", what, bad, kSize * kSize, at, got);
    got = 0; at = 0;
    bad = fill_and_check(c, tex, c.atomic, true, &got, &at);
    CHECK(bad == 0, "%s: UAV InterlockedMax lands (%u/%u wrong; first texel %u = 0x%08x)", what, bad, kSize * kSize, at, got);
}

int main() {
    /* Pin the sparse module on: an environment that disabled it would turn
     * this into a second run of the committed control. */
    setenv("DMN_SPARSE", "1", 1);
    Ctx c;
    if (!c.init()) { printf("no device\n"); return 1; }

    D3D12_RESOURCE_DESC rd = tex_desc();
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* committed = nullptr;
    HRESULT hr = c.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                nullptr, __uuidof(ID3D12Resource), (void**)&committed);
    CHECK(SUCCEEDED(hr) && committed, "committed %ux%u R32_UINT UAV texture hr=0x%08x", kSize, kSize, (unsigned)hr);
    if (committed) run_kernels(c, committed, "committed (control)");

    rd.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    ID3D12Resource* reserved = nullptr;
    hr = c.dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                       __uuidof(ID3D12Resource), (void**)&reserved);
    CHECK(SUCCEEDED(hr) && reserved, "reserved %ux%u R32_UINT UAV texture hr=0x%08x", kSize, kSize, (unsigned)hr);
    if (!reserved) return 1;

    /* Map all four 128x128 tiles. */
    D3D12_HEAP_DESC hd{}; hd.SizeInBytes = 1ull << 20; hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    ID3D12Heap* pool = nullptr;
    c.dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&pool);
    void** vt = *reinterpret_cast<void***>(c.q.ptr());
    UTM utm = reinterpret_cast<UTM>(vt[8]);
    D3D12_TILED_RESOURCE_COORDINATE co{0, 0, 0, 0};
    D3D12_TILE_REGION_SIZE sz{}; sz.NumTiles = 4; sz.UseBox = TRUE; sz.Width = 2; sz.Height = 2; sz.Depth = 1;
    D3D12_TILE_RANGE_FLAGS fl = D3D12_TILE_RANGE_FLAG_NONE; UINT off = 0, cnt = 4;
    utm(c.q.ptr(), reserved, 1, &co, &sz, pool, 1, &fl, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);

    run_kernels(c, reserved, "reserved");
    CHECK(c.dev->GetDeviceRemovedReason() == S_OK, "device not removed (0x%08x)",
          (unsigned)c.dev->GetDeviceRemovedReason());

    printf(fails ? "SPARSE-UAV: %d FAILED\n" : "SPARSE-UAV: PASS\n", fails);
    return fails ? 1 : 0;
}
