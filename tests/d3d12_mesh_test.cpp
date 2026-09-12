/*
 * Mesh-shader pipeline through the shim, the way the guest UMD builds it:
 * an amplification-less mesh PSO packed as a pipeline-state stream
 * (ID3D12Device2::CreatePipelineState) and launched with DispatchMesh,
 * drawing one full-screen green triangle into an 8x8 RGBA8 target that is
 * read back.  The stream layout is the CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT
 * one: each record pointer-aligned, its payload at the payload's alignment.
 *
 * The bundled mingw headers stop at ID3D12Device1 / GraphicsCommandList2, so
 * the two newer methods are reached through their vtable slots.
 *
 * Prints "MESH12: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <time.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "MESH12"
#include "common/check.h"

#include "d3d12_mesh_shaders.h"

namespace {

/* SDK 10.0.19041 additions the bundled d3d12.h lacks. */
struct PIPELINE_STATE_STREAM_DESC { SIZE_T SizeInBytes; void* pPipelineStateSubobjectStream; };
enum : UINT {
    SUBOBJ_ROOT_SIGNATURE = 0, SUBOBJ_PS = 2, SUBOBJ_BLEND = 8, SUBOBJ_SAMPLE_MASK = 9,
    SUBOBJ_RASTERIZER = 10, SUBOBJ_DEPTH_STENCIL = 11, SUBOBJ_PRIMITIVE_TOPOLOGY = 14,
    SUBOBJ_RENDER_TARGET_FORMATS = 15, SUBOBJ_DEPTH_STENCIL_FORMAT = 16,
    SUBOBJ_SAMPLE_DESC = 17, SUBOBJ_NODE_MASK = 18, SUBOBJ_AS = 24, SUBOBJ_MS = 25,
};
struct RT_FORMAT_ARRAY { DXGI_FORMAT RTFormats[8]; UINT NumRenderTargets; };
const GUID kIID_Device2 = {0x30baa41e, 0xb15b, 0x475c, {0xa0, 0xbb, 0x1a, 0xf5, 0xc5, 0xb6, 0x43, 0x28}};
const GUID kIID_List6 = {0xc3827890, 0xe548, 0x4cfa, {0x96, 0xcf, 0x56, 0x89, 0xa9, 0x37, 0x0f, 0x80}};
const unsigned kSlot_Device2_CreatePipelineState = 47;
const unsigned kSlot_List6_DispatchMesh = 79;
typedef HRESULT (STDMETHODCALLTYPE* PFN_CreatePipelineState)(void*, const PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
typedef void (STDMETHODCALLTYPE* PFN_DispatchMesh)(void*, UINT, UINT, UINT);

template <class T> T vslot(void* obj, unsigned slot) {
    return reinterpret_cast<T>((*reinterpret_cast<void***>(obj))[slot]);
}

size_t alignUp(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

void put(unsigned char* buf, size_t* off, UINT type, const void* payload, size_t size, size_t align) {
    size_t at = alignUp(*off, sizeof(void*));
    memcpy(buf + at, &type, sizeof(type));
    at = alignUp(at + sizeof(type), align);
    memcpy(buf + at, payload, size);
    *off = alignUp(at + size, sizeof(void*));
}
#define PUT(tag, ptr, T) put(buf, &off, SUBOBJ_##tag, (ptr), sizeof(T), alignof(T))

ID3D12Device* g_dev;
ID3D12CommandQueue* g_q;
ID3D12CommandAllocator* g_alloc;
ID3D12GraphicsCommandList* g_cl;
ID3D12Fence* g_fence;
UINT64 g_fenceVal;

bool execAndWait() {
    if (FAILED(g_cl->Close())) return false;
    ID3D12CommandList* lists[] = {g_cl};
    g_q->ExecuteCommandLists(1, lists);
    if (FAILED(g_q->Signal(g_fence, ++g_fenceVal))) return false;
    for (int i = 0; i < 15000; i++) {
        if (g_fence->GetCompletedValue() >= g_fenceVal) return true;
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, nullptr);
    }
    printf("MESH12: GPU wait TIMEOUT\n");
    return false;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "MESH12: dmn_init FAILED\n");
        return 1;
    }
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&g_dev),
       "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd = {};
    CK(g_dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&g_q), "CreateCommandQueue");
    CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                     (void**)&g_alloc), "CreateCommandAllocator");
    CK(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc, nullptr,
                                __uuidof(ID3D12GraphicsCommandList), (void**)&g_cl), "CreateCommandList");
    CK(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&g_fence), "CreateFence");

    void* dev2 = nullptr;
    CK(g_dev->QueryInterface(kIID_Device2, &dev2), "QueryInterface(ID3D12Device2)");
    void* cl6 = nullptr;
    CK(g_cl->QueryInterface(kIID_List6, &cl6), "QueryInterface(ID3D12GraphicsCommandList6)");

    D3D12_ROOT_SIGNATURE_DESC rs = {};
    Com<ID3DBlob> sig, err;
    CK(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err),
       "D3D12SerializeRootSignature");
    Com<ID3D12RootSignature> rootSig;
    CK(g_dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                  __uuidof(ID3D12RootSignature), (void**)&rootSig), "CreateRootSignature");

    /* The stream, packed as the UMD packs it. */
    unsigned char buf[1024];
    size_t off = 0;
    ID3D12RootSignature* rsPtr = rootSig.ptr();
    PUT(ROOT_SIGNATURE, &rsPtr, ID3D12RootSignature*);
    D3D12_SHADER_BYTECODE ms = {kMeshDxil, sizeof(kMeshDxil)};
    D3D12_SHADER_BYTECODE ps = {kPixDxil, sizeof(kPixDxil)};
    PUT(MS, &ms, D3D12_SHADER_BYTECODE);
    PUT(PS, &ps, D3D12_SHADER_BYTECODE);
    D3D12_BLEND_DESC blend = {};
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    PUT(BLEND, &blend, D3D12_BLEND_DESC);
    UINT sampleMask = 0xffffffffu;
    PUT(SAMPLE_MASK, &sampleMask, UINT);
    D3D12_RASTERIZER_DESC rast = {};
    rast.FillMode = D3D12_FILL_MODE_SOLID;
    rast.CullMode = D3D12_CULL_MODE_NONE;
    rast.DepthClipEnable = TRUE;
    PUT(RASTERIZER, &rast, D3D12_RASTERIZER_DESC);
    D3D12_DEPTH_STENCIL_DESC ds = {};
    PUT(DEPTH_STENCIL, &ds, D3D12_DEPTH_STENCIL_DESC);
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topo = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    PUT(PRIMITIVE_TOPOLOGY, &topo, D3D12_PRIMITIVE_TOPOLOGY_TYPE);
    RT_FORMAT_ARRAY rts = {};
    rts.NumRenderTargets = 1;
    rts.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    put(buf, &off, SUBOBJ_RENDER_TARGET_FORMATS, &rts, sizeof(rts), alignof(UINT));
    DXGI_FORMAT dsv = DXGI_FORMAT_UNKNOWN;
    PUT(DEPTH_STENCIL_FORMAT, &dsv, DXGI_FORMAT);
    DXGI_SAMPLE_DESC sd = {1, 0};
    PUT(SAMPLE_DESC, &sd, DXGI_SAMPLE_DESC);
    UINT nodeMask = 0;
    PUT(NODE_MASK, &nodeMask, UINT);
    PIPELINE_STATE_STREAM_DESC streamDesc = {off, buf};
    ID3D12PipelineState* pso = nullptr;
    CK(vslot<PFN_CreatePipelineState>(dev2, kSlot_Device2_CreatePipelineState)(
           dev2, &streamDesc, __uuidof(ID3D12PipelineState), (void**)&pso),
       "ID3D12Device2::CreatePipelineState(mesh stream)");
    printf("MESH12: mesh PSO created (stream %zu bytes)\n", off);

    const UINT W = 8, H = 8;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = W; rd.Height = H; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv = {};
    cv.Format = rd.Format;
    cv.Color[0] = 1.0f; cv.Color[3] = 1.0f;
    Com<ID3D12Resource> rt;
    CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                      &cv, __uuidof(ID3D12Resource), (void**)&rt), "CreateCommittedResource(rt)");
    D3D12_DESCRIPTOR_HEAP_DESC rh = {};
    rh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rh.NumDescriptors = 1;
    Com<ID3D12DescriptorHeap> rtvHeap;
    CK(g_dev->CreateDescriptorHeap(&rh, __uuidof(ID3D12DescriptorHeap), (void**)&rtvHeap), "CreateDescriptorHeap");
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    g_dev->CreateRenderTargetView(rt.ptr(), nullptr, rtv);

    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = 256 * H; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Com<ID3D12Resource> rb;
    CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                      nullptr, __uuidof(ID3D12Resource), (void**)&rb), "CreateCommittedResource(readback)");

    float clear[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    g_cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    g_cl->ClearRenderTargetView(rtv, clear, 0, nullptr);
    D3D12_VIEWPORT vp = {0, 0, (float)W, (float)H, 0, 1};
    D3D12_RECT sc = {0, 0, (LONG)W, (LONG)H};
    g_cl->RSSetViewports(1, &vp);
    g_cl->RSSetScissorRects(1, &sc);
    g_cl->SetGraphicsRootSignature(rootSig.ptr());
    g_cl->SetPipelineState(pso);
    vslot<PFN_DispatchMesh>(cl6, kSlot_List6_DispatchMesh)(cl6, 1, 1, 1);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = rt.ptr();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    g_cl->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst = {};
    dst.pResource = rb.ptr();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = rd.Format;
    dst.PlacedFootprint.Footprint.Width = W;
    dst.PlacedFootprint.Footprint.Height = H;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = 256;
    D3D12_TEXTURE_COPY_LOCATION src = {};
    src.pResource = rt.ptr();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    g_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (!execAndWait()) {
        printf("MESH12: submit FAILED\n");
        return 1;
    }
    void* m = nullptr;
    CK(rb->Map(0, nullptr, &m), "Map(readback)");
    bool ok = true;
    for (UINT y = 0; y < H && ok; y++)
        for (UINT x = 0; x < W; x++) {
            const unsigned char* px = (const unsigned char*)m + y * 256 + x * 4;
            if (!(px[0] == 0 && px[1] == 255 && px[2] == 0)) {
                printf("MESH12: pixel (%u,%u) = %u,%u,%u (want 0,255,0)\n", x, y, px[0], px[1], px[2]);
                ok = false;
                break;
            }
        }
    rb->Unmap(0, nullptr);
    printf(ok ? "MESH12: PASS\n" : "MESH12: FAIL\n");
    return ok ? 0 : 1;
}
