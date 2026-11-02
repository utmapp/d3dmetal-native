/* QuadAny/QuadAll across all 16 input patterns, plus quad partner mapping.
 * Fixtures: dxc -T cs_6_7 -E main -D TWO_D=0|1 -Fh quadvote0.h|quadvote1.h
 *           -Vn g_quadvote0|g_quadvote1 shaders/quadvote.hlsl
 */
#include <cstdio>
#include <cstring>
#include <time.h>
#include <d3d12.h>
#include "d3dmetal_native.h"
#include "dmn_d3d12_up.h"
#define T_TAG "QUADVOTE"
#include "common/check.h"
typedef unsigned char BYTE;
#include "shaders/quadvote0.h"
#include "shaders/quadvote1.h"

namespace {
ID3D12Device *device;
ID3D12CommandQueue *queue;
ID3D12CommandAllocator *allocator;
ID3D12GraphicsCommandList *commands;
ID3D12Fence *fence;
UINT64 serial;
constexpr UINT kBytes = 64 * 4 * sizeof(UINT);

ID3D12Resource *buffer(D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                       D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES heap = {}; heap.Type = type;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kBytes; desc.Height = 1; desc.DepthOrArraySize = 1;
    desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
    ID3D12Resource *result = nullptr;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &desc, state, nullptr, __uuidof(ID3D12Resource), (void **)&result)))
        return nullptr;
    return result;
}

void transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from; b.Transition.StateAfter = to;
    commands->ResourceBarrier(1, &b);
}

bool check(ID3D12PipelineState *state, ID3D12RootSignature *root, bool twoD,
           const char *path) {
    ID3D12Resource *out = buffer(D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource *seed = buffer(D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource *readback = buffer(D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    if (!out || !seed || !readback) return false;
    void *mapped = nullptr;
    if (FAILED(seed->Map(0, nullptr, &mapped))) return false;
    memset(mapped, 0xcd, kBytes); seed->Unmap(0, nullptr);
    if (FAILED(allocator->Reset()) || FAILED(commands->Reset(allocator, state))) return false;
    commands->CopyResource(out, seed);
    transition(out, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    commands->SetComputeRootSignature(root);
    commands->SetComputeRootUnorderedAccessView(0, out->GetGPUVirtualAddress());
    commands->Dispatch(1, 1, 1);
    transition(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands->CopyResource(readback, out);
    if (FAILED(commands->Close())) return false;
    ID3D12CommandList *lists[] = {commands}; queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence, ++serial))) return false;
    const timespec pause = {0, 1000000};
    unsigned waits = 0;
    while (fence->GetCompletedValue() < serial && waits++ < 15000)
        nanosleep(&pause, nullptr);
    if (fence->GetCompletedValue() < serial) return false;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    const UINT *values = static_cast<const UINT *>(mapped);
    unsigned badVotes = 0, badPartners = 0;
    for (UINT i = 0; i < 64; ++i) {
        const UINT x = i % 8, y = i / 8;
        const UINT pattern = twoD ? (y / 2) * 4 + x / 2 : i / 4;
        const UINT vote = (pattern != 0) | ((pattern == 15) << 1);
        if (values[i * 4] != vote) ++badVotes;
        const UINT partners[] = {i ^ 1, i ^ (twoD ? 8 : 2), i ^ (twoD ? 9 : 3)};
        for (UINT p = 0; p < 3; ++p)
            if (values[i * 4 + p + 1] != partners[p]) ++badPartners;
    }
    readback->Unmap(0, nullptr);
    printf(T_TAG ": %s %s votes=%u/64 wrong, partners=%u/192 wrong\n",
           path, twoD ? "8x8" : "64x1", badVotes, badPartners);
    readback->Release(); seed->Release(); out->Release();
    return badVotes == 0 && badPartners == 0;
}

template<class T> struct alignas(void *) Subobject { UINT type; T value; };
}

int main() {
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
        __uuidof(ID3D12Device), (void **)&device), "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd = {};
    CK(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void **)&queue), "queue");
    CK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        __uuidof(ID3D12CommandAllocator), (void **)&allocator), "allocator");
    CK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator,
        nullptr, __uuidof(ID3D12GraphicsCommandList), (void **)&commands), "commands");
    CK(commands->Close(), "initial Close");
    CK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        __uuidof(ID3D12Fence), (void **)&fence), "fence");
    D3D12_ROOT_PARAMETER parameter = {};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC rd = {}; rd.NumParameters = 1; rd.pParameters = &parameter;
    ID3DBlob *blob = nullptr;
    CK(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "serialize root");
    ID3D12RootSignature *root = nullptr;
    CK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
        __uuidof(ID3D12RootSignature), (void **)&root), "root");
    blob->Release();
    ID3D12Device2 *device2 = nullptr;
    CK(device->QueryInterface(__uuidof(ID3D12Device2), (void **)&device2), "Device2");
    ID3D12PipelineLibrary *library = nullptr;
    const HRESULT libraryHr = device2->CreatePipelineLibrary(nullptr, 0,
        __uuidof(ID3D12PipelineLibrary), (void **)&library);
    if (FAILED(libraryHr))
        printf(T_TAG ": pipeline library unavailable hr=0x%08x\n", (unsigned)libraryHr);
    ID3D12PipelineLibrary1 *library1 = nullptr;
    if (library) library->QueryInterface(__uuidof(ID3D12PipelineLibrary1), (void **)&library1);
    const D3D12_SHADER_BYTECODE shaders[] = {
        {g_quadvote0, sizeof(g_quadvote0)}, {g_quadvote1, sizeof(g_quadvote1)}};
    for (unsigned mode = 0; mode < 2; ++mode) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = root; desc.CS = shaders[mode];
        ID3D12PipelineState *state = nullptr;
        CK(device->CreateComputePipelineState(&desc,
            __uuidof(ID3D12PipelineState), (void **)&state), "compute pipeline");
        EXPECT(check(state, root, mode != 0, "compute"), "compute quad semantics");
        const WCHAR name[] = {'q', 'u', 'a', 'd', WCHAR('1' + mode), 'd', 0};
        if (library) CK(library->StorePipeline(name, state), "StorePipeline");
        state->Release();
        struct { Subobject<ID3D12RootSignature *> root; Subobject<D3D12_SHADER_BYTECODE> cs; }
            stream = {{0, root}, {6, shaders[mode]}};
        const D3D12_PIPELINE_STATE_STREAM_DESC sd = {sizeof(stream), &stream};
        CK(device2->CreatePipelineState(&sd,
            __uuidof(ID3D12PipelineState), (void **)&state), "stream pipeline");
        EXPECT(check(state, root, mode != 0, "stream"), "stream quad semantics");
        state->Release();
        if (library) {
            CK(library->LoadComputePipeline(name, &desc,
                __uuidof(ID3D12PipelineState), (void **)&state), "LoadComputePipeline");
            EXPECT(check(state, root, mode != 0, "library-compute"), "library quad semantics");
            state->Release();
        }
        if (library1) {
            CK(library1->LoadPipeline(name, &sd,
                __uuidof(ID3D12PipelineState), (void **)&state), "LoadPipeline");
            EXPECT(check(state, root, mode != 0, "library-stream"), "library stream quad semantics");
            state->Release();
        }
    }
    if (library1) library1->Release();
    if (library) library->Release();
    device2->Release(); root->Release(); fence->Release(); commands->Release();
    allocator->Release(); queue->Release(); device->Release();
    T_PASS();
    return 0;
}
