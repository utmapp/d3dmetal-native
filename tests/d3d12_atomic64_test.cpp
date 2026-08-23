/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Host twin of the guest's `atomic64.exe`: the same dxc cs_6_6 kernel
 * (shaders/a64sh/, built by builda64.cmd on the dev box) runs 64-bit
 * integer atomics on a typed R32G32_UINT UAV, group shared memory and a
 * raw buffer; every accumulation crosses 2^32.  Passing here backs
 * Triton's SHADER.AtomicInt64On{TypedResource,GroupShared} = host truth.
 *
 * Prints "A64: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>
#include <time.h>

#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"

#define T_TAG "A64"
#include "common/check.h"

typedef unsigned char BYTE;
#include "shaders/a64sh/cs66.h"
#include "shaders/a64sh/cs66b.h" /* typed atomics replaced by plain stores: isolates the typed path */

/* The vendored d3d12.h stops at OPTIONS7; the backend answers later ids
 * regardless (its ceiling is not the header's). */
struct FeatureOptions9 {
    BOOL MeshShaderPipelineStatsSupported;
    BOOL MeshShaderSupportsFullRangeRenderTargetArrayIndex;
    BOOL AtomicInt64OnTypedResourceSupported;
    BOOL AtomicInt64OnGroupSharedSupported;
    BOOL DerivativesInMeshAndAmplificationShadersSupported;
    UINT WaveMMATier;
};
static const UINT kFeatureOptions9 = 37;

namespace {

ID3D12Device* g_dev;
ID3D12CommandQueue* g_q;
ID3D12CommandAllocator* g_alloc;
ID3D12GraphicsCommandList* g_cl;
ID3D12Fence* g_fence;
UINT64 g_fenceVal;

ID3D12Resource* makeBuffer(UINT64 bytes, D3D12_HEAP_TYPE type,
                           D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = type;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = bytes; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = flags;
    ID3D12Resource* r = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                              nullptr, __uuidof(ID3D12Resource), (void**)&r)))
        return nullptr;
    return r;
}

bool execAndWait() {
    if (FAILED(g_cl->Close())) return false;
    ID3D12CommandList* lists[] = {g_cl};
    g_q->ExecuteCommandLists(1, lists);
    ++g_fenceVal;
    if (FAILED(g_q->Signal(g_fence, g_fenceVal))) return false;
    for (int i = 0; i < 15000; i++) {
        if (g_fence->GetCompletedValue() >= g_fenceVal) return true;
        struct timespec ts = {0, 1000000}; nanosleep(&ts, nullptr);
    }
    printf(T_TAG ": GPU wait TIMEOUT\n");
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const bool variantB = argc > 1 && argv[1][0] == 'b';
    /* "-f file.cso": run an arbitrary DXIL container (bisection aid); the
     * pass/fail line is then only informational. */
    static unsigned char fileBlob[1 << 20]; size_t fileLen = 0;
    if (argc > 2 && !strcmp(argv[1], "-f")) {
        FILE* fp = fopen(argv[2], "rb");
        EXPECT(fp != nullptr, "cannot open -f file");
        fileLen = fread(fileBlob, 1, sizeof(fileBlob), fp); fclose(fp);
    }
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&g_dev),
       "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CK(g_dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&g_q), "CreateCommandQueue");
    CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&g_alloc), "CreateCommandAllocator");
    CK(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&g_cl), "CreateCommandList");
    CK(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&g_fence), "CreateFence");

    FeatureOptions9 o9 = {};
    CK(g_dev->CheckFeatureSupport((D3D12_FEATURE)kFeatureOptions9, &o9, sizeof(o9)), "CheckFeatureSupport(OPTIONS9)");
    printf(T_TAG ": host AtomicInt64OnTypedResource=%d OnGroupShared=%d\n",
           (int)o9.AtomicInt64OnTypedResourceSupported, (int)o9.AtomicInt64OnGroupSharedSupported);

    D3D12_DESCRIPTOR_RANGE range = {};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; range.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER rp[3] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[0].DescriptorTable.NumDescriptorRanges = 1; rp[0].DescriptorTable.pDescriptorRanges = &range;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[2].Descriptor.ShaderRegister = 2;
    D3D12_ROOT_SIGNATURE_DESC rd = {}; rd.NumParameters = 3; rd.pParameters = rp;
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr; ID3D12RootSignature* rs = nullptr;
    CK(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err), "D3D12SerializeRootSignature");
    CK(g_dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&rs), "CreateRootSignature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {}; cd.pRootSignature = rs;
    cd.CS.pShaderBytecode = fileLen ? (const void*)fileBlob : variantB ? (const void*)g_a64b_cs : (const void*)g_a64_cs;
    cd.CS.BytecodeLength = fileLen ? fileLen : variantB ? sizeof(g_a64b_cs) : sizeof(g_a64_cs);
    ID3D12PipelineState* pso = nullptr;
    CK_OK(g_dev->CreateComputePipelineState(&cd, __uuidof(ID3D12PipelineState), (void**)&pso),
          "CreateComputePipelineState(int64 atomics cs_6_6)");

    ID3D12Resource* T = makeBuffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource* G = makeBuffer(8, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource* B = makeBuffer(8, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource* rb = makeBuffer(32, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    EXPECT(T && G && B && rb, "buffer create failed");
    ID3D12DescriptorHeap* heap = nullptr;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 1; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CK(g_dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&heap), "CreateDescriptorHeap");
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = DXGI_FORMAT_R32G32_UINT;
    uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; uv.Buffer.NumElements = 2;
    g_dev->CreateUnorderedAccessView(T, nullptr, &uv, heap->GetCPUDescriptorHandleForHeapStart());

    g_cl->SetPipelineState(pso);
    g_cl->SetComputeRootSignature(rs);
    g_cl->SetDescriptorHeaps(1, &heap);
    g_cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    g_cl->SetComputeRootUnorderedAccessView(1, G->GetGPUVirtualAddress());
    g_cl->SetComputeRootUnorderedAccessView(2, B->GetGPUVirtualAddress());
    g_cl->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER bars[3] = {};
    ID3D12Resource* srcs[3] = {T, G, B};
    for (int i = 0; i < 3; i++) {
        bars[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bars[i].Transition.pResource = srcs[i];
        bars[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        bars[i].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        bars[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    g_cl->ResourceBarrier(3, bars);
    ID3D12Resource* rbT = makeBuffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* rbG = makeBuffer(8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    ID3D12Resource* rbB = makeBuffer(8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    g_cl->CopyResource(rbT, T);
    g_cl->CopyResource(rbG, G);
    g_cl->CopyResource(rbB, B);
    EXPECT(execAndWait(), "execute/wait failed");

    UINT64 m[4] = {0, 0, 0, 0};
    {
        UINT64* p = nullptr;
        CK(rbT->Map(0, nullptr, (void**)&p), "Map(rbT)"); m[0] = p[0]; m[1] = p[1]; rbT->Unmap(0, nullptr);
        CK(rbG->Map(0, nullptr, (void**)&p), "Map(rbG)"); m[2] = p[0]; rbG->Unmap(0, nullptr);
        CK(rbB->Map(0, nullptr, (void**)&p), "Map(rbB)"); m[3] = p[0]; rbB->Unmap(0, nullptr);
    }
    const UINT64 wantAdd = 64ull * 0x100000001ull;
    const UINT64 wantMax = variantB ? 0x200000002ull : (64ull << 33);
    const UINT64 wantTypedAdd = variantB ? 0x100000001ull : wantAdd;
    printf(T_TAG ": typed add=0x%llx (want 0x%llx) typed max=0x%llx (want 0x%llx) groupshared=0x%llx raw=0x%llx (want 0x%llx)\n",
           (unsigned long long)m[0], (unsigned long long)wantAdd, (unsigned long long)m[1],
           (unsigned long long)wantMax, (unsigned long long)m[2], (unsigned long long)m[3],
           (unsigned long long)wantAdd);
    const bool ok = m[0] == wantTypedAdd && m[1] == wantMax && m[2] == wantAdd && m[3] == wantAdd;
    (void)rb;
    EXPECT(ok, "64-bit atomics produced wrong values");
    T_PASS();
    return 0;
}
