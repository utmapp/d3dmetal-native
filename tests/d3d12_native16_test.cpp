/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Host twin of the guest's `native16.exe`: the same dxc cs_6_2
 * -enable-16bit-types kernel (shaders/n16sh/, built by buildn16.cmd on the
 * dev box) writes asuint16() of float16_t results to a UAV; the readback
 * must match IEEE half rounding computed on the CPU.  Passing here backs
 * Triton's SHADER.Native16BitOps = host truth.
 *
 * Prints "N16: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>
#include <time.h>

#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"

#define T_TAG "N16"
#include "common/check.h"

typedef unsigned char BYTE;
#include "shaders/n16sh/cs62.h"

namespace {

unsigned short f2h(float f) {
    unsigned int x; memcpy(&x, &f, 4);
    unsigned int sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    unsigned int mant = x & 0x7fffff;
    if (exp >= 31) return (unsigned short)(sign | 0x7c00);
    if (exp <= 0) return (unsigned short)sign;
    unsigned int h = sign | (exp << 10) | (mant >> 13);
    unsigned int rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (unsigned short)h;
}
float h2f(unsigned short h) {
    unsigned int sign = (h & 0x8000) << 16; int exp = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    unsigned int x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; memcpy(&f, &x, 4); return f;
}

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

int main() {
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&g_dev),
       "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CK(g_dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&g_q), "CreateCommandQueue");
    CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&g_alloc), "CreateCommandAllocator");
    CK(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&g_cl), "CreateCommandList");
    CK(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&g_fence), "CreateFence");

    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4 = {};
    CK(g_dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4)), "CheckFeatureSupport(OPTIONS4)");
    printf(T_TAG ": host Native16BitShaderOpsSupported=%d\n", (int)o4.Native16BitShaderOpsSupported);

    D3D12_ROOT_PARAMETER rp = {}; rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC rd = {}; rd.NumParameters = 1; rd.pParameters = &rp;
    ID3DBlob* blob = nullptr; ID3DBlob* err = nullptr; ID3D12RootSignature* rs = nullptr;
    CK(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err), "D3D12SerializeRootSignature");
    CK(g_dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&rs), "CreateRootSignature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {}; cd.pRootSignature = rs;
    cd.CS.pShaderBytecode = g_n16_cs; cd.CS.BytecodeLength = sizeof(g_n16_cs);
    ID3D12PipelineState* pso = nullptr;
    CK_OK(g_dev->CreateComputePipelineState(&cd, __uuidof(ID3D12PipelineState), (void**)&pso),
          "CreateComputePipelineState(float16_t cs_6_2)");

    const UINT N = 64 * 2, BYTES = N * 4;
    ID3D12Resource* uav = makeBuffer(BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ID3D12Resource* rb = makeBuffer(BYTES, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    EXPECT(uav && rb, "buffer create failed");
    g_cl->SetPipelineState(pso);
    g_cl->SetComputeRootSignature(rs);
    g_cl->SetComputeRootUnorderedAccessView(0, uav->GetGPUVirtualAddress());
    g_cl->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = uav; b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    g_cl->ResourceBarrier(1, &b);
    g_cl->CopyResource(rb, uav);
    EXPECT(execAndWait(), "execute/wait failed");

    UINT* m = nullptr;
    CK(rb->Map(0, nullptr, (void**)&m), "Map(readback)");
    UINT bad = 0;
    const unsigned short wantB = f2h(h2f(f2h(1.001f)) * h2f(f2h(3.0f)));
    for (UINT i = 0; i < 64; i++) {
        unsigned short wantA = f2h(2049.0f + (float)i);
        if ((unsigned short)m[i * 2] != wantA || (unsigned short)m[i * 2 + 1] != wantB) {
            if (bad < 4) printf(T_TAG ": thread %u: got (0x%04x,0x%04x) want (0x%04x,0x%04x)\n", i, m[i * 2], m[i * 2 + 1], wantA, wantB);
            bad++;
        }
    }
    rb->Unmap(0, nullptr);
    printf(T_TAG ": %u of 64 threads wrong\n", bad);
    EXPECT(bad == 0, "float16_t results do not round as IEEE half");
    T_PASS();
    return 0;
}
