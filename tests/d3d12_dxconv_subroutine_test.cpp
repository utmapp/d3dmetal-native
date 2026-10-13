/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Shaders as the D3D12 runtime's in-driver DXBC conversion hands them over
 * -- call/label subroutines left as internal functions -- must build and
 * run: Metal Shader Converter crashes the process on a module defining one,
 * so the shim inlines it first (dmn_dxil_inline_subroutines).  Each pixel
 * shader case draws a fullscreen triangle and reads the target back:
 *
 *   - the captured ps_2_x shader;
 *   - an indexable temp shared with a subroutine, which stays a global;
 *   - a pass-through geometry shader whose vertices cross subroutines.
 *
 * A compute shader whose subroutine computes the stored value is dispatched
 * and its UAV read back.
 *
 * Prints "DXCONV: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>

#include <d3d12.h>
#include <windows.h>

#include <time.h>

#include "d3dmetal_native.h"

#define T_TAG "DXCONV"
#include "common/check.h"

typedef unsigned char BYTE;
#include "shaders/dxconvsh/vs_texcoord.h"
#include "shaders/dxconvsh/ps_label_calls.h"
#include "shaders/dxconvsh/ps_indexable_temp.h"
#include "shaders/dxconvsh/cs_label_calls.h"
#include "shaders/dxconvsh/gs_label_calls.h"

namespace {

const UINT W = 64, H = 64;

struct Case {
    const char* name;
    const unsigned char* ps;
    size_t psSize;
    BYTE want[4];
    const unsigned char* gs;
    size_t gsSize;
};
const Case kCases[] = {
    {"ps_2_x label calls", g_ps_label_calls, sizeof(g_ps_label_calls),
     {255, 255, 0, 0}},
    {"indexable temp shared with a subroutine", g_ps_indexable_temp,
     sizeof(g_ps_indexable_temp), {255, 255, 0, 0}},
    {"gs label calls", g_ps_label_calls, sizeof(g_ps_label_calls),
     {255, 255, 0, 0}, g_gs_label_calls, sizeof(g_gs_label_calls)},
};

ID3D12Device* g_dev;
ID3D12CommandQueue* g_q;
ID3D12CommandAllocator* g_alloc;
ID3D12GraphicsCommandList* g_cl;
ID3D12Fence* g_fence;
UINT64 g_fenceVal;

ID3D12Resource* makeBuffer(UINT64 bytes, D3D12_HEAP_TYPE type,
                           D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = type;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              state, nullptr,
                                              __uuidof(ID3D12Resource),
                                              (void**)&r)))
        return nullptr;
    return r;
}

/* Poll rather than SetEventOnCompletion: the host-side windows shim has no
 * waitable event objects (same idiom as d3d12_defbuf_test). */
bool execAndWait() {
    if (FAILED(g_cl->Close()))
        return false;
    ID3D12CommandList* lists[] = {g_cl};
    g_q->ExecuteCommandLists(1, lists);
    ++g_fenceVal;
    if (FAILED(g_q->Signal(g_fence, g_fenceVal)))
        return false;
    for (int i = 0; i < 15000; i++) {
        if (g_fence->GetCompletedValue() >= g_fenceVal)
            return true;
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, nullptr);
    }
    printf(T_TAG ": GPU wait TIMEOUT\n");
    return false;
}

} // namespace

int main() {
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                         __uuidof(ID3D12Device), (void**)&g_dev),
       "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CK(g_dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                 (void**)&g_q),
       "CreateCommandQueue");
    CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     __uuidof(ID3D12CommandAllocator),
                                     (void**)&g_alloc),
       "CreateCommandAllocator");
    CK(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc,
                                nullptr, __uuidof(ID3D12GraphicsCommandList),
                                (void**)&g_cl),
       "CreateCommandList");
    CK(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                          (void**)&g_fence),
       "CreateFence");

    ID3D12RootSignature* rs = nullptr;
    {
        D3D12_ROOT_SIGNATURE_DESC rd = {};
        ID3DBlob* blob = nullptr;
        ID3DBlob* err = nullptr;
        CK(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                                       &blob, &err),
           "D3D12SerializeRootSignature");
        CK(g_dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                      blob->GetBufferSize(),
                                      __uuidof(ID3D12RootSignature),
                                      (void**)&rs),
           "CreateRootSignature");
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = rs;
    pd.VS.pShaderBytecode = g_vs_texcoord;
    pd.VS.BytecodeLength = sizeof(g_vs_texcoord);
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    pd.SampleMask = 0xFFFFFFFFu;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pd.SampleDesc.Count = 1;

    /* One pipeline per case. */
    const int kNumCases = (int)(sizeof(kCases) / sizeof(kCases[0]));
    ID3D12PipelineState* psos[kNumCases] = {};
    for (int i = 0; i < kNumCases; i++) {
        pd.PS.pShaderBytecode = kCases[i].ps;
        pd.PS.BytecodeLength = kCases[i].psSize;
        pd.GS.pShaderBytecode = kCases[i].gs;
        pd.GS.BytecodeLength = kCases[i].gsSize;
        HRESULT hr = g_dev->CreateGraphicsPipelineState(
            &pd, __uuidof(ID3D12PipelineState), (void**)&psos[i]);
        printf(T_TAG ": CreateGraphicsPipelineState(%s) -> 0x%08x\n",
               kCases[i].name, (unsigned)hr);
        EXPECT(SUCCEEDED(hr), "converted shader failed to build");
    }

    ID3D12Resource* rt = nullptr;
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC td = {};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = W;
        td.Height = H;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv = {};
        cv.Format = td.Format;
        cv.Color[2] = 1.0f; /* blue clear; every draw must overwrite it */
        cv.Color[3] = 1.0f;
        CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                          D3D12_RESOURCE_STATE_RENDER_TARGET,
                                          &cv, __uuidof(ID3D12Resource),
                                          (void**)&rt),
           "CreateCommittedResource(rt)");
    }
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = 1;
        CK(g_dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap),
                                       (void**)&rtvHeap),
           "CreateDescriptorHeap(rtv)");
        g_dev->CreateRenderTargetView(
            rt, nullptr, rtvHeap->GetCPUDescriptorHandleForHeapStart());
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    D3D12_RESOURCE_DESC td = rt->GetDesc();
    g_dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
    ID3D12Resource* rb = makeBuffer(total, D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
    EXPECT(rb != nullptr, "readback buffer create failed");

    D3D12_CPU_DESCRIPTOR_HANDLE rtv =
        rtvHeap->GetCPUDescriptorHandleForHeapStart();
    const float blue[4] = {0.f, 0.f, 1.f, 1.f};
    D3D12_VIEWPORT vp = {0.f, 0.f, (float)W, (float)H, 0.f, 1.f};
    D3D12_RECT sc = {0, 0, (LONG)W, (LONG)H};
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = rt;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = rt;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = rb;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;

    bool first = true;
    for (int i = 0; i < kNumCases; i++) {
        const Case& cs = kCases[i];
        if (!first) {
            CK(g_alloc->Reset(), "allocator Reset");
            CK(g_cl->Reset(g_alloc, nullptr), "command list Reset");
        }
        first = false;
        g_cl->SetPipelineState(psos[i]);
        g_cl->SetGraphicsRootSignature(rs);
        g_cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        g_cl->ClearRenderTargetView(rtv, blue, 0, nullptr);
        g_cl->RSSetViewports(1, &vp);
        g_cl->RSSetScissorRects(1, &sc);
        g_cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_cl->DrawInstanced(3, 1, 0, 0);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_cl->ResourceBarrier(1, &b);
        g_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        g_cl->ResourceBarrier(1, &b);
        EXPECT(execAndWait(), "execute/wait failed");

        void* m = nullptr;
        CK(rb->Map(0, nullptr, &m), "Map(readback)");
        const BYTE* c = (const BYTE*)m + fp.Footprint.RowPitch * (H / 2) + (W / 2) * 4;
        const BYTE* k = (const BYTE*)m;
        printf(T_TAG ": %s: center=(%u,%u,%u,%u) corner=(%u,%u,%u,%u) "
               "want (%u,%u,%u,%u)\n",
               cs.name, c[0], c[1], c[2],
               c[3], k[0], k[1], k[2], k[3], cs.want[0], cs.want[1],
               cs.want[2], cs.want[3]);
        const bool ok = memcmp(c, cs.want, 4) == 0 && memcmp(k, cs.want, 4) == 0;
        rb->Unmap(0, nullptr);
        EXPECT(ok, "converted subroutine shader produced the wrong pixels");
    }

    /* Compute: root UAV u0, one 64-thread group. */
    {
        const UINT kN = 64;
        ID3D12RootSignature* crs = nullptr;
        D3D12_ROOT_PARAMETER rp = {};
        rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rd = {};
        rd.NumParameters = 1;
        rd.pParameters = &rp;
        ID3DBlob* blob = nullptr;
        ID3DBlob* err = nullptr;
        CK(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                                       &blob, &err),
           "D3D12SerializeRootSignature(cs)");
        CK(g_dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                      blob->GetBufferSize(),
                                      __uuidof(ID3D12RootSignature),
                                      (void**)&crs),
           "CreateRootSignature(cs)");
        D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {};
        cd.pRootSignature = crs;
        cd.CS.pShaderBytecode = g_cs_label_calls;
        cd.CS.BytecodeLength = sizeof(g_cs_label_calls);
        ID3D12PipelineState* cpso = nullptr;
        HRESULT hr = g_dev->CreateComputePipelineState(
            &cd, __uuidof(ID3D12PipelineState), (void**)&cpso);
        printf(T_TAG ": CreateComputePipelineState(cs label calls) -> 0x%08x\n",
               (unsigned)hr);
        EXPECT(SUCCEEDED(hr), "converted compute shader failed to build");

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC bd = {};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = kN * 4;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ID3D12Resource* uav = nullptr;
        CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          nullptr, __uuidof(ID3D12Resource),
                                          (void**)&uav),
           "CreateCommittedResource(uav)");
        ID3D12Resource* crb = makeBuffer(kN * 4, D3D12_HEAP_TYPE_READBACK,
                                         D3D12_RESOURCE_STATE_COPY_DEST);
        EXPECT(crb != nullptr, "compute readback buffer create failed");

        CK(g_alloc->Reset(), "allocator Reset(cs)");
        CK(g_cl->Reset(g_alloc, cpso), "command list Reset(cs)");
        g_cl->SetComputeRootSignature(crs);
        g_cl->SetComputeRootUnorderedAccessView(0, uav->GetGPUVirtualAddress());
        g_cl->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER ub = {};
        ub.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        ub.Transition.pResource = uav;
        ub.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ub.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        ub.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_cl->ResourceBarrier(1, &ub);
        g_cl->CopyBufferRegion(crb, 0, uav, 0, kN * 4);
        EXPECT(execAndWait(), "compute execute/wait failed");

        void* m = nullptr;
        CK(crb->Map(0, nullptr, &m), "Map(compute readback)");
        const unsigned* v = (const unsigned*)m;
        UINT bad = 0;
        for (UINT i = 0; i < kN; i++)
            if (v[i] != 3 * i + 7) {
                if (!bad)
                    printf(T_TAG ": cs label calls: [%u] = %u, want %u\n", i,
                           v[i], 3 * i + 7);
                bad++;
            }
        printf(T_TAG ": cs label calls: %u/%u wrong\n", bad, kN);
        crb->Unmap(0, nullptr);
        EXPECT(bad == 0, "converted subroutine compute shader stored wrong values");
    }

    T_PASS();
    return 0;
}
