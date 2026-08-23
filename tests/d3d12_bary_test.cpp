/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Host twin of the guest's `barydraw.exe`: the same dxc SM 6.1 pair
 * (shaders/barysh/, built by buildbary.cmd on the dev box) draws a
 * fullscreen triangle from SV_VertexID whose pixel shader writes
 * SV_Barycentrics as colour.  The centre pixel -- NDC (0,0) of the
 * (-1,-1),(3,-1),(-1,3) triangle -- must read back (128,64,64) within
 * +-3 and the (0,0) corner, NDC (-1,+1) on the vertex-0/vertex-2 edge,
 * (128,0,128).  Passing here
 * backs Triton's OPTIONS3.BarycentricsSupported = host truth.
 *
 * Prints "BARY: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>

#include <d3d12.h>
#include <windows.h>

#include <time.h>

#include "d3dmetal_native.h"

#define T_TAG "BARY"
#include "common/check.h"

typedef unsigned char BYTE;
#include "shaders/barysh/vs61.h"
#include "shaders/barysh/ps61.h"

namespace {

const UINT W = 64, H = 64;

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

    /* Host truth for the cap this test backs. */
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 o3 = {};
    CK(g_dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &o3,
                                  sizeof(o3)),
       "CheckFeatureSupport(OPTIONS3)");
    printf(T_TAG ": host BarycentricsSupported=%d\n",
           (int)o3.BarycentricsSupported);

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

    ID3D12PipelineState* pso = nullptr;
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rs;
        pd.VS.pShaderBytecode = g_bary_vs;
        pd.VS.BytecodeLength = sizeof(g_bary_vs);
        pd.PS.pShaderBytecode = g_bary_ps;
        pd.PS.BytecodeLength = sizeof(g_bary_ps);
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
        pd.SampleMask = 0xFFFFFFFFu;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.SampleDesc.Count = 1;
        CK_OK(g_dev->CreateGraphicsPipelineState(&pd,
                                                 __uuidof(ID3D12PipelineState),
                                                 (void**)&pso),
              "CreateGraphicsPipelineState(SV_Barycentrics vs/ps)");
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
        cv.Color[0] = 1.0f; /* red clear; the draw must overwrite it */
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

    D3D12_CPU_DESCRIPTOR_HANDLE rtv =
        rtvHeap->GetCPUDescriptorHandleForHeapStart();
    const float blue[4] = {0.f, 0.f, 1.f, 1.f};
    D3D12_VIEWPORT vp = {0.f, 0.f, (float)W, (float)H, 0.f, 1.f};
    D3D12_RECT sc = {0, 0, (LONG)W, (LONG)H};
    g_cl->SetPipelineState(pso);
    g_cl->SetGraphicsRootSignature(rs);
    g_cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    g_cl->ClearRenderTargetView(rtv, blue, 0, nullptr);
    g_cl->RSSetViewports(1, &vp);
    g_cl->RSSetScissorRects(1, &sc);
    g_cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_cl->DrawInstanced(3, 1, 0, 0);
    {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = rt;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_cl->ResourceBarrier(1, &b);
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    D3D12_RESOURCE_DESC td = rt->GetDesc();
    g_dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &total);
    ID3D12Resource* rb = makeBuffer(total, D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
    EXPECT(rb != nullptr, "readback buffer create failed");
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = rt;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = rb;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    g_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    EXPECT(execAndWait(), "execute/wait failed");

    void* m = nullptr;
    CK(rb->Map(0, nullptr, &m), "Map(readback)");
    const BYTE* c = (const BYTE*)m + fp.Footprint.RowPitch * (H / 2) + (W / 2) * 4;
    const BYTE* k = (const BYTE*)m;
    printf(T_TAG ": center=(%u,%u,%u,%u) corner=(%u,%u,%u,%u) want center (128,64,64), corner (128,0,128) +-3\n",
           c[0], c[1], c[2], c[3], k[0], k[1], k[2], k[3]);
    auto nearv = [](unsigned v, unsigned want, unsigned tol) {
        return (v >= want ? v - want : want - v) <= tol;
    };
    const bool ok = nearv(c[0], 128, 3) && nearv(c[1], 64, 3) &&
                    nearv(c[2], 64, 3) && c[3] == 255 && nearv(k[0], 128, 3) &&
                    nearv(k[1], 0, 3) && nearv(k[2], 128, 3);
    rb->Unmap(0, nullptr);
    EXPECT(ok, "SV_Barycentrics draw produced the wrong pixels");

    T_PASS();
    return 0;
}
