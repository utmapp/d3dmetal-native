/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Host twin of the guest's `sm6draw.exe -minprec`: the same fxc-compiled
 * vs_5_0/ps_5_0 pair with min16float vertex inputs and varyings
 * (shaders/sm6sh/, built by buildsm6draw.cmd on the dev box) draws a
 * fullscreen green triangle to an offscreen RT; the readback must be
 * exactly 0xFF00FF00 at the center and the corner.  Passing here means
 * the host backend executes minimum-precision DXBC, which is what
 * Triton's D3D12 SHADER.MinPrecision = 16_BIT claims.
 *
 * Prints "MINPREC: PASS" and exits 0 on success.
 */

#include <cstdio>
#include <cstring>

#include <d3d12.h>
#include <windows.h>

#include <time.h>

#include "d3dmetal_native.h"

#define T_TAG "MINPREC"
#include "common/check.h"

typedef unsigned char BYTE;
#include "shaders/sm6sh/vs50min.h"
#include "shaders/sm6sh/ps50min.h"

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
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts = {};
    CK(g_dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts,
                                  sizeof(opts)),
       "CheckFeatureSupport(OPTIONS)");
    printf(T_TAG ": host MinPrecisionSupport=0x%x\n",
           (unsigned)opts.MinPrecisionSupport);

    ID3D12RootSignature* rs = nullptr;
    {
        D3D12_ROOT_SIGNATURE_DESC rd = {};
        rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
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
        D3D12_INPUT_ELEMENT_DESC elems[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rs;
        pd.VS.pShaderBytecode = g_vs_50min;
        pd.VS.BytecodeLength = sizeof(g_vs_50min);
        pd.PS.pShaderBytecode = g_ps_50min;
        pd.PS.BytecodeLength = sizeof(g_ps_50min);
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
        pd.SampleMask = 0xFFFFFFFFu;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.InputLayout.pInputElementDescs = elems;
        pd.InputLayout.NumElements = 2;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.SampleDesc.Count = 1;
        CK_OK(g_dev->CreateGraphicsPipelineState(&pd,
                                                 __uuidof(ID3D12PipelineState),
                                                 (void**)&pso),
              "CreateGraphicsPipelineState(min16float vs/ps)");
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

    static const float verts[3][6] = {
        {-1.f, -1.f, 0.f, 0.f, 1.f, 0.f},
        {3.f, -1.f, 0.f, 0.f, 1.f, 0.f},
        {-1.f, 3.f, 0.f, 0.f, 1.f, 0.f},
    };
    ID3D12Resource* vb = makeBuffer(sizeof(verts), D3D12_HEAP_TYPE_UPLOAD,
                                    D3D12_RESOURCE_STATE_GENERIC_READ);
    EXPECT(vb != nullptr, "vertex buffer create failed");
    {
        void* m = nullptr;
        CK(vb->Map(0, nullptr, &m), "Map(vb)");
        memcpy(m, verts, sizeof(verts));
        vb->Unmap(0, nullptr);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE rtv =
        rtvHeap->GetCPUDescriptorHandleForHeapStart();
    const float red[4] = {1.f, 0.f, 0.f, 1.f};
    D3D12_VIEWPORT vp = {0.f, 0.f, (float)W, (float)H, 0.f, 1.f};
    D3D12_RECT sc = {0, 0, (LONG)W, (LONG)H};
    D3D12_VERTEX_BUFFER_VIEW vbv = {vb->GetGPUVirtualAddress(),
                                    3 * 6 * sizeof(float), 6 * sizeof(float)};
    g_cl->SetPipelineState(pso);
    g_cl->SetGraphicsRootSignature(rs);
    g_cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    g_cl->ClearRenderTargetView(rtv, red, 0, nullptr);
    g_cl->RSSetViewports(1, &vp);
    g_cl->RSSetScissorRects(1, &sc);
    g_cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    g_cl->IASetVertexBuffers(0, 1, &vbv);
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
    const UINT* row =
        (const UINT*)((const BYTE*)m + fp.Footprint.RowPitch * (H / 2));
    UINT center = row[W / 2];
    UINT corner = ((const UINT*)m)[0];
    rb->Unmap(0, nullptr);
    printf(T_TAG ": center=0x%08X corner=0x%08X (want 0xFF00FF00)\n", center,
           corner);
    EXPECT(center == 0xFF00FF00u && corner == 0xFF00FF00u,
           "min16float draw produced the wrong pixels");

    T_PASS();
    return 0;
}
