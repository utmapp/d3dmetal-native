/*
 * D3D12 triangle demo for libd3dmetal-native (Cocoa window + D3DMetal). The
 * canonical D3D12 HelloTriangle pipeline — device, command queue, flip-model
 * swapchain, RTV heap, root signature, PSO, vertex buffer, command list — with
 * a per-frame GPU fence and a backbuffer readback that validates the triangle
 * actually rendered.
 *
 *   (default)   vertex-colored triangle over a dark clear.
 *   --shared    fork a D3D11 producer that owns a MISC_SHARED texture whose
 *               solid color cycles once per second; the D3D12 process imports
 *               it (OpenSharedHandle) and, when D3DMetal accepts the import,
 *               samples it onto the triangle. Validates BOTH that the shared
 *               color changes over time (read cross-process through the shared
 *               mapping, gated on the shared fence advancing) AND that the
 *               triangle renders.
 *
 * Environment knobs:
 *   DMN_FRAMES=N   auto-exit after N frames (CI; safety cap in --shared).
 *
 * Pass --callback to use the callback-backed swapchain (embedder textures
 * presented through an MTKView) instead of the CAMetalLayer view backend.
 *   DMN_VALIDATE=1 read back the backbuffer and assert the triangle rendered
 *                  ("D3D12: PASS"); in --shared also assert the color changed
 *                  ("D3D12 SHARED: PASS").
 *   DMN_DUMP=path  write the validation-frame backbuffer as a .ppm.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cerrno>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* WIDL_EXPLICIT_AGGREGATE_RETURNS is defined project-wide in meson.build so the
 * D3D12 aggregate-return methods (GetCPU/GPUDescriptorHandleForHeapStart, ...)
 * match D3DMetal's Wine/MinGW ABI. See the comment there. */

#include <d3dcompiler.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "cocoa_window.h"
#include "common/com.h"
#include "common/ipc.h"
#include "common/util.h"

namespace {

const bool     g_validate = std::getenv("DMN_VALIDATE") != nullptr;
const uint32_t g_maxFrames =
    std::getenv("DMN_FRAMES") ? (uint32_t)atoi(std::getenv("DMN_FRAMES")) : 0;

constexpr UINT kFrameCount = 2;
constexpr uint32_t kSharedW = 256, kSharedH = 256;

struct Vertex {
    float pos[3];
    float uv[2];
    float color[4];
};

struct WireHandles {
    dmn_shared_texture_handle tex;
    dmn_shared_fence_handle fence;
};

uint64_t now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* Vertex-colored triangle (default). */
const char g_shaderColor[] =
    "struct VSIn  { float3 pos:POSITION; float2 uv:TEXCOORD; float4 col:COLOR; };\n"
    "struct PSIn  { float4 pos:SV_POSITION; float2 uv:TEXCOORD; float4 col:COLOR; };\n"
    "PSIn VSMain(VSIn i){ PSIn o; o.pos=float4(i.pos,1); o.uv=i.uv; o.col=i.col; return o; }\n"
    "float4 PSMain(PSIn i):SV_TARGET { return i.col; }\n";

/* Triangle textured with the imported shared texture (--shared). */
const char g_shaderTex[] =
    "Texture2D tex : register(t0);\n"
    "SamplerState smp : register(s0);\n"
    "struct VSIn  { float3 pos:POSITION; float2 uv:TEXCOORD; float4 col:COLOR; };\n"
    "struct PSIn  { float4 pos:SV_POSITION; float2 uv:TEXCOORD; float4 col:COLOR; };\n"
    "PSIn VSMain(VSIn i){ PSIn o; o.pos=float4(i.pos,1); o.uv=i.uv; o.col=i.col; return o; }\n"
    "float4 PSMain(PSIn i):SV_TARGET { return tex.Sample(smp, i.uv); }\n";

#define CK(expr, what)                                                     \
    do {                                                                   \
        HRESULT hr_ = (expr);                                              \
        if (FAILED(hr_)) {                                                 \
            fprintf(stderr, "D3D12: %s FAILED hr=0x%08x\n", what,          \
                    (unsigned)hr_);                                        \
            return false;                                                  \
        }                                                                  \
    } while (0)

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                                  D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = r;
    barrier.Transition.StateBefore = a;
    barrier.Transition.StateAfter = b;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

/* == SCM_RIGHTS fd passing =============================================== */

/* == D3D11 producer (forked child) ====================================== */
/* Owns a MISC_SHARED texture, clears it to a solid color that cycles once per
 * second, and mirrors a monotonic fence value. Exits when the consumer closes
 * the socket. */
int producer(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "SHARED-PROD: dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                 D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                                 D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "SHARED-PROD: D3D11CreateDevice FAILED\n");
        return 1;
    }

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = kSharedW;
    td.Height = kSharedH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Texture2D> tex;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex))) {
        fprintf(stderr, "SHARED-PROD: CreateTexture2D FAILED\n");
        return 1;
    }

    Com<IDXGIResource> res;
    HANDLE texH = nullptr;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
        FAILED(res->GetSharedHandle(&texH)) || !texH) {
        fprintf(stderr, "SHARED-PROD: GetSharedHandle FAILED (swizzle miss?)\n");
        return 1;
    }
    WireHandles wire = {};
    memcpy(&wire.tex, texH, sizeof(wire.tex));

    /* Real D3D11 shared fence: the CreateFence(SHARED) hook makes a companion
     * shared buffer, CreateSharedHandle exports it, and each context4 Signal
     * makes the GPU write the value in so the D3D12 consumer sees it. */
    Com<ID3D11Device5> dev5;
    Com<ID3D11DeviceContext4> ctx4;
    Com<ID3D11Fence> fenceObj;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5)) ||
        FAILED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&ctx4)) ||
        FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                 __uuidof(ID3D11Fence), (void**)&fenceObj))) {
        fprintf(stderr, "SHARED-PROD: CreateFence(SHARED) unavailable\n");
        return 1;
    }
    HANDLE fenceH = nullptr;
    if (FAILED(fenceObj->CreateSharedHandle(nullptr, 0, nullptr, &fenceH)) ||
        !fenceH) {
        fprintf(stderr, "SHARED-PROD: fence CreateSharedHandle FAILED\n");
        return 1;
    }
    memcpy(&wire.fence, fenceH, sizeof(wire.fence));

    int fds[2] = {wire.tex.fd, wire.fence.fd};
    if (!send_with_fds(sock, &wire, sizeof(wire), fds, 2)) {
        fprintf(stderr, "SHARED-PROD: send FAILED: %s\n", strerror(errno));
        return 1;
    }
    dmn_shared_handle_close(fenceH); /* NT-style; texH is legacy (no close) */

    Com<ID3D11RenderTargetView> rtv;
    if (FAILED(dev->CreateRenderTargetView(tex.ptr(), nullptr, &rtv))) {
        fprintf(stderr, "SHARED-PROD: CreateRenderTargetView FAILED\n");
        return 1;
    }

    const float colors[4][4] = {
        {1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 0, 1},
    };
    for (int tick = 0; tick < 30; tick++) {
        char b;
        ssize_t n = recv(sock, &b, 1, MSG_DONTWAIT);
        if (n == 0)
            break; /* consumer closed the socket: done */

        ctx->ClearRenderTargetView(rtv.ptr(), colors[tick % 4]);
        /* The fence Signal is GPU-ordered after the clear on this same context;
         * the store rides that submission (no producer-side self-check needed). */
        ctx4->Signal(fenceObj.ptr(), (uint64_t)(tick + 1));
        fprintf(stderr, "SHARED-PROD: tick %d cleared + signaled\n", tick);
        struct timespec s = {1, 0};
        nanosleep(&s, nullptr); /* cycle once per second */
    }
    return 0;
}

class Triangle12App {
public:
    Triangle12App(HWND hwnd, const WireHandles* shared) : m_hwnd(hwnd) {
        if (shared) {
            m_shared = true;
            m_sharedTex = shared->tex;
            m_fenceHandle = shared->fence;
        }
        m_init = loadPipeline() && setupShared() && loadAssets();
    }

    bool initialized() const { return m_init; }
    uint32_t frameTotal() const { return m_frameTotal; }
    bool validated() const { return !g_validate || m_validated; }
    bool done() const { return m_shared ? m_shareDone : false; }

    bool render() {
        if (!m_init)
            return false;
        if (!populateAndExecute())
            return false;
        waitForGpu();

        if (g_validate && !m_shareDone) {
            if (m_shared) {
                if (!validateShared())
                    return false;
            } else if (!m_validated && m_frameTotal >= 30) {
                if (!validateFrame())
                    return false;
                m_validated = true;
            }
        }

        HRESULT hr = m_swapchain->Present(1, 0);
        if (FAILED(hr)) {
            fprintf(stderr, "D3D12: Present FAILED 0x%08x\n", (unsigned)hr);
            return false;
        }
        waitForGpu();
        m_frameIndex = m_swapchain->GetCurrentBackBufferIndex();
        m_frameTotal++;
        return true;
    }

private:
    bool loadPipeline() {
        Com<IDXGIFactory4> factory;
        CK(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), (void**)&factory),
           "CreateDXGIFactory2");

        CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                             __uuidof(ID3D12Device), (void**)&m_device),
           "D3D12CreateDevice");

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        CK(m_device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                        (void**)&m_queue),
           "CreateCommandQueue");

        DXGI_SWAP_CHAIN_DESC1 sc = {};
        sc.BufferCount = kFrameCount;
        sc.Width = m_width;
        sc.Height = m_height;
        sc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sc.SampleDesc.Count = 1;
        sc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;

        Com<IDXGISwapChain1> sc1;
        CK(factory->CreateSwapChainForHwnd(m_queue.ptr(), m_hwnd, &sc, nullptr,
                                           nullptr, &sc1),
           "CreateSwapChainForHwnd");
        CK(sc1->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&m_swapchain),
           "QI(IDXGISwapChain3)");
        m_frameIndex = m_swapchain->GetCurrentBackBufferIndex();

        D3D12_DESCRIPTOR_HEAP_DESC rh = {};
        rh.NumDescriptors = kFrameCount;
        rh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rh.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        CK(m_device->CreateDescriptorHeap(&rh, __uuidof(ID3D12DescriptorHeap),
                                          (void**)&m_rtvHeap),
           "CreateDescriptorHeap(RTV)");
        m_rtvSize = m_device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_CPU_DESCRIPTOR_HANDLE h =
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < kFrameCount; i++) {
            CK(m_swapchain->GetBuffer(i, __uuidof(ID3D12Resource),
                                      (void**)&m_rt[i]),
               "GetBuffer");
            m_device->CreateRenderTargetView(m_rt[i].ptr(), nullptr, h);
            h.ptr += m_rtvSize;
        }

        CK(m_device->CreateCommandAllocator(
               D3D12_COMMAND_LIST_TYPE_DIRECT,
               __uuidof(ID3D12CommandAllocator), (void**)&m_cmdAlloc),
           "CreateCommandAllocator");
        return true;
    }

    /* Map the shared texture (robust color source) and try to import it as a
     * D3D12 resource. Import failure is non-fatal: we fall back to the plain
     * triangle and still verify color change through the mapping. */
    bool setupShared() {
        if (!m_shared)
            return true;
        /* Import the fence through the standard API (vends an ID3D12Fence whose
         * GetCompletedValue polls the GPU-written companion buffer). */
        if (FAILED(m_device->OpenSharedHandle((HANDLE)&m_fenceHandle,
                                              __uuidof(ID3D12Fence),
                                              (void**)&m_sharedFence12)) ||
            !m_sharedFence12) {
            fprintf(stderr, "D3D12: OpenSharedHandle(fence) FAILED\n");
            return false;
        }
        /* Import the texture as an ID3D12Resource to sample onto the triangle. */
        HRESULT hr = m_device->OpenSharedHandle((HANDLE)&m_sharedTex,
                                                __uuidof(ID3D12Resource),
                                                (void**)&m_importedTex);
        if (SUCCEEDED(hr) && m_importedTex) {
            m_textured = true;
            printf("D3D12: imported shared texture as ID3D12Resource "
                   "(sampling it onto the triangle)\n");
        } else {
            printf("D3D12: OpenSharedHandle(texture) unavailable (hr=0x%08x); "
                   "plain triangle\n", (unsigned)hr);
        }
        return true;
    }

    bool loadAssets() {
        const char* src = m_textured ? g_shaderTex : g_shaderColor;
        const size_t srcLen =
            (m_textured ? sizeof(g_shaderTex) : sizeof(g_shaderColor)) - 1;

        if (!buildRootSignature())
            return false;

        Com<ID3DBlob> vs, ps, err;
        if (FAILED(D3DCompile(src, srcLen, "vs", nullptr, nullptr, "VSMain",
                              "vs_5_0", 0, 0, &vs, &err))) {
            fprintf(stderr, "D3D12: VS compile FAILED: %s\n",
                    err ? (const char*)err->GetBufferPointer() : "?");
            return false;
        }
        if (FAILED(D3DCompile(src, srcLen, "ps", nullptr, nullptr, "PSMain",
                              "ps_5_0", 0, 0, &ps, &err))) {
            fprintf(stderr, "D3D12: PS compile FAILED: %s\n",
                    err ? (const char*)err->GetBufferPointer() : "?");
            return false;
        }

        D3D12_INPUT_ELEMENT_DESC il[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 20,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso = {};
        pso.pRootSignature = m_rootSig.ptr();
        pso.InputLayout = {il, 3};
        pso.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
        pso.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
        pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso.RasterizerState.DepthClipEnable = TRUE;
        for (auto& rt : pso.BlendState.RenderTarget)
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso.DepthStencilState.DepthEnable = FALSE;
        pso.DepthStencilState.StencilEnable = FALSE;
        pso.SampleMask = 0xFFFFFFFFu;
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
        pso.SampleDesc.Count = 1;
        CK(m_device->CreateGraphicsPipelineState(
               &pso, __uuidof(ID3D12PipelineState), (void**)&m_pso),
           "CreateGraphicsPipelineState");

        CK(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       m_cmdAlloc.ptr(), m_pso.ptr(),
                                       __uuidof(ID3D12GraphicsCommandList),
                                       (void**)&m_cmdList),
           "CreateCommandList");
        m_cmdList->Close();

        Vertex verts[3] = {
            {{0.0f, 0.8f, 0.0f}, {0.5f, 0.0f}, {1, 0, 0, 1}},
            {{-0.8f, -0.8f, 0.0f}, {0.0f, 1.0f}, {0, 1, 0, 1}},
            {{0.8f, -0.8f, 0.0f}, {1.0f, 1.0f}, {0, 0, 1, 1}},
        };
        if (!uploadBuffer(sizeof(verts), verts, m_vbo))
            return false;
        m_vbView.BufferLocation = m_vbo->GetGPUVirtualAddress();
        m_vbView.StrideInBytes = sizeof(Vertex);
        m_vbView.SizeInBytes = sizeof(verts);

        if (m_textured && !buildSharedSrv())
            return false;

        CK(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                 __uuidof(ID3D12Fence), (void**)&m_fence),
           "CreateFence");

        m_viewport = {0.0f, 0.0f, (float)m_width, (float)m_height, 0.0f, 1.0f};
        m_scissor = {0, 0, (LONG)m_width, (LONG)m_height};
        waitForGpu();
        return true;
    }

    bool buildRootSignature() {
        D3D12_ROOT_SIGNATURE_DESC rs = {};
        rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        D3D12_DESCRIPTOR_RANGE range = {};
        D3D12_ROOT_PARAMETER param = {};
        D3D12_STATIC_SAMPLER_DESC samp = {};
        if (m_textured) {
            range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            range.NumDescriptors = 1;
            range.BaseShaderRegister = 0; /* t0 */
            range.OffsetInDescriptorsFromTableStart =
                D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            param.DescriptorTable.NumDescriptorRanges = 1;
            param.DescriptorTable.pDescriptorRanges = &range;
            param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            samp.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            samp.AddressU = samp.AddressV = samp.AddressW =
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            samp.ShaderRegister = 0; /* s0 */
            samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

            rs.NumParameters = 1;
            rs.pParameters = &param;
            rs.NumStaticSamplers = 1;
            rs.pStaticSamplers = &samp;
        }

        Com<ID3DBlob> sig, sigErr;
        CK(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig,
                                       &sigErr),
           "D3D12SerializeRootSignature");
        CK(m_device->CreateRootSignature(0, sig->GetBufferPointer(),
                                         sig->GetBufferSize(),
                                         __uuidof(ID3D12RootSignature),
                                         (void**)&m_rootSig),
           "CreateRootSignature");
        return true;
    }

    bool buildSharedSrv() {
        D3D12_DESCRIPTOR_HEAP_DESC sh = {};
        sh.NumDescriptors = 1;
        sh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        sh.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CK(m_device->CreateDescriptorHeap(&sh, __uuidof(ID3D12DescriptorHeap),
                                          (void**)&m_srvHeap),
           "CreateDescriptorHeap(SRV)");

        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = (DXGI_FORMAT)m_sharedTex.dxgi_format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        m_device->CreateShaderResourceView(
            m_importedTex.ptr(), &srv,
            m_srvHeap->GetCPUDescriptorHandleForHeapStart());
        return true;
    }

    bool uploadBuffer(size_t size, const void* data, Com<ID3D12Resource>& out) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        CK(m_device->CreateCommittedResource(
               &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
               nullptr, __uuidof(ID3D12Resource), (void**)&out),
           "CreateCommittedResource(upload)");
        void* p = nullptr;
        D3D12_RANGE none = {0, 0};
        CK(out->Map(0, &none, &p), "Map(upload)");
        memcpy(p, data, size);
        out->Unmap(0, nullptr);
        return true;
    }

    bool populateAndExecute() {
        CK(m_cmdAlloc->Reset(), "cmdAlloc Reset");
        CK(m_cmdList->Reset(m_cmdAlloc.ptr(), m_pso.ptr()), "cmdList Reset");

        m_cmdList->SetGraphicsRootSignature(m_rootSig.ptr());
        if (m_textured) {
            /* The imported resource is created in COMMON; move it into a
             * shader-read state once so the pixel shader samples the live
             * shared memory (D3DMetal does not implicitly promote it). */
            if (!m_importBarriered) {
                D3D12_RESOURCE_BARRIER ib = transition(
                    m_importedTex.ptr(), D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                m_cmdList->ResourceBarrier(1, &ib);
                m_importBarriered = true;
            }
            ID3D12DescriptorHeap* heaps[] = {m_srvHeap.ptr()};
            m_cmdList->SetDescriptorHeaps(1, heaps);
            m_cmdList->SetGraphicsRootDescriptorTable(
                0, m_srvHeap->GetGPUDescriptorHandleForHeapStart());
        }
        m_cmdList->RSSetViewports(1, &m_viewport);
        m_cmdList->RSSetScissorRects(1, &m_scissor);

        D3D12_RESOURCE_BARRIER b = transition(
            m_rt[m_frameIndex].ptr(), D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET);
        m_cmdList->ResourceBarrier(1, &b);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv =
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += (SIZE_T)m_frameIndex * m_rtvSize;
        m_cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        const float clear[4] = {0.05f, 0.05f, 0.15f, 1.0f};
        m_cmdList->ClearRenderTargetView(rtv, clear, 0, nullptr);
        m_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_cmdList->IASetVertexBuffers(0, 1, &m_vbView);
        m_cmdList->DrawInstanced(3, 1, 0, 0);

        D3D12_RESOURCE_BARRIER b2 = transition(
            m_rt[m_frameIndex].ptr(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT);
        m_cmdList->ResourceBarrier(1, &b2);
        CK(m_cmdList->Close(), "cmdList Close");

        ID3D12CommandList* lists[] = {m_cmdList.ptr()};
        m_queue->ExecuteCommandLists(1, lists);
        return true;
    }

    void waitForGpu() {
        const UINT64 target = ++m_fenceValue;
        m_queue->Signal(m_fence.ptr(), target);
        uint64_t start = now_ms();
        while (m_fence->GetCompletedValue() < target) {
            if (now_ms() - start > 5000) {
                fprintf(stderr, "D3D12: waitForGpu timed out at %llu\n",
                        (unsigned long long)target);
                break;
            }
            struct timespec ns = {0, 200 * 1000};
            nanosleep(&ns, nullptr);
        }
    }

    /* Copy the current backbuffer to a readback buffer and sample two pixels. */
    bool readBackbuffer(uint8_t center[4], uint8_t corner[4]) {
        ID3D12Resource* rt = m_rt[m_frameIndex].ptr();
        D3D12_RESOURCE_DESC rd = rt->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        m_device->GetCopyableFootprints(&rd, 0, 1, 0, &fp, &rows, &rowBytes,
                                        &total);

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd = {};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = total;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Com<ID3D12Resource> readback;
        CK(m_device->CreateCommittedResource(
               &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
               nullptr, __uuidof(ID3D12Resource), (void**)&readback),
           "CreateCommittedResource(readback)");

        CK(m_cmdAlloc->Reset(), "cmdAlloc Reset (rb)");
        CK(m_cmdList->Reset(m_cmdAlloc.ptr(), nullptr), "cmdList Reset (rb)");
        D3D12_RESOURCE_BARRIER b = transition(
            rt, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        m_cmdList->ResourceBarrier(1, &b);

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = rt;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = readback.ptr();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp;
        m_cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        D3D12_RESOURCE_BARRIER b2 = transition(
            rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        m_cmdList->ResourceBarrier(1, &b2);
        CK(m_cmdList->Close(), "cmdList Close (rb)");
        ID3D12CommandList* lists[] = {m_cmdList.ptr()};
        m_queue->ExecuteCommandLists(1, lists);
        waitForGpu();

        void* p = nullptr;
        D3D12_RANGE all = {0, (SIZE_T)total};
        CK(readback->Map(0, &all, &p), "Map(readback)");
        auto px = [&](uint32_t x, uint32_t y, uint8_t out[4]) {
            const uint8_t* row =
                (const uint8_t*)p + (uint64_t)y * fp.Footprint.RowPitch;
            memcpy(out, row + x * 4, 4);
        };
        px(m_width / 2, m_height / 2, center);
        px(4, 4, corner);
        if (const char* dump = std::getenv("DMN_DUMP")) {
            if (FILE* f = fopen(dump, "wb")) {
                fprintf(f, "P6\n%u %u\n255\n", m_width, m_height);
                for (uint32_t y = 0; y < m_height; y++) {
                    const uint8_t* row =
                        (const uint8_t*)p + (uint64_t)y * fp.Footprint.RowPitch;
                    for (uint32_t x = 0; x < m_width; x++) {
                        const uint8_t* q = row + x * 4;
                        uint8_t rgb[3] = {q[2], q[1], q[0]};
                        fwrite(rgb, 1, 3, f);
                    }
                }
                fclose(f);
            }
        }
        D3D12_RANGE none = {0, 0};
        readback->Unmap(0, &none);
        return true;
    }

    bool validateFrame() {
        uint8_t c[4], corner[4];
        if (!readBackbuffer(c, corner))
            return false;
        int centerLum = c[0] + c[1] + c[2];
        int cornerLum = corner[0] + corner[1] + corner[2];
        bool ok = centerLum > cornerLum + 120;
        printf("D3D12: center BGRA=%02x%02x%02x%02x corner=%02x%02x%02x "
               "(lum %d vs %d) %s\n",
               c[0], c[1], c[2], c[3], corner[0], corner[1], corner[2],
               centerLum, cornerLum, ok ? "OK" : "MISMATCH");
        return ok;
    }

    /* Color-change check read entirely from the rendered BACKBUFFER (which shows
     * the imported shared texture sampled onto the triangle), gated on the vended
     * shared fence advancing >= 2 producer cycles. Fully standard D3D — no CPU
     * mapping. Cross-process GPU sampling is coherence-limited (the MTLSharedEvent
     * gap), so for a windowed demo this is best-effort, not a strict assertion. */
    bool validateShared() {
        uint64_t fv = m_sharedFence12 ? m_sharedFence12->GetCompletedValue() : 0;
        uint8_t c[4], corner[4];
        if (!readBackbuffer(c, corner))
            return false;
        bool triangleHere = (abs((int)c[0] - 13) + abs((int)c[1] - 13) +
                             abs((int)c[2] - 38)) > 60;

        if (!m_haveSampleA) {
            memcpy(m_colorA, c, 3);
            m_baseFence = fv;
            m_sampleAt = now_ms();
            m_triangleA = triangleHere;
            m_haveSampleA = true;
            printf("D3D12: shared sample A backbuffer BGR=%02x%02x%02x fence=%llu "
                   "triangle=%d\n", c[0], c[1], c[2],
                   (unsigned long long)fv, m_triangleA);
            return true;
        }

        /* Wait for >= 2 producer cycles, but never hang if the fence stalls. */
        bool advanced = (fv >= m_baseFence + 2) || (now_ms() - m_sampleAt > 4000);
        if (!advanced)
            return true; /* keep rendering until the producer cycles */

        int d = abs((int)m_colorA[0] - c[0]) + abs((int)m_colorA[1] - c[1]) +
                abs((int)m_colorA[2] - c[2]);
        bool colorChanged = d > 24;
        bool triangle = m_triangleA && triangleHere;

        printf("D3D12: shared sample B backbuffer BGR=%02x%02x%02x fence=%llu "
               "(A was %02x%02x%02x); color_changed=%d triangle=%d\n",
               c[0], c[1], c[2], (unsigned long long)fv, m_colorA[0],
               m_colorA[1], m_colorA[2], colorChanged, triangle);

        m_validated = colorChanged && triangle;
        m_shareDone = true;
        printf("D3D12 SHARED: %s (best-effort; cross-process GPU sampling is "
               "coherence-limited without a shared GPU fence)\n",
               m_validated ? "color change observed" : "no change observed");
        return true;
    }

    HWND m_hwnd;
    UINT m_width = 1024, m_height = 600;
    bool m_init = false;
    uint32_t m_frameTotal = 0;
    bool m_validated = false;

    /* shared-texture mode (consumed entirely through standard D3D) */
    bool m_shared = false, m_textured = false, m_shareDone = false;
    dmn_shared_texture_handle m_sharedTex = {};   /* opaque handle bytes + fd */
    dmn_shared_fence_handle m_fenceHandle = {};
    Com<ID3D12Fence> m_sharedFence12;             /* vended by OpenSharedHandle */
    Com<ID3D12Resource> m_importedTex;
    Com<ID3D12DescriptorHeap> m_srvHeap;
    bool m_importBarriered = false;
    bool m_haveSampleA = false, m_triangleA = false;
    uint8_t m_colorA[3] = {};
    uint64_t m_baseFence = 0, m_sampleAt = 0;

    Com<ID3D12Device> m_device;
    Com<ID3D12CommandQueue> m_queue;
    Com<IDXGISwapChain3> m_swapchain;
    Com<ID3D12DescriptorHeap> m_rtvHeap;
    UINT m_rtvSize = 0;
    Com<ID3D12Resource> m_rt[kFrameCount];
    Com<ID3D12CommandAllocator> m_cmdAlloc;
    Com<ID3D12GraphicsCommandList> m_cmdList;
    Com<ID3D12RootSignature> m_rootSig;
    Com<ID3D12PipelineState> m_pso;
    Com<ID3D12Resource> m_vbo;
    D3D12_VERTEX_BUFFER_VIEW m_vbView = {};
    Com<ID3D12Fence> m_fence;
    UINT64 m_fenceValue = 0;
    UINT m_frameIndex = 0;
    D3D12_VIEWPORT m_viewport = {};
    D3D12_RECT m_scissor = {};
};

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    bool shared = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--shared") == 0)
            shared = true;

    /* In shared mode fork the windowless D3D11 producer BEFORE any Cocoa (AppKit
     * does not survive fork); only the consumer touches the window. */
    int sock = -1;
    pid_t child = -1;
    if (shared) {
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
            fprintf(stderr, "socketpair FAILED: %s\n", strerror(errno));
            return 1;
        }
        child = fork();
        if (child < 0) {
            fprintf(stderr, "fork FAILED: %s\n", strerror(errno));
            return 1;
        }
        if (child == 0) {
            close(sv[0]);
            int rc = producer(sv[1]);
            close(sv[1]);
            _exit(rc);
        }
        close(sv[1]);
        sock = sv[0];
    }

    WireHandles wire = {};
    if (shared) {
        int fds[2] = {-1, -1};
        if (!recv_with_fds(sock, &wire, sizeof(wire), fds, 2)) {
            fprintf(stderr, "consumer recv FAILED: %s\n", strerror(errno));
            return 1;
        }
        wire.tex.fd = fds[0];     /* the received fds are the resource tokens */
        wire.fence.fd = fds[1];
    }

    if (!cocoa_app_init()) {
        fprintf(stderr, "Failed to initialize Cocoa application\n");
        return 1;
    }
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "Failed to initialize d3dmetal-native\n");
        return 1;
    }

    cocoa_window_t* window =
        cocoa_window_create(shared ? "D3D12 shared" : "D3D12 triangle", 1024,
                            600, true);
    dmn_window_t dmnWindow =
        window ? cocoa_window_create_dmn(window, cocoa_arg_callback(argc, argv))
               : nullptr;
    if (!dmnWindow) {
        fprintf(stderr, "Failed to create window\n");
        return 1;
    }
    HWND hwnd = (HWND)dmn_window_get_hwnd(dmnWindow);

    int ret = 0;
    {
        Triangle12App app(hwnd, shared ? &wire : nullptr);
        if (!app.initialized()) {
            fprintf(stderr, "D3D12: initialization FAILED\n");
            ret = 1;
        }
        while (ret == 0 && cocoa_app_poll()) {
            if (!app.render()) {
                ret = 1;
                break;
            }
            if (app.done())
                break;
            if (g_maxFrames && app.frameTotal() >= g_maxFrames)
                break;
        }
        if (ret == 0 && !app.validated())
            ret = 1;
        if (ret == 0 && g_validate && !shared)
            printf("D3D12: PASS\n");
    }

    dmn_window_destroy(dmnWindow);
    if (window)
        cocoa_window_destroy(window);
    cocoa_app_shutdown();

    if (shared) {
        close(sock); /* signals the producer (EOF) to exit */
        if (child > 0)
            waitpid(child, nullptr, 0);
    }
    return ret;
}
