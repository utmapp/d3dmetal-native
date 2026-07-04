/*
 * Cross-process, cross-API, on-screen end-to-end demo.
 *
 *   Process 1 (child, exec'd): D3D12. Owns a shared texture and a shared fence.
 *     Every interval a compute shader fills the whole texture with a solid color
 *     (cycling red/green/blue), then the queue signals the fence.
 *   Process 2 (parent): D3D11 + a Cocoa window. Imports the texture (as an SRV)
 *     and the fence through the standard D3D APIs, and each frame draws a
 *     triangle that samples the shared texture — so the on-screen triangle
 *     changes color in lock-step with the D3D12 producer. The fence is waited on
 *     (GPU Wait + GetCompletedValue) so the sample follows the producer's fill.
 *
 * The consumer also does a fence-gated readback of the imported texture and
 * asserts the color actually cycles ("VERIFY: PASS"), so the data path is
 * checkable headlessly even where on-screen Present is unavailable.
 *
 * Pass --callback to use the callback-backed swapchain (embedder textures
 * presented through an MTKView) instead of the CAMetalLayer view backend.
 *
 * Env: DMN_INTERVAL_MS (producer color period, default 1000),
 *      DMN_FRAMES=N (consumer auto-exit after N frames).
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

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

constexpr uint32_t kTexW = 256, kTexH = 256;

struct Rgba { float r, g, b, a; };
const Rgba kColors[3] = {
    {1, 0, 0, 1}, /* red   */
    {0, 1, 0, 1}, /* green */
    {0, 0, 1, 1}, /* blue  */
};

struct WireHandles {
    dmn_shared_texture_handle tex;
    dmn_shared_fence_handle fence;
};

uint64_t now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

/* == Producer: D3D12 compute fills a shared texture, signals a shared fence == */

const char* kComputeCS =
    "RWTexture2D<float4> tex : register(u0);\n"
    "cbuffer C : register(b0) { float4 gColor; uint gW; uint gH; };\n"
    "[numthreads(8,8,1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) {\n"
    "    if (id.x < gW && id.y < gH) tex[id.xy] = gColor;\n"
    "}\n";

#define PCK(expr, what)                                                      \
    do {                                                                     \
        HRESULT hr_ = (expr);                                                \
        if (FAILED(hr_)) {                                                   \
            fprintf(stderr, "PROD: %s FAILED 0x%08x\n", what, (unsigned)hr_);\
            return 1;                                                        \
        }                                                                    \
    } while (0)

int producer(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "PROD: dmn_init FAILED\n");
        return 1;
    }
    const int intervalMs =
        getenv("DMN_INTERVAL_MS") ? atoi(getenv("DMN_INTERVAL_MS")) : 1000;

    Com<ID3D12Device> device;
    PCK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
                          (void**)&device), "D3D12CreateDevice");
    Com<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    PCK(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue),
        "CreateCommandQueue");
    Com<ID3D12CommandAllocator> alloc;
    PCK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       __uuidof(ID3D12CommandAllocator), (void**)&alloc),
        "CreateCommandAllocator");
    Com<ID3D12GraphicsCommandList> cmd;
    PCK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.ptr(),
                                  nullptr, __uuidof(ID3D12GraphicsCommandList),
                                  (void**)&cmd), "CreateCommandList");
    cmd->Close();

    /* Shared, UAV-writable texture. */
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = kTexW;
    td.Height = kTexH;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Com<ID3D12Resource> tex;
    PCK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &td,
                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        __uuidof(ID3D12Resource), (void**)&tex),
        "CreateCommittedResource(shared tex)");

    /* Private UAV texture the compute shader fills; then copied into the shared
     * texture. (A compute UAV write straight to the *substituted* shared texture
     * does not reach its shared backing under D3DMetal — same as shared buffers
     * — but a copy/blit into it does, like a render clear.) */
    Com<ID3D12Resource> priv;
    PCK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                        __uuidof(ID3D12Resource), (void**)&priv),
        "CreateCommittedResource(private tex)");

    Com<ID3D12Fence> fence;
    PCK(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
                            (void**)&fence), "CreateFence(SHARED)");

    /* Shader-visible UAV descriptor heap + the texture UAV. */
    Com<ID3D12DescriptorHeap> uavHeap;
    D3D12_DESCRIPTOR_HEAP_DESC dh = {};
    dh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dh.NumDescriptors = 1;
    dh.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    PCK(device->CreateDescriptorHeap(&dh, __uuidof(ID3D12DescriptorHeap), (void**)&uavHeap),
        "CreateDescriptorHeap");
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(priv.ptr(), nullptr, &ud,
                                      uavHeap->GetCPUDescriptorHandleForHeapStart());

    /* Root signature: [0] UAV table (u0), [1] 6 root constants (b0). */
    D3D12_DESCRIPTOR_RANGE range = {};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &range;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.ShaderRegister = 0;
    params[1].Constants.Num32BitValues = 6;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 2;
    rsd.pParameters = params;
    Com<ID3DBlob> rsBlob, rsErr;
    PCK(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr),
        "SerializeRootSignature");
    Com<ID3D12RootSignature> rootSig;
    PCK(device->CreateRootSignature(0, rsBlob->GetBufferPointer(),
                                    rsBlob->GetBufferSize(),
                                    __uuidof(ID3D12RootSignature), (void**)&rootSig),
        "CreateRootSignature");

    Com<ID3DBlob> csBlob, csErr;
    if (FAILED(D3DCompile(kComputeCS, strlen(kComputeCS), "fill_cs", nullptr,
                          nullptr, "main", "cs_5_0", 0, 0, &csBlob, &csErr))) {
        fprintf(stderr, "PROD: compute compile: %s\n",
                csErr ? (const char*)csErr->GetBufferPointer() : "?");
        return 1;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = rootSig.ptr();
    pd.CS.pShaderBytecode = csBlob->GetBufferPointer();
    pd.CS.BytecodeLength = csBlob->GetBufferSize();
    Com<ID3D12PipelineState> pso;
    PCK(device->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&pso),
        "CreateComputePipelineState");

    /* Export both handles and ship them to the consumer. */
    HANDLE texH = nullptr, fenceH = nullptr;
    PCK(device->CreateSharedHandle(tex.ptr(), nullptr, 0, nullptr, &texH),
        "CreateSharedHandle(tex)");
    PCK(device->CreateSharedHandle(fence.ptr(), nullptr, 0, nullptr, &fenceH),
        "CreateSharedHandle(fence)");
    WireHandles wire = {};
    memcpy(&wire.tex, texH, sizeof(wire.tex));
    memcpy(&wire.fence, fenceH, sizeof(wire.fence));
    int fds[2] = {wire.tex.fd, wire.fence.fd};
    if (!send_with_fds(sock, &wire, sizeof(wire), fds, 2)) {
        fprintf(stderr, "PROD: send FAILED: %s\n", strerror(errno));
        return 1;
    }
    printf("PROD: D3D12 shared %ux%u texture + fence exported (interval %dms)\n",
           kTexW, kTexH, intervalMs);
    dmn_shared_handle_close(texH);   /* NT-style handles; PODs + fds shipped */
    dmn_shared_handle_close(fenceH);

    uint64_t v = 0;
    for (int frame = 0;; frame++) {
        char b;
        if (recv(sock, &b, 1, MSG_DONTWAIT) == 0)
            break; /* consumer exited */

        const Rgba& col = kColors[frame % 3];
        PCK(alloc->Reset(), "alloc Reset");
        PCK(cmd->Reset(alloc.ptr(), pso.ptr()), "cmd Reset");

        auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                           D3D12_RESOURCE_STATES b_) {
            D3D12_RESOURCE_BARRIER bar = {};
            bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bar.Transition.pResource = r;
            bar.Transition.StateBefore = a;
            bar.Transition.StateAfter = b_;
            bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd->ResourceBarrier(1, &bar);
        };
        if (frame == 0)
            barrier(tex.ptr(), D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST);

        /* Compute-fill the private texture with the solid color. */
        cmd->SetComputeRootSignature(rootSig.ptr());
        ID3D12DescriptorHeap* heaps[] = {uavHeap.ptr()};
        cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootDescriptorTable(0, uavHeap->GetGPUDescriptorHandleForHeapStart());
        uint32_t consts[6];
        memcpy(&consts[0], &col, 16);
        consts[4] = kTexW;
        consts[5] = kTexH;
        cmd->SetComputeRoot32BitConstants(1, 6, consts, 0);
        cmd->Dispatch((kTexW + 7) / 8, (kTexH + 7) / 8, 1);

        /* Copy the filled private texture into the shared texture. */
        barrier(priv.ptr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->CopyResource(tex.ptr(), priv.ptr());
        barrier(priv.ptr(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        PCK(cmd->Close(), "cmd Close");
        ID3D12CommandList* lists[] = {cmd.ptr()};
        queue->ExecuteCommandLists(1, lists);
        PCK(queue->Signal(fence.ptr(), ++v), "queue Signal");

        uint64_t t0 = now_ms();
        while (fence->GetCompletedValue() < v && now_ms() - t0 < 5000) {
            struct timespec ns = {0, 500 * 1000};
            nanosleep(&ns, nullptr);
        }
        struct timespec s = {intervalMs / 1000, (long)(intervalMs % 1000) * 1000000};
        nanosleep(&s, nullptr);
    }
    return 0;
}

/* == Consumer: D3D11 window, triangle samples the shared texture ============ */

const char* kVS =
    "float4 main(float2 pos : POS) : SV_POSITION { return float4(pos, 0, 1); }\n";
const char* kPS =
    "Texture2D tex0 : register(t0);\n"
    "SamplerState smp : register(s0);\n"
    "float4 main() : SV_TARGET { return tex0.Sample(smp, float2(0.5, 0.5)); }\n";

#define CCK(expr, what)                                                      \
    do {                                                                     \
        HRESULT hr_ = (expr);                                                \
        if (FAILED(hr_)) {                                                   \
            fprintf(stderr, "CONS: %s FAILED 0x%08x\n", what, (unsigned)hr_);\
            return false;                                                    \
        }                                                                    \
    } while (0)

class ConsumerApp {
public:
    ConsumerApp(cocoa_window_t* w, HWND hwnd, const WireHandles& wire)
        : m_win(w), m_hwnd(hwnd), m_wire(wire) {}

    bool init() {
        m_headless = getenv("DMN_HEADLESS") != nullptr;
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
        CCK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                              D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                              D3D11_SDK_VERSION, &m_device, &flo, &m_ctx),
            "D3D11CreateDevice");
        CCK(m_ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&m_ctx4),
            "QI(DeviceContext4)");
        CCK(m_device->QueryInterface(__uuidof(ID3D11Device5), (void**)&m_dev5),
            "QI(Device5)");

        /* Import the shared texture + fence — pure D3D. */
        CCK(m_device->OpenSharedResource((HANDLE)&m_wire.tex,
                                         __uuidof(ID3D11Texture2D), (void**)&m_sharedTex),
            "OpenSharedResource(texture)");
        CCK(m_device->CreateShaderResourceView(m_sharedTex.ptr(), nullptr, &m_srv),
            "CreateShaderResourceView");
        CCK(m_dev5->OpenSharedFence((HANDLE)&m_wire.fence, __uuidof(ID3D11Fence),
                                    (void**)&m_fence), "OpenSharedFence");

        /* Staging copy of the shared texture, for the headless readback check. */
        D3D11_TEXTURE2D_DESC sd = {};
        m_sharedTex->GetDesc(&sd);
        sd.BindFlags = 0;
        sd.MiscFlags = 0;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        CCK(m_device->CreateTexture2D(&sd, nullptr, &m_staging), "staging tex");

        /* Headless mode (CI / no display): import + readback verify only. */
        if (m_headless)
            return true;
        if (!buildSwapchain() || !buildPipeline())
            return false;
        return true;
    }

    /* One frame: GPU-wait the fence to the latest fill, sample it onto a
     * triangle, present. Also drives the headless readback verify. */
    bool frame() {
        /* Use the shared fence via GetCompletedValue (a CPU poll of the GPU-
         * written companion buffer): it tells us the producer has filled up to
         * this value. (The vended fence is our object; a context GPU Wait on it
         * is not valid — GetCompletedValue is the supported consumer path.) */
        uint64_t fv = m_fence->GetCompletedValue();
        verifyStep(fv);
        if (m_headless) { m_frames++; return true; }

        Com<ID3D11Texture2D> back;
        Com<ID3D11RenderTargetView> rtv;
        if (m_swap &&
            SUCCEEDED(m_swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back)) &&
            SUCCEEDED(m_device->CreateRenderTargetView(back.ptr(), nullptr, &rtv))) {
            FLOAT clear[4] = {0.1f, 0.1f, 0.12f, 1};
            m_ctx->OMSetRenderTargets(1, &rtv, nullptr);
            m_ctx->ClearRenderTargetView(rtv.ptr(), clear);
            D3D11_VIEWPORT vp = {0, 0, (float)m_w, (float)m_h, 0, 1};
            m_ctx->RSSetViewports(1, &vp);
            m_ctx->VSSetShader(m_vs.ptr(), nullptr, 0);
            m_ctx->PSSetShader(m_ps.ptr(), nullptr, 0);
            m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_ctx->IASetInputLayout(m_layout.ptr());
            UINT stride = sizeof(float) * 2, off = 0;
            m_ctx->IASetVertexBuffers(0, 1, &m_vbo, &stride, &off);
            ID3D11ShaderResourceView* srvs[] = {m_srv.ptr()};
            m_ctx->PSSetShaderResources(0, 1, srvs);
            ID3D11SamplerState* samps[] = {m_samp.ptr()};
            m_ctx->PSSetSamplers(0, 1, samps);
            m_ctx->Draw(3, 0);
        }
        m_swap->Present(1, 0); /* may be a no-op where there's no display */
        m_frames++;
        return true;
    }

    bool verified() const { return m_verified; }
    uint32_t frames() const { return m_frames; }
    uint64_t lastFence() const { return m_lastFv; }

private:
    /* Read the imported texture's center every time the fence advances; assert
     * the color cycles across producer fills (warm-up-tolerant, since the first
     * cross-process GPU reads of a shared surface are coherence-limited). */
    void verifyStep(uint64_t fv) {
        if (m_verified || fv == m_lastFv)
            return;
        m_lastFv = fv;
        m_ctx->CopyResource(m_staging.ptr(), m_sharedTex.ptr());
        m_ctx->Flush();
        D3D11_MAPPED_SUBRESOURCE m = {};
        if (FAILED(m_ctx->Map(m_staging.ptr(), 0, D3D11_MAP_READ, 0, &m)))
            return;
        const uint8_t* px =
            (const uint8_t*)m.pData + (kTexH / 2) * m.RowPitch + (kTexW / 2) * 4;
        uint8_t r = px[0], g = px[1], b = px[2]; /* R8G8B8A8 */
        m_ctx->Unmap(m_staging.ptr(), 0);

        int idx = -1; /* which pure color, if any */
        if (r > 200 && g < 60 && b < 60) idx = 0;
        else if (r < 60 && g > 200 && b < 60) idx = 1;
        else if (r < 60 && g < 60 && b > 200) idx = 2;
        if (idx < 0)
            return; /* warm-up / stale frame, keep going */

        if (m_seenIdx >= 0 && idx != m_seenIdx)
            m_distinct++;
        m_seenIdx = idx;
        printf("VERIFY: fence=%llu sampled RGB=%02x%02x%02x (color %d), distinct=%d\n",
               (unsigned long long)fv, r, g, b, idx, m_distinct);
        if (m_distinct >= 2) {
            m_verified = true;
            printf("VERIFY: PASS (shared texture cycles color, fence-gated)\n");
        }
    }

    bool buildSwapchain() {
        Com<IDXGIDevice> dxgiDev;
        Com<IDXGIAdapter> adapter;
        Com<IDXGIFactory2> factory;
        CCK(m_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev), "QI dxgi");
        CCK(dxgiDev->GetAdapter(&adapter), "GetAdapter");
        CCK(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory), "GetParent");

        cocoa_window_get_content_size(m_win, &m_w, &m_h);
        DXGI_SWAP_CHAIN_DESC1 sc = {};
        sc.Width = m_w;
        sc.Height = m_h;
        sc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sc.SampleDesc = {1, 0};
        sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sc.BufferCount = 2;
        sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        Com<IDXGISwapChain1> sc1;
        CCK(factory->CreateSwapChainForHwnd(m_device.ptr(), m_hwnd, &sc, nullptr,
                                            nullptr, &sc1), "CreateSwapChainForHwnd");
        CCK(sc1->QueryInterface(__uuidof(IDXGISwapChain3), (void**)&m_swap), "QI swapchain");
        return true;
    }

    bool buildPipeline() {
        Com<ID3DBlob> vs, ps, err;
        CCK(D3DCompile(kVS, strlen(kVS), "vs", nullptr, nullptr, "main", "vs_5_0",
                       0, 0, &vs, &err), "VS compile");
        CCK(D3DCompile(kPS, strlen(kPS), "ps", nullptr, nullptr, "main", "ps_5_0",
                       0, 0, &ps, &err), "PS compile");
        CCK(m_device->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(),
                                         nullptr, &m_vs), "CreateVertexShader");
        CCK(m_device->CreatePixelShader(ps->GetBufferPointer(), ps->GetBufferSize(),
                                        nullptr, &m_ps), "CreatePixelShader");
        D3D11_INPUT_ELEMENT_DESC ie = {"POS", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,
                                       D3D11_INPUT_PER_VERTEX_DATA, 0};
        CCK(m_device->CreateInputLayout(&ie, 1, vs->GetBufferPointer(),
                                        vs->GetBufferSize(), &m_layout), "InputLayout");
        float verts[6] = {-0.6f, -0.6f, 0.0f, 0.7f, 0.6f, -0.6f};
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = sizeof(verts);
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA sr = {verts, 0, 0};
        CCK(m_device->CreateBuffer(&bd, &sr, &m_vbo), "vbo");
        D3D11_SAMPLER_DESC smp = {};
        smp.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        smp.AddressU = smp.AddressV = smp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        CCK(m_device->CreateSamplerState(&smp, &m_samp), "sampler");
        return true;
    }

    cocoa_window_t* m_win;
    HWND m_hwnd;
    WireHandles m_wire;
    uint32_t m_w = 640, m_h = 480, m_frames = 0;
    bool m_headless = false;

    Com<ID3D11Device> m_device;
    Com<ID3D11DeviceContext> m_ctx;
    Com<ID3D11DeviceContext4> m_ctx4;
    Com<ID3D11Device5> m_dev5;
    Com<IDXGISwapChain3> m_swap;
    Com<ID3D11Texture2D> m_sharedTex, m_staging;
    Com<ID3D11ShaderResourceView> m_srv;
    Com<ID3D11Fence> m_fence;
    Com<ID3D11VertexShader> m_vs;
    Com<ID3D11PixelShader> m_ps;
    Com<ID3D11InputLayout> m_layout;
    Com<ID3D11Buffer> m_vbo;
    Com<ID3D11SamplerState> m_samp;

    uint64_t m_lastFv = 0;
    int m_seenIdx = -1, m_distinct = 0;
    bool m_verified = false;
};

bool g_callback_swapchain = false; /* --callback: MTKView presenter */

int consumer(int sock) {
    if (!cocoa_app_init()) {
        fprintf(stderr, "CONS: cocoa_app_init FAILED\n");
        return 1;
    }
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "CONS: dmn_init FAILED\n");
        return 1;
    }

    WireHandles wire = {};
    int fds[2] = {-1, -1};
    if (!recv_with_fds(sock, &wire, sizeof(wire), fds, 2)) {
        fprintf(stderr, "CONS: recv FAILED: %s\n", strerror(errno));
        return 1;
    }
    wire.tex.fd = fds[0];
    wire.fence.fd = fds[1];
    printf("CONS: received shared texture + fence\n");

    const bool headless = getenv("DMN_HEADLESS") != nullptr;
    cocoa_window_t* win = headless ? nullptr
        : cocoa_window_create("D3D12 compute -> D3D11 triangle", 640, 480, true);
    dmn_window_t dmnWin =
        win ? cocoa_window_create_dmn(win, g_callback_swapchain) : nullptr;
    HWND hwnd = dmnWin ? (HWND)dmn_window_get_hwnd(dmnWin) : nullptr;
    fprintf(stderr, "CONS: window=%p dmnWin=%p hwnd=%p headless=%d\n", (void*)win,
            (void*)dmnWin, (void*)hwnd, headless);

    const uint32_t maxFrames =
        getenv("DMN_FRAMES") ? (uint32_t)atoi(getenv("DMN_FRAMES")) : 0;

    int rc = 0;
    {
        ConsumerApp app(win, hwnd, wire);
        if (!app.init()) {
            fprintf(stderr, "CONS: init FAILED\n");
            rc = 1;
        } else {
            printf("CONS: %s, sampling the shared texture onto a triangle\n",
                   headless ? "headless verify" : "window up");
            while (headless || cocoa_app_poll()) {
                if (!app.frame())
                    break;
                if (maxFrames && app.frames() >= maxFrames)
                    break;
                struct timespec ns = {0, 16 * 1000 * 1000}; /* ~60 fps cap */
                nanosleep(&ns, nullptr);
            }
            if (!app.verified()) {
                fprintf(stderr, "VERIFY: FAIL (never observed a color cycle)\n");
                rc = 1;
            }
        }
    }

    if (dmnWin) dmn_window_destroy(dmnWin);
    if (win) cocoa_window_destroy(win);
    cocoa_app_shutdown();
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    g_callback_swapchain = cocoa_arg_callback(argc, argv);
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (argc >= 3 && strcmp(argv[1], "--producer") == 0)
        return producer(atoi(argv[2]));

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "socketpair FAILED: %s\n", strerror(errno));
        return 1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork FAILED\n");
        return 1;
    }
    if (pid == 0) {
        /* Producer runs in a FRESH process (exec) so Metal's compute-pipeline
         * compiler is reachable (it is not, across a bare fork). */
        close(sv[0]);
        char fdbuf[16];
        snprintf(fdbuf, sizeof(fdbuf), "%d", sv[1]);
        execl(argv[0], argv[0], "--producer", fdbuf, (char*)nullptr);
        _exit(127);
    }
    close(sv[1]);
    int rc = consumer(sv[0]);
    close(sv[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    return rc;
}
