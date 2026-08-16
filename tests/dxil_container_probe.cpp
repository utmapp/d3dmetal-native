/*
 * dxil_container_probe: what container shapes does D3DMetal's D3D12 accept
 * for SM 6.0 DXIL shaders?
 *
 * The Windows D3D12 runtime hands a DXIL-consumer UMD the BARE DXIL program
 * part (DxilProgramHeader + bitcode) -- no container, no ISG1/OSG1/PSV0.
 * The guest UMD must re-wrap that into something this backend accepts, so
 * measure exactly how much container is needed:
 *
 *   A  compute PSO + dispatch + readback, FULL dxc container      (control)
 *   B  same, MINIMAL container: header + lone DXIL part, zero digest
 *   C  same, BARE program part (no container at all)
 *   D  graphics PSO create, FULL containers + POSITION/COLOR layout (control)
 *   E  graphics PSO create, MINIMAL containers + same layout
 *   F  graphics PSO create, MINIMAL + synthesized fake-name ISG1 + matching
 *      fake-name layout (only interesting if E fails)
 *   G  graphics draw + readback on the best-passing graphics variant
 *   H  graphics draw with WRONG layout names: proves matching is NAME-based
 *      against the module metadata and that a mismatch is a SILENT no-op PSO
 *   R  dmn_dxil_input_semantics() directly on both container shapes
 *   P  the production contract: NPTA<reg> placeholder layout names resolved
 *      to real semantics by the CreateGraphicsPipelineState hook -- must draw
 *
 * The compute path executes (WaveActiveSum: 32 lanes must read 496) so a
 * lazily-failing PSO cannot masquerade as a pass.  Prints one PASS/FAIL line
 * per case; exit 0 iff the controls (A, D), the minimal-container shapes the
 * guest depends on (B, E, G) and the placeholder contract (P) pass.  C is
 * opt-in (-bare) because a bare program part crashes D3DMetal; F and H are
 * informational (F only runs if E fails; H is expected to fail silently, which
 * is what makes P necessary).
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#include "../src/dmn_dxil_reflect.h"

#include "dxil_probe_shaders.h"

#define FOURCC(a, b, c, d) \
    ((uint32_t)(uint8_t)(a) | ((uint32_t)(uint8_t)(b) << 8) | \
     ((uint32_t)(uint8_t)(c) << 16) | ((uint32_t)(uint8_t)(d) << 24))

static const uint32_t kDxbcMagic = FOURCC('D', 'X', 'B', 'C');
static const uint32_t kDxilFourcc = FOURCC('D', 'X', 'I', 'L');
static const uint32_t kIsg1Fourcc = FOURCC('I', 'S', 'G', '1');

struct Span { const uint8_t *p; uint32_t n; };

static Span
findPart(const void *container, uint32_t cb, uint32_t fourcc)
{
    const uint8_t *base = (const uint8_t *)container;
    Span none = {nullptr, 0};
    if (cb < 32 || *(const uint32_t *)base != kDxbcMagic)
        return none;
    uint32_t nPart = *(const uint32_t *)(base + 28);
    const uint32_t *offs = (const uint32_t *)(base + 32);
    for (uint32_t i = 0; i < nPart; i++) {
        const uint8_t *ph = base + offs[i];
        if (*(const uint32_t *)ph == fourcc) {
            Span s = {ph + 8, *(const uint32_t *)(ph + 4)};
            return s;
        }
    }
    return none;
}

/* Build a container from parts; caller frees.  Digest left zero. */
static uint8_t *
buildContainer(const uint32_t *fourccs, const Span *parts, uint32_t nParts,
               uint32_t *pcb)
{
    uint32_t headerSize = 32 + 4 * nParts;
    uint32_t total = headerSize;
    for (uint32_t i = 0; i < nParts; i++)
        total += 8 + ((parts[i].n + 3) & ~3u);
    uint8_t *c = (uint8_t *)calloc(1, total);
    *(uint32_t *)c = kDxbcMagic;
    *(uint16_t *)(c + 20) = 1; /* MajorVersion */
    *(uint32_t *)(c + 24) = total;
    *(uint32_t *)(c + 28) = nParts;
    uint32_t off = headerSize;
    for (uint32_t i = 0; i < nParts; i++) {
        ((uint32_t *)(c + 32))[i] = off;
        *(uint32_t *)(c + off) = fourccs[i];
        *(uint32_t *)(c + off + 4) = parts[i].n;
        memcpy(c + off + 8, parts[i].p, parts[i].n);
        off += 8 + ((parts[i].n + 3) & ~3u);
    }
    *pcb = total;
    return c;
}

/* Synthesized ISG1 with two elements and caller-chosen names (reg 0/1,
 * float3 masks). */
static uint8_t *
buildFakeIsg1(const char *name0, const char *name1, uint32_t *pcb)
{
    uint32_t l0 = (uint32_t)strlen(name0) + 1, l1 = (uint32_t)strlen(name1) + 1;
    uint32_t cb = 8 + 2 * 32 + l0 + l1;
    cb = (cb + 3) & ~3u;
    uint8_t *d = (uint8_t *)calloc(1, cb);
    *(uint32_t *)d = 2;      /* ParamCount */
    *(uint32_t *)(d + 4) = 8; /* ParamOffset */
    uint32_t nameBase = 8 + 2 * 32;
    for (int i = 0; i < 2; i++) {
        uint8_t *el = d + 8 + i * 32;
        *(uint32_t *)(el + 0) = 0;                     /* Stream */
        *(uint32_t *)(el + 4) = nameBase + (i ? l0 : 0); /* SemanticName */
        *(uint32_t *)(el + 8) = 0;                     /* SemanticIndex */
        *(uint32_t *)(el + 12) = 0;                    /* SystemValue undefined */
        *(uint32_t *)(el + 16) = 3;                    /* CompType float32 */
        *(uint32_t *)(el + 20) = (uint32_t)i;          /* Register */
        el[24] = 0x7;                                  /* Mask xyz */
        el[25] = 0x7;
    }
    memcpy(d + nameBase, name0, l0);
    memcpy(d + nameBase + l0, name1, l1);
    *pcb = cb;
    return d;
}

static ID3D12Device *g_dev;
static ID3D12CommandQueue *g_q;
static ID3D12CommandAllocator *g_alloc;
static ID3D12GraphicsCommandList *g_cl;
static ID3D12Fence *g_fence;
static uint64_t g_fv;

static bool
execAndWait(void)
{
    if (FAILED(g_cl->Close()))
        return false;
    ID3D12CommandList *ls[] = {g_cl};
    g_q->ExecuteCommandLists(1, ls);
    ++g_fv;
    if (FAILED(g_q->Signal(g_fence, g_fv)))
        return false;
    {
        /* Poll like the other host tests (no Win32 event plumbing here). */
        int spins = 0;
        while (g_fence->GetCompletedValue() < g_fv) {
            struct timespec ns = {0, 1000000};
            nanosleep(&ns, nullptr);
            if (++spins > 15000) {
                printf("  GPU wait TIMEOUT\n");
                return false;
            }
        }
    }
    return g_alloc->Reset() == S_OK && g_cl->Reset(g_alloc, nullptr) == S_OK;
}

static ID3D12Resource *
makeBuffer(uint64_t size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
           D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ID3D12Resource *r = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              state, nullptr,
                                              __uuidof(ID3D12Resource),
                                              (void **)&r)))
        return nullptr;
    return r;
}

/* Compute create+dispatch+readback with the given CS blob. */
static bool
tryCompute(const void *cs, uint32_t cbCs, const char *tag)
{
    bool ok = false;
    ID3D12RootSignature *rs = nullptr;
    ID3D12PipelineState *pso = nullptr;
    ID3D12Resource *buf = nullptr, *rb = nullptr;

    D3D12_ROOT_PARAMETER p = {};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.NumParameters = 1;
    rd.pParameters = &p;
    ID3DBlob *blob = nullptr, *err = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &blob, &err))) {
        printf("%s: serialize RS FAILED\n", tag);
        goto out;
    }
    if (FAILED(g_dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                          blob->GetBufferSize(),
                                          __uuidof(ID3D12RootSignature),
                                          (void **)&rs))) {
        printf("%s: CreateRootSignature FAILED\n", tag);
        goto out;
    }

    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC cd = {};
        cd.pRootSignature = rs;
        cd.CS.pShaderBytecode = cs;
        cd.CS.BytecodeLength = cbCs;
        HRESULT hr = g_dev->CreateComputePipelineState(
            &cd, __uuidof(ID3D12PipelineState), (void **)&pso);
        printf("%s: CreateComputePipelineState hr=0x%08lx\n", tag,
               (unsigned long)hr);
        if (FAILED(hr))
            goto out;
    }

    buf = makeBuffer(32 * 4, D3D12_HEAP_TYPE_DEFAULT,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    rb = makeBuffer(32 * 4, D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!buf || !rb)
        goto out;
    g_cl->SetPipelineState(pso);
    g_cl->SetComputeRootSignature(rs);
    g_cl->SetComputeRootUnorderedAccessView(0, buf->GetGPUVirtualAddress());
    g_cl->Dispatch(1, 1, 1);
    {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = buf;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_cl->ResourceBarrier(1, &b);
    }
    g_cl->CopyResource(rb, buf);
    if (!execAndWait())
        goto out;
    {
        void *m = nullptr;
        if (FAILED(rb->Map(0, nullptr, &m)))
            goto out;
        const uint32_t *v = (const uint32_t *)m;
        ok = true;
        for (int i = 0; i < 32; i++)
            if (v[i] != 496u) {
                printf("%s: out[%d]=%u want 496\n", tag, i, v[i]);
                ok = false;
                break;
            }
        rb->Unmap(0, nullptr);
        if (ok)
            printf("%s: dispatch verified (all lanes 496)\n", tag);
    }

out:
    if (rb) rb->Release();
    if (buf) buf->Release();
    if (pso) pso->Release();
    if (rs) rs->Release();
    if (blob) blob->Release();
    if (err) err->Release();
    return ok;
}

/* Graphics PSO create (and optional draw+readback) with given VS/PS blobs and
 * input-layout semantic names. */
static bool
tryGraphics(const void *vs, uint32_t cbVs, const void *ps, uint32_t cbPs,
            const char *sem0, const char *sem1, bool draw, const char *tag)
{
    bool ok = false;
    ID3D12RootSignature *rs = nullptr;
    ID3D12PipelineState *pso = nullptr;
    ID3D12Resource *rt = nullptr, *vb = nullptr, *rb = nullptr;
    ID3D12DescriptorHeap *rtvHeap = nullptr;
    ID3DBlob *blob = nullptr, *err = nullptr;

    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1,
                                           &blob, &err)) ||
        FAILED(g_dev->CreateRootSignature(0, blob->GetBufferPointer(),
                                          blob->GetBufferSize(),
                                          __uuidof(ID3D12RootSignature),
                                          (void **)&rs))) {
        printf("%s: root signature FAILED\n", tag);
        goto out;
    }

    {
        D3D12_INPUT_ELEMENT_DESC elems[2] = {};
        elems[0].SemanticName = sem0;
        elems[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
        elems[0].AlignedByteOffset = 0;
        elems[1].SemanticName = sem1;
        elems[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
        elems[1].AlignedByteOffset = 12;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = rs;
        pd.VS.pShaderBytecode = vs;
        pd.VS.BytecodeLength = cbVs;
        pd.PS.pShaderBytecode = ps;
        pd.PS.BytecodeLength = cbPs;
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
        pd.SampleMask = 0xFFFFFFFFu;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.InputLayout.pInputElementDescs = elems;
        pd.InputLayout.NumElements = 2;
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.SampleDesc.Count = 1;
        HRESULT hr = g_dev->CreateGraphicsPipelineState(
            &pd, __uuidof(ID3D12PipelineState), (void **)&pso);
        printf("%s: CreateGraphicsPipelineState hr=0x%08lx\n", tag,
               (unsigned long)hr);
        if (FAILED(hr))
            goto out;
    }
    if (!draw) {
        ok = true;
        goto out;
    }

    {
        const UINT W = 64, H = 64;
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
        cv.Color[0] = 1.f;
        cv.Color[3] = 1.f;
        if (FAILED(g_dev->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &td,
                D3D12_RESOURCE_STATE_RENDER_TARGET, &cv,
                __uuidof(ID3D12Resource), (void **)&rt)))
            goto out;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = 1;
        if (FAILED(g_dev->CreateDescriptorHeap(
                &hd, __uuidof(ID3D12DescriptorHeap), (void **)&rtvHeap)))
            goto out;
        g_dev->CreateRenderTargetView(
            rt, nullptr, rtvHeap->GetCPUDescriptorHandleForHeapStart());

        static const float verts[3][6] = {
            {-1.f, -1.f, 0.f, 0.f, 1.f, 0.f},
            {3.f, -1.f, 0.f, 0.f, 1.f, 0.f},
            {-1.f, 3.f, 0.f, 0.f, 1.f, 0.f},
        };
        vb = makeBuffer(sizeof(verts), D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_GENERIC_READ);
        if (!vb)
            goto out;
        void *m = nullptr;
        vb->Map(0, nullptr, &m);
        memcpy(m, verts, sizeof(verts));
        vb->Unmap(0, nullptr);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv =
            rtvHeap->GetCPUDescriptorHandleForHeapStart();
        const float red[4] = {1.f, 0.f, 0.f, 1.f};
        D3D12_VIEWPORT vp = {0.f, 0.f, (float)W, (float)H, 0.f, 1.f};
        D3D12_RECT sc = {0, 0, (LONG)W, (LONG)H};
        D3D12_VERTEX_BUFFER_VIEW vbv = {vb->GetGPUVirtualAddress(),
                                        sizeof(verts), 6 * sizeof(float)};
        g_cl->SetPipelineState(pso);
        g_cl->SetGraphicsRootSignature(rs);
        g_cl->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        g_cl->ClearRenderTargetView(rtv, red, 0, nullptr);
        g_cl->RSSetViewports(1, &vp);
        g_cl->RSSetScissorRects(1, &sc);
        g_cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        g_cl->IASetVertexBuffers(0, 1, &vbv);
        g_cl->DrawInstanced(3, 1, 0, 0);

        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = rt;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_cl->ResourceBarrier(1, &b);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        UINT64 total = 0;
        D3D12_RESOURCE_DESC tdq = rt->GetDesc();
        g_dev->GetCopyableFootprints(&tdq, 0, 1, 0, &fp, nullptr, nullptr,
                                     &total);
        rb = makeBuffer(total, D3D12_HEAP_TYPE_READBACK,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COPY_DEST);
        if (!rb)
            goto out;
        D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
        src.pResource = rt;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = rb;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp;
        g_cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        if (!execAndWait())
            goto out;
        m = nullptr;
        if (FAILED(rb->Map(0, nullptr, &m)))
            goto out;
        uint32_t center = *(const uint32_t *)((const uint8_t *)m +
                                              fp.Footprint.RowPitch * (H / 2) +
                                              4 * (W / 2));
        rb->Unmap(0, nullptr);
        printf("%s: center=0x%08x (want 0xff00ff00)\n", tag, center);
        ok = (center == 0xff00ff00u);
    }

out:
    if (rb) rb->Release();
    if (vb) vb->Release();
    if (rtvHeap) rtvHeap->Release();
    if (rt) rt->Release();
    if (pso) pso->Release();
    if (rs) rs->Release();
    if (blob) blob->Release();
    if (err) err->Release();
    return ok;
}

static bool g_runBare;

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-bare"))
            g_runBare = true;
    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "DXILPROBE: dmn_init FAILED\n");
        return 1;
    }
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void **)&g_dev))) {
        fprintf(stderr, "DXILPROBE: D3D12CreateDevice FAILED\n");
        return 1;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    if (FAILED(g_dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                         (void **)&g_q)) ||
        FAILED(g_dev->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
            (void **)&g_alloc)) ||
        FAILED(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        g_alloc, nullptr,
                                        __uuidof(ID3D12GraphicsCommandList),
                                        (void **)&g_cl)) ||
        FAILED(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  __uuidof(ID3D12Fence), (void **)&g_fence))) {
        fprintf(stderr, "DXILPROBE: setup FAILED\n");
        return 1;
    }

    /* Slice the DXIL parts out of the dxc containers. */
    Span csPart = findPart(g_cs_60, sizeof(g_cs_60), kDxilFourcc);
    Span vsPart = findPart(g_vs_60, sizeof(g_vs_60), kDxilFourcc);
    Span psPart = findPart(g_ps_60, sizeof(g_ps_60), kDxilFourcc);
    if (!csPart.p || !vsPart.p || !psPart.p) {
        fprintf(stderr, "DXILPROBE: DXIL part not found in dxc output\n");
        return 1;
    }
    printf("dxc containers: cs=%zu (dxil part %u), vs=%zu (%u), ps=%zu (%u)\n",
           sizeof(g_cs_60), csPart.n, sizeof(g_vs_60), vsPart.n,
           sizeof(g_ps_60), psPart.n);

    uint32_t cbCsMin, cbVsMin, cbPsMin;
    uint8_t *csMin = buildContainer(&kDxilFourcc, &csPart, 1, &cbCsMin);
    uint8_t *vsMin = buildContainer(&kDxilFourcc, &vsPart, 1, &cbVsMin);
    uint8_t *psMin = buildContainer(&kDxilFourcc, &psPart, 1, &cbPsMin);

    bool A = tryCompute(g_cs_60, sizeof(g_cs_60), "A full-container CS");
    bool B = tryCompute(csMin, cbCsMin, "B minimal-container CS");
    /* A bare program part with no container crashes D3DMetal inside
     * CreateComputePipelineState.  Keep it opt-in so D-G still run; the UMD
     * must never forward a bare part. */
    bool C = false;
    if (g_runBare)
        C = tryCompute(csPart.p, csPart.n, "C bare-part CS");
    else
        printf("C bare-part CS: skipped (crashes D3DMetal; run with -bare)\n");

    bool D = tryGraphics(g_vs_60, sizeof(g_vs_60), g_ps_60, sizeof(g_ps_60),
                         "POSITION", "COLOR", false, "D full-container GFX");
    bool E = tryGraphics(vsMin, cbVsMin, psMin, cbPsMin, "POSITION", "COLOR",
                         false, "E minimal-container GFX");

    bool F = true;
    if (!E) {
        uint32_t cbIsg1;
        uint8_t *isg1 = buildFakeIsg1("XA", "XB", &cbIsg1);
        uint32_t fcs[2] = {kIsg1Fourcc, kDxilFourcc};
        Span parts[2] = {{isg1, cbIsg1}, vsPart};
        uint32_t cbVsF;
        uint8_t *vsF = buildContainer(fcs, parts, 2, &cbVsF);
        F = tryGraphics(vsF, cbVsF, psMin, cbPsMin, "XA", "XB", false,
                        "F fake-ISG1 GFX");
        free(vsF);
        free(isg1);
    } else {
        printf("F fake-ISG1 GFX: skipped (E passed)\n");
    }

    /* Execution check on the best graphics variant. */
    bool G;
    if (E)
        G = tryGraphics(vsMin, cbVsMin, psMin, cbPsMin, "POSITION", "COLOR",
                        true, "G draw minimal");
    else
        G = tryGraphics(g_vs_60, sizeof(g_vs_60), g_ps_60, sizeof(g_ps_60),
                        "POSITION", "COLOR", true, "G draw full");

    /* H: does input-layout matching go by semantic NAME (create fails or
     * misbinds with wrong names) or by register/order (fake names work)?
     * Decides whether the guest UMD needs real names out of the bitcode
     * metadata or can synthesize its own.  D3DMetal matches by NAME, and the
     * mismatch is a SILENT no-op PSO (S_OK, nothing draws). */
    bool H = tryGraphics(vsMin, cbVsMin, psMin, cbPsMin, "XA", "XB", true,
                         "H draw wrong-names");
    printf("H verdict: %s\n",
           H ? "matching is register/order-based (fake names OK)"
             : "matching is NAME-based (guest needs real names)");

    /* R: exercise the reflection helper directly on both container shapes
     * (full dxc output vs our minimal re-wrap) to attribute failures:
     * both fail => ABI/slot problem; only minimal fails => shape problem. */
    {
        dmn_dxil_semantic sem[8];
        int nFull = dmn_dxil_input_semantics(g_vs_60, sizeof(g_vs_60), sem, 8);
        printf("R reflect full: %d entries", nFull);
        for (int i = 0; i < nFull; i++)
            printf("  [reg %u]=%s idx=%u", sem[i].reg, sem[i].name,
                   sem[i].index);
        printf("\n");
        int nMin = dmn_dxil_input_semantics(vsMin, cbVsMin, sem, 8);
        printf("R reflect minimal: %d entries", nMin);
        for (int i = 0; i < nMin; i++)
            printf("  [reg %u]=%s idx=%u", sem[i].reg, sem[i].name,
                   sem[i].index);
        printf("\n");
    }

    /* P: the production contract -- the guest sends register-tagged
     * placeholders (NPTA<reg>) and the CreateGraphicsPipelineState hook
     * resolves them to real names via DXC reflection.  Must render. */
    bool P = tryGraphics(vsMin, cbVsMin, psMin, cbPsMin, "NPTA0", "NPTA1",
                         true, "P draw placeholders");
    printf("P verdict: NPTA placeholder resolution %s\n",
           P ? "WORKS" : "FAILED");

    printf("DXILPROBE: A=%d B=%d C=%d D=%d E=%d F=%d G=%d P=%d\n", A, B, C, D,
           E, F, G, P);
    free(csMin);
    free(vsMin);
    free(psMin);
    const bool ok = A && B && D && E && G && P;
    printf("DXILPROBE: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}
