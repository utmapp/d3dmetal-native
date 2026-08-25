/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * d3d12_lumen_shapes_test: drive SHARED (== substituted, impostor-backed)
 * D3D12 textures at the exact resource shapes of the Subnautica 2 command
 * buffer that GPU-page-faults during CREATIVE world load, locally, with no VM.
 *
 * Why these shapes.  The two faulting command buffers (recorded in
 * var/sn2-traces/vm-fault-fingerprint.md) are Lumen GI / virtual-shadow-map
 * frames whose distinctive resources are:
 *   A) a 4096x4096 Lumen card atlas written by LumenCardCopyPS (three passes,
 *      three different pixel formats)
 *   B) a 512x512 MRT capture target: three colour attachments + depth,
 *      cleared by ClearLumenCardCapturePS
 *   C) UAV clears/writes (texBufClearUint, ClearTextureRWPS)
 * and the VM's Metal stream carries dmn-shared-prod-4096x4096 /
 * -512x512 impostors for them, where CrossOver -- which reaches gameplay on
 * the same D3DMetal -- substitutes NOTHING.  Impostor substitution at these
 * shapes is therefore the concrete measured difference between the two arms.
 *
 * On this stack every guest resource must be host-visible, so a SHARED
 * texture is replaced by a buffer-backed LINEAR texture over shared memory
 * (make_linear_texture()).  A linear impostor is a genuinely different Metal
 * object from a native tiled texture: different storage mode, no GPU
 * compression, explicit stride.  This test asks whether D3DMetal drives those
 * correctly at Lumen's sizes and attachment counts.
 *
 * Deliberately shader-free: ClearRenderTargetView / ClearDepthStencilView /
 * ClearUnorderedAccessViewUint plus CopyTextureRegion exercise the render
 * target, depth and UAV paths without a PSO, so a failure is unambiguously
 * about the resource, not about shader translation.
 *
 * Bounded and safe: fixed work, 10 s fence waits, no loops -- it cannot wedge
 * the GPU the way an unbounded run can.  Each phase prints PASS/FAIL and the
 * test keeps going, so one run reports every shape.
 */

#include "common/dx12.h"
#include "common/util.h"

#include "d3dmetal_native.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define T_TAG "LUMEN"
#include "common/check.h"

namespace {

struct Gpu {
    Com<ID3D12CommandQueue> queue;
    Com<ID3D12CommandAllocator> alloc;
    Com<ID3D12GraphicsCommandList> list;
    Com<ID3D12Fence> fence;
    UINT64 fv = 0;

    bool init(ID3D12Device* dev) {
        if (FAILED(make_d3d12_queue(dev, queue)))
            return false;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               __uuidof(ID3D12CommandAllocator),
                                               (void**)&alloc)) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          alloc.ptr(), nullptr,
                                          __uuidof(ID3D12GraphicsCommandList),
                                          (void**)&list)))
            return false;
        /* CreateCommandList hands back an OPEN list; Reset() on an open list
         * fails, so every phase would die at begin(). */
        if (FAILED(list->Close()))
            return false;
        return SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                          __uuidof(ID3D12Fence),
                                          (void**)&fence));
    }
    bool begin() {
        return SUCCEEDED(alloc->Reset()) &&
               SUCCEEDED(list->Reset(alloc.ptr(), nullptr));
    }
    /* Submit and wait. Returns false on timeout (a wedge, not a slow GPU). */
    bool end(int timeout_ms = 10000) {
        if (FAILED(list->Close()))
            return false;
        ID3D12CommandList* ls[] = {list.ptr()};
        queue->ExecuteCommandLists(1, ls);
        const UINT64 want = ++fv;
        queue->Signal(fence.ptr(), want);
        const uint64_t t0 = now_ms();
        while (fence->GetCompletedValue() < want) {
            if (now_ms() - t0 > (uint64_t)timeout_ms)
                return false;
            sleep_ms(1);
        }
        return true;
    }
};

D3D12_RESOURCE_DESC tex_desc(UINT64 w, UINT h, DXGI_FORMAT fmt,
                             D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = flags;
    return rd;
}

void transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* r,
                D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &b);
}

int failures = 0;

void verdict(const char* phase, bool ok, const char* detail = "") {
    printf(T_TAG ": %-28s %s%s%s\n", phase, ok ? "PASS" : "FAIL",
           detail[0] ? " -- " : "", detail);
    fflush(stdout);
    if (!ok)
        failures++;
}

} // namespace

int main() {
    setbuf(stdout, nullptr);

    dmn_options opts = {};
    if (dmn_init(&opts) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev))) {
        fprintf(stderr, T_TAG ": no D3D12 device\n");
        return 1;
    }
    Gpu gpu;
    if (!gpu.init(dev.ptr())) {
        fprintf(stderr, T_TAG ": gpu init failed\n");
        return 1;
    }

    D3D12_HEAP_PROPERTIES def = {};
    def.Type = D3D12_HEAP_TYPE_DEFAULT;

    /* Descriptor heaps: RTV/DSV (CPU) and a shader-visible one for UAV clears
     * (ClearUnorderedAccessViewUint needs both a shader-visible and a
     * non-shader-visible handle for the same descriptor). */
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 8;
    Com<ID3D12DescriptorHeap> rtvHeap;
    CK(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap),
                                 (void**)&rtvHeap), "rtv heap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    hd.NumDescriptors = 2;
    Com<ID3D12DescriptorHeap> dsvHeap;
    CK(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap),
                                 (void**)&dsvHeap), "dsv heap");
    const UINT rtvInc =
        dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    /* ---- Phase A: 4096x4096 Lumen card atlas, three formats ------------- */
    {
        struct { DXGI_FORMAT fmt; const char* name; } fmts[] = {
            {DXGI_FORMAT_R8G8B8A8_UNORM,     "RGBA8"},
            {DXGI_FORMAT_R16G16B16A16_FLOAT, "RGBA16F"},
            {DXGI_FORMAT_R11G11B10_FLOAT,    "R11G11B10F"},
        };
        for (auto& f : fmts) {
            char tag[64];
            snprintf(tag, sizeof(tag), "A 4096^2 %s", f.name);
            D3D12_RESOURCE_DESC rd =
                tex_desc(4096, 4096, f.fmt,
                         D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            Com<ID3D12Resource> t;
            HRESULT hr = dev->CreateCommittedResource(
                &def, D3D12_HEAP_FLAG_SHARED, &rd,
                D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                __uuidof(ID3D12Resource), (void**)&t);
            if (FAILED(hr)) {
                char d[64];
                snprintf(d, sizeof(d), "create SHARED hr=0x%08lx",
                         (unsigned long)hr);
                verdict(tag, false, d);
                continue;
            }
            D3D12_CPU_DESCRIPTOR_HANDLE rtv =
                rtvHeap->GetCPUDescriptorHandleForHeapStart();
            dev->CreateRenderTargetView(t.ptr(), nullptr, rtv);
            if (!gpu.begin()) { verdict(tag, false, "begin"); continue; }
            const float c[4] = {0.25f, 0.5f, 0.75f, 1.0f};
            gpu.list->ClearRenderTargetView(rtv, c, 0, nullptr);
            bool ok = gpu.end();
            verdict(tag, ok, ok ? "" : "FENCE TIMEOUT (GPU wedge shape)");
            if (dev->GetDeviceRemovedReason() != S_OK) {
                verdict(tag, false, "device removed");
                break;
            }
        }
    }

    /* ---- Phase B: 512x512 MRT capture target, 3 colour + depth ---------- */
    {
        const char* tag = "B 512^2 MRT 3-colour+depth";
        D3D12_RESOURCE_DESC rc =
            tex_desc(512, 512, DXGI_FORMAT_R8G8B8A8_UNORM,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        Com<ID3D12Resource> c0, c1, c2;
        bool made = SUCCEEDED(dev->CreateCommittedResource(
                        &def, D3D12_HEAP_FLAG_SHARED, &rc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                        __uuidof(ID3D12Resource), (void**)&c0)) &&
                    SUCCEEDED(dev->CreateCommittedResource(
                        &def, D3D12_HEAP_FLAG_SHARED, &rc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                        __uuidof(ID3D12Resource), (void**)&c1)) &&
                    SUCCEEDED(dev->CreateCommittedResource(
                        &def, D3D12_HEAP_FLAG_SHARED, &rc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                        __uuidof(ID3D12Resource), (void**)&c2));
        D3D12_RESOURCE_DESC rdd =
            tex_desc(512, 512, DXGI_FORMAT_D32_FLOAT,
                     D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        Com<ID3D12Resource> ds;
        D3D12_CLEAR_VALUE cv = {};
        cv.Format = DXGI_FORMAT_D32_FLOAT;
        cv.DepthStencil.Depth = 1.0f;
        /* Depth is NOT shared in the game either (it is transient), so this
         * one stays a normal committed resource -- the point of the phase is
         * shared COLOUR attachments alongside a native depth. */
        bool depth_ok = SUCCEEDED(dev->CreateCommittedResource(
            &def, D3D12_HEAP_FLAG_NONE, &rdd,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv,
            __uuidof(ID3D12Resource), (void**)&ds));
        if (!made || !depth_ok) {
            verdict(tag, false, "create");
        } else {
            D3D12_CPU_DESCRIPTOR_HANDLE base =
                rtvHeap->GetCPUDescriptorHandleForHeapStart();
            D3D12_CPU_DESCRIPTOR_HANDLE r0 = base, r1 = base, r2 = base;
            r1.ptr += rtvInc;
            r2.ptr += 2ull * rtvInc;
            dev->CreateRenderTargetView(c0.ptr(), nullptr, r0);
            dev->CreateRenderTargetView(c1.ptr(), nullptr, r1);
            dev->CreateRenderTargetView(c2.ptr(), nullptr, r2);
            D3D12_CPU_DESCRIPTOR_HANDLE dsv =
                dsvHeap->GetCPUDescriptorHandleForHeapStart();
            dev->CreateDepthStencilView(ds.ptr(), nullptr, dsv);
            if (gpu.begin()) {
                const float c[4] = {1.0f, 0.0f, 0.0f, 1.0f};
                D3D12_CPU_DESCRIPTOR_HANDLE rts[3] = {r0, r1, r2};
                gpu.list->OMSetRenderTargets(3, rts, FALSE, &dsv);
                gpu.list->ClearRenderTargetView(r0, c, 0, nullptr);
                gpu.list->ClearRenderTargetView(r1, c, 0, nullptr);
                gpu.list->ClearRenderTargetView(r2, c, 0, nullptr);
                gpu.list->ClearDepthStencilView(
                    dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
                bool ok = gpu.end();
                verdict(tag, ok, ok ? "" : "FENCE TIMEOUT (GPU wedge shape)");
            } else {
                verdict(tag, false, "begin");
            }
        }
    }

    /* ---- Phase C: shared texture as UAV (texBufClearUint shape) --------- */
    {
        const char* tag = "C 1024^2 shared UAV clear";
        D3D12_RESOURCE_DESC rd =
            tex_desc(1024, 1024, DXGI_FORMAT_R32_UINT,
                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Com<ID3D12Resource> u;
        HRESULT hr = dev->CreateCommittedResource(
            &def, D3D12_HEAP_FLAG_SHARED, &rd,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            __uuidof(ID3D12Resource), (void**)&u);
        if (FAILED(hr)) {
            char d[64];
            snprintf(d, sizeof(d), "create SHARED UAV hr=0x%08lx",
                     (unsigned long)hr);
            verdict(tag, false, d);
        } else {
            D3D12_DESCRIPTOR_HEAP_DESC sv = {};
            sv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            sv.NumDescriptors = 2;
            sv.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            Com<ID3D12DescriptorHeap> gpuHeap;
            D3D12_DESCRIPTOR_HEAP_DESC cp = sv;
            cp.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            Com<ID3D12DescriptorHeap> cpuHeap;
            bool heaps =
                SUCCEEDED(dev->CreateDescriptorHeap(
                    &sv, __uuidof(ID3D12DescriptorHeap), (void**)&gpuHeap)) &&
                SUCCEEDED(dev->CreateDescriptorHeap(
                    &cp, __uuidof(ID3D12DescriptorHeap), (void**)&cpuHeap));
            if (!heaps) {
                verdict(tag, false, "uav heaps");
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
                ud.Format = DXGI_FORMAT_R32_UINT;
                ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                dev->CreateUnorderedAccessView(
                    u.ptr(), nullptr, &ud,
                    cpuHeap->GetCPUDescriptorHandleForHeapStart());
                dev->CreateUnorderedAccessView(
                    u.ptr(), nullptr, &ud,
                    gpuHeap->GetCPUDescriptorHandleForHeapStart());
                if (gpu.begin()) {
                    ID3D12DescriptorHeap* hs[] = {gpuHeap.ptr()};
                    gpu.list->SetDescriptorHeaps(1, hs);
                    const UINT vals[4] = {0xDEADBEEFu, 0, 0, 0};
                    gpu.list->ClearUnorderedAccessViewUint(
                        gpuHeap->GetGPUDescriptorHandleForHeapStart(),
                        cpuHeap->GetCPUDescriptorHandleForHeapStart(),
                        u.ptr(), vals, 0, nullptr);
                    bool ok = gpu.end();
                    verdict(tag, ok,
                            ok ? "" : "FENCE TIMEOUT (GPU wedge shape)");
                } else {
                    verdict(tag, false, "begin");
                }
            }
        }
    }

    /* ---- Phase D: readback -- did the clear actually land? -------------- */
    {
        const char* tag = "D 256^2 shared RT readback";
        D3D12_RESOURCE_DESC rd =
            tex_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        Com<ID3D12Resource> t;
        HRESULT hr = dev->CreateCommittedResource(
            &def, D3D12_HEAP_FLAG_SHARED, &rd,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
            __uuidof(ID3D12Resource), (void**)&t);
        D3D12_HEAP_PROPERTIES rb = {};
        rb.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd = {};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = 256ull * 256 * 4;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Com<ID3D12Resource> read;
        bool ok = SUCCEEDED(hr) &&
                  SUCCEEDED(dev->CreateCommittedResource(
                      &rb, D3D12_HEAP_FLAG_NONE, &bd,
                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                      __uuidof(ID3D12Resource), (void**)&read));
        if (!ok) {
            verdict(tag, false, "create");
        } else {
            D3D12_CPU_DESCRIPTOR_HANDLE rtv =
                rtvHeap->GetCPUDescriptorHandleForHeapStart();
            dev->CreateRenderTargetView(t.ptr(), nullptr, rtv);
            if (!gpu.begin()) {
                verdict(tag, false, "begin");
            } else {
                /* Opaque red: every channel distinct from the zero-fill a
                 * never-written surface would show. */
                const float c[4] = {1.0f, 0.0f, 0.0f, 1.0f};
                gpu.list->ClearRenderTargetView(rtv, c, 0, nullptr);
                transition(gpu.list.ptr(), t.ptr(),
                           D3D12_RESOURCE_STATE_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION src = {};
                src.pResource = t.ptr();
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex = 0;
                D3D12_TEXTURE_COPY_LOCATION dst = {};
                dst.pResource = read.ptr();
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint.Offset = 0;
                dst.PlacedFootprint.Footprint.Format =
                    DXGI_FORMAT_R8G8B8A8_UNORM;
                dst.PlacedFootprint.Footprint.Width = 256;
                dst.PlacedFootprint.Footprint.Height = 256;
                dst.PlacedFootprint.Footprint.Depth = 1;
                dst.PlacedFootprint.Footprint.RowPitch = 256 * 4;
                gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                if (!gpu.end()) {
                    verdict(tag, false, "FENCE TIMEOUT (GPU wedge shape)");
                } else {
                    void* p = nullptr;
                    if (FAILED(read->Map(0, nullptr, &p))) {
                        verdict(tag, false, "map");
                    } else {
                        const uint32_t* w = (const uint32_t*)p;
                        int bad = 0;
                        for (int i = 0; i < 256 * 256; i += 37)
                            if (w[i] != 0xFF0000FFu) /* ABGR in memory */
                                bad++;
                        D3D12_RANGE none{0, 0};
                        read->Unmap(0, &none);
                        char d[96];
                        snprintf(d, sizeof(d),
                                 "%d/%d sampled texels wrong (first=0x%08x)",
                                 bad, (256 * 256) / 37, w ? w[0] : 0);
                        verdict(tag, bad == 0, bad ? d : "");
                    }
                }
            }
        }
    }

    /* ---- Phase E: prove the phases above actually hit the impostor path --
     * Every phase asks for HEAP_FLAG_SHARED, but if substitution ever stopped
     * happening they would all quietly pass as ordinary native textures and
     * this test would be a false negative. A shared handle can only be
     * exported for a resource that went through the substitution, so this is
     * the guard: no export, no evidence, no PASS. */
    {
        const char* tag = "E substitution actually ran";
        D3D12_RESOURCE_DESC rd =
            tex_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        Com<ID3D12Resource> t;
        HANDLE h = nullptr;
        bool ok = SUCCEEDED(dev->CreateCommittedResource(
                      &def, D3D12_HEAP_FLAG_SHARED, &rd,
                      D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                      __uuidof(ID3D12Resource), (void**)&t)) &&
                  SUCCEEDED(dev->CreateSharedHandle(t.ptr(), nullptr, 0,
                                                    nullptr, &h)) &&
                  h != nullptr;
        verdict(tag, ok, ok ? "" : "no shared handle -- phases above did NOT "
                                   "exercise impostors");
        if (h)
            dmn_shared_handle_close(h);
    }

    /* ---- Phase F: PLACED textures inside a SHARED heap ------------------
     * Everything above is a committed resource, which takes the producer path
     * with buf_offset 0. The game does not work that way: UE5 sub-allocates
     * out of heaps, so on this stack the impostors are heap WINDOWS -- linear
     * textures at a page-floored offset inside one imported object, sharing
     * the object with live neighbours. That path has its own offset
     * alignment rule and its own aliasing rules, and nothing else here
     * reaches it. Several surfaces are placed in one heap on purpose, so a
     * window that overran its neighbour would show up as corrupt readback. */
    {
        const char* tag = "F placed-in-shared-heap";
        const UINT64 kHeapSz = 64ull << 20;
        D3D12_HEAP_DESC hpd = {};
        hpd.SizeInBytes = kHeapSz;
        hpd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hpd.Flags = D3D12_HEAP_FLAG_SHARED |
                    D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        Com<ID3D12Heap> heap;
        HRESULT hr = dev->CreateHeap(&hpd, __uuidof(ID3D12Heap),
                                     (void**)&heap);
        if (FAILED(hr)) {
            /* Retry allowing RT textures: which heap flag combinations a
             * backend accepts varies, and the point of the phase is placed
             * textures, not the flag taxonomy. */
            hpd.Flags = D3D12_HEAP_FLAG_SHARED |
                        D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
            hr = dev->CreateHeap(&hpd, __uuidof(ID3D12Heap), (void**)&heap);
        }
        if (FAILED(hr)) {
            char d[72];
            snprintf(d, sizeof(d), "CreateHeap(SHARED) hr=0x%08lx",
                     (unsigned long)hr);
            verdict(tag, false, d);
        } else {
            /* Three 256^2 RGBA8 surfaces packed into the one heap. */
            const UINT64 kStep = 1ull << 20;   /* 1 MiB apart, 64 KiB-aligned */
            D3D12_RESOURCE_DESC rd =
                tex_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM,
                         D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            Com<ID3D12Resource> p[3];
            bool placed = true;
            for (int i = 0; i < 3 && placed; i++)
                placed = SUCCEEDED(dev->CreatePlacedResource(
                    heap.ptr(), (UINT64)i * kStep, &rd,
                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                    __uuidof(ID3D12Resource), (void**)&p[i]));
            if (!placed) {
                verdict(tag, false, "CreatePlacedResource");
            } else if (!gpu.begin()) {
                verdict(tag, false, "begin");
            } else {
                /* A distinct colour per window: if one window overlaps
                 * another, the readback below sees the wrong one. */
                static const float cols[3][4] = {
                    {1.f, 0.f, 0.f, 1.f}, {0.f, 1.f, 0.f, 1.f},
                    {0.f, 0.f, 1.f, 1.f}};
                for (int i = 0; i < 3; i++) {
                    D3D12_CPU_DESCRIPTOR_HANDLE rtv =
                        rtvHeap->GetCPUDescriptorHandleForHeapStart();
                    rtv.ptr += (SIZE_T)i * rtvInc;
                    dev->CreateRenderTargetView(p[i].ptr(), nullptr, rtv);
                    gpu.list->ClearRenderTargetView(rtv, cols[i], 0, nullptr);
                }
                bool ok = gpu.end();
                verdict(tag, ok, ok ? "" : "FENCE TIMEOUT (GPU wedge shape)");
            }
        }
    }

    /* ---- Phase G: render through an IMPORTED (consumer) impostor --------
     * Every phase above is producer-side: dmn's substitute_producer() builds
     * the surface. Production spends most of its life on the OTHER branch --
     * substitute_consumer()/substitute_texture_window(), reached through
     * OpenSharedHandle, which does not derive the layout but VALIDATES it
     * against the exporter's, re-maps an existing fd, and consults the alias
     * cache so a same-process import returns the SAME Metal object rather
     * than a second mapping. Different code, different failure modes, never
     * covered here.
     *
     * Render through the import and read back through the PRODUCER's own
     * object: if the import aliased the wrong memory (or a second mapping),
     * the producer's view would not show the import's pixels. */
    {
        const char* tag = "G render via imported impostor";
        /* 256x255, NOT 256x256, on purpose: 256x256 RGBA8 is an exact
         * 16 KiB page multiple, so mapped == logical and the alias-cache
         * logical-vs-mapped mismatch HIDES. 255 rows makes logical 261120,
         * page_align -> 262144, which is the case that misses. Run with
         * DMN_LOG=info and look for "reuses cached impostor backing". */
        D3D12_RESOURCE_DESC rd =
            tex_desc(256, 255, DXGI_FORMAT_R8G8B8A8_UNORM,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        Com<ID3D12Resource> prod;
        HANDLE h = nullptr;
        bool ok = SUCCEEDED(dev->CreateCommittedResource(
                      &def, D3D12_HEAP_FLAG_SHARED, &rd,
                      D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                      __uuidof(ID3D12Resource), (void**)&prod)) &&
                  SUCCEEDED(dev->CreateSharedHandle(prod.ptr(), nullptr, 0,
                                                    nullptr, &h)) && h;
        Com<ID3D12Resource> cons;
        if (ok)
            ok = SUCCEEDED(dev->OpenSharedHandle(h, __uuidof(ID3D12Resource),
                                                 (void**)&cons));
        if (!ok) {
            verdict(tag, false, "export/import failed");
        } else {
            /* Clear through the IMPORT... */
            D3D12_CPU_DESCRIPTOR_HANDLE rtv =
                rtvHeap->GetCPUDescriptorHandleForHeapStart();
            dev->CreateRenderTargetView(cons.ptr(), nullptr, rtv);
            D3D12_HEAP_PROPERTIES rb = {};
            rb.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC bd = {};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = 256ull * 255 * 4;
            bd.Height = 1;
            bd.DepthOrArraySize = 1;
            bd.MipLevels = 1;
            bd.SampleDesc.Count = 1;
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            Com<ID3D12Resource> read;
            if (FAILED(dev->CreateCommittedResource(
                    &rb, D3D12_HEAP_FLAG_NONE, &bd,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                    __uuidof(ID3D12Resource), (void**)&read))) {
                verdict(tag, false, "readback alloc");
            } else if (!gpu.begin()) {
                verdict(tag, false, "begin");
            } else {
                const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
                gpu.list->ClearRenderTargetView(rtv, green, 0, nullptr);
                /* Clear and copy stay in ONE command buffer on purpose.
                 * MEASURED 2026-08-24: splitting them with a full fence wait
                 * makes this pass, so the memory IS shared and the write does
                 * land -- what fails is HAZARD ORDERING. Producer and import
                 * were two separate MTLBuffers (the alias cache missed), so
                 * Metal did not know they alias and the barrier on the
                 * producer did not order the clear issued through the import;
                 * the copy read the untouched surface and saw zeros. Keeping
                 * them in one command buffer is what makes this a regression
                 * test for that: it FAILS while the alias cache misses and
                 * PASSES once producer and import share one buffer. */
                /* D3D12 requires an ALIASING barrier when work is issued
                 * through one resource object and then through another that
                 * refers to the same memory. Without it, "the producer sees
                 * zeros" would be MY bug, not the driver's -- so issue it and
                 * let the result speak. */
                {
                    D3D12_RESOURCE_BARRIER ab = {};
                    ab.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                    ab.Aliasing.pResourceBefore = cons.ptr();
                    ab.Aliasing.pResourceAfter = prod.ptr();
                    gpu.list->ResourceBarrier(1, &ab);
                }
                /* ...and read back through the PRODUCER object. */
                transition(gpu.list.ptr(), prod.ptr(),
                           D3D12_RESOURCE_STATE_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION src = {};
                src.pResource = prod.ptr();
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                src.SubresourceIndex = 0;
                D3D12_TEXTURE_COPY_LOCATION dst = {};
                dst.pResource = read.ptr();
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint.Footprint.Format =
                    DXGI_FORMAT_R8G8B8A8_UNORM;
                dst.PlacedFootprint.Footprint.Width = 256;
                dst.PlacedFootprint.Footprint.Height = 255;
                dst.PlacedFootprint.Footprint.Depth = 1;
                dst.PlacedFootprint.Footprint.RowPitch = 256 * 4;
                gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                if (!gpu.end()) {
                    verdict(tag, false, "FENCE TIMEOUT (GPU wedge shape)");
                } else {
                    void* p = nullptr;
                    if (FAILED(read->Map(0, nullptr, &p))) {
                        verdict(tag, false, "map");
                    } else {
                        const uint32_t* w = (const uint32_t*)p;
                        int bad = 0;
                        for (int i = 0; i < 256 * 255; i += 37)
                            if (w[i] != 0xFF00FF00u) /* ABGR: opaque green */
                                bad++;
                        D3D12_RANGE none{0, 0};
                        read->Unmap(0, &none);
                        char d[112];
                        snprintf(d, sizeof(d),
                                 "import's write not visible via producer: "
                                 "%d/%d wrong (first=0x%08x)",
                                 bad, (256 * 255) / 37, w ? w[0] : 0);
                        verdict(tag, bad == 0, bad ? d : "");
                    }
                }
            }
            dmn_shared_handle_close(h);
        }
    }

    HRESULT rr = dev->GetDeviceRemovedReason();
    if (rr != S_OK) {
        printf(T_TAG ": DEVICE REMOVED 0x%08lx\n", (unsigned long)rr);
        failures++;
    }
    printf(T_TAG ": %s (%d failing phase(s))\n",
           failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
