/*
 * Multi-threaded stress for the sharing interception. MSDN makes device
 * methods free-threaded (D3D11 unless CREATE_DEVICE_SINGLETHREADED; D3D12
 * always), so apps may create devices AND shared resources from many threads
 * concurrently. This exercises exactly that:
 *
 *  1. Barrier-started concurrent device creation on both APIs — races the
 *     vtable patching in dmn_hooks_after_*, which must be atomic per
 *     (vtable, slot) or a losing thread records the winner's thunk as the
 *     "original" and the first hooked call recurses.
 *  2. One D3D11 device, N threads churning MISC_SHARED textures + buffers.
 *     Every thread uses a thread-unique width/size, and verifies the exported
 *     POD (and the desc of an imported open-back) matches: a cross-thread
 *     mis-capture of the thread-local arm surfaces as a dimension mismatch.
 *  3. Same for D3D12 committed SHARED textures (NT handle round trip).
 *  4. Shared-fence create/export/destroy churn on both APIs (registry +
 *     eviction under concurrency).
 *
 * Prints "MT-STRESS: PASS" and exits 0 on success.
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

namespace {

constexpr int kThreads = 4;
constexpr int kIters   = 24;

std::atomic<int> g_fail{0};

#define TFAIL(...)                                                           \
    do {                                                                     \
        fprintf(stderr, "MT-STRESS: " __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                               \
        g_fail.fetch_add(1);                                                 \
        return;                                                              \
    } while (0)

/* Spin barrier (C++17: no std::barrier). Reusable is not needed — one shot
 * per phase, constructed fresh. */
struct Barrier {
    std::atomic<int> arrived{0};
    int expected;
    explicit Barrier(int n) : expected(n) {}
    void wait() {
        arrived.fetch_add(1, std::memory_order_acq_rel);
        while (arrived.load(std::memory_order_acquire) < expected)
            std::this_thread::yield();
    }
};

/* Thread-unique dimensions: any cross-thread capture mismatch is visible in
 * the exported POD / reimported desc. */
uint32_t tex_width(int tid)             { return 32u * (uint32_t)(tid + 1); }
uint32_t tex_height(int iter)           { return 16u + 16u * (uint32_t)(iter & 7); }
uint32_t buf_bytes(int tid, int iter)   { return 1024u * (uint32_t)(tid + 1) +
                                                 256u * (uint32_t)(iter & 3); }

/* == Phase 1: concurrent device creation ================================== */

void mk_device_d3d11(Barrier* b) {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    b->wait();
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx)))
        TFAIL("concurrent D3D11CreateDevice FAILED");
    /* One shared create+export through the freshly patched vtables: a lost
     * patch race would recurse right here. */
    D3D11_TEXTURE2D_DESC td{};
    td.Width = 64;
    td.Height = 64;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Texture2D> tex;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex)))
        TFAIL("device-phase CreateTexture2D FAILED");
    Com<IDXGIResource> res;
    HANDLE h = nullptr;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
        FAILED(res->GetSharedHandle(&h)) || !h ||
        ((dmn_shared_texture_handle*)h)->magic != DMN_SHARED_TEXTURE_MAGIC)
        TFAIL("device-phase texture export FAILED");
}

void mk_device_d3d12(Barrier* b) {
    Com<ID3D12Device> dev;
    b->wait();
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&dev)))
        TFAIL("concurrent D3D12CreateDevice FAILED");
    Com<ID3D12Fence> fence;
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                __uuidof(ID3D12Fence), (void**)&fence)))
        TFAIL("device-phase D3D12 CreateFence(SHARED) FAILED");
    HANDLE h = nullptr;
    if (FAILED(dev->CreateSharedHandle(fence.ptr(), nullptr, 0, nullptr, &h)) ||
        !h || ((dmn_shared_fence_handle*)h)->magic != DMN_SHARED_FENCE_MAGIC)
        TFAIL("device-phase D3D12 fence export FAILED");
    if (dmn_shared_handle_close(h) != DMN_SUCCESS)
        TFAIL("device-phase D3D12 fence handle close FAILED");
}

void phase_concurrent_devices() {
    Barrier b(2 * kThreads);
    std::vector<std::thread> ts;
    for (int i = 0; i < kThreads; i++) {
        ts.emplace_back(mk_device_d3d11, &b);
        ts.emplace_back(mk_device_d3d12, &b);
    }
    for (auto& t : ts)
        t.join();
}

/* == Phase 2: one D3D11 device, N threads churning shared resources ======= */

void churn_d3d11(ID3D11Device* dev, Barrier* b, int tid) {
    b->wait();
    for (int i = 0; i < kIters; i++) {
        const uint32_t W = tex_width(tid), H = tex_height(i);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = W;
        td.Height = H;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        Com<ID3D11Texture2D> tex;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex)))
            TFAIL("t%d i%d CreateTexture2D FAILED", tid, i);

        Com<IDXGIResource> res;
        HANDLE h = nullptr;
        if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
            FAILED(res->GetSharedHandle(&h)) || !h)
            TFAIL("t%d i%d texture export FAILED", tid, i);
        auto* pod = (dmn_shared_texture_handle*)h;
        if (pod->magic != DMN_SHARED_TEXTURE_MAGIC)
            TFAIL("t%d i%d bad texture POD magic", tid, i);
        if (pod->width != W || pod->height != H)
            TFAIL("t%d i%d POD is %ux%u, created %ux%u — arm cross-capture!",
                  tid, i, pod->width, pod->height, W, H);

        /* Open it back on this thread (concurrent consumer arms). */
        Com<ID3D11Texture2D> imp;
        if (FAILED(dev->OpenSharedResource(h, __uuidof(ID3D11Texture2D),
                                           (void**)&imp)))
            TFAIL("t%d i%d OpenSharedResource FAILED", tid, i);
        D3D11_TEXTURE2D_DESC id{};
        imp->GetDesc(&id);
        if (id.Width != W || id.Height != H)
            TFAIL("t%d i%d import is %ux%u, expected %ux%u — arm cross-capture!",
                  tid, i, id.Width, id.Height, W, H);

        /* Thread-unique shared buffer, same verification through its POD. */
        const uint32_t BW = buf_bytes(tid, i);
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = BW;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        Com<ID3D11Buffer> buf;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, &buf)))
            TFAIL("t%d i%d CreateBuffer FAILED", tid, i);
        Com<IDXGIResource> bres;
        HANDLE bh = nullptr;
        if (FAILED(buf->QueryInterface(__uuidof(IDXGIResource), (void**)&bres)) ||
            FAILED(bres->GetSharedHandle(&bh)) || !bh)
            TFAIL("t%d i%d buffer export FAILED", tid, i);
        auto* bpod = (dmn_shared_buffer_handle*)bh;
        if (bpod->magic != DMN_SHARED_BUFFER_MAGIC || bpod->size != BW)
            TFAIL("t%d i%d buffer POD size %llu, created %u — arm cross-capture!",
                  tid, i, (unsigned long long)bpod->size, BW);
    }
}

void phase_same_device_d3d11() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "MT-STRESS: phase-2 D3D11CreateDevice FAILED\n");
        g_fail.fetch_add(1);
        return;
    }
    Barrier b(kThreads);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; t++)
        ts.emplace_back(churn_d3d11, dev.ptr(), &b, t);
    for (auto& t : ts)
        t.join();
    ctx->Flush();
}

/* == Phase 3: one D3D12 device, N threads, NT-handle round trips ========== */

void churn_d3d12(ID3D12Device* dev, Barrier* b, int tid) {
    b->wait();
    for (int i = 0; i < kIters; i++) {
        const uint32_t W = tex_width(tid), H = tex_height(i);
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = W;
        rd.Height = H;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Com<ID3D12Resource> tex;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                                D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, __uuidof(ID3D12Resource),
                                                (void**)&tex)))
            TFAIL("t%d i%d D3D12 shared texture FAILED", tid, i);

        HANDLE h = nullptr;
        if (FAILED(dev->CreateSharedHandle(tex.ptr(), nullptr, 0, nullptr, &h)) ||
            !h)
            TFAIL("t%d i%d D3D12 CreateSharedHandle FAILED", tid, i);
        auto* pod = (dmn_shared_texture_handle*)h;
        if (pod->magic != DMN_SHARED_TEXTURE_MAGIC)
            TFAIL("t%d i%d bad D3D12 texture POD magic", tid, i);
        if (pod->width != W || pod->height != H)
            TFAIL("t%d i%d D3D12 POD is %ux%u, created %ux%u — arm cross-capture!",
                  tid, i, pod->width, pod->height, W, H);

        Com<ID3D12Resource> imp;
        if (FAILED(dev->OpenSharedHandle(h, __uuidof(ID3D12Resource),
                                         (void**)&imp)))
            TFAIL("t%d i%d D3D12 OpenSharedHandle FAILED", tid, i);
        D3D12_RESOURCE_DESC id = imp->GetDesc();
        if (id.Width != W || id.Height != H)
            TFAIL("t%d i%d D3D12 import is %llux%u, expected %ux%u",
                  tid, i, (unsigned long long)id.Width, id.Height, W, H);
        if (dmn_shared_handle_close(h) != DMN_SUCCESS)
            TFAIL("t%d i%d D3D12 handle close FAILED", tid, i);
    }
}

void phase_same_device_d3d12() {
    Com<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&dev))) {
        fprintf(stderr, "MT-STRESS: phase-3 D3D12CreateDevice FAILED\n");
        g_fail.fetch_add(1);
        return;
    }
    Barrier b(kThreads);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; t++)
        ts.emplace_back(churn_d3d12, dev.ptr(), &b, t);
    for (auto& t : ts)
        t.join();
}

/* == Phase 4: shared-fence churn on both APIs ============================= */

void churn_fences(ID3D11Device5* d11, ID3D12Device* d12, Barrier* b, int tid) {
    b->wait();
    for (int i = 0; i < kIters; i++) {
        Com<ID3D11Fence> f11;
        if (FAILED(d11->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                    __uuidof(ID3D11Fence), (void**)&f11)))
            TFAIL("t%d i%d D3D11 CreateFence(SHARED) FAILED", tid, i);
        HANDLE h11 = nullptr;
        if (FAILED(f11->CreateSharedHandle(nullptr, 0, nullptr, &h11)) || !h11 ||
            ((dmn_shared_fence_handle*)h11)->magic != DMN_SHARED_FENCE_MAGIC)
            TFAIL("t%d i%d D3D11 fence export FAILED", tid, i);
        if (dmn_shared_handle_close(h11) != DMN_SUCCESS)
            TFAIL("t%d i%d D3D11 fence handle close FAILED", tid, i);

        Com<ID3D12Fence> f12;
        if (FAILED(d12->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                    __uuidof(ID3D12Fence), (void**)&f12)))
            TFAIL("t%d i%d D3D12 CreateFence(SHARED) FAILED", tid, i);
        HANDLE h12 = nullptr;
        if (FAILED(d12->CreateSharedHandle(f12.ptr(), nullptr, 0, nullptr,
                                           &h12)) || !h12 ||
            ((dmn_shared_fence_handle*)h12)->magic != DMN_SHARED_FENCE_MAGIC)
            TFAIL("t%d i%d D3D12 fence export FAILED", tid, i);
        if (dmn_shared_handle_close(h12) != DMN_SUCCESS)
            TFAIL("t%d i%d D3D12 fence handle close FAILED", tid, i);
    }
}

void phase_fences() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "MT-STRESS: phase-4 D3D11CreateDevice FAILED\n");
        g_fail.fetch_add(1);
        return;
    }
    Com<ID3D11Device5> dev5;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5))) {
        fprintf(stderr, "MT-STRESS: phase-4 ID3D11Device5 unavailable\n");
        g_fail.fetch_add(1);
        return;
    }
    Com<ID3D12Device> d12;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&d12))) {
        fprintf(stderr, "MT-STRESS: phase-4 D3D12CreateDevice FAILED\n");
        g_fail.fetch_add(1);
        return;
    }
    Barrier b(kThreads);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; t++)
        ts.emplace_back(churn_fences, dev5.ptr(), d12.ptr(), &b, t);
    for (auto& t : ts)
        t.join();
    ctx->Flush();
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "MT-STRESS: dmn_init FAILED\n");
        return 1;
    }

    printf("MT-STRESS: phase 1: %d concurrent device creations per API\n",
           kThreads);
    phase_concurrent_devices();
    printf("MT-STRESS: phase 2: D3D11 same-device churn, %d threads x %d\n",
           kThreads, kIters);
    phase_same_device_d3d11();
    printf("MT-STRESS: phase 3: D3D12 same-device churn, %d threads x %d\n",
           kThreads, kIters);
    phase_same_device_d3d12();
    printf("MT-STRESS: phase 4: shared-fence churn, %d threads x %d\n",
           kThreads, kIters);
    phase_fences();

    int fails = g_fail.load();
    if (fails) {
        fprintf(stderr, "MT-STRESS: FAIL (%d thread failures)\n", fails);
        return 1;
    }
    printf("MT-STRESS: PASS\n");
    return 0;
}
