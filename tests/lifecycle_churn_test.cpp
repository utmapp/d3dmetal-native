/*
 * Shared-resource lifecycle churn: create and destroy shared textures (with
 * and without keyed mutex), shared buffers, and shared fences in a loop, then
 * assert the process did not accumulate fds or memory. This is the regression
 * test for destruction-driven eviction: a per-iteration leak of the texture's
 * shm fd + mapping, its registry PODs, its keyed mutex, or the fence's helper
 * machinery lands an order of magnitude above these thresholds.
 *
 * Windowless, single process. D3D11 churn plus a D3D12 segment (committed
 * SHARED texture + queue-signaled shared fence, exercising the helper-queue
 * drain in the fence teardown). Prints "CHURN: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <mach/mach.h>
#include <time.h>
#include <unistd.h>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

namespace {

constexpr int kIters11 = 200;
constexpr int kIters12 = 100;
constexpr uint32_t kW = 512, kH = 512; /* 1 MiB per BGRA texture */

/* DMN_CHURN_MODE=unshared runs the same loop without any SHARED flags (no
 * interception at all) — the control for separating D3DMetal's own
 * per-resource footprint behavior from the sharing machinery's. */
bool g_shared = true;

int count_fds() {
    int n = 0;
    for (int fd = 0; fd < 1024; fd++)
        if (fcntl(fd, F_GETFD) != -1)
            n++;
    return n;
}

uint64_t phys_footprint() {
    task_vm_info_data_t vm{};
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &cnt) !=
        KERN_SUCCESS)
        return 0;
    return vm.phys_footprint;
}

void settle(ID3D11DeviceContext* ctx) {
    /* Let deferred destruction drain: flush pending work, then give Metal's
     * completion handlers (which drop the last buffer refs) a moment. */
    if (ctx)
        ctx->Flush();
    struct timespec ns = {0, 300 * 1000 * 1000};
    nanosleep(&ns, nullptr);
}

int churn_d3d11() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                 D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                                 D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "CHURN: D3D11CreateDevice FAILED\n");
        return -1;
    }
    Com<ID3D11Device5> dev5;
    Com<ID3D11DeviceContext4> ctx4;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5)) ||
        FAILED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&ctx4))) {
        fprintf(stderr, "CHURN: D3D11 device5/context4 unavailable\n");
        return -1;
    }

    for (int i = 0; i < kIters11; i++) {
        /* Shared texture, keyed mutex on every other iteration. */
        D3D11_TEXTURE2D_DESC td{};
        td.Width = kW;
        td.Height = kH;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc = {1, 0};
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        td.MiscFlags = !g_shared ? 0
                     : (i & 1)   ? D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX
                                 : D3D11_RESOURCE_MISC_SHARED;
        Com<ID3D11Texture2D> tex;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex))) {
            fprintf(stderr, "CHURN: iter %d CreateTexture2D FAILED\n", i);
            return -1;
        }
        Com<IDXGIResource> res;
        HANDLE h1 = nullptr, h2 = nullptr;
        if (g_shared &&
            (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
             FAILED(res->GetSharedHandle(&h1)) || !h1 ||
             FAILED(res->GetSharedHandle(&h2)))) {
            fprintf(stderr, "CHURN: iter %d texture export FAILED\n", i);
            return -1;
        }
        if (g_shared && h1 != h2) {
            fprintf(stderr, "CHURN: iter %d repeated export returned distinct "
                    "handles (%p vs %p)\n", i, h1, h2);
            return -1;
        }
        if (g_shared &&
            ((dmn_shared_texture_handle*)h1)->magic != DMN_SHARED_TEXTURE_MAGIC) {
            fprintf(stderr, "CHURN: iter %d bad texture POD magic\n", i);
            return -1;
        }
        if (g_shared && (i & 1)) {
            Com<IDXGIKeyedMutex> km;
            if (FAILED(tex->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&km))) {
                fprintf(stderr, "CHURN: iter %d keyed-mutex QI FAILED\n", i);
                return -1;
            }
            if (FAILED(km->AcquireSync(0, 1000)) || FAILED(km->ReleaseSync(1))) {
                fprintf(stderr, "CHURN: iter %d keyed-mutex cycle FAILED\n", i);
                return -1;
            }
        }

        /* Shared buffer. */
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = 65536;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = g_shared ? D3D11_RESOURCE_MISC_SHARED : 0;
        Com<ID3D11Buffer> buf;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, &buf))) {
            fprintf(stderr, "CHURN: iter %d CreateBuffer FAILED\n", i);
            return -1;
        }

        /* Shared fence: create, signal once (drives the GPU slot store), export. */
        Com<ID3D11Fence> fence;
        if (FAILED(dev5->CreateFence(0, g_shared ? D3D11_FENCE_FLAG_SHARED
                                                 : D3D11_FENCE_FLAG_NONE,
                                     __uuidof(ID3D11Fence), (void**)&fence))) {
            fprintf(stderr, "CHURN: iter %d CreateFence FAILED\n", i);
            return -1;
        }
        HANDLE fh = nullptr;
        if (g_shared &&
            (FAILED(fence->CreateSharedHandle(nullptr, 0, nullptr, &fh)) || !fh ||
             ((dmn_shared_fence_handle*)fh)->magic != DMN_SHARED_FENCE_MAGIC)) {
            fprintf(stderr, "CHURN: iter %d fence export FAILED\n", i);
            return -1;
        }
        /* NT-style handle: without this close, every iteration leaks a POD +
         * fd and the fd-growth assertion below trips. */
        if (fh && dmn_shared_handle_close(fh) != DMN_SUCCESS) {
            fprintf(stderr, "CHURN: iter %d fence handle close FAILED\n", i);
            return -1;
        }
        ctx4->Signal(fence.ptr(), 1);

        /* Everything drops here (Com dtors): texture + buffer + fence must
         * take their registry entries, PODs, mutex, and backing with them. */
    }
    settle(ctx.ptr());
    return 0;
}

int churn_d3d12() {
    Com<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&dev))) {
        fprintf(stderr, "CHURN: D3D12CreateDevice FAILED\n");
        return -1;
    }
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Com<ID3D12CommandQueue> queue;
    if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                       (void**)&queue))) {
        fprintf(stderr, "CHURN: D3D12 CreateCommandQueue FAILED\n");
        return -1;
    }

    for (int i = 0; i < kIters12; i++) {
        /* Shared committed texture. */
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = kW;
        rd.Height = kH;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Com<ID3D12Resource> tex;
        if (FAILED(dev->CreateCommittedResource(&hp,
                                                g_shared ? D3D12_HEAP_FLAG_SHARED
                                                         : D3D12_HEAP_FLAG_NONE,
                                                &rd,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                __uuidof(ID3D12Resource),
                                                (void**)&tex))) {
            fprintf(stderr, "CHURN: iter %d D3D12 shared texture FAILED\n", i);
            return -1;
        }
        HANDLE th = nullptr;
        if (g_shared &&
            (FAILED(dev->CreateSharedHandle(tex.ptr(), nullptr, 0, nullptr, &th)) ||
             !th || ((dmn_shared_texture_handle*)th)->magic != DMN_SHARED_TEXTURE_MAGIC)) {
            fprintf(stderr, "CHURN: iter %d D3D12 texture export FAILED\n", i);
            return -1;
        }
        if (th && dmn_shared_handle_close(th) != DMN_SUCCESS) {
            fprintf(stderr, "CHURN: iter %d D3D12 handle close FAILED\n", i);
            return -1;
        }

        /* Shared fence, queue-signaled (helper-queue write + drained teardown). */
        Com<ID3D12Fence> fence;
        if (FAILED(dev->CreateFence(0, g_shared ? D3D12_FENCE_FLAG_SHARED
                                                : D3D12_FENCE_FLAG_NONE,
                                    __uuidof(ID3D12Fence), (void**)&fence))) {
            fprintf(stderr, "CHURN: iter %d D3D12 CreateFence(SHARED) FAILED\n", i);
            return -1;
        }
        queue->Signal(fence.ptr(), 1);
        /* Wait for the signal so teardown's drain has nothing pending. */
        for (int spin = 0; spin < 5000 && fence->GetCompletedValue() < 1; spin++) {
            struct timespec ns = {0, 1000 * 1000};
            nanosleep(&ns, nullptr);
        }
    }
    settle(nullptr);
    return 0;
}

/* Warm up both APIs (one-time costs: shader caches, device-global pools, the
 * eviction probes), then run a measured round. Prints a parseable growth line. */
int run_measured(int64_t* fd_growth, int64_t* mem_growth) {
    if (churn_d3d11() != 0 || churn_d3d12() != 0)
        return -1;

    const int fd0 = count_fds();
    const uint64_t mem0 = phys_footprint();
    printf("CHURN: baseline fds=%d footprint=%llu MiB\n", fd0,
           (unsigned long long)(mem0 >> 20));

    if (churn_d3d11() != 0 || churn_d3d12() != 0)
        return -1;

    const int fd1 = count_fds();
    const uint64_t mem1 = phys_footprint();
    *fd_growth = fd1 - fd0;
    *mem_growth = (int64_t)(mem1 >> 20) - (int64_t)(mem0 >> 20);
    printf("CHURN: after %d D3D11 + %d D3D12 iterations: fds=%d "
           "footprint=%llu MiB\n",
           kIters11, kIters12, fd1, (unsigned long long)(mem1 >> 20));
    printf("CHURN-GROWTH fds=%lld mem_mib=%lld\n", (long long)*fd_growth,
           (long long)*mem_growth);
    return 0;
}

/* Re-exec self in control mode and parse its CHURN-GROWTH line. */
int run_control(const char* self, int64_t* fd_growth, int64_t* mem_growth) {
    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "DMN_CHURN_MODE=unshared '%s' 2>&1", self);
    FILE* p = popen(cmd, "r");
    if (!p) {
        fprintf(stderr, "CHURN: control popen FAILED\n");
        return -1;
    }
    char line[512];
    bool got = false;
    long long fds = 0, mem = 0;
    while (fgets(line, sizeof(line), p)) {
        if (sscanf(line, "CHURN-GROWTH fds=%lld mem_mib=%lld", &fds, &mem) == 2)
            got = true;
    }
    int st = pclose(p);
    if (!got || st != 0) {
        fprintf(stderr, "CHURN: control run failed (status=%d parsed=%d)\n",
                st, (int)got);
        return -1;
    }
    *fd_growth = fds;
    *mem_growth = mem;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    (void)argc;
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* mode = getenv("DMN_CHURN_MODE");
    g_shared = !(mode && strcmp(mode, "unshared") == 0);

    /* Control child: measure D3DMetal's own churn behavior with the sharing
     * machinery never engaged; report growth and exit. */
    if (!g_shared) {
        printf("CHURN: control mode — no SHARED flags\n");
        if (dmn_init(nullptr) != DMN_SUCCESS)
            return 1;
        int64_t fdg = 0, memg = 0;
        return run_measured(&fdg, &memg) != 0 ? 1 : 0;
    }

    int64_t ctl_fd = 0, ctl_mem = 0;
    if (run_control(argv[0], &ctl_fd, &ctl_mem) != 0)
        return 1;
    printf("CHURN: control growth fds=%lld mem=%lld MiB\n",
           (long long)ctl_fd, (long long)ctl_mem);

    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "CHURN: dmn_init FAILED\n");
        return 1;
    }
    int64_t fd_growth = 0, mem_growth = 0;
    if (run_measured(&fd_growth, &mem_growth) != 0)
        return 1;

    /* The sharing machinery must not add growth beyond D3DMetal's own churn
     * behavior (the control): a leaked backing would be ~1 fd + ~1 MiB per
     * texture iteration ON TOP of it — 300+ fds / 300+ MiB here. fds are
     * held absolute (the control leaks none); memory is differential because
     * D3DMetal's internal pools grow on churn even with sharing disabled. */
    int rc = 0;
    if (fd_growth > 32) {
        fprintf(stderr, "CHURN: FAIL — fd count grew by %lld\n",
                (long long)fd_growth);
        rc = 1;
    }
    if (mem_growth > ctl_mem + 128) {
        fprintf(stderr, "CHURN: FAIL — footprint grew %lld MiB vs control's "
                "%lld MiB\n", (long long)mem_growth, (long long)ctl_mem);
        rc = 1;
    }

    /* Debug aid: DMN_CHURN_HOLD=<sec> keeps the process alive after the
     * measurement so its regions can be inspected (vmmap/footprint). */
    if (const char* hold = getenv("DMN_CHURN_HOLD")) {
        printf("CHURN: holding %s s (pid=%d)\n", hold, getpid());
        sleep((unsigned)atoi(hold));
    }

    if (rc == 0)
        printf("CHURN: PASS\n");
    return rc;
}
