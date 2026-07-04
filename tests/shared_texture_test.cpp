/*
 * Cross-process shared-texture + shared-fence round trip.
 *
 * One binary that forks:
 *   parent  = D3D11 producer. Creates a MISC_SHARED texture (intercepted so its
 *             Metal backing is anonymous shared memory), renders a per-tick
 *             solid color into it, exports the texture + a shared fence, and
 *             ships both PODs plus their fds to the child over a socketpair
 *             (SCM_RIGHTS). Each tick it Flushes, confirms the GPU write landed
 *             in the shared mapping, signals the fence, and waits for the
 *             child's ack before overwriting.
 *   child   = consumer. Receives the PODs, patches the received fds, maps the
 *             texture directly (the backing is MTLStorageModeShared, so the
 *             producer's GPU writes are CPU-visible), waits on the fence for
 *             each tick, samples the center pixel, and acks. Asserts each tick
 *             shows the expected color and that colors differ across ticks.
 *
 * Prints "SHARED: PASS" and exits 0 on success. Set DMN_SHARE_DUMP=<dir> to
 * write per-tick .ppm frames for visual inspection.
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

#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"
#include "common/ipc.h"
#include "common/util.h"

namespace {

constexpr uint32_t kW = 256, kH = 256;
constexpr int kTicks = 3;
/* Cross-process GPU reads of a freshly shared surface are stale until a few
 * producer submissions have propagated (a CPU-polled fence carries no GPU-side
 * cross-process barrier — the MTLSharedEvent gap). Prime with warm-up frames so
 * every measured frame is coherent. */
constexpr int kWarmup = 3;

struct Rgba { float r, g, b, a; };
/* Pure channels: gamma-invariant, so validation needs no sRGB guesswork. */
const Rgba kColors[kTicks] = {
    {1.0f, 0.0f, 0.0f, 1.0f}, /* red   -> BGRA 00 00 ff ff */
    {0.0f, 1.0f, 0.0f, 1.0f}, /* green -> BGRA 00 ff 00 ff */
    {0.0f, 0.0f, 1.0f, 1.0f}, /* blue  -> BGRA ff 00 00 ff */
};

void expected_bgr(const Rgba& c, uint8_t out[3]) {
    out[0] = (uint8_t)(c.b * 255.0f + 0.5f);
    out[1] = (uint8_t)(c.g * 255.0f + 0.5f);
    out[2] = (uint8_t)(c.r * 255.0f + 0.5f);
}

/* == SCM_RIGHTS fd passing =============================================== */

struct WireHandles {
    dmn_shared_texture_handle tex;
    dmn_shared_fence_handle fence;
};

void dump_ppm(const char* dir, int tick, const uint8_t* base, uint64_t stride) {
    if (!dir)
        return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/shared_tick%d.ppm", dir, tick);
    FILE* f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "P6\n%u %u\n255\n", kW, kH);
    for (uint32_t y = 0; y < kH; y++) {
        const uint8_t* row = base + (uint64_t)y * stride;
        for (uint32_t x = 0; x < kW; x++) {
            const uint8_t* px = row + x * 4; /* BGRA */
            uint8_t rgb[3] = { px[2], px[1], px[0] };
            fwrite(rgb, 1, 3, f);
        }
    }
    fclose(f);
    printf("SHARED: dumped %s\n", path);
}

/* == Producer (parent) =================================================== */
int producer(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "SHARED: producer dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D11Device> device;
    Com<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flOut;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                                   D3D11_SDK_VERSION, &device, &flOut, &context);
    if (FAILED(hr)) {
        fprintf(stderr, "SHARED: producer D3D11CreateDevice FAILED 0x%08x\n",
                (unsigned)hr);
        return 1;
    }

    /* Shared render-target texture. */
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    Com<ID3D11Texture2D> tex;
    if (FAILED(hr = device->CreateTexture2D(&td, nullptr, &tex))) {
        fprintf(stderr, "SHARED: CreateTexture2D(MISC_SHARED) FAILED 0x%08x\n",
                (unsigned)hr);
        return 1;
    }

    /* Export the texture handle (our intercepted GetSharedHandle). */
    Com<IDXGIResource> dxgiRes;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgiRes))) {
        fprintf(stderr, "SHARED: QI(IDXGIResource) FAILED\n");
        return 1;
    }
    HANDLE texH = nullptr;
    if (FAILED(hr = dxgiRes->GetSharedHandle(&texH)) || !texH) {
        fprintf(stderr, "SHARED: GetSharedHandle FAILED 0x%08x — the shared "
                "texture was not intercepted (swizzle miss?)\n", (unsigned)hr);
        return 1;
    }
    WireHandles wire = {};
    memcpy(&wire.tex, texH, sizeof(wire.tex));

    /* Fence: a real D3D11 shared fence. The CreateFence(SHARED) hook allocates a
     * companion shared buffer; CreateSharedHandle exports it as our fence POD,
     * and each context Signal makes the GPU write the value into that buffer. */
    Com<ID3D11Device5> device5;
    Com<ID3D11DeviceContext4> context4;
    Com<ID3D11Fence> d3dFence;
    if (FAILED(device->QueryInterface(__uuidof(ID3D11Device5), (void**)&device5)) ||
        FAILED(context->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&context4)) ||
        FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                    __uuidof(ID3D11Fence), (void**)&d3dFence))) {
        fprintf(stderr, "SHARED: ID3D11Device5::CreateFence(SHARED) unavailable\n");
        return 1;
    }
    HANDLE fenceH = nullptr;
    if (FAILED(hr = d3dFence->CreateSharedHandle(nullptr, 0, nullptr, &fenceH)) ||
        !fenceH) {
        fprintf(stderr, "SHARED: fence CreateSharedHandle FAILED 0x%08x\n",
                (unsigned)hr);
        return 1;
    }
    memcpy(&wire.fence, fenceH, sizeof(wire.fence));
    printf("SHARED: producer using D3D11 shared fence (buf fd=%d)\n",
           wire.fence.fd);

    /* Ship both PODs + their fds to the consumer. */
    int fds[2] = { wire.tex.fd, wire.fence.fd };
    if (!send_with_fds(sock, &wire, sizeof(wire), fds, 2)) {
        fprintf(stderr, "SHARED: producer send_with_fds FAILED: %s\n",
                strerror(errno));
        return 1;
    }
    dmn_shared_handle_close(fenceH); /* NT-style; texH is legacy (no close) */

    Com<ID3D11RenderTargetView> rtv;
    if (FAILED(device->CreateRenderTargetView(tex.ptr(), nullptr, &rtv))) {
        fprintf(stderr, "SHARED: CreateRenderTargetView FAILED\n");
        return 1;
    }

    /* Ticks [-kWarmup, 0) are warm-up frames (see consumer): they prime the
     * shared surface so every measured frame is coherent for the consumer's GPU. */
    for (int tick = -kWarmup; tick < kTicks; tick++) {
        const Rgba& col = kColors[tick < 0 ? 0 : tick];
        FLOAT c[4] = { col.r, col.g, col.b, col.a };
        context->ClearRenderTargetView(rtv.ptr(), c);

        /* Don't flush here: let the fence-store (injected right after Signal on
         * this same context) ride the same submission as the clear, so the value
         * write is GPU-ordered strictly after the render. The consumer's fence
         * wait completing then means the render has landed in the shared texture. */
        uint64_t value = (uint64_t)(tick + kWarmup + 1);
        context4->Signal(d3dFence.ptr(), value); /* hook writes value via GPU */
        if (tick >= 0)
            printf("SHARED: producer signaled tick %d (value=%llu)\n", tick,
                   (unsigned long long)value);

        /* Wait for the child's ack before overwriting next tick. */
        char ack = 0;
        if (read(sock, &ack, 1) != 1) {
            fprintf(stderr, "SHARED: producer ack read FAILED\n");
            return 1;
        }
    }

    printf("SHARED: producer done\n");
    return 0;
}

/* == Consumer (child) ==================================================== */
/* Consumes the shared texture + fence through the STANDARD D3D APIs only:
 * OpenSharedResource -> ID3D11Texture2D, OpenSharedFence -> ID3D11Fence, then
 * GetCompletedValue to wait and CopyResource->staging->Map to read back. The
 * only non-D3D step is receiving the fd (SCM_RIGHTS) and putting it in the
 * handle the producer built — the OS handle-transport, not a sharing API. */
int consumer(int sock) {
    const char* dumpDir = getenv("DMN_SHARE_DUMP");
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "SHARED: consumer dmn_init FAILED\n");
        return 1;
    }

    WireHandles wire = {};
    int fds[2] = { -1, -1 };
    if (!recv_with_fds(sock, &wire, sizeof(wire), fds, 2)) {
        fprintf(stderr, "SHARED: consumer recv_with_fds FAILED: %s\n",
                strerror(errno));
        return 1;
    }
    wire.tex.fd = fds[0];     /* the received fd is the resource's token */
    wire.fence.fd = fds[1];

    if (wire.tex.magic != DMN_SHARED_TEXTURE_MAGIC ||
        wire.fence.magic != DMN_SHARED_FENCE_MAGIC) {
        fprintf(stderr, "SHARED: consumer bad magic (tex=0x%08x fence=0x%08x)\n",
                wire.tex.magic, wire.fence.magic);
        return 1;
    }

    Com<ID3D11Device> device;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                 D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                                 D3D11_SDK_VERSION, &device, &flo, &ctx))) {
        fprintf(stderr, "SHARED: consumer D3D11CreateDevice FAILED\n");
        return 1;
    }
    Com<ID3D11Device5> device5;
    if (FAILED(device->QueryInterface(__uuidof(ID3D11Device5), (void**)&device5))) {
        fprintf(stderr, "SHARED: consumer QI(ID3D11Device5) FAILED\n");
        return 1;
    }

    /* Import both shared objects through the standard D3D entry points. */
    Com<ID3D11Texture2D> shared;
    HRESULT hr = device->OpenSharedResource((HANDLE)&wire.tex,
                                            __uuidof(ID3D11Texture2D), (void**)&shared);
    if (FAILED(hr) || !shared) {
        fprintf(stderr, "SHARED: consumer OpenSharedResource(texture) 0x%08x\n",
                (unsigned)hr);
        return 1;
    }
    Com<ID3D11Fence> fence;
    hr = device5->OpenSharedFence((HANDLE)&wire.fence, __uuidof(ID3D11Fence),
                                  (void**)&fence);
    if (FAILED(hr) || !fence) {
        fprintf(stderr, "SHARED: consumer OpenSharedFence 0x%08x\n", (unsigned)hr);
        return 1;
    }

    /* Learn the texture's shape from the imported object (not the wire), then a
     * CPU-readable staging texture to copy it into for validation. */
    D3D11_TEXTURE2D_DESC id = {};
    shared->GetDesc(&id);
    D3D11_TEXTURE2D_DESC sd = id;
    sd.BindFlags = 0; sd.MiscFlags = 0;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&sd, nullptr, &staging))) {
        fprintf(stderr, "SHARED: consumer staging CreateTexture2D FAILED\n");
        return 1;
    }
    printf("SHARED: consumer imported %ux%u tex + fence via D3D\n", id.Width, id.Height);

    const uint32_t cx = id.Width / 2, cy = id.Height / 2;
    uint8_t observed[kTicks][3] = {};
    int rc = 0;

    for (int tick = -kWarmup; tick < kTicks; tick++) {
        uint64_t value = (uint64_t)(tick + kWarmup + 1);
        /* Wait on the imported fence — pure D3D. */
        uint64_t t0 = now_ms();
        bool signaled = false;
        while (now_ms() - t0 < 10000) {
            if (fence->GetCompletedValue() >= value) { signaled = true; break; }
            struct timespec ns = {0, 1000 * 1000};
            nanosleep(&ns, nullptr);
        }
        if (!signaled) {
            fprintf(stderr, "SHARED: consumer fence wait tick %d timed out "
                    "(completed=%llu)\n", tick,
                    (unsigned long long)fence->GetCompletedValue());
            rc = 1;
            break;
        }

        /* Read the imported texture back the standard way (GPU copy -> staging
         * -> Map). Frame -1 is a warm-up: the first cross-process GPU read of a
         * freshly shared surface is stale (a CPU-polled fence carries no GPU-side
         * cross-process cache barrier), so it primes rather than validates. */
        ctx->CopyResource(staging.ptr(), shared.ptr());
        ctx->Flush();
        D3D11_MAPPED_SUBRESOURCE m = {};
        if (FAILED(ctx->Map(staging.ptr(), 0, D3D11_MAP_READ, 0, &m))) {
            fprintf(stderr, "SHARED: consumer Map(staging) FAILED\n");
            rc = 1;
            break;
        }
        const uint8_t* px = (const uint8_t*)m.pData + (uint64_t)cy * m.RowPitch + cx * 4;
        uint8_t got[3] = { px[0], px[1], px[2] };
        if (tick >= 0) dump_ppm(dumpDir, tick, (const uint8_t*)m.pData, m.RowPitch);
        ctx->Unmap(staging.ptr(), 0);

        if (tick >= 0) {
            observed[tick][0] = got[0]; observed[tick][1] = got[1]; observed[tick][2] = got[2];
            uint8_t want[3];
            expected_bgr(kColors[tick], want);
            bool match = true;
            for (int k = 0; k < 3; k++)
                if (abs((int)got[k] - (int)want[k]) > 12)
                    match = false;
            printf("SHARED: consumer tick %d center BGR=%02x%02x%02x want %02x%02x%02x %s\n",
                   tick, got[0], got[1], got[2], want[0], want[1], want[2],
                   match ? "OK" : "MISMATCH");
            if (!match)
                rc = 1;
        }

        char ack = 1;
        if (write(sock, &ack, 1) != 1) {
            fprintf(stderr, "SHARED: consumer ack write FAILED\n");
            rc = 1;
            break;
        }
    }

    /* Distinct frames prove the fence ordered separate producer writes. */
    if (rc == 0) {
        bool differ = memcmp(observed[0], observed[kTicks - 1], 3) != 0;
        if (!differ) {
            fprintf(stderr, "SHARED: colors did not change across ticks\n");
            rc = 1;
        }
    }

    if (rc == 0)
        printf("SHARED: PASS\n");
    return rc;
}

} // namespace

int main(int argc, char** argv) {
    /* Unbuffered so parent and child output stays ordered and survives the
     * child's _exit (which skips stdio flushing). */
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    /* Re-exec'd consumer instance: the child runs in a FRESH process (exec, not
     * a bare fork) so D3DMetal's texture copy/map path can reach the Metal
     * shader compiler (MTLCompilerService is unreachable across a bare fork). */
    if (argc >= 3 && strcmp(argv[1], "--consumer") == 0)
        return consumer(atoi(argv[2]));

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "SHARED: socketpair FAILED: %s\n", strerror(errno));
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "SHARED: fork FAILED: %s\n", strerror(errno));
        return 1;
    }

    if (pid == 0) {
        close(sv[0]);
        /* Pass the inherited socket fd (not CLOEXEC) to the fresh consumer. */
        char fdbuf[16];
        snprintf(fdbuf, sizeof(fdbuf), "%d", sv[1]);
        execl(argv[0], argv[0], "--consumer", fdbuf, (char*)nullptr);
        fprintf(stderr, "SHARED: execl consumer FAILED: %s\n", strerror(errno));
        _exit(127);
    }

    close(sv[1]);
    int prc = producer(sv[0]);
    close(sv[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    int crc = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    if (prc != 0 || crc != 0) {
        fprintf(stderr, "SHARED: FAIL (producer=%d consumer=%d)\n", prc, crc);
        return 1;
    }
    return 0;
}
