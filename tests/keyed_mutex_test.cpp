/*
 * Cross-process keyed mutex over a shared texture. The parent (producer)
 * creates a D3D11 texture with D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, gets an
 * IDXGIKeyedMutex via QueryInterface, and ships the texture POD + fd. The child
 * (consumer) opens the texture, QIs its own IDXGIKeyedMutex, and the two hand
 * the lock back and forth by key to serialize access.
 *
 * The mutex protects a counter written into the first bytes of the shared
 * surface: each side, while holding the lock, reads the counter, checks it
 * matches the number of completed hand-offs, bumps it, and releases with the
 * key the OTHER side waits on. If the mutex did not actually serialize, the
 * interleaved read-modify-writes would desync and the run fails. Prints
 * "KMUTEX: PASS".
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
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

constexpr uint32_t kW = 64, kH = 64;
constexpr int kRounds = 8;
/* Keys: producer acquires with kKeyP and releases with kKeyC (handing to the
 * consumer); consumer acquires with kKeyC and releases with kKeyP. Start key
 * is 0 (Windows convention: a freshly created keyed mutex is released at 0), so
 * the producer takes the first turn with an AcquireSync(0). */
constexpr uint64_t kKeyStart = 0, kKeyP = 0, kKeyC = 0x1111;

/* The shared surface is StorageModeShared; a staging copy is not needed to see
 * the bytes because the producer maps the same fd. We read/write the counter
 * through a CPU mapping the library exposes via the fd, mirroring how the
 * fence consumer polls its slot. Keeping it simple: both sides mmap the fd. */
#include <sys/mman.h>
volatile uint32_t* map_counter(int fd, uint64_t size) {
    void* p = mmap(nullptr, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return p == MAP_FAILED ? nullptr : reinterpret_cast<volatile uint32_t*>(p);
}

Com<ID3D11Device> make_device() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                 &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx)))
        return nullptr;
    return dev;
}

int producer(int sock) {
    Com<ID3D11Device> dev = make_device();
    if (!dev) { fprintf(stderr, "KMUTEX: prod device FAILED\n"); return 1; }

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = kW; td.Height = kH; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    Com<ID3D11Texture2D> tex;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex))) {
        fprintf(stderr, "KMUTEX: prod CreateTexture2D(KEYEDMUTEX) FAILED\n");
        return 1;
    }
    Com<IDXGIResource> res;
    HANDLE h = nullptr;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
        FAILED(res->GetSharedHandle(&h)) || !h) {
        fprintf(stderr, "KMUTEX: prod GetSharedHandle FAILED\n");
        return 1;
    }
    Com<IDXGIKeyedMutex> km;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&km)) || !km) {
        fprintf(stderr, "KMUTEX: prod QI(IDXGIKeyedMutex) FAILED — keyed mutex "
                "not vended\n");
        return 1;
    }
    dmn_shared_texture_handle pod;
    memcpy(&pod, h, sizeof(pod));
    if (!send_with_fd(sock, &pod, sizeof(pod), pod.fd)) {
        fprintf(stderr, "KMUTEX: prod send FAILED: %s\n", strerror(errno));
        return 1;
    }
    volatile uint32_t* counter = map_counter(pod.fd, pod.stride * kH);
    if (!counter) { fprintf(stderr, "KMUTEX: prod map FAILED\n"); return 1; }
    counter[0] = 0;

    for (int r = 0; r < kRounds; r++) {
        uint64_t key = (r == 0) ? kKeyStart : kKeyP;
        if (FAILED(km->AcquireSync(key, 5000))) {
            fprintf(stderr, "KMUTEX: prod AcquireSync(%llu) round %d FAILED/timeout\n",
                    (unsigned long long)key, r);
            return 1;
        }
        uint32_t seen = counter[0];
        uint32_t expect = (uint32_t)(2 * r);
        if (seen != expect) {
            fprintf(stderr, "KMUTEX: prod desync round %d: counter=%u expect=%u\n",
                    r, seen, expect);
            return 1;
        }
        counter[0] = seen + 1;
        km->ReleaseSync(kKeyC); /* hand to consumer */
    }
    printf("KMUTEX: producer done\n");
    return 0;
}

int consumer(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS) return 1;
    Com<ID3D11Device> dev = make_device();
    if (!dev) { fprintf(stderr, "KMUTEX: cons device FAILED\n"); return 1; }

    dmn_shared_texture_handle pod;
    int fd = -1;
    if (!recv_with_fd(sock, &pod, sizeof(pod), &fd)) {
        fprintf(stderr, "KMUTEX: cons recv FAILED\n"); return 1;
    }
    pod.fd = fd;
    if (pod.magic != DMN_SHARED_TEXTURE_MAGIC) {
        fprintf(stderr, "KMUTEX: cons bad magic 0x%08x\n", pod.magic); return 1;
    }
    Com<ID3D11Texture2D> tex;
    if (FAILED(dev->OpenSharedResource((HANDLE)&pod, __uuidof(ID3D11Texture2D),
                                       (void**)&tex))) {
        fprintf(stderr, "KMUTEX: cons OpenSharedResource FAILED\n"); return 1;
    }
    Com<IDXGIKeyedMutex> km;
    if (FAILED(tex->QueryInterface(__uuidof(IDXGIKeyedMutex), (void**)&km)) || !km) {
        fprintf(stderr, "KMUTEX: cons QI(IDXGIKeyedMutex) FAILED\n"); return 1;
    }
    volatile uint32_t* counter = map_counter(pod.fd, pod.stride * kH);
    if (!counter) { fprintf(stderr, "KMUTEX: cons map FAILED\n"); return 1; }

    for (int r = 0; r < kRounds; r++) {
        if (FAILED(km->AcquireSync(kKeyC, 5000))) {
            fprintf(stderr, "KMUTEX: cons AcquireSync round %d FAILED/timeout\n", r);
            return 1;
        }
        uint32_t seen = counter[0];
        uint32_t expect = (uint32_t)(2 * r + 1);
        if (seen != expect) {
            fprintf(stderr, "KMUTEX: cons desync round %d: counter=%u expect=%u\n",
                    r, seen, expect);
            return 1;
        }
        counter[0] = seen + 1;
        km->ReleaseSync(kKeyP); /* hand back to producer */
    }
    printf("KMUTEX: PASS\n");
    return 0;
}

} // namespace

namespace {
int producer_main(int sock) {
    if (dmn_init(nullptr) != DMN_SUCCESS)
        return 1;
    return producer(sock);
}
} // namespace

int main() {
    return run_fork_pair("KMUTEX", producer_main, consumer);
}
