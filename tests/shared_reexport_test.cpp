/*
 * create -> OpenShared* -> release-creator -> re-export.
 *
 * MSDN (IDXGIResource::GetSharedHandle remarks): handle validity is tied to
 * the underlying memory, and "to extend the lifetime of the handle and video
 * memory, you must open the shared resource on a device" — an opened resource
 * keeps the sharing alive after the creator releases, and re-exporting from
 * an opened resource is explicitly supported. This test drives that sequence
 * same-process for both APIs (D3D12 has no legacy handles — its CreateShared/
 * OpenSharedHandle pair is NT-handle-shaped, same lifetime rule):
 *
 *   1. create a SHARED texture, export it, open the handle in the same
 *      process, and retain a raw mapping of the original object,
 *   2. release the CREATOR and let its Metal backing (and its fd) actually
 *      die — the retained mapping outlives the fd,
 *   3. re-export from the opened resource; the vended fd must be alive, a
 *      marker written through it must appear through the retained original
 *      mapping (same shm object), and the new handle must itself open.
 *
 * Prints "REEXPORT: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

namespace {

constexpr uint32_t kW = 64, kH = 64;
constexpr uint32_t kMarker = 0x5AFEC0DEu;

void settle_ms(unsigned ms) {
    struct timespec ns = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000L};
    nanosleep(&ns, nullptr);
}

/* The vended fd must be a live descriptor AND reference the same shm object
 * as `viewA` — a mapping of the ORIGINAL export retained from before the
 * creator was released (a mapping outlives its fd). Proven by writing a
 * marker through the re-exported fd and reading it back through viewA. */
bool check_reexport_fd(const char* api, int fd, uint64_t size,
                       volatile uint32_t* viewA) {
    if (fcntl(fd, F_GETFD) < 0) {
        fprintf(stderr, "REEXPORT: %s re-exported fd=%d is DEAD\n", api, fd);
        return false;
    }
    void* m = mmap(nullptr, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
    if (m == MAP_FAILED) {
        fprintf(stderr, "REEXPORT: %s re-exported fd=%d does not map\n", api, fd);
        return false;
    }
    *reinterpret_cast<volatile uint32_t*>(m) = kMarker;
    uint32_t got = *viewA;
    munmap(m, (size_t)size);
    if (got != kMarker) {
        fprintf(stderr, "REEXPORT: %s re-exported fd=%d is a DIFFERENT object "
                "(0x%08x != 0x%08x through the original mapping)\n",
                api, fd, got, kMarker);
        return false;
    }
    return true;
}

int test_d3d11() {
    Com<ID3D11Device> dev;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                 D3D11_CREATE_DEVICE_BGRA_SUPPORT, &fl, 1,
                                 D3D11_SDK_VERSION, &dev, &flo, &ctx))) {
        fprintf(stderr, "REEXPORT: D3D11CreateDevice FAILED\n");
        return 1;
    }

    D3D11_TEXTURE2D_DESC td{};
    td.Width = kW;
    td.Height = kH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Texture2D> creator;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &creator))) {
        fprintf(stderr, "REEXPORT: D3D11 CreateTexture2D(SHARED) FAILED\n");
        return 1;
    }
    Com<IDXGIResource> res;
    HANDLE h = nullptr;
    if (FAILED(creator->QueryInterface(__uuidof(IDXGIResource), (void**)&res)) ||
        FAILED(res->GetSharedHandle(&h)) || !h) {
        fprintf(stderr, "REEXPORT: D3D11 export FAILED\n");
        return 1;
    }
    dmn_shared_texture_handle podA;
    memcpy(&podA, h, sizeof(podA));

    /* Legacy handle: dmn_shared_handle_close must reject it untouched. */
    if (dmn_shared_handle_close(h) == DMN_SUCCESS) {
        fprintf(stderr, "REEXPORT: close ACCEPTED a legacy handle\n");
        return 1;
    }

    Com<ID3D11Texture2D> opened;
    if (FAILED(dev->OpenSharedResource((HANDLE)&podA, __uuidof(ID3D11Texture2D),
                                       (void**)&opened)) || !opened) {
        fprintf(stderr, "REEXPORT: D3D11 OpenSharedResource FAILED\n");
        return 1;
    }
    /* Retain a view of the original object while its fd is still alive; the
     * mapping survives the creator's fd being closed. */
    void* viewA = mmap(nullptr, (size_t)podA.size, PROT_READ, MAP_SHARED,
                       podA.fd, 0);
    if (viewA == MAP_FAILED) {
        fprintf(stderr, "REEXPORT: D3D11 original mapping FAILED\n");
        return 1;
    }

    /* Release the creator and give its Metal backing time to actually die
     * (this is what closes the creator's fd). */
    res = nullptr;
    creator = nullptr;
    ctx->Flush();
    settle_ms(700);

    Com<IDXGIResource> res2;
    HANDLE h2 = nullptr;
    if (FAILED(opened->QueryInterface(__uuidof(IDXGIResource), (void**)&res2)) ||
        FAILED(res2->GetSharedHandle(&h2)) || !h2) {
        fprintf(stderr, "REEXPORT: D3D11 re-export after creator release FAILED\n");
        return 1;
    }
    dmn_shared_texture_handle podB;
    memcpy(&podB, h2, sizeof(podB));
    bool ok = podB.magic == DMN_SHARED_TEXTURE_MAGIC &&
              check_reexport_fd("D3D11", podB.fd, podB.size,
                                reinterpret_cast<volatile uint32_t*>(viewA));
    munmap(viewA, (size_t)podA.size);
    if (!ok)
        return 1;

    /* And the re-exported handle must itself open. */
    Com<ID3D11Texture2D> opened2;
    if (FAILED(dev->OpenSharedResource((HANDLE)&podB, __uuidof(ID3D11Texture2D),
                                       (void**)&opened2)) || !opened2) {
        fprintf(stderr, "REEXPORT: D3D11 open of re-exported handle FAILED\n");
        return 1;
    }

    printf("REEXPORT: D3D11 OK (creator fd=%d, re-export fd=%d)\n",
           podA.fd, podB.fd);
    return 0;
}

int test_d3d12() {
    Com<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                 __uuidof(ID3D12Device), (void**)&dev))) {
        fprintf(stderr, "REEXPORT: D3D12CreateDevice FAILED\n");
        return 1;
    }

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
    Com<ID3D12Resource> creator;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                            __uuidof(ID3D12Resource),
                                            (void**)&creator))) {
        fprintf(stderr, "REEXPORT: D3D12 shared create FAILED\n");
        return 1;
    }
    HANDLE h = nullptr;
    if (FAILED(dev->CreateSharedHandle(creator.ptr(), nullptr, 0, nullptr, &h)) ||
        !h) {
        fprintf(stderr, "REEXPORT: D3D12 export FAILED\n");
        return 1;
    }
    dmn_shared_texture_handle podA;
    memcpy(&podA, h, sizeof(podA));

    Com<ID3D12Resource> opened;
    if (FAILED(dev->OpenSharedHandle((HANDLE)&podA, __uuidof(ID3D12Resource),
                                     (void**)&opened)) || !opened) {
        fprintf(stderr, "REEXPORT: D3D12 OpenSharedHandle FAILED\n");
        return 1;
    }
    /* Retain a view of the original object while its fd is still alive. */
    void* viewA = mmap(nullptr, (size_t)podA.size, PROT_READ, MAP_SHARED,
                       podA.fd, 0);
    if (viewA == MAP_FAILED) {
        fprintf(stderr, "REEXPORT: D3D12 original mapping FAILED\n");
        return 1;
    }

    /* NT-style handle: owned by us; close once its fd is no longer needed. */
    if (dmn_shared_handle_close(h) != DMN_SUCCESS) {
        fprintf(stderr, "REEXPORT: D3D12 close(h) FAILED\n");
        return 1;
    }

    creator = nullptr;
    settle_ms(700);

    HANDLE h2 = nullptr;
    if (FAILED(dev->CreateSharedHandle(opened.ptr(), nullptr, 0, nullptr, &h2)) ||
        !h2) {
        fprintf(stderr, "REEXPORT: D3D12 re-export after creator release FAILED\n");
        return 1;
    }
    dmn_shared_texture_handle podB;
    memcpy(&podB, h2, sizeof(podB));
    bool ok = podB.magic == DMN_SHARED_TEXTURE_MAGIC &&
              check_reexport_fd("D3D12", podB.fd, podB.size,
                                reinterpret_cast<volatile uint32_t*>(viewA));
    munmap(viewA, (size_t)podA.size);
    if (!ok)
        return 1;

    Com<ID3D12Resource> opened2;
    if (FAILED(dev->OpenSharedHandle((HANDLE)&podB, __uuidof(ID3D12Resource),
                                     (void**)&opened2)) || !opened2) {
        fprintf(stderr, "REEXPORT: D3D12 open of re-exported handle FAILED\n");
        return 1;
    }

    if (dmn_shared_handle_close(h2) != DMN_SUCCESS) {
        fprintf(stderr, "REEXPORT: D3D12 close(h2) FAILED\n");
        return 1;
    }
    if (dmn_shared_handle_close(h2) == DMN_SUCCESS) {
        fprintf(stderr, "REEXPORT: double close ACCEPTED\n");
        return 1;
    }

    /* NT handles must outlive their resource: create -> export -> RELEASE ->
     * open from the handle itself — the sequence legacy handles cannot do
     * (the handle's own dup fd is what keeps the allocation alive). */
    Com<ID3D12Resource> late;
    HANDLE h3 = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                            __uuidof(ID3D12Resource),
                                            (void**)&late)) ||
        FAILED(dev->CreateSharedHandle(late.ptr(), nullptr, 0, nullptr, &h3)) ||
        !h3) {
        fprintf(stderr, "REEXPORT: NT-lifetime setup FAILED\n");
        return 1;
    }
    late = nullptr;
    settle_ms(700);
    Com<ID3D12Resource> lateOpened;
    if (FAILED(dev->OpenSharedHandle(h3, __uuidof(ID3D12Resource),
                                     (void**)&lateOpened)) || !lateOpened) {
        fprintf(stderr, "REEXPORT: open AFTER creator release via NT handle "
                "FAILED\n");
        return 1;
    }
    if (dmn_shared_handle_close(h3) != DMN_SUCCESS) {
        fprintf(stderr, "REEXPORT: D3D12 close(h3) FAILED\n");
        return 1;
    }

    printf("REEXPORT: D3D12 OK (creator fd=%d, re-export fd=%d, "
           "NT handle outlived its resource)\n", podA.fd, podB.fd);
    return 0;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "REEXPORT: dmn_init FAILED\n");
        return 1;
    }
    if (test_d3d11() != 0)
        return 1;
    if (test_d3d12() != 0)
        return 1;
    printf("REEXPORT: PASS\n");
    return 0;
}
