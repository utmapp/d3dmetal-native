/*
 * D3D12 shared heaps + placed resources, windowless, single process. The
 * placed create routes through the lazily swizzled Metal HEAP class (a
 * different interception path than committed resources), so this covers:
 *
 *  1. CreateHeap(D3D12_HEAP_FLAG_SHARED) -> heap tracked.
 *  2. CreatePlacedResource on the shared heap -> substituted shm backing;
 *     export with CreateSharedHandle, POD dims verified.
 *  3. OpenSharedHandle round trip; a marker written through the exported fd
 *     must be readable through a re-export from the OPENED resource (same
 *     memory), proving the substitution actually backs both objects.
 *  4. CreateCommittedResource1 (ID3D12Device4) shared create, when available.
 *
 * Prints "HEAP: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "dmn_d3d12_up.h" /* ID3D12Device4 (vendored d3d12.h stops at Device1) */
#include "common/com.h"

#define T_TAG "HEAP"
#include "common/check.h"
#include "common/dx12.h"

static D3D12_RESOURCE_DESC tex_desc(UINT w, UINT h) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    return rd;
}

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "HEAP: dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");

    /* 1) Shared heap. */
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = 8ull * 1024 * 1024;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = 0;
    hd.Flags = D3D12_HEAP_FLAG_SHARED;
    Com<ID3D12Heap> heap;
    CK(dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&heap), "CreateHeap(SHARED)");

    /* 2) Placed texture on it -> export. */
    const UINT kW = 192, kH = 96;
    D3D12_RESOURCE_DESC rd = tex_desc(kW, kH);
    Com<ID3D12Resource> placed;
    CK(dev->CreatePlacedResource(heap.ptr(), 0, &rd, D3D12_RESOURCE_STATE_COMMON,
                                 nullptr, __uuidof(ID3D12Resource),
                                 (void**)&placed), "CreatePlacedResource");
    HANDLE h = nullptr;
    CK(dev->CreateSharedHandle(placed.ptr(), nullptr, 0, nullptr, &h),
       "CreateSharedHandle(placed)");
    auto* pod = (dmn_shared_texture_handle*)h;
    EXPECT(pod->magic == DMN_SHARED_TEXTURE_MAGIC, "bad POD magic");
    EXPECT(pod->width == kW && pod->height == kH, "POD dims mismatch");

    /* 3) Open + prove shared backing via a marker through the fds. */
    Com<ID3D12Resource> opened;
    CK(dev->OpenSharedHandle(h, __uuidof(ID3D12Resource), (void**)&opened),
       "OpenSharedHandle(placed)");
    D3D12_RESOURCE_DESC od = opened->GetDesc();
    EXPECT(od.Width == kW && od.Height == kH, "opened desc mismatch");

    void* map1 = mmap(nullptr, (size_t)pod->size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, pod->fd, 0);
    EXPECT(map1 != MAP_FAILED, "mmap(exported fd) failed");
    const uint32_t kMarker = 0xC0FFEE42u;
    memcpy(map1, &kMarker, sizeof(kMarker));

    HANDLE h2 = nullptr;
    CK(dev->CreateSharedHandle(opened.ptr(), nullptr, 0, nullptr, &h2),
       "CreateSharedHandle(opened)");
    auto* pod2 = (dmn_shared_texture_handle*)h2;
    void* map2 = mmap(nullptr, (size_t)pod2->size, PROT_READ, MAP_SHARED,
                      pod2->fd, 0);
    EXPECT(map2 != MAP_FAILED, "mmap(re-exported fd) failed");
    uint32_t got = 0;
    memcpy(&got, map2, sizeof(got));
    EXPECT(got == kMarker, "marker not visible through the re-export — "
           "placed resource not backed by the shared memory");
    munmap(map1, (size_t)pod->size);
    munmap(map2, (size_t)pod2->size);
    CK(dmn_shared_handle_close(h2) == DMN_SUCCESS ? S_OK : E_FAIL, "close h2");
    CK(dmn_shared_handle_close(h) == DMN_SUCCESS ? S_OK : E_FAIL, "close h");
    printf("HEAP: placed round trip ok (%ux%u, stride %llu)\n", kW, kH,
           (unsigned long long)pod2->stride);

    /* 4) CreateCommittedResource1, when the device exposes ID3D12Device4. */
    {
        Com<ID3D12Device4> dev4;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device4),
                                          (void**)&dev4)) && dev4) {
            D3D12_HEAP_PROPERTIES hp{};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd1 = tex_desc(64, 64);
            Com<ID3D12Resource> tex;
            CK(dev4->CreateCommittedResource1(&hp, D3D12_HEAP_FLAG_SHARED, &rd1,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, nullptr,
                                              __uuidof(ID3D12Resource),
                                              (void**)&tex),
               "CreateCommittedResource1(SHARED)");
            HANDLE h1 = nullptr;
            CK(dev->CreateSharedHandle(tex.ptr(), nullptr, 0, nullptr, &h1),
               "CreateSharedHandle(CCR1)");
            auto* p1 = (dmn_shared_texture_handle*)h1;
            EXPECT(p1->magic == DMN_SHARED_TEXTURE_MAGIC && p1->width == 64,
                   "CCR1 POD mismatch");
            CK(dmn_shared_handle_close(h1) == DMN_SUCCESS ? S_OK : E_FAIL,
               "close h1");
            printf("HEAP: CreateCommittedResource1 shared create ok\n");
        } else {
            printf("HEAP: ID3D12Device4 unavailable; CCR1 skipped\n");
        }
    }

    T_PASS();
    return 0;
}
