/*
 * D3D12 shared buffers through the STANDARD APIs only: create with
 * D3D12_HEAP_FLAG_SHARED, export with ID3D12Device::CreateSharedHandle, re-import
 * with ID3D12Device::OpenSharedHandle. No dmn_* sharing calls beyond closing the
 * handle.
 *
 *  1. The opaque handle round-trips and yields a resource describing the same
 *     buffer.
 *  2. Opening a buffer leaves the producer's bytes intact (an import must never
 *     initialise the memory it aliases).
 *  3. Two shared buffers opened in one process stay distinct: a GPU write
 *     through one import lands in that buffer's memory and in no other's, and
 *     is visible through the producer's own view of the same buffer.
 *
 * Prints "SHAREDBUF: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "SHAREDBUF"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

#include <cstring>
#include <sys/mman.h>

namespace {

/* Submit one CopyBufferRegion and wait for it. */
struct Copier {
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
        list->Close();
        return SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                          __uuidof(ID3D12Fence), (void**)&fence));
    }
    bool copy(ID3D12Resource* dst, ID3D12Resource* src, uint64_t bytes) {
        if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.ptr(), nullptr)))
            return false;
        list->CopyBufferRegion(dst, 0, src, 0, bytes);
        if (FAILED(list->Close()))
            return false;
        ID3D12CommandList* ls[] = {list.ptr()};
        queue->ExecuteCommandLists(1, ls);
        const UINT64 want = ++fv;
        queue->Signal(fence.ptr(), want);
        const uint64_t t0 = now_ms();
        while (fence->GetCompletedValue() < want) {
            if (now_ms() - t0 > 10000)
                return false;
            sleep_ms(1);
        }
        return true;
    }
};

D3D12_RESOURCE_DESC buf_desc(uint64_t w, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = w;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    return rd;
}

uint32_t pattern(uint32_t seed, uint64_t i) {
    return seed ^ (uint32_t)(i * 2654435761u);
}

/* Count mismatches against pattern(seed) over a committed buffer's object (a
 * committed buffer owns its whole object, so it starts at offset 0). */
int count_bad(const dmn_shared_buffer_handle* pod, uint32_t seed, size_t bytes) {
    void* m = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, pod->fd, 0);
    if (m == MAP_FAILED)
        return -1;
    int bad = 0;
    const uint32_t* w = (const uint32_t*)m;
    for (uint64_t i = 0; i < bytes / 4; i += 61)
        if (w[i] != pattern(seed, i))
            bad++;
    munmap(m, bytes);
    return bad;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "SHAREDBUF: dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D12Device> device;
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
                         (void**)&device),
       "D3D12CreateDevice");

    /* Standard D3D12 shared buffer: CreateCommittedResource + HEAP_FLAG_SHARED. */
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 4096;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Com<ID3D12Resource> buf;
    CK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                       __uuidof(ID3D12Resource), (void**)&buf),
       "CreateCommittedResource(SHARED buffer)");

    /* Export + re-import through the standard APIs; the HANDLE is opaque. */
    HANDLE shared = nullptr;
    CK(device->CreateSharedHandle(buf.ptr(), nullptr, 0, nullptr, &shared),
       "CreateSharedHandle");
    Com<ID3D12Resource> imported;
    CK(device->OpenSharedHandle(shared, __uuidof(ID3D12Resource), (void**)&imported),
       "OpenSharedHandle");
    CK(dmn_shared_handle_close(shared) == DMN_SUCCESS ? S_OK : E_FAIL,
       "dmn_shared_handle_close");

    D3D12_RESOURCE_DESC id = imported->GetDesc();
    EXPECT(imported && id.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
           id.Width == rd.Width, "imported buffer does not describe the original");
    printf("SHAREDBUF: export+import round trip ok (%llu-byte buffer)\n",
           (unsigned long long)id.Width);

    /* 2 + 3: GPU data flow through the handles. */
    {
        const uint64_t kSz = 1ull << 20;
        Copier gpu;
        EXPECT(gpu.init(device.ptr()), "copier init failed");
        D3D12_HEAP_PROPERTIES up = {};
        up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC srd = buf_desc(kSz, D3D12_RESOURCE_FLAG_NONE);
        Com<ID3D12Resource> srcA, srcB;
        CK(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &srd,
                                           D3D12_RESOURCE_STATE_GENERIC_READ,
                                           nullptr, __uuidof(ID3D12Resource),
                                           (void**)&srcA), "upload src A");
        CK(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &srd,
                                           D3D12_RESOURCE_STATE_GENERIC_READ,
                                           nullptr, __uuidof(ID3D12Resource),
                                           (void**)&srcB), "upload src B");
        for (int k = 0; k < 2; k++) {
            ID3D12Resource* src = k ? srcB.ptr() : srcA.ptr();
            void* p = nullptr;
            D3D12_RANGE none{0, 0};
            CK(src->Map(0, &none, &p), "Map(src)");
            auto* w = (uint32_t*)p;
            for (uint64_t i = 0; i < kSz / 4; i++)
                w[i] = pattern(k ? 0xB0B00000u : 0xA0A00000u, i);
            src->Unmap(0, nullptr);
        }

        D3D12_RESOURCE_DESC drd = buf_desc(kSz, D3D12_RESOURCE_FLAG_NONE);
        Com<ID3D12Resource> a, b;
        CK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &drd,
                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           __uuidof(ID3D12Resource), (void**)&a),
           "CreateCommittedResource(SHARED A)");
        CK(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &drd,
                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           __uuidof(ID3D12Resource), (void**)&b),
           "CreateCommittedResource(SHARED B)");
        HANDLE ha = nullptr, hb = nullptr;
        CK(device->CreateSharedHandle(a.ptr(), nullptr, 0, nullptr, &ha), "export A");
        CK(device->CreateSharedHandle(b.ptr(), nullptr, 0, nullptr, &hb), "export B");
        auto* pa = (const dmn_shared_buffer_handle*)ha;
        auto* pb = (const dmn_shared_buffer_handle*)hb;

        /* 2) Fill A through the producer, open it, and check the bytes
         * survived the open. */
        EXPECT(gpu.copy(a.ptr(), srcA.ptr(), kSz), "GPU copy into A");
        Com<ID3D12Resource> a2;
        CK(device->OpenSharedHandle(ha, __uuidof(ID3D12Resource), (void**)&a2),
           "OpenSharedHandle(A)");
        EXPECT(count_bad(pa, 0xA0A00000u, (size_t)kSz) == 0,
               "opening A disturbed the bytes the producer wrote");
        printf("SHAREDBUF: producer bytes intact across an open\n");

        /* 3) Write B THROUGH its import; only B's memory may change, and A's
         * own view must still see A's bytes. */
        Com<ID3D12Resource> b2;
        CK(device->OpenSharedHandle(hb, __uuidof(ID3D12Resource), (void**)&b2),
           "OpenSharedHandle(B)");
        EXPECT(gpu.copy(b2.ptr(), srcB.ptr(), kSz), "GPU copy into B via import");
        EXPECT(count_bad(pb, 0xB0B00000u, (size_t)kSz) == 0,
               "GPU write through B's import did not reach B's memory");
        EXPECT(count_bad(pa, 0xA0A00000u, (size_t)kSz) == 0,
               "GPU write through B's import changed A — two shared buffers "
               "share one Metal backing");
        /* And back through the producer's own resource: copy B into a
         * readback via the producer-side object. */
        D3D12_HEAP_PROPERTIES rb = {};
        rb.Type = D3D12_HEAP_TYPE_READBACK;
        Com<ID3D12Resource> read;
        CK(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &drd,
                                           D3D12_RESOURCE_STATE_COPY_DEST,
                                           nullptr, __uuidof(ID3D12Resource),
                                           (void**)&read), "readback");
        EXPECT(gpu.copy(read.ptr(), b.ptr(), kSz), "GPU copy B -> readback");
        {
            void* p = nullptr;
            CK(read->Map(0, nullptr, &p), "Map(readback)");
            int bad = 0;
            const uint32_t* w = (const uint32_t*)p;
            for (uint64_t i = 0; i < kSz / 4; i += 61)
                if (w[i] != pattern(0xB0B00000u, i))
                    bad++;
            D3D12_RANGE none{0, 0};
            read->Unmap(0, &none);
            EXPECT(bad == 0, "producer view of B does not see the import's write");
        }
        dmn_shared_handle_close(ha);
        dmn_shared_handle_close(hb);
        printf("SHAREDBUF: GPU write via import lands in its own object only\n");
    }

    T_PASS();
    return 0;
}
