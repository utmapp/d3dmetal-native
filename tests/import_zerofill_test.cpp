/*
 * An import must deliver the producer's bytes no matter what the process
 * did before it.
 *
 * This framework memsets a committed resource's contents inside the
 * create, so every hooked create whose allocation is substituted over
 * EXISTING bytes must pass D3D12_HEAP_FLAG_CREATE_NOT_ZEROED — including
 * the import creates, which alias a producer's data by definition.
 *
 * Phase 1 runs a placed-buffer create on an imported heap over a nonzero
 * window (create history a streaming title generates constantly).  Phase 2
 * round-trips a GPU-written pattern through CreateSharedHandle/
 * OpenSharedHandle and requires it back intact.  On the pre-fix code the
 * open's unflagged committed create memset the shared window and the
 * pattern came back zeros (verified: FAILS, 16384/16384 words); with the
 * flag the bytes are never touched.
 *
 * Prints "IMPORTZF: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "dmn_share.h" /* dmn_share_anon_file */
#include "common/com.h"

#define T_TAG "IMPORTZF"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

static const IID kIidHeap = __uuidof(ID3D12Heap);

namespace {

const uint64_t kBufSize = 64 * 1024;

D3D12_RESOURCE_DESC buf_desc(uint64_t bytes) {
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return rd;
}

/* One executed-and-waited CopyBufferRegion (heap_import_test's shape). */
struct GpuCopier {
    Com<ID3D12Device> dev;
    Com<ID3D12CommandQueue> queue;
    Com<ID3D12CommandAllocator> alloc;
    Com<ID3D12GraphicsCommandList> list;
    Com<ID3D12Fence> fence;
    UINT64 fence_value = 0;

    bool init(ID3D12Device* d) {
        dev = d;
        dev->AddRef();
        if (FAILED(make_d3d12_queue(dev.ptr(), queue)))
            return false;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               __uuidof(ID3D12CommandAllocator),
                                               (void**)&alloc)))
            return false;
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          alloc.ptr(), nullptr,
                                          __uuidof(ID3D12GraphicsCommandList),
                                          (void**)&list)))
            return false;
        list->Close();
        return SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                          __uuidof(ID3D12Fence),
                                          (void**)&fence));
    }

    bool copy(ID3D12Resource* dst, ID3D12Resource* src, uint64_t bytes) {
        if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.ptr(), nullptr)))
            return false;
        list->CopyBufferRegion(dst, 0, src, 0, bytes);
        if (FAILED(list->Close()))
            return false;
        ID3D12CommandList* lists[] = {list.ptr()};
        queue->ExecuteCommandLists(1, lists);
        const UINT64 target = ++fence_value;
        queue->Signal(fence.ptr(), target);
        const uint64_t start = now_ms();
        while (fence->GetCompletedValue() < target) {
            if (now_ms() - start > 5000) {
                fprintf(stderr, T_TAG ": GPU copy timed out\n");
                return false;
            }
            sleep_ms(1);
        }
        return true;
    }
};

HRESULT make_committed(ID3D12Device* dev, D3D12_HEAP_TYPE type,
                       D3D12_HEAP_FLAGS flags, D3D12_RESOURCE_STATES state,
                       Com<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    D3D12_RESOURCE_DESC rd = buf_desc(kBufSize);
    return dev->CreateCommittedResource(&hp, flags, &rd, state, nullptr,
                                        __uuidof(ID3D12Resource), (void**)&out);
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");
    GpuCopier gpu;
    EXPECT(gpu.init(dev.ptr()), "copier init failed");

    /* Phase 1: prior create history — a placed create over a nonzero
     * imported window, as a streaming title generates constantly. */
    {
        const uint64_t kWinSize = 256 * 1024;
        const int fd = dmn_share_anon_file((size_t)kWinSize);
        EXPECT(fd >= 0, "dmn_share_anon_file failed");
        void* map = mmap(nullptr, (size_t)kWinSize, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
        EXPECT(map != MAP_FAILED, "mmap(shm fd) failed");
        memset(map, 0xA5, (size_t)kWinSize);

        Com<ID3D12Heap> heap;
        CK(dmn_open_existing_heap_from_fd(dev.ptr(), fd, 0, kWinSize,
                                          (uint32_t)D3D12_HEAP_TYPE_UPLOAD, 0,
                                          &kIidHeap, (void**)&heap),
           "dmn_open_existing_heap_from_fd");
        D3D12_RESOURCE_DESC rd = buf_desc(kBufSize);
        Com<ID3D12Resource> placed;
        CK(dev->CreatePlacedResource(heap.ptr(), 0, &rd,
                                     D3D12_RESOURCE_STATE_GENERIC_READ,
                                     nullptr, __uuidof(ID3D12Resource),
                                     (void**)&placed),
           "CreatePlacedResource");
        munmap(map, (size_t)kWinSize);
        close(fd);
    }

    /* Phase 2: shared round trip.  GPU-write a pattern into a shared
     * buffer, wait it out, then open the shared handle — the open's create
     * must not cost the producer's bytes. */
    Com<ID3D12Resource> shared_buf;
    CK(make_committed(dev.ptr(), D3D12_HEAP_TYPE_DEFAULT,
                      D3D12_HEAP_FLAG_SHARED,
                      D3D12_RESOURCE_STATE_COMMON, shared_buf),
       "CreateCommittedResource(SHARED)");

    Com<ID3D12Resource> upload;
    CK(make_committed(dev.ptr(), D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ, upload),
       "CreateCommittedResource(UPLOAD)");
    {
        void* p = nullptr;
        D3D12_RANGE none{0, 0};
        CK(upload->Map(0, &none, &p), "Map(upload)");
        uint32_t* w = (uint32_t*)p;
        for (uint64_t i = 0; i < kBufSize / 4; i++)
            w[i] = 0x5AFEB10Bu ^ (uint32_t)i;
        upload->Unmap(0, nullptr);
    }
    EXPECT(gpu.copy(shared_buf.ptr(), upload.ptr(), kBufSize),
           "upload copy failed");

    HANDLE shared = nullptr;
    CK(dev->CreateSharedHandle(shared_buf.ptr(), nullptr, 0, nullptr, &shared),
       "CreateSharedHandle");
    Com<ID3D12Resource> imported;
    CK(dev->OpenSharedHandle(shared, __uuidof(ID3D12Resource),
                             (void**)&imported),
       "OpenSharedHandle");

    Com<ID3D12Resource> readback;
    CK(make_committed(dev.ptr(), D3D12_HEAP_TYPE_READBACK,
                      D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                      readback),
       "CreateCommittedResource(READBACK)");
    EXPECT(gpu.copy(readback.ptr(), imported.ptr(), kBufSize),
           "readback copy failed");
    {
        void* p = nullptr;
        D3D12_RANGE all{0, (SIZE_T)kBufSize};
        CK(readback->Map(0, &all, &p), "Map(readback)");
        const uint32_t* r = (const uint32_t*)p;
        uint64_t bad = 0;
        for (uint64_t i = 0; i < kBufSize / 4; i++)
            if (r[i] != (0x5AFEB10Bu ^ (uint32_t)i))
                bad++;
        D3D12_RANGE nowrite{0, 0};
        readback->Unmap(0, &nowrite);
        printf(T_TAG ": %llu of %llu words wrong through the import "
               "(first=0x%08x)\n", (unsigned long long)bad,
               (unsigned long long)(kBufSize / 4), r[0]);
        EXPECT(bad == 0, "the import wiped the producer's bytes");
    }

    T_PASS();
    return 0;
}
