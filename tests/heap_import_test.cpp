/*
 * dmn_open_existing_heap_from_fd: a window of a shared-memory fd imported as
 * an ID3D12Heap whose placed buffers alias the caller's own pages — the host
 * half of the Neptune shmem heap import (guest UPLOAD/READBACK rings Mapped
 * persistently). Windowless, single process:
 *
 *  1. Import [1 MiB, +2 MiB) of a 3 MiB shm object as an UPLOAD-typed heap;
 *     GetDesc reflects the window.
 *  2. CPU->GPU: bytes written through the caller's mmap appear in a GPU
 *     CopyBufferRegion out of a buffer placed at a nonzero heap offset —
 *     offsets compose (import offset + heap offset) and the GPU reads the
 *     caller's pages, not a shadow.
 *  3. GPU->CPU: a copy into a buffer placed on a DEFAULT-typed import lands
 *     in the caller's mmap, and the first heap's window is untouched.
 *  3b. A buffer with ALLOW_UNORDERED_ACCESS places on the UPLOAD-typed import
 *     (a flag D3D12 forbids on CPU heaps) and still aliases the window.
 *  4. Releasing the heap does not tear down a live placed buffer's backing
 *     (the copy still round-trips), matching the guest-side pin semantics.
 *  5. Validation refuses: unaligned offset, window past EOF, zero/huge size,
 *     non-heap iid, non-device object, texture placements, out-of-window
 *     placements — and none of it leaks an fd (poll, D3DMetal defers).
 *
 * Prints "HEAPIMP: PASS" and exits 0 on success.
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

#define T_TAG "HEAPIMP"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

namespace {

constexpr uint64_t kFileSize = 3ull << 20;
constexpr uint64_t kImpOff   = 1ull << 20; /* import window start in the fd */
constexpr uint64_t kImpSize  = 2ull << 20;
constexpr uint64_t kPlaceOff = 64 * 1024;  /* placed offset inside the heap */
constexpr uint64_t kBufSize  = 128 * 1024;

/* dmn_open_existing_heap_from_fd takes a GUID pointer; __uuidof here yields a
 * temporary, so give the two IIDs the test passes an address. */
const GUID kIidHeap     = __uuidof(ID3D12Heap);
const GUID kIidResource = __uuidof(ID3D12Resource);

D3D12_RESOURCE_DESC buf_desc(uint64_t width) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = width;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return rd;
}

struct GpuCopier {
    Com<ID3D12Device> dev;
    Com<ID3D12CommandQueue> queue;
    Com<ID3D12CommandAllocator> alloc;
    Com<ID3D12GraphicsCommandList> list;
    Com<ID3D12Fence> fence;
    UINT64 fence_value = 0;

    bool init(ID3D12Device* d, ID3D12CommandQueue* q) {
        dev = d;
        dev->AddRef();
        queue = q;
        queue->AddRef();
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

    /* One CopyBufferRegion, executed and CPU-waited. */
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
                fprintf(stderr, "HEAPIMP: GPU copy timed out\n");
                return false;
            }
            sleep_ms(1);
        }
        return true;
    }
};

HRESULT make_committed(ID3D12Device* dev, D3D12_HEAP_TYPE type,
                       D3D12_RESOURCE_STATES state, uint64_t width,
                       Com<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    D3D12_RESOURCE_DESC rd = buf_desc(width);
    return dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                        nullptr, __uuidof(ID3D12Resource),
                                        (void**)&out);
}

void fill_pattern(void* dst, uint64_t bytes, uint32_t seed) {
    auto* p = static_cast<uint32_t*>(dst);
    for (uint64_t i = 0; i < bytes / 4; i++)
        p[i] = seed ^ (uint32_t)(i * 2654435761u);
}

bool check_pattern(const void* src, uint64_t bytes, uint32_t seed) {
    auto* p = static_cast<const uint32_t*>(src);
    for (uint64_t i = 0; i < bytes / 4; i++)
        if (p[i] != (seed ^ (uint32_t)(i * 2654435761u)))
            return false;
    return true;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "HEAPIMP: dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");
    Com<ID3D12CommandQueue> queue;
    CK(make_d3d12_queue(dev.ptr(), queue), "CreateCommandQueue");
    GpuCopier gpu;
    EXPECT(gpu.init(dev.ptr(), queue.ptr()), "copier init failed");

    /* Committed staging pair, created before the fd baseline so their (and
     * the copier's) lazy D3DMetal allocations do not read as import leaks. */
    Com<ID3D12Resource> readback, upload;
    CK(make_committed(dev.ptr(), D3D12_HEAP_TYPE_READBACK,
                      D3D12_RESOURCE_STATE_COPY_DEST, kBufSize, readback),
       "CreateCommittedResource(READBACK)");
    CK(make_committed(dev.ptr(), D3D12_HEAP_TYPE_UPLOAD,
                      D3D12_RESOURCE_STATE_GENERIC_READ, kBufSize, upload),
       "CreateCommittedResource(UPLOAD)");

    const int fds0 = t_count_fds();

    const int fd = dmn_share_anon_file((size_t)kFileSize);
    EXPECT(fd >= 0, "dmn_share_anon_file failed");
    void* my_map = mmap(nullptr, (size_t)kFileSize, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    EXPECT(my_map != MAP_FAILED, "mmap(shm fd) failed");
    auto* win = static_cast<uint8_t*>(my_map) + kImpOff; /* the imported window */

    /* 1) Import a window at a nonzero fd offset. */
    Com<ID3D12Heap> heap;
    CK(dmn_open_existing_heap_from_fd(dev.ptr(), fd, kImpOff, kImpSize,
                                      (uint32_t)D3D12_HEAP_TYPE_UPLOAD, 0,
                                      &kIidHeap, (void**)&heap),
       "dmn_open_existing_heap_from_fd");
    D3D12_HEAP_DESC hd = heap->GetDesc();
    EXPECT(hd.SizeInBytes == kImpSize, "heap desc size mismatch");

    /* 2) CPU -> GPU through a placed buffer at a nonzero heap offset. */
    D3D12_RESOURCE_DESC rd = buf_desc(kBufSize);
    Com<ID3D12Resource> placed;
    CK(dev->CreatePlacedResource(heap.ptr(), kPlaceOff, &rd,
                                 D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                 __uuidof(ID3D12Resource), (void**)&placed),
       "CreatePlacedResource(imported heap)");
    fill_pattern(win + kPlaceOff, kBufSize, 0xA11A5EDu);
    EXPECT(gpu.copy(readback.ptr(), placed.ptr(), kBufSize), "copy out failed");
    {
        void* rb = nullptr;
        D3D12_RANGE all{0, (SIZE_T)kBufSize};
        CK(readback->Map(0, &all, &rb), "Map(readback)");
        EXPECT(check_pattern(rb, kBufSize, 0xA11A5EDu),
               "CPU writes not visible to the GPU — placed buffer is not "
               "aliasing the caller's pages");
        D3D12_RANGE none{0, 0};
        readback->Unmap(0, &none);
    }
    printf("HEAPIMP: CPU->GPU alias ok (import off %llu + heap off %llu)\n",
           (unsigned long long)kImpOff, (unsigned long long)kPlaceOff);

    /* 3) GPU -> CPU through a DEFAULT-typed import over a second window. */
    Com<ID3D12Heap> heap2;
    CK(dmn_open_existing_heap_from_fd(dev.ptr(), fd, 0, kImpOff,
                                      (uint32_t)D3D12_HEAP_TYPE_DEFAULT, 0,
                                      &kIidHeap, (void**)&heap2),
       "dmn_open_existing_heap_from_fd(#2)");
    Com<ID3D12Resource> placed2;
    CK(dev->CreatePlacedResource(heap2.ptr(), 0, &rd,
                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                 __uuidof(ID3D12Resource), (void**)&placed2),
       "CreatePlacedResource(#2)");
    {
        void* up = nullptr;
        D3D12_RANGE none{0, 0};
        CK(upload->Map(0, &none, &up), "Map(upload)");
        fill_pattern(up, kBufSize, 0xB0BCA7u);
        upload->Unmap(0, nullptr);
    }
    EXPECT(gpu.copy(placed2.ptr(), upload.ptr(), kBufSize), "copy in failed");
    EXPECT(check_pattern(my_map, kBufSize, 0xB0BCA7u),
           "GPU writes not visible through the caller's mmap");
    EXPECT(check_pattern(win + kPlaceOff, kBufSize, 0xA11A5EDu),
           "first window disturbed by a copy into the second");
    printf("HEAPIMP: GPU->CPU alias ok, windows independent\n");

    /* 3b) A UAV buffer placed on the UPLOAD-typed import. D3D12 forbids
     * ALLOW_UNORDERED_ACCESS on UPLOAD/READBACK heaps, but the guest may
     * place one on a CUSTOM heap that arrives here clamped to UPLOAD; the
     * create must still succeed and the placement must still alias. */
    {
        D3D12_RESOURCE_DESC ud = buf_desc(kBufSize);
        ud.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Com<ID3D12Resource> uav;
        CK(dev->CreatePlacedResource(heap.ptr(), kPlaceOff, &ud,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&uav),
           "CreatePlacedResource(UAV buffer on UPLOAD-typed import)");
        fill_pattern(win + kPlaceOff, kBufSize, 0x0AF0AF00u);
        EXPECT(gpu.copy(readback.ptr(), uav.ptr(), kBufSize), "copy out (UAV) failed");
        void* rb = nullptr;
        D3D12_RANGE all{0, (SIZE_T)kBufSize};
        CK(readback->Map(0, &all, &rb), "Map(readback UAV)");
        EXPECT(check_pattern(rb, kBufSize, 0x0AF0AF00u),
               "UAV placement on the imported heap does not alias the window");
        D3D12_RANGE none{0, 0};
        readback->Unmap(0, &none);
        printf("HEAPIMP: UAV placement on a CPU-typed import ok\n");
    }

    /* 4) Heap released first: the placed buffer's backing must survive. */
    heap = nullptr;
    fill_pattern(win + kPlaceOff, kBufSize, 0x5EC02Du);
    EXPECT(gpu.copy(readback.ptr(), placed.ptr(), kBufSize),
           "copy after heap release failed");
    {
        void* rb = nullptr;
        D3D12_RANGE all{0, (SIZE_T)kBufSize};
        CK(readback->Map(0, &all, &rb), "Map(readback #2)");
        EXPECT(check_pattern(rb, kBufSize, 0x5EC02Du),
               "aliasing lost after the heap was released");
        D3D12_RANGE none{0, 0};
        readback->Unmap(0, &none);
    }
    printf("HEAPIMP: placed buffer survives heap release\n");

    /* 5) Validation. Nothing below may succeed or leak. */
    Com<ID3D12Heap> bad;
    EXPECT(FAILED(dmn_open_existing_heap_from_fd(
               dev.ptr(), fd, 1024, kImpSize, 0, 0, &kIidHeap,
               (void**)&bad)) && !bad, "unaligned offset accepted");
    EXPECT(FAILED(dmn_open_existing_heap_from_fd(
               dev.ptr(), fd, 2ull << 20, kImpSize, 0, 0,
               &kIidHeap, (void**)&bad)) && !bad,
           "window past EOF accepted");
    EXPECT(FAILED(dmn_open_existing_heap_from_fd(
               dev.ptr(), fd, 0, 0, 0, 0, &kIidHeap,
               (void**)&bad)) && !bad, "zero size accepted");
    EXPECT(FAILED(dmn_open_existing_heap_from_fd(
               dev.ptr(), fd, 0, 5ull << 30, 0, 0, &kIidHeap,
               (void**)&bad)) && !bad, "5 GiB size accepted");
    EXPECT(FAILED(dmn_open_existing_heap_from_fd(
               dev.ptr(), fd, 0, kImpOff, 0, 0, &kIidResource,
               (void**)&bad)) && !bad, "non-heap iid accepted");
    EXPECT(FAILED(dmn_open_existing_heap_from_fd(
               queue.ptr(), fd, 0, kImpOff, 0, 0, &kIidHeap,
               (void**)&bad)) && !bad, "non-device object accepted");
    {
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = 64;
        td.Height = 64;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        Com<ID3D12Resource> tex;
        EXPECT(FAILED(dev->CreatePlacedResource(
                   heap2.ptr(), 0, &td, D3D12_RESOURCE_STATE_COMMON, nullptr,
                   __uuidof(ID3D12Resource), (void**)&tex)) && !tex,
               "texture placement on an imported heap accepted");
    }
    {
        D3D12_RESOURCE_DESC big = buf_desc(kImpOff); /* == heap2's whole size */
        Com<ID3D12Resource> over;
        EXPECT(FAILED(dev->CreatePlacedResource(
                   heap2.ptr(), 64 * 1024, &big,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   __uuidof(ID3D12Resource), (void**)&over)) && !over,
               "out-of-window placement accepted");
    }
    printf("HEAPIMP: validation refusals ok\n");

    /* Teardown; every import-owned fd must come back (D3DMetal defers some
     * destruction, so poll). The caller's own fd + mmap go last. */
    placed = nullptr;
    placed2 = nullptr;
    heap2 = nullptr;
    munmap(my_map, (size_t)kFileSize);
    close(fd);
    int fds_now = -1;
    const uint64_t start = now_ms();
    do {
        fds_now = t_count_fds();
        if (fds_now <= fds0)
            break;
        sleep_ms(50);
    } while (now_ms() - start < 3000);
    EXPECT(fds_now <= fds0, "fd count did not return to baseline — an "
           "imported heap's dup leaked");

    T_PASS();
    return 0;
}
