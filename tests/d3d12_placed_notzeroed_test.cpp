/*
 * A placed buffer create on an imported (synthetic) heap must never write
 * to the heap's memory.
 *
 * D3D12 gives placed resources UNDEFINED initial contents, and the memory
 * behind an imported heap is the caller's — live guest pages the guest CPU
 * writes through a persistent map while creates stream in.  This framework
 * zero-fills committed resources inside the create, so the placed->committed
 * translation (d12_place_on_synth) must pass
 * D3D12_HEAP_FLAG_CREATE_NOT_ZEROED: no CPU-side repair can save a writer
 * racing the memset (any snapshot predates the writes).
 *
 * The test pins the contract from outside: pre-fill an imported window with
 * a nonzero pattern, run placed creates over it in a loop while a checker
 * thread continuously scans the window, and require that NOT ONE zero word
 * is ever observable.  Without CREATE_NOT_ZEROED every create memsets the
 * window and the checker catches the transient zeros within a few hundred
 * iterations; with it the memory is never touched.  A final GPU copy-out
 * checks the placed buffer still aliases the caller's pages, so the flag
 * cannot silently trade the memset for a lost substitution.
 *
 * Prints "NOTZEROED: PASS" and exits 0 on success.
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

#include <sys/mman.h>
#include <unistd.h>

#include <d3d12.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "dmn_share.h" /* dmn_share_anon_file */
#include "common/com.h"

#define T_TAG "NOTZEROED"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

static const IID kIidHeap = __uuidof(ID3D12Heap);

namespace {

const uint64_t kWinSize = 256 * 1024; /* imported window */
const uint64_t kBufSize = 64 * 1024;  /* placed buffer, at offset 0 */
const int kCreates = 2000;

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

/* Minimal executed-and-waited CopyBufferRegion (heap_import_test's shape). */
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
                       D3D12_RESOURCE_STATES state, uint64_t width,
                       Com<ID3D12Resource>& out) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    D3D12_RESOURCE_DESC rd = buf_desc(width);
    return dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                        nullptr, __uuidof(ID3D12Resource),
                                        (void**)&out);
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, T_TAG ": dmn_init FAILED\n");
        return 1;
    }
    Com<ID3D12Device> dev;
    CK(make_d3d12_device(dev), "D3D12CreateDevice");
    Com<ID3D12CommandQueue> queue;
    CK(make_d3d12_queue(dev.ptr(), queue), "CreateCommandQueue");
    GpuCopier gpu;
    EXPECT(gpu.init(dev.ptr(), queue.ptr()), "copier init failed");

    Com<ID3D12Resource> readback;
    CK(make_committed(dev.ptr(), D3D12_HEAP_TYPE_READBACK,
                      D3D12_RESOURCE_STATE_COPY_DEST, kBufSize, readback),
       "CreateCommittedResource(READBACK)");

    const int fd = dmn_share_anon_file((size_t)kWinSize);
    EXPECT(fd >= 0, "dmn_share_anon_file failed");
    void* map = mmap(nullptr, (size_t)kWinSize, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    EXPECT(map != MAP_FAILED, "mmap(shm fd) failed");
    volatile uint64_t* words = (volatile uint64_t*)map;
    const size_t nwords = kBufSize / sizeof(uint64_t);

    /* Live nonzero contents, as a guest's persistent map would hold. */
    for (size_t i = 0; i < kWinSize / sizeof(uint64_t); i++)
        words[i] = 0xA5A5A5A5A5A5A5A5ull;

    Com<ID3D12Heap> heap;
    CK(dmn_open_existing_heap_from_fd(dev.ptr(), fd, 0, kWinSize,
                                      (uint32_t)D3D12_HEAP_TYPE_UPLOAD, 0,
                                      &kIidHeap, (void**)&heap),
       "dmn_open_existing_heap_from_fd");

    /* Checker: any zero word in the placed range, at any instant, is the
     * create writing to memory it does not own. */
    std::atomic<bool> running{true};
    std::atomic<uint64_t> zero_sightings{0};
    std::atomic<uint64_t> scans{0};
    std::thread checker([&] {
        while (running.load(std::memory_order_relaxed)) {
            for (size_t i = 0; i < nwords; i++) {
                if (words[i] == 0)
                    zero_sightings.fetch_add(1, std::memory_order_relaxed);
            }
            scans.fetch_add(1, std::memory_order_relaxed);
        }
    });

    D3D12_RESOURCE_DESC rd = buf_desc(kBufSize);
    int create_fail = 0;
    Com<ID3D12Resource> last; /* keep the final one for the GPU check */
    for (int i = 0; i < kCreates; i++) {
        Com<ID3D12Resource> placed;
        if (FAILED(dev->CreatePlacedResource(
                heap.ptr(), 0, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr, __uuidof(ID3D12Resource), (void**)&placed))) {
            create_fail++;
            continue;
        }
        if (i == kCreates - 1)
            last = placed;
    }
    running.store(false);
    checker.join();

    printf(T_TAG ": %d creates (%d failed), %llu checker scans, "
           "%llu zero-word sightings\n",
           kCreates, create_fail, (unsigned long long)scans.load(),
           (unsigned long long)zero_sightings.load());
    EXPECT(create_fail == 0, "placed creates failed");
    EXPECT(scans.load() > 100, "checker starved; result not meaningful");
    EXPECT(zero_sightings.load() == 0,
           "a create wrote zeros into the imported window");

    /* Contents intact end to end... */
    for (size_t i = 0; i < nwords; i++)
        EXPECT(words[i] == 0xA5A5A5A5A5A5A5A5ull, "window contents changed");
    /* ...and the placed buffer still aliases the caller's pages on the GPU
     * side (substitution captured; the flag must not cost the import). */
    EXPECT(last.ptr() != nullptr, "no final placed resource");
    EXPECT(gpu.copy(readback.ptr(), last.ptr(), kBufSize), "copy out failed");
    {
        void* rb = nullptr;
        D3D12_RANGE all{0, (SIZE_T)kBufSize};
        CK(readback->Map(0, &all, &rb), "Map(readback)");
        EXPECT(memcmp(rb, (const void*)map, kBufSize) == 0,
               "GPU does not see the caller's pages");
        D3D12_RANGE none{0, 0};
        readback->Unmap(0, &none);
    }

    T_PASS();
    return 0;
}
