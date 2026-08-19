/*
 * A placed BUFFER in a plain (non-shared, non-imported) DEFAULT heap, used as
 * a copy source and destination.
 *
 * Triton turns every non-shared D3D12 buffer create into CreateHeap +
 * CreatePlacedResource, so this is the shape every device-local buffer in the
 * guest actually has -- and it is the shape no existing test covered:
 * heap-placed-test only exercises SHARED/imported heaps, which our own
 * substitution layer replaces with synthetic ones.
 *
 * Sequence: UPLOAD -> placed DEFAULT -> READBACK, with the readback buffer
 * pre-poisoned so "the copy never ran" is distinguishable from "the copy ran
 * and moved zeros".
 *
 * Prints "DEFBUF: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <sys/mman.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "dmn_share.h" /* dmn_share_anon_file */
#include "common/com.h"

#define T_TAG "DEFBUF"
#include "common/check.h"

/* dmn_open_existing_heap_from_fd takes a GUID pointer. */
static const IID IID_ID3D12Heap_local = __uuidof(ID3D12Heap);

namespace {

const UINT kN = 32;
const UINT64 kBytes = kN * sizeof(UINT);

ID3D12Device* g_dev;
ID3D12CommandQueue* g_q;
ID3D12CommandAllocator* g_alloc;
ID3D12GraphicsCommandList* g_cl;
ID3D12Fence* g_fence;
UINT64 g_fenceVal;

ID3D12Resource* plainBuffer(D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = type;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = kBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                              state, nullptr,
                                              __uuidof(ID3D12Resource),
                                              (void**)&r)))
        return nullptr;
    return r;
}

bool execAndWait() {
    if (FAILED(g_cl->Close()))
        return false;
    ID3D12CommandList* lists[] = {g_cl};
    g_q->ExecuteCommandLists(1, lists);
    ++g_fenceVal;
    if (FAILED(g_q->Signal(g_fence, g_fenceVal)))
        return false;
    /* Poll rather than SetEventOnCompletion: the host-side windows shim has no
     * Win32 event API, and polling keeps the test dependency-free. */
    for (int i = 0; i < 15000; i++) {
        if (g_fence->GetCompletedValue() >= g_fenceVal)
            break;
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, nullptr);
        if (i == 14999) {
            printf("DEFBUF: GPU wait TIMEOUT\n");
            return false;
        }
    }
    if (FAILED(g_alloc->Reset()) || FAILED(g_cl->Reset(g_alloc, nullptr)))
        return false;
    return true;
}

} // namespace

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "DEFBUF: dmn_init FAILED\n");
        return 1;
    }
    CK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                         __uuidof(ID3D12Device), (void**)&g_dev),
       "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd = {};
    CK(g_dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&g_q),
       "CreateCommandQueue");
    CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                     __uuidof(ID3D12CommandAllocator),
                                     (void**)&g_alloc), "CreateCommandAllocator");
    CK(g_dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc,
                                nullptr, __uuidof(ID3D12GraphicsCommandList),
                                (void**)&g_cl), "CreateCommandList");
    CK(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                          (void**)&g_fence), "CreateFence");

    /* The device-local middle buffer, PLACED in a plain DEFAULT heap -- the
     * shape Triton produces for every guest buffer. */
    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = 64 * 1024;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = 0;
    hd.Flags = D3D12_HEAP_FLAG_NONE;
    ID3D12Heap* heap = nullptr;
    CK(g_dev->CreateHeap(&hd, __uuidof(ID3D12Heap), (void**)&heap), "CreateHeap");

    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = kBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* mid = nullptr;
    CK(g_dev->CreatePlacedResource(heap, 0, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                   nullptr, __uuidof(ID3D12Resource),
                                   (void**)&mid), "CreatePlacedResource");
    printf("DEFBUF: placed buffer gpuva=0x%llx\n",
           (unsigned long long)mid->GetGPUVirtualAddress());

    ID3D12Resource* up = plainBuffer(D3D12_HEAP_TYPE_UPLOAD,
                                     D3D12_RESOURCE_STATE_GENERIC_READ);
    /* The readback destination is a placed buffer in an IMPORTED shmem heap --
     * the shape the guest always has, because the Neptune client routes every
     * CPU-visible buffer through CREATE_HEAP_FROM_SHMEM. Its Metal backing is
     * a substituted window over the shm object rather than one of D3DMetal's
     * own allocations, and that is the one difference left between this test
     * (which passes) and the guest (which does not). */
    const size_t kFileSize = 1u << 20;
    /* Which heap type the imported window claims. The guest's CPU-visible
     * buffers arrive as UPLOAD/READBACK/CUSTOM; try each so the failure is
     * attributed to a type rather than to "imported heaps" as a class. */
    uint32_t impType = (uint32_t)D3D12_HEAP_TYPE_READBACK;
    if (const char* e = getenv("DEFBUF_HEAP_TYPE"))
        impType = (uint32_t)atoi(e);
    printf("DEFBUF: imported-heap type=%u\n", impType);
    const int fd = dmn_share_anon_file(kFileSize);
    if (fd < 0) {
        printf("DEFBUF: dmn_share_anon_file FAILED\n");
        return 1;
    }
    ID3D12Heap* impHeap = nullptr;
    HRESULT ihr = dmn_open_existing_heap_from_fd(g_dev, fd, 0, kFileSize,
                                                 impType,
                                                 0, &IID_ID3D12Heap_local,
                                                 (void**)&impHeap);
    if (FAILED(ihr) || !impHeap) {
        printf("DEFBUF: dmn_open_existing_heap_from_fd FAILED 0x%08lx\n",
               (unsigned long)ihr);
        return 1;
    }
    ID3D12Resource* rb = nullptr;
    CK(g_dev->CreatePlacedResource(impHeap, 0, &rd,
                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                   __uuidof(ID3D12Resource), (void**)&rb),
       "CreatePlacedResource(imported readback heap)");
    if (!up || !rb) {
        printf("DEFBUF: staging buffer create FAILED\n");
        return 1;
    }
    {
        void* m = nullptr;
        CK(up->Map(0, nullptr, &m), "Map(upload)");
        for (UINT i = 0; i < kN; i++)
            ((UINT*)m)[i] = 8000 + i;
        up->Unmap(0, nullptr);
    }
    {   /* poison: distinguishes "copy never ran" from "copy moved zeros" */
        void* m = nullptr;
        CK(rb->Map(0, nullptr, &m), "Map(readback)");
        for (UINT i = 0; i < kN; i++)
            ((UINT*)m)[i] = 0xDEADBEEFu;
        rb->Unmap(0, nullptr);
    }

    printf("DEFBUF: src(DEFAULT placed) va=0x%llx  dst(imported placed) va=0x%llx"
           "  up va=0x%llx\n",
           (unsigned long long)mid->GetGPUVirtualAddress(),
           (unsigned long long)rb->GetGPUVirtualAddress(),
           (unsigned long long)up->GetGPUVirtualAddress());
    /* DEFBUF_DIRECT=1: skip the device-local hop entirely and copy straight
     * from the UPLOAD staging buffer into the substituted destination. If that
     * works, the substituted destination is fine and the broken ingredient is
     * the device-local SOURCE. */
    if (getenv("DEFBUF_DIRECT")) {
        /* Skip the device-local hop: copy straight from UPLOAD into the
         * substituted destination. If this works, the substituted destination
         * is fine and the broken ingredient is the device-local SOURCE. */
        g_cl->CopyResource(rb, up);
    } else {
        g_cl->CopyResource(mid, up);
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = mid;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_cl->ResourceBarrier(1, &b);
        g_cl->CopyResource(rb, mid);
    }
    if (!execAndWait()) {
        printf("DEFBUF: submit FAILED\n");
        return 1;
    }

    /* Read the shm object DIRECTLY as well. If the bytes are here but Map()
     * shows the poison, the copy ran and Map is looking at different memory --
     * a coherence bug, not a dropped copy. The two are indistinguishable from
     * Map() alone. */
    void* raw = mmap(nullptr, kFileSize, PROT_READ, MAP_SHARED, fd, 0);
    if (raw != MAP_FAILED) {
        const UINT* rv = (const UINT*)raw;
        printf("DEFBUF: shm[0]=0x%08X shm[1]=0x%08X (want 8000, 8001)\n",
               rv[0], rv[1]);
        munmap(raw, kFileSize);
    }

    void* m = nullptr;
    CK(rb->Map(0, nullptr, &m), "Map(readback,2)");
    const UINT* v = (const UINT*)m;
    printf("DEFBUF: Map() ptr=%p\n", m);
    bool ok = true;
    for (UINT i = 0; i < kN; i++) {
        if (v[i] != 8000 + i) {
            printf("DEFBUF: out[%u]=0x%08X (want %u)%s\n", i, v[i], 8000 + i,
                   v[i] == 0xDEADBEEFu ? "  <- copy never ran" : "");
            ok = false;
            break;
        }
    }
    rb->Unmap(0, nullptr);
    printf("DEFBUF: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}
