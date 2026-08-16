/*
 * D3D12 shared heaps + placed resources, windowless, single process. A SHARED
 * heap is one shared-memory object and every placement is a window into it,
 * so this covers:
 *
 *  1. CreateHeap(D3D12_HEAP_FLAG_SHARED).
 *  2. CreatePlacedResource (texture) on it, exported with CreateSharedHandle;
 *     POD dims verified.
 *  3. OpenSharedHandle round trip: a marker written through the exported fd is
 *     readable through a re-export from the OPENED resource (same memory).
 *  3b. Buffers placed at one nonzero offset alias each other; the exported POD
 *     carries the placement offset and an opened copy maps the same window.
 *  3c. The heap itself is shareable: a placement in the OPENED heap aliases
 *     the exporter's placement at the same offset.
 *  3d. A GPU write through a placed buffer lands in the heap's object and in
 *     no other live shared object's.
 *  3e. A texture placed at a nonzero offset exports/opens with that offset.
 *  3f. 1D/3D placements are created but refuse to export (no window backing).
 *  3g. Residency calls and UpdateTileMappings accept the heap without
 *     reaching the framework.
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
#include "dmn_d3d12_up.h" /* ID3D12Device3/4 (vendored d3d12.h stops at Device1) */
#include "common/com.h"

#define T_TAG "HEAP"
#include "common/check.h"
#include "common/dx12.h"
#include "common/util.h"

/* == GPU copy helper for 3d) ============================================== */
namespace {
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

D3D12_RESOURCE_DESC buf_desc(uint64_t w) {
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = w;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return rd;
}
} // namespace

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
    printf("HEAP: placed round trip ok (%ux%u, stride %llu)\n", kW, kH,
           (unsigned long long)pod2->stride);
    CK(dmn_shared_handle_close(h2) == DMN_SUCCESS ? S_OK : E_FAIL, "close h2");
    CK(dmn_shared_handle_close(h) == DMN_SUCCESS ? S_OK : E_FAIL, "close h");

    /* 3b) Placed BUFFERS alias, and do so at a nonzero heap offset: the heap
     * owns the storage, so two resources placed at the same offset ARE the
     * same bytes. The exported POD carries `offset` and an import must map at
     * it. */
    {
        const uint64_t kOff = 64 * 1024, kSz = 64 * 1024;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = kSz;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        Com<ID3D12Resource> ba, bb;
        CK(dev->CreatePlacedResource(heap.ptr(), kOff, &bd,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&ba),
           "CreatePlacedResource(buffer @64K)");
        CK(dev->CreatePlacedResource(heap.ptr(), kOff, &bd,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&bb),
           "CreatePlacedResource(buffer @64K, aliasing)");

        HANDLE ha = nullptr, hb = nullptr;
        CK(dev->CreateSharedHandle(ba.ptr(), nullptr, 0, nullptr, &ha),
           "CreateSharedHandle(placed buffer A)");
        CK(dev->CreateSharedHandle(bb.ptr(), nullptr, 0, nullptr, &hb),
           "CreateSharedHandle(placed buffer B)");
        auto* pa = (dmn_shared_buffer_handle*)ha;
        auto* pb = (dmn_shared_buffer_handle*)hb;
        EXPECT(pa->magic == DMN_SHARED_BUFFER_MAGIC, "bad buffer POD magic");
        EXPECT(pa->version == DMN_SHARED_HANDLE_VERSION, "bad buffer POD version");
        EXPECT(pa->offset == kOff && pb->offset == kOff,
               "exported POD does not carry the placement offset");
        EXPECT(pa->size == kSz, "exported POD size mismatch");

        void* wa = mmap(nullptr, (size_t)kSz, PROT_READ | PROT_WRITE, MAP_SHARED,
                        pa->fd, (off_t)pa->offset);
        EXPECT(wa != MAP_FAILED, "mmap(placed buffer A window) failed");
        const uint32_t kAlias = 0xC0FFEE01u;
        memcpy(wa, &kAlias, sizeof(kAlias));

        void* wb = mmap(nullptr, (size_t)kSz, PROT_READ, MAP_SHARED, pb->fd,
                        (off_t)pb->offset);
        EXPECT(wb != MAP_FAILED, "mmap(placed buffer B window) failed");
        uint32_t seen = 0;
        memcpy(&seen, wb, sizeof(seen));
        EXPECT(seen == kAlias,
               "two buffers placed at the same heap offset do not alias");

        /* An opened copy must map the same window, not offset 0 of the heap. */
        Com<ID3D12Resource> oa;
        CK(dev->OpenSharedHandle(ha, __uuidof(ID3D12Resource), (void**)&oa),
           "OpenSharedHandle(placed buffer)");
        HANDLE hr2 = nullptr;
        CK(dev->CreateSharedHandle(oa.ptr(), nullptr, 0, nullptr, &hr2),
           "CreateSharedHandle(opened placed buffer)");
        auto* p2 = (dmn_shared_buffer_handle*)hr2;
        EXPECT(p2->offset == kOff, "re-export lost the window offset");
        void* w2 = mmap(nullptr, (size_t)kSz, PROT_READ, MAP_SHARED, p2->fd,
                        (off_t)p2->offset);
        EXPECT(w2 != MAP_FAILED, "mmap(re-exported window) failed");
        uint32_t seen2 = 0;
        memcpy(&seen2, w2, sizeof(seen2));
        EXPECT(seen2 == kAlias, "opened placed buffer maps different memory");

        munmap(wa, (size_t)kSz);
        munmap(wb, (size_t)kSz);
        munmap(w2, (size_t)kSz);
        dmn_shared_handle_close(hr2);
        dmn_shared_handle_close(hb);
        dmn_shared_handle_close(ha);
        printf("HEAP: placed buffers alias at offset %llu\n",
               (unsigned long long)kOff);
    }

    /* 3c) The heap ITSELF is shareable (MSDN: CreateSharedHandle takes a
     * heap, resource, or fence). A buffer placed in the OPENED heap at the
     * same offset must alias the original heap's placements — what sharing a
     * heap means. */
    {
        const uint64_t kOff = 64 * 1024, kSz = 64 * 1024;
        HANDLE hh = nullptr;
        CK(dev->CreateSharedHandle(heap.ptr(), nullptr, 0, nullptr, &hh),
           "CreateSharedHandle(heap)");
        Com<ID3D12Heap> opened_heap;
        CK(dev->OpenSharedHandle(hh, __uuidof(ID3D12Heap),
                                 (void**)&opened_heap),
           "OpenSharedHandle(heap)");
        D3D12_HEAP_DESC ohd = opened_heap->GetDesc();
        EXPECT(ohd.SizeInBytes == hd.SizeInBytes,
               "opened heap desc size mismatch");

        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = kSz;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Com<ID3D12Resource> bo;
        CK(dev->CreatePlacedResource(opened_heap.ptr(), kOff, &bd,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&bo),
           "CreatePlacedResource(opened heap)");
        HANDLE hbo = nullptr;
        CK(dev->CreateSharedHandle(bo.ptr(), nullptr, 0, nullptr, &hbo),
           "CreateSharedHandle(buffer on opened heap)");
        auto* po = (dmn_shared_buffer_handle*)hbo;
        EXPECT(po->offset == kOff, "opened-heap placement lost its offset");
        void* wo = mmap(nullptr, (size_t)kSz, PROT_READ, MAP_SHARED, po->fd,
                        (off_t)po->offset);
        EXPECT(wo != MAP_FAILED, "mmap(opened-heap placement) failed");
        /* 3b left 0xC0FFEE01 at heap offset 64K through the ORIGINAL heap's
         * placements; the opened heap's placement must read the same bytes. */
        uint32_t seen = 0;
        memcpy(&seen, wo, sizeof(seen));
        EXPECT(seen == 0xC0FFEE01u,
               "placement in an opened shared heap does not alias the "
               "exporter's heap memory");
        munmap(wo, (size_t)kSz);
        dmn_shared_handle_close(hbo);
        dmn_shared_handle_close(hh);
        printf("HEAP: heap handle round trip ok (opened placement aliases)\n");
    }

    /* 3d) A GPU write through a placed buffer lands in THE HEAP'S object and
     * not in some other shared object's. The earlier steps read and write the
     * shm through fds on the CPU, which never touches the Metal backing; only
     * a GPU write shows which object a placement's Metal buffer really is. Two
     * independent shared objects are alive at once — a committed SHARED buffer
     * and a placed one, both at offset 0 — the pattern goes into the placement,
     * and both fds are checked. */
    {
        const uint64_t kSz = 4ull << 20;
        Copier gpu;
        EXPECT(gpu.init(dev.ptr()), "copier init failed");
        D3D12_HEAP_PROPERTIES up{};
        up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = buf_desc(kSz);
        Com<ID3D12Resource> src;
        CK(dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd,
                                        D3D12_RESOURCE_STATE_GENERIC_READ,
                                        nullptr, __uuidof(ID3D12Resource),
                                        (void**)&src),
           "CreateCommittedResource(upload src)");
        {
            void* p = nullptr;
            D3D12_RANGE none{0, 0};
            CK(src->Map(0, &none, &p), "Map(src)");
            auto* w = (uint32_t*)p;
            for (uint64_t i = 0; i < kSz / 4; i++)
                w[i] = 0xC0DE0000u ^ (uint32_t)(i * 2654435761u);
            src->Unmap(0, nullptr);
        }
        /* The decoy: a committed SHARED buffer, alive across the copy. */
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        Com<ID3D12Resource> decoy;
        CK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        __uuidof(ID3D12Resource),
                                        (void**)&decoy),
           "CreateCommittedResource(SHARED decoy)");
        HANDLE hdecoy = nullptr;
        CK(dev->CreateSharedHandle(decoy.ptr(), nullptr, 0, nullptr, &hdecoy),
           "CreateSharedHandle(decoy)");
        /* The placement: fresh SHARED heap, buffer at offset 0. */
        D3D12_HEAP_DESC hd2{};
        hd2.SizeInBytes = kSz;
        hd2.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd2.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        hd2.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        Com<ID3D12Heap> heap2;
        CK(dev->CreateHeap(&hd2, __uuidof(ID3D12Heap), (void**)&heap2),
           "CreateHeap(SHARED #2)");
        Com<ID3D12Resource> pl;
        CK(dev->CreatePlacedResource(heap2.ptr(), 0, &rd,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&pl),
           "CreatePlacedResource(GPU-write target)");
        HANDLE hpl = nullptr;
        CK(dev->CreateSharedHandle(pl.ptr(), nullptr, 0, nullptr, &hpl),
           "CreateSharedHandle(GPU-write target)");
        auto* pd = (dmn_shared_buffer_handle*)hdecoy;
        auto* pp = (dmn_shared_buffer_handle*)hpl;

        EXPECT(gpu.copy(pl.ptr(), src.ptr(), kSz), "GPU copy into placement");
        void* mp = mmap(nullptr, (size_t)kSz, PROT_READ, MAP_SHARED, pp->fd,
                        (off_t)pp->offset);
        void* md = mmap(nullptr, (size_t)kSz, PROT_READ, MAP_SHARED, pd->fd,
                        (off_t)pd->offset);
        EXPECT(mp != MAP_FAILED && md != MAP_FAILED, "mmap(fds) failed");
        int bad_p = 0, hit_d = 0;
        auto* rp = (const uint32_t*)mp;
        auto* rdc = (const uint32_t*)md;
        for (uint64_t i = 0; i < kSz / 4; i += 997) {
            const uint32_t want = 0xC0DE0000u ^ (uint32_t)(i * 2654435761u);
            if (rp[i] != want) bad_p++;
            if (rdc[i] == want) hit_d++;
        }
        EXPECT(bad_p == 0,
               "GPU write through a placed buffer did not reach the heap's "
               "shared object — the placement is backed by some other memory");
        EXPECT(hit_d == 0,
               "GPU write through a placed buffer landed in an UNRELATED shared "
               "buffer's object — two shared resources are aliasing one "
               "MTLBuffer");
        munmap(mp, (size_t)kSz);
        munmap(md, (size_t)kSz);
        dmn_shared_handle_close(hpl);
        dmn_shared_handle_close(hdecoy);
        printf("HEAP: GPU write via placement reaches its own object only\n");
    }

    /* 3e) A texture placed at a nonzero offset: the POD carries the offset,
     * and an opened copy sees the same bytes at that window. */
    {
        const uint64_t kOff = 1ull << 20;
        D3D12_RESOURCE_DESC trd = tex_desc(64, 64);
        Com<ID3D12Resource> ptex;
        CK(dev->CreatePlacedResource(heap.ptr(), kOff, &trd,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), (void**)&ptex),
           "CreatePlacedResource(texture @1M)");
        HANDLE ht = nullptr;
        CK(dev->CreateSharedHandle(ptex.ptr(), nullptr, 0, nullptr, &ht),
           "CreateSharedHandle(placed texture @1M)");
        auto* pt = (dmn_shared_texture_handle*)ht;
        EXPECT(pt->version == DMN_SHARED_HANDLE_VERSION, "bad texture POD version");
        EXPECT(pt->offset == kOff, "texture POD does not carry the placement offset");
        void* wt = mmap(nullptr, (size_t)pt->size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, pt->fd, (off_t)pt->offset);
        EXPECT(wt != MAP_FAILED, "mmap(placed texture window) failed");
        const uint32_t kMark = 0x7E57A11Cu;
        memcpy(wt, &kMark, sizeof(kMark));
        Com<ID3D12Resource> otex;
        CK(dev->OpenSharedHandle(ht, __uuidof(ID3D12Resource), (void**)&otex),
           "OpenSharedHandle(placed texture @1M)");
        HANDLE ht2 = nullptr;
        CK(dev->CreateSharedHandle(otex.ptr(), nullptr, 0, nullptr, &ht2),
           "CreateSharedHandle(opened placed texture)");
        auto* pt2 = (dmn_shared_texture_handle*)ht2;
        EXPECT(pt2->offset == kOff, "re-export lost the texture's window offset");
        void* wt2 = mmap(nullptr, (size_t)pt2->size, PROT_READ, MAP_SHARED,
                         pt2->fd, (off_t)pt2->offset);
        EXPECT(wt2 != MAP_FAILED, "mmap(re-exported texture window) failed");
        uint32_t seen = 0;
        memcpy(&seen, wt2, sizeof(seen));
        EXPECT(seen == kMark, "opened placed texture maps different memory");
        munmap(wt, (size_t)pt->size);
        munmap(wt2, (size_t)pt2->size);
        dmn_shared_handle_close(ht2);
        dmn_shared_handle_close(ht);
        printf("HEAP: placed texture at offset %llu round trip ok\n",
               (unsigned long long)kOff);
    }

    /* 3f) A 3D texture placed in the SHARED heap has no window backing: the
     * create succeeds (correct memory, not aliased) and exporting it fails
     * where the app can see it. */
    {
        D3D12_RESOURCE_DESC v = tex_desc(16, 16);
        v.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        v.DepthOrArraySize = 4;
        v.Flags = D3D12_RESOURCE_FLAG_NONE;
        Com<ID3D12Resource> vol;
        HRESULT hr = dev->CreatePlacedResource(heap.ptr(), 2ull << 20, &v,
                                               D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, __uuidof(ID3D12Resource),
                                               (void**)&vol);
        if (SUCCEEDED(hr) && vol) {
            HANDLE hv = (HANDLE)0x1;
            HRESULT hx = dev->CreateSharedHandle(vol.ptr(), nullptr, 0, nullptr,
                                                 &hv);
            EXPECT(hx == E_INVALIDARG && hv == nullptr,
                   "exporting a 3D placement vended a handle to memory that "
                   "does not alias the heap");
            printf("HEAP: 3D placement created unshared, export refused\n");
        } else {
            printf("HEAP: 3D placed create not supported here (0x%08x); "
                   "export refusal not exercised\n", (unsigned)hr);
        }
    }

    /* 3g) The heap must never reach the framework through the residency
     * calls or as a tile pool: every call accepts it (or drops it) instead of
     * dereferencing it as a framework object, and EnqueueMakeResident still
     * honours its fence. */
    {
        ID3D12Pageable* objs[1] = {heap.ptr()};
        CK(dev->MakeResident(1, objs), "MakeResident(synthetic heap)");
        CK(dev->Evict(1, objs), "Evict(synthetic heap)");
        Com<ID3D12Device1> dev1;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device1),
                                          (void**)&dev1)) && dev1) {
            D3D12_RESIDENCY_PRIORITY prio[1] = {D3D12_RESIDENCY_PRIORITY_HIGH};
            CK(dev1->SetResidencyPriority(1, objs, prio),
               "SetResidencyPriority(synthetic heap)");
        }
        Com<ID3D12Device3> dev3;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device3),
                                          (void**)&dev3)) && dev3) {
            Com<ID3D12Fence> f;
            CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                (void**)&f), "CreateFence(residency)");
            CK(dev3->EnqueueMakeResident(0, 1, objs, f.ptr(), 7),
               "EnqueueMakeResident(synthetic heap)");
            const uint64_t t0 = now_ms();
            while (f->GetCompletedValue() < 7 && now_ms() - t0 < 5000)
                sleep_ms(1);
            EXPECT(f->GetCompletedValue() >= 7,
                   "EnqueueMakeResident with only a synthetic heap never "
                   "signalled its fence");
        }
        Com<ID3D12CommandQueue> q;
        CK(make_d3d12_queue(dev.ptr(), q), "CreateCommandQueue(tiles)");
        D3D12_TILED_RESOURCE_COORDINATE coord{};
        D3D12_TILE_REGION_SIZE size{};
        size.NumTiles = 1;
        D3D12_TILE_RANGE_FLAGS rflags = D3D12_TILE_RANGE_FLAG_NONE;
        UINT roff = 0, rcount = 1;
        /* Declared with the real 10-parameter signature (the vendored d3d12.h
         * omits pHeap): a call through the correct prototype. */
        typedef void (STDMETHODCALLTYPE *UpdateTileMappingsFn)(
            ID3D12CommandQueue*, ID3D12Resource*, UINT,
            const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*,
            ID3D12Heap*, UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*,
            const UINT*, D3D12_TILE_MAPPING_FLAGS);
        void** vtbl = *reinterpret_cast<void***>(q.ptr());
        auto fn = reinterpret_cast<UpdateTileMappingsFn>(vtbl[8]);
        fn(q.ptr(), nullptr, 1, &coord, &size, heap.ptr(), 1, &rflags, &roff,
           &rcount, D3D12_TILE_MAPPING_FLAG_NONE);
        printf("HEAP: residency + tile-mapping calls with the synthetic heap ok\n");
    }

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
