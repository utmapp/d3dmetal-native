/*
 * D3D12 shared-buffer round trip through the STANDARD APIs only: create a buffer
 * with D3D12_HEAP_FLAG_SHARED, export it with ID3D12Device::CreateSharedHandle,
 * and re-import it with ID3D12Device::OpenSharedHandle — confirming the opaque
 * handle round-trips and yields a resource describing the same buffer. No
 * dmn_* sharing calls.
 *
 * Data-flow across the handle (GPU writes visible to a peer) is validated
 * end-to-end by the fence tests, whose companion buffer is exactly a D3D12
 * shared buffer written with WriteBufferImmediate and read by the consumer.
 * (A GPU CopyBufferRegion read of the substituted buffer does NOT round-trip
 * under D3DMetal — cross-resource coherence needs a shared GPU fence — so this
 * test does not attempt one.)
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
    bool ok = imported && id.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
              id.Width == rd.Width;
    printf("SHAREDBUF: export+import round trip -> %s (imported %llu-byte buffer)\n",
           ok ? "OK" : "MISMATCH", (unsigned long long)id.Width);
    printf("SHAREDBUF: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
