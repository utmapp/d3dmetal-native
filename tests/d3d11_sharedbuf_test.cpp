/*
 * D3D11 shared-buffer round trip through the standard D3D11 APIs: create a
 * buffer with D3D11_RESOURCE_MISC_SHARED, GPU-write a value into it via
 * ClearUnorderedAccessViewUint, export it with IDXGIResource::GetSharedHandle,
 * confirm the GPU write reached the shared memory (read via the handle's fd),
 * and re-import it with OpenSharedResource.
 *
 * Prints "SHAREDBUF11: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <time.h>

#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <windows.h>

#include "d3dmetal_native.h"
#include "common/com.h"

#define T_TAG "SHAREDBUF11"
#include "common/check.h"

int main() {
    if (dmn_init(nullptr) != DMN_SUCCESS) {
        fprintf(stderr, "SHAREDBUF11: dmn_init FAILED\n");
        return 1;
    }

    Com<ID3D11Device> device;
    Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    CK(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1,
                         D3D11_SDK_VERSION, &device, &flo, &ctx),
       "D3D11CreateDevice");

    /* Shared buffer, typed for a UAV so the GPU can clear a value into it. */
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = 4096;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    Com<ID3D11Buffer> buf;
    CK(device->CreateBuffer(&bd, nullptr, &buf), "CreateBuffer(MISC_SHARED)");

    D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_R32_UINT;
    ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.FirstElement = 0;
    ud.Buffer.NumElements = bd.ByteWidth / 4;
    Com<ID3D11UnorderedAccessView> uav;
    CK(device->CreateUnorderedAccessView(buf.ptr(), &ud, &uav), "CreateUAV");

    /* GPU-write a value into the shared buffer. */
    const uint32_t kMagic = 0x11BEEF11u;
    UINT clearVals[4] = {kMagic, kMagic, kMagic, kMagic};
    ctx->ClearUnorderedAccessViewUint(uav.ptr(), clearVals);
    ctx->Flush();

    /* Export + re-import through the STANDARD D3D APIs (the HANDLE is opaque). */
    Com<IDXGIResource> dxgi;
    CK(buf->QueryInterface(__uuidof(IDXGIResource), (void**)&dxgi),
       "QI(IDXGIResource)");
    HANDLE shared = nullptr;
    CK(dxgi->GetSharedHandle(&shared), "GetSharedHandle");
    Com<ID3D11Buffer> imported;
    HRESULT ih = device->OpenSharedResource(shared, __uuidof(ID3D11Buffer),
                                            (void**)&imported);
    if (FAILED(ih) || !imported) {
        fprintf(stderr, "SHAREDBUF11: OpenSharedResource FAILED 0x%08x\n",
                (unsigned)ih);
        return 1;
    }

    /* Read the imported buffer back the ordinary way (copy to a STAGING buffer
     * and Map it) to confirm the GPU write is visible through the shared handle. */
    D3D11_BUFFER_DESC stagingDesc = {};
    stagingDesc.ByteWidth = bd.ByteWidth;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    Com<ID3D11Buffer> staging;
    CK(device->CreateBuffer(&stagingDesc, nullptr, &staging),
       "CreateBuffer(staging)");
    ctx->CopyResource(staging.ptr(), imported.ptr());
    ctx->Flush();
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    CK(ctx->Map(staging.ptr(), 0, D3D11_MAP_READ, 0, &mapped), "Map(staging)");
    uint32_t got = *(const uint32_t*)mapped.pData;
    ctx->Unmap(staging.ptr(), 0);

    bool ok = got == kMagic;
    printf("SHAREDBUF11: GPU wrote 0x%08x, read 0x%08x via imported handle -> %s\n",
           kMagic, got, ok ? "OK" : "MISMATCH");
    printf("SHAREDBUF11: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
