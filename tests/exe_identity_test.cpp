/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * dmn_set_executable_path reaches D3DMetal's per-app profile matcher.
 *
 * D3DMetal picks its per-app profile once per process, the first time a
 * factory or device needs it, from what GFXTPathInterface::getExecutablePath
 * reports.  bf1.exe is one of its profiles, and the only visible effect of
 * that one is the adapter identity: vendor 0x106b instead of the default
 * profile's AMD 0x1002.
 *
 *   exe-identity-test bf1.exe   set the override, expect vendor 0x106b
 *   exe-identity-test           no override (this process is not in the
 *                               table), expect anything but 0x106b
 *
 * Prints "EXE-IDENTITY: PASS" and exits 0 on success.
 */

#include <cstdio>

#include <dxgi1_6.h>

#include "d3dmetal_native.h"

int main(int argc, char **argv) {
    const char *name = argc > 1 ? argv[1] : nullptr;
    if (name && dmn_set_executable_path(name) != DMN_SUCCESS) {
        fprintf(stderr, "EXE-IDENTITY: dmn_set_executable_path failed\n");
        return 1;
    }

    IDXGIFactory1 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&factory))) {
        fprintf(stderr, "EXE-IDENTITY: CreateDXGIFactory1 failed\n");
        return 1;
    }
    IDXGIAdapter1 *adapter = nullptr;
    DXGI_ADAPTER_DESC1 desc = {};
    HRESULT hr = factory->EnumAdapters1(0, &adapter);
    if (SUCCEEDED(hr))
        hr = adapter->GetDesc1(&desc);
    if (adapter)
        adapter->Release();
    factory->Release();
    if (FAILED(hr)) {
        fprintf(stderr, "EXE-IDENTITY: adapter 0 desc failed 0x%08x\n", (unsigned)hr);
        return 1;
    }

    const bool apple = desc.VendorId == 0x106b;
    printf("EXE-IDENTITY: override %s -> adapter vendor 0x%04x\n",
           name ? name : "(none)", desc.VendorId);
    if (apple != (name != nullptr)) {
        fprintf(stderr, "EXE-IDENTITY: FAIL (want vendor %s0x106b)\n", name ? "" : "!= ");
        return 1;
    }
    printf("EXE-IDENTITY: PASS\n");
    return 0;
}
