/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Smoke test (pure C): initialize the library, which dlopens
 * D3DMetal.framework and runs the GFXT_Initialize handshake. Then verify
 * that the framework's exported GFXTOSInterface::Instance static holds
 * our singleton (proving D3DMetal accepted the interface).
 */

#include <dlfcn.h>
#include <stdio.h>

#include "d3dmetal_native.h"

int main(void) {
    dmn_result res = dmn_init(NULL);
    if (res != DMN_SUCCESS) {
        fprintf(stderr, "SMOKE: dmn_init FAILED (%d)\n", (int)res);
        return 1;
    }
    printf("SMOKE: dmn_init: OK\n");

    if (!dmn_is_initialized()) {
        fprintf(stderr, "SMOKE: dmn_is_initialized FAILED\n");
        return 1;
    }

    const char* path = dmn_framework_path();
    printf("SMOKE: framework: %s\n", path ? path : "(null)");
    if (!path)
        return 1;

    /* dlopen is refcounted; this returns the already-loaded image. */
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!handle) {
        fprintf(stderr, "SMOKE: re-dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    /* GFXTOSInterface::Instance — D3DMetal stores the host interface
     * pointer it received in GFXT_Initialize here. */
    void** instance = (void**)dlsym(handle, "_ZN15GFXTOSInterface8InstanceE");
    if (!instance) {
        fprintf(stderr, "SMOKE: GFXTOSInterface::Instance not found: %s\n",
                dlerror());
        return 1;
    }
    if (!*instance) {
        fprintf(stderr, "SMOKE: GFXTOSInterface::Instance is NULL — "
                        "GFXT_Initialize did not store the interface\n");
        return 1;
    }
    printf("SMOKE: GFXTOSInterface::Instance = %p\n", *instance);

    printf("SMOKE: PASS\n");
    return 0;
}
