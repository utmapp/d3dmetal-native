/*
 * Known-answer test for dmn_vindex (dmn_hook.h): the Itanium member-pointer
 * decoding that every vtable patch relies on. IUnknown's slots are fixed by
 * the COM contract (QI/AddRef/Release = 0/1/2); the derived-interface slots
 * are hand-counted from the vendored headers' declaration order and cover the
 * inheritance-chain accumulation, including SetPrivateDataInterface — the
 * slot the eviction sentinel machinery depends on.
 *
 * The encoding is fixed at compile time, so a build that passes this test
 * cannot mis-patch at runtime; there is no runtime re-check. If it ever fails
 * (e.g. a compiler switches to the ARM-variant encoding, where the virtual
 * bit lives in the adjustment word), hooking must not ship.
 *
 * Prints "ABI: PASS" and exits 0 on success.
 */

#include <cstdio>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>

#include "dmn_hook.h"

static int g_failures = 0;

#define CHECK_SLOT(expr, want)                                               \
    do {                                                                     \
        size_t got_ = (expr);                                                \
        if (got_ != (size_t)(want)) {                                        \
            fprintf(stderr, "ABI: %s = %zu, want %d\n", #expr, got_, want);  \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

int main() {
    /* COM contract: every interface starts with these three. */
    CHECK_SLOT(dmn_vindex_qi(), 0);
    CHECK_SLOT(dmn_vindex(&IUnknown::AddRef), 1);
    CHECK_SLOT(dmn_vindex(&IUnknown::Release), 2);

    /* d3d11.h declaration order on ID3D11DeviceChild. */
    CHECK_SLOT(dmn_vindex(&ID3D11DeviceChild::GetDevice), 3);
    CHECK_SLOT(dmn_vindex(&ID3D11DeviceChild::GetPrivateData), 4);
    CHECK_SLOT(dmn_vindex(&ID3D11DeviceChild::SetPrivateData), 5);
    CHECK_SLOT(dmn_vindex(&ID3D11DeviceChild::SetPrivateDataInterface), 6);

    /* d3d12.h declaration order on ID3D12Object. */
    CHECK_SLOT(dmn_vindex(&ID3D12Object::GetPrivateData), 3);
    CHECK_SLOT(dmn_vindex(&ID3D12Object::SetPrivateData), 4);
    CHECK_SLOT(dmn_vindex(&ID3D12Object::SetPrivateDataInterface), 5);
    CHECK_SLOT(dmn_vindex(&ID3D12Object::SetName), 6);

    if (g_failures) {
        fprintf(stderr, "ABI: FAIL (%d slot mismatches)\n", g_failures);
        return 1;
    }
    printf("ABI: PASS\n");
    return 0;
}
