/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTAdapter — adapter LUIDs from Metal device registry IDs.
 * The system default device sorts first so adapter 0 is the GPU
 * D3DMetal should prefer.
 */

#import <Metal/Metal.h>

#include <algorithm>

#include "dmn_gfxt.h"
#include "dmn_private.h"

namespace {

LUID luid_from_registry_id(uint64_t reg) {
    LUID luid;
    luid.LowPart  = (uint32_t)(reg & 0xffffffffu);
    luid.HighPart = (int32_t)(reg >> 32);
    return luid;
}

} // namespace

DmnGFXTAdapter::~DmnGFXTAdapter() = default;

uint32_t DmnGFXTAdapter::getAdapterLUIDs(LUID* outArray, size_t capacity) {
    if (!outArray || capacity == 0)
        return 0;

    @autoreleasepool {
        /* MTLCopyAllDevices returns a +1 array (even when empty). */
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        const NSUInteger count = devices ? devices.count : 0;

        if (count == 0) {
            [devices release];
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (!dev) {
                DMN_ERROR("adapter: no Metal devices available");
                return 0;
            }
            outArray[0] = luid_from_registry_id(dev.registryID);
            DMN_INFO("adapter: 1 LUID (default device %s)",
                     dev.name.UTF8String);
            [dev release];
            return 1;
        }

        uint64_t default_reg = 0;
        if (id<MTLDevice> def = MTLCreateSystemDefaultDevice()) {
            default_reg = def.registryID;
            [def release];
        }

        uint32_t n = (uint32_t)std::min<size_t>(count, capacity);
        uint32_t slot = 0;
        /* Default device first, then the rest in enumeration order. */
        for (NSUInteger pass = 0; pass < 2 && slot < n; pass++) {
            for (NSUInteger i = 0; i < count && slot < n; i++) {
                uint64_t reg = devices[i].registryID;
                bool is_default = reg == default_reg;
                if ((pass == 0) == is_default) {
                    outArray[slot++] = luid_from_registry_id(reg);
                    DMN_INFO("adapter: LUID[%u] = %08x:%08x (%s)", slot - 1,
                             (uint32_t)(reg >> 32),
                             (uint32_t)(reg & 0xffffffffu),
                             devices[i].name.UTF8String);
                }
            }
        }
        [devices release];
        return slot;
    }
}
