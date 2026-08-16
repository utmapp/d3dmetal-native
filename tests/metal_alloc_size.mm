/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * MTLDevice.currentAllocatedSize for the footprint test, in its own ObjC++ TU
 * because the DirectX headers and <Metal/Metal.h> cannot share one (windows_
 * base.h's BOOL conflicts with objc.h) — the same split the library uses.
 *
 * The system default device is a singleton, so the device this returns is the
 * one D3DMetal allocates from: the counter covers the framework's allocations
 * as well as the substituted shared backings this library makes.
 */

#import <Metal/Metal.h>
#include <mach/mach.h>

extern "C" {

static id<MTLDevice> g_dev;

static id<MTLDevice> dev(void) {
    if (!g_dev)
        g_dev = MTLCreateSystemDefaultDevice();
    return g_dev;
}

unsigned long long dmn_test_metal_allocated_size(void) {
    return (unsigned long long)dev().currentAllocatedSize;
}

const char* dmn_test_metal_device_name(void) {
    return dev().name.UTF8String;
}

} /* extern "C" */

extern "C" unsigned long long dmn_test_phys_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return (unsigned long long)info.phys_footprint;
}
