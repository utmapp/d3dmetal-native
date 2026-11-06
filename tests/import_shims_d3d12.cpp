/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * D3D12 half of import_shims_test: create a buffer in the
 * RAYTRACING_ACCELERATION_STRUCTURE state, which D3DMetal backs with a Metal
 * acceleration structure allocated through
 * MTLAccelerationStructureAllocationDescriptor (or, in a patched framework,
 * DMNAccelerationStructureAllocationDescriptor). Kept apart from the
 * Objective-C++ half because windows.h and objc.h disagree about BOOL.
 */

#include <d3d12.h>
#include <windows.h>

#include "common/com.h"
#include "common/dx12.h"

/* Newer than the vendored d3d12.h. */
static const D3D12_RESOURCE_STATES kStateRaytracingAS =
    (D3D12_RESOURCE_STATES)0x400000;

extern "C" int import_shims_create_as(unsigned long long* gpu_va)
{
    Com<ID3D12Device> dev;
    if (FAILED(make_d3d12_device(dev)))
        return 1;

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5,
                                        sizeof(o5))) ||
        o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
        return -1;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 64 * 1024;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    Com<ID3D12Resource> as;
    if (FAILED(dev->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd,
            kStateRaytracingAS, nullptr,
            __uuidof(ID3D12Resource), (void**)&as)))
        return 2;
    *gpu_va = as->GetGPUVirtualAddress();
    return 0;
}
