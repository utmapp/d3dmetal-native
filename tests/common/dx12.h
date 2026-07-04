/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * D3D12 device/queue bootstrap for the test programs.
 */

#pragma once

#include <d3d12.h>

#include "com.h"

static inline HRESULT make_d3d12_device(Com<ID3D12Device>& dev) {
    return D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                             __uuidof(ID3D12Device), (void**)&dev);
}

static inline HRESULT make_d3d12_queue(ID3D12Device* dev,
                                       Com<ID3D12CommandQueue>& queue) {
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    return dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue),
                                   (void**)&queue);
}
