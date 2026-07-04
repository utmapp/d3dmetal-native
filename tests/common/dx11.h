/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * D3D11 device bootstrap for the test programs.
 */

#pragma once

#include <d3d11_4.h>

#include "com.h"

static inline HRESULT make_d3d11_device(Com<ID3D11Device>& dev,
                                        Com<ID3D11DeviceContext>& ctx,
                                        UINT flags = 0) {
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_1, flo;
    return D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                             &fl, 1, D3D11_SDK_VERSION, &dev, &flo, &ctx);
}
