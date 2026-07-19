/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DXGI format capability reporting. D3DMetal's own CheckFormatSupport answers
 * 0xffffffff for every format it recognizes and 0 for every one it does not,
 * so each capability bit is wrong in one direction or the other; apps that
 * branch on RENDER_TARGET / BLENDABLE / TYPED_UNORDERED_ACCESS_VIEW pick paths
 * Metal cannot express and render wrong pixels instead of failing. The real
 * answers are computed in dmn_formats.mm from a DXGI -> MTLPixelFormat table
 * plus a Metal capability table, and dmn_com_hooks.cpp serves them.
 *
 * This header is the facade between the two header worlds: dmn_formats.mm is
 * Objective-C++ and must use dmn_directx_types.h (which #errors if the
 * vendored DirectX headers are also present), while dmn_com_hooks.cpp needs
 * the real d3d11.h to patch vtables. So the interface is uint32_t in,
 * uint32_t out, with no DirectX and no Metal types crossing it — the same
 * split dmn_share.h uses.
 */

#pragma once

#include <cstdint>

/* == D3D11_FORMAT_SUPPORT mirrors ========================================= */
/* Values copied from include/native/directx/d3d11.h. dmn_com_hooks.cpp is the
 * only TU that sees both these and the real enum, so it static_asserts every
 * pair — drift is a compile error, not a silently wrong capability mask. Only
 * the bits this implementation can actually produce are mirrored. */
enum DmnFormatSupport : uint32_t {
    DMN_FMT_SUPPORT_BUFFER                       = 0x1,
    DMN_FMT_SUPPORT_IA_VERTEX_BUFFER             = 0x2,
    DMN_FMT_SUPPORT_IA_INDEX_BUFFER              = 0x4,
    DMN_FMT_SUPPORT_SO_BUFFER                    = 0x8,
    DMN_FMT_SUPPORT_TEXTURE1D                    = 0x10,
    DMN_FMT_SUPPORT_TEXTURE2D                    = 0x20,
    DMN_FMT_SUPPORT_TEXTURE3D                    = 0x40,
    DMN_FMT_SUPPORT_TEXTURECUBE                  = 0x80,
    DMN_FMT_SUPPORT_SHADER_LOAD                  = 0x100,
    DMN_FMT_SUPPORT_SHADER_SAMPLE                = 0x200,
    DMN_FMT_SUPPORT_SHADER_SAMPLE_COMPARISON     = 0x400,
    DMN_FMT_SUPPORT_MIP                          = 0x1000,
    DMN_FMT_SUPPORT_MIP_AUTOGEN                  = 0x2000,
    DMN_FMT_SUPPORT_RENDER_TARGET                = 0x4000,
    DMN_FMT_SUPPORT_BLENDABLE                    = 0x8000,
    DMN_FMT_SUPPORT_DEPTH_STENCIL                = 0x10000,
    DMN_FMT_SUPPORT_CPU_LOCKABLE                 = 0x20000,
    DMN_FMT_SUPPORT_MULTISAMPLE_RESOLVE          = 0x40000,
    DMN_FMT_SUPPORT_DISPLAY                      = 0x80000,
    DMN_FMT_SUPPORT_CAST_WITHIN_BIT_LAYOUT       = 0x100000,
    DMN_FMT_SUPPORT_MULTISAMPLE_RENDERTARGET     = 0x200000,
    DMN_FMT_SUPPORT_MULTISAMPLE_LOAD             = 0x400000,
    DMN_FMT_SUPPORT_SHADER_GATHER                = 0x800000,
    DMN_FMT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW  = 0x2000000,
    DMN_FMT_SUPPORT_SHADER_GATHER_COMPARISON     = 0x4000000,
};

/* == D3D11_FORMAT_SUPPORT2 mirrors ======================================== */
enum DmnFormatSupport2 : uint32_t {
    DMN_FMT_SUPPORT2_UAV_ATOMIC_ADD                                  = 0x1,
    DMN_FMT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS                          = 0x2,
    DMN_FMT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE    = 0x4,
    DMN_FMT_SUPPORT2_UAV_ATOMIC_EXCHANGE                             = 0x8,
    DMN_FMT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX                    = 0x10,
    DMN_FMT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX                  = 0x20,
    DMN_FMT_SUPPORT2_UAV_TYPED_LOAD                                  = 0x40,
    DMN_FMT_SUPPORT2_UAV_TYPED_STORE                                 = 0x80,
    DMN_FMT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP                          = 0x100,
    DMN_FMT_SUPPORT2_TILED                                           = 0x200,
    DMN_FMT_SUPPORT2_SHAREABLE                                       = 0x400,
};

/* == Queries ============================================================== */

/* Fill *out_support with the D3D11_FORMAT_SUPPORT mask for a DXGI_FORMAT.
 * Returns false when the format has no Metal representation at all (video,
 * palettized, R1_UNORM, ...), which the caller reports as E_INVALIDARG.
 * DXGI_FORMAT_UNKNOWN is *not* such a case: it is legal and answers
 * BUFFER | CPU_LOCKABLE. */
bool dmn_format_support(uint32_t dxgi_format, uint32_t* out_support);

/* Same contract for the D3D11_FORMAT_SUPPORT2 mask (typed-UAV load/store,
 * UAV atomics, tiled). */
bool dmn_format_support2(uint32_t dxgi_format, uint32_t* out_support2);
