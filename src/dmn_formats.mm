/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DXGI format capability tables (see dmn_formats.h for why this TU exists).
 *
 * Three layers, each with a different source — worth knowing which, because
 * they are trustworthy to different degrees:
 *
 *   1. describe() — DXGI_FORMAT -> MTLPixelFormat. Transcribed from
 *      D3DMetal's own table (see below). Authoritative.
 *   2. caps_for() — MTLPixelFormat -> capability bits, built once and branched
 *      on GPU family. Transcribed from Apple's Metal Feature Set Tables; Metal
 *      has no per-format capability query, so there is nothing to ask the
 *      driver at runtime beyond the family and BC support. Authoritative for
 *      the hardware, though see the note on which columns are verifiable.
 *   3. dmn_format_support() — capability bits -> D3D11_FORMAT_SUPPORT mask.
 *      dxmt's design: which Metal capability implies which D3D bit, which bits
 *      are granted unconditionally, the vertex-attribute and backbuffer/
 *      DISPLAY sets in describe(), and returning E_INVALIDARG for an
 *      unrepresentable format. Neither Apple's tables nor D3DMetal say
 *      anything about D3D semantics, so this layer is reasoned, not sourced —
 *      treat it as the least certain of the three.
 *
 * The mapping mirrors the format table D3DMetal itself uses, which is static
 * and device-independent — only the per-entry capability flags vary with the
 * GPU. Two patterns run through it: every _TYPELESS format resolves to the
 * Uint flavour of its group, and the 96-bit formats widen to RGBA32.
 *
 * The one entry deliberately NOT followed is the D24 group; see the comment at
 * that case for why the table's answer is right for Intel Macs and wrong for
 * the hardware we run on.
 *
 * The capability data comes from Apple's Metal Feature Set Tables ("Texture
 * capabilities by pixel format" / "Texture buffer pixel formats", May 21 2026
 * revision), which are the authority for it — in preference to experiment,
 * because most of it cannot be measured. Only Color, Blend and MSAA can be
 * probed at all, since Metal validates those when you build the pipeline state
 * or the multisample texture. Write, Filter, Sparse and Resolve cannot: Metal
 * accepts the descriptor (a BC texture will happily take
 * MTLTextureUsageShaderWrite) and only misbehaves later, so a creation probe
 * answers "yes" for every format. Probing Color further needs the fragment
 * output type to match the attachment — float4 into an R8Uint attachment is
 * rejected for the type mismatch, not for lack of renderability, which reads
 * as missing support if taken at face value.
 */

#import <Metal/Metal.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "dmn_directx_types.h"
#include "dmn_formats.h"
#include "dmn_log.h"

namespace {

/* == Metal format capabilities ============================================ */
/* Plain constants and plain &/|: this project has no enum-bitop header and
 * leans C-ish, and one file does not justify introducing one. */
/* Recorded for completeness but not currently consumed: D3D11_FORMAT_SUPPORT
 * has no "filterable" bit (SHADER_SAMPLE covers sampling generally), so
 * nothing in the mask assembly reads this one. */
constexpr uint32_t kCapFilter       = 0x001; /* filterable when sampled */
constexpr uint32_t kCapWrite        = 0x002; /* writable from a shader */
constexpr uint32_t kCapColor        = 0x004; /* usable as a color attachment */
constexpr uint32_t kCapBlend        = 0x008;
constexpr uint32_t kCapMSAA         = 0x010;
constexpr uint32_t kCapResolve      = 0x020;
constexpr uint32_t kCapSparse       = 0x040;
constexpr uint32_t kCapAtomic       = 0x080;
constexpr uint32_t kCapDepthStencil = 0x100;
constexpr uint32_t kCapBufferRead   = 0x200; /* texture-buffer read  */
constexpr uint32_t kCapBufferWrite  = 0x400; /* texture-buffer write */
constexpr uint32_t kCapBufferReadWrite = 0x800; /* texture-buffer read_write */

/* Recurring combinations, named for the group of formats each describes. */
constexpr uint32_t kCapAll =
    kCapFilter | kCapWrite | kCapColor | kCapMSAA | kCapBlend | kCapSparse |
    kCapResolve;
constexpr uint32_t kCapBufAll = kCapBufferRead | kCapBufferWrite | kCapBufferReadWrite;
/* "Read Write" in the tables: separate read and write access, but NOT
 * read_write access -- that is kCapBufferReadWrite. */
constexpr uint32_t kCapBufRdWr = kCapBufferRead | kCapBufferWrite;
/* Packed 16-bit formats: everything except Write. */
constexpr uint32_t kCapPacked16 =
    kCapFilter | kCapColor | kCapMSAA | kCapResolve | kCapBlend | kCapSparse;
/* Integer color formats: never filterable, never blendable, never resolvable. */
constexpr uint32_t kCapInt = kCapWrite | kCapColor | kCapMSAA | kCapSparse;
/* Single-channel 32-bit integers only: same, plus atomics from Apple7. RG32
 * gets atomics later (Apple8, applied below) and RGBA32 never does, so both
 * of those use plain kCapInt. */
constexpr uint32_t kCapInt32 = kCapInt | kCapAtomic;
/* 32-bit floats before Apple9: blendable and MSAA-able but not filterable. */
constexpr uint32_t kCapFloat32 =
    kCapWrite | kCapColor | kCapMSAA | kCapBlend | kCapSparse;

/* GPU family constants spelled numerically so this TU builds against SDKs
 * predating MTLGPUFamilyApple9 (macOS 14). Values are ABI-stable. */
constexpr MTLGPUFamily kFamilyApple7 = (MTLGPUFamily)1007; /* M1  */
constexpr MTLGPUFamily kFamilyApple8 = (MTLGPUFamily)1008; /* M2  */
constexpr MTLGPUFamily kFamilyApple9 = (MTLGPUFamily)1009; /* M3+ */

using CapTable = std::unordered_map<uint32_t, uint32_t>;

/* Build the capability table for `dev`. D3DMetal is Apple-Silicon-only, so
 * there is no Intel/AMD (Mac2) branch — that path is unreachable here. The
 * Apple7 set is the baseline; Apple8 and Apple9 are additive upgrades. */
void build_table(id<MTLDevice> dev, CapTable& t) {
    auto cap = [&t](MTLPixelFormat f, uint32_t c) {
        t[(uint32_t)f] |= c;
    };

    /* Keyed by Metal format, so it also covers formats describe() can never
     * produce: ones DXGI has no equivalent for (A1BGR5Unorm, BGR10A2Unorm, the
     * XR formats, R8/RG8 _sRGB), the 4:2:2 pair (D3DMetal leaves R8G8_B8G8 and
     * G8R8_G8B8 unmapped, so they are unsupported), and Stencil8. Harmless,
     * and keeps the table diffable against the published one. */

    /* -- Apple7 (M1) baseline -------------------------------------------- */

    cap(MTLPixelFormatA8Unorm, kCapAll | kCapBufAll);

    cap(MTLPixelFormatR8Unorm, kCapAll | kCapBufAll);
    cap(MTLPixelFormatR8Unorm_sRGB, kCapAll);
    cap(MTLPixelFormatR8Snorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatR8Uint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatR8Sint, kCapInt | kCapBufAll);

    cap(MTLPixelFormatR16Unorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatR16Snorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatR16Uint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatR16Sint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatR16Float, kCapAll | kCapBufAll);

    cap(MTLPixelFormatRG8Unorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRG8Unorm_sRGB, kCapAll);
    cap(MTLPixelFormatRG8Snorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRG8Uint, kCapInt | kCapBufRdWr);
    cap(MTLPixelFormatRG8Sint, kCapInt | kCapBufRdWr);

    /* Packed 16-bit: resolvable, but not shader-writable. */
    cap(MTLPixelFormatB5G6R5Unorm, kCapPacked16);
    cap(MTLPixelFormatA1BGR5Unorm, kCapPacked16);
    cap(MTLPixelFormatABGR4Unorm, kCapPacked16);
    cap(MTLPixelFormatBGR5A1Unorm, kCapPacked16);

    cap(MTLPixelFormatR32Uint, kCapInt32 | kCapBufAll);
    cap(MTLPixelFormatR32Sint, kCapInt32 | kCapBufAll);
    cap(MTLPixelFormatR32Float, kCapFloat32 | kCapBufAll);

    cap(MTLPixelFormatRG16Unorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRG16Snorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRG16Uint, kCapInt | kCapBufRdWr);
    cap(MTLPixelFormatRG16Sint, kCapInt | kCapBufRdWr);
    cap(MTLPixelFormatRG16Float, kCapAll | kCapBufRdWr);

    cap(MTLPixelFormatRGBA8Unorm, kCapAll | kCapBufAll);
    cap(MTLPixelFormatRGBA8Unorm_sRGB, kCapAll);
    cap(MTLPixelFormatRGBA8Snorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRGBA8Uint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatRGBA8Sint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatBGRA8Unorm, kCapAll | kCapBufferRead);
    cap(MTLPixelFormatBGRA8Unorm_sRGB, kCapAll);

    /* 32-bit packed */
    cap(MTLPixelFormatRGB10A2Unorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatBGR10A2Unorm, kCapAll);
    cap(MTLPixelFormatRGB10A2Uint, kCapInt | kCapBufRdWr);
    cap(MTLPixelFormatRG11B10Float, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRGB9E5Float, kCapAll);

    /* 64-bit */
    cap(MTLPixelFormatRG32Uint, kCapInt | kCapBufRdWr);
    cap(MTLPixelFormatRG32Sint, kCapInt | kCapBufRdWr);
    cap(MTLPixelFormatRG32Float, kCapFloat32 | kCapBufRdWr);
    cap(MTLPixelFormatRGBA16Unorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRGBA16Snorm, kCapAll | kCapBufRdWr);
    cap(MTLPixelFormatRGBA16Uint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatRGBA16Sint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatRGBA16Float, kCapAll | kCapBufAll);

    /* 128-bit */
    cap(MTLPixelFormatRGBA32Uint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatRGBA32Sint, kCapInt | kCapBufAll);
    cap(MTLPixelFormatRGBA32Float, kCapFloat32 | kCapBufAll);

    /* Block compressed. Apple9+ always has BC; on Apple7/Apple8 the tables say
     * "Varies" — every macOS device in those families has it, but iPadOS ones
     * may not, so the runtime query is the correct gate either way. */
    if ([dev respondsToSelector:@selector(supportsBCTextureCompression)] &&
        [dev supportsBCTextureCompression]) {
        const uint32_t bc = kCapFilter | kCapSparse;
        cap(MTLPixelFormatBC1_RGBA, bc);
        cap(MTLPixelFormatBC1_RGBA_sRGB, bc);
        cap(MTLPixelFormatBC2_RGBA, bc);
        cap(MTLPixelFormatBC2_RGBA_sRGB, bc);
        cap(MTLPixelFormatBC3_RGBA, bc);
        cap(MTLPixelFormatBC3_RGBA_sRGB, bc);
        cap(MTLPixelFormatBC4_RUnorm, bc);
        cap(MTLPixelFormatBC4_RSnorm, bc);
        cap(MTLPixelFormatBC5_RGUnorm, bc);
        cap(MTLPixelFormatBC5_RGSnorm, bc);
        cap(MTLPixelFormatBC6H_RGBUfloat, bc);
        cap(MTLPixelFormatBC6H_RGBFloat, bc);
        cap(MTLPixelFormatBC7_RGBAUnorm, bc);
        cap(MTLPixelFormatBC7_RGBAUnorm_sRGB, bc);
    }

    /* Packed 4:2:2 */
    cap(MTLPixelFormatGBGR422, kCapFilter);
    cap(MTLPixelFormatBGRG422, kCapFilter);

    /* Depth & stencil. Depth24Unorm_Stencil8 and X24_Stencil8 are absent on
     * Apple silicon entirely — the D24 DXGI formats are emulated on top of
     * Depth32Float_Stencil8 (see describe()). Sparse is Apple7 here, not
     * Apple8: the tables footnote it as available on every Apple7 macOS
     * device (only iPadOS varies, where supportsPlacementSparse decides). */
    cap(MTLPixelFormatDepth16Unorm,
        kCapFilter | kCapMSAA | kCapResolve | kCapSparse | kCapDepthStencil);
    cap(MTLPixelFormatDepth32Float,
        kCapMSAA | kCapResolve | kCapSparse | kCapDepthStencil);
    /* Latent — no DXGI format
     * reaches MTLPixelFormatStencil8 (D3D's stencil-only views are the
     * X24/X32 typeless ones, which land on X32_Stencil8), so this entry is
     * carried for completeness rather than consulted. */
    cap(MTLPixelFormatStencil8,
        kCapMSAA | kCapResolve | kCapSparse | kCapDepthStencil);
    cap(MTLPixelFormatDepth32Float_Stencil8,
        kCapMSAA | kCapResolve | kCapDepthStencil);
    /* X32_Stencil8 is MSAA-only on every family — no resolve, no sparse. */
    cap(MTLPixelFormatX32_Stencil8, kCapMSAA | kCapDepthStencil);

    /* Extended range */
    cap(MTLPixelFormatBGRA10_XR, kCapAll);
    cap(MTLPixelFormatBGRA10_XR_sRGB, kCapAll);
    cap(MTLPixelFormatBGR10_XR, kCapAll);
    cap(MTLPixelFormatBGR10_XR_sRGB, kCapAll);

    if (![dev supportsFamily:kFamilyApple8])
        return;

    /* -- Apple8 (M2) additions: 64-bit integer atomics -------------------- */
    cap(MTLPixelFormatRG32Uint, kCapAtomic);
    cap(MTLPixelFormatRG32Sint, kCapAtomic);

    if (![dev supportsFamily:kFamilyApple9])
        return;

    /* -- Apple9 (M3+): 32-bit floats become filterable/resolvable, and the
     *    float depth formats gain Filter. ------------------------------- */
    cap(MTLPixelFormatR32Float, kCapAll);
    cap(MTLPixelFormatRG32Float, kCapAll);
    cap(MTLPixelFormatRGBA32Float, kCapAll);
    /* Latent — nothing reads kCapFilter (see its declaration). */
    cap(MTLPixelFormatDepth32Float, kCapFilter);
    cap(MTLPixelFormatDepth32Float_Stencil8, kCapFilter);
}

/* Capability lookup. The table is immutable after the first call, so the
 * once_flag is the whole of the synchronisation. */
uint32_t caps_for(MTLPixelFormat format) {
    static CapTable  g_table;
    static std::once_flag g_once;
    std::call_once(g_once, [] {
        /* The same accessor dmn_gfxt_adapter.mm and dmn_share_metal.mm use;
         * D3DMetal renders on the system default device. */
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            DMN_WARN("formats: no Metal device — reporting no format caps");
            return;
        }
        if (![dev supportsFamily:kFamilyApple7]) {
            /* Should be unreachable: D3DMetal requires Apple silicon. Fall
             * through with the Apple7 baseline rather than refusing to
             * answer; this library does not use exceptions. */
            DMN_WARN("formats: %s is not Apple7+ — assuming the Apple7 "
                     "baseline format capabilities",
                     [[dev name] UTF8String]);
        }
        build_table(dev, g_table);
        DMN_INFO("formats: capability table built for %s (%zu formats)",
                 [[dev name] UTF8String], g_table.size());
        [dev release];
    });
    auto it = g_table.find((uint32_t)format);
    return it == g_table.end() ? 0 : it->second;
}

/* == DXGI -> Metal ======================================================== */

constexpr uint32_t kFmtTypeless     = 0x1;
constexpr uint32_t kFmtBC           = 0x2;
constexpr uint32_t kFmtBackbuffer   = 0x4; /* legal swapchain/display format */
constexpr uint32_t kFmtDepthPlane   = 0x8;
constexpr uint32_t kFmtStencilPlane = 0x10;

struct DmnFormatDesc {
    MTLPixelFormat pixel;     /* MTLPixelFormatInvalid if none */
    bool           attribute; /* legal as a vertex attribute */
    uint32_t       flags;
};

/* Translate a DXGI_FORMAT. Returns false for formats with no Metal
 * representation, which the caller turns into E_INVALIDARG.
 *
 * The pixel format comes from D3DMetal's table. The two other fields do not,
 * because that table does not record them, so both are dxmt's:
 *  - `attribute`: whether the format is legal as a vertex attribute. dxmt
 *    carries a full attribute-format enum per entry because it builds real
 *    vertex descriptors; we only need the yes/no, so it collapses to a bool.
 *  - kFmtBackbuffer: which formats may be presented, i.e. earn DISPLAY.
 * Neither is verifiable against the sources above — if either looks wrong,
 * there is nothing authoritative backing it. */
bool describe(uint32_t dxgi, DmnFormatDesc& out) {
    out.pixel     = MTLPixelFormatInvalid;
    out.attribute = false;
    out.flags     = 0;

    switch (dxgi) {
    /* -- 128-bit ---------------------------------------------------------- */
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        out.pixel = MTLPixelFormatRGBA32Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R32G32B32A32_UINT:
        out.pixel = MTLPixelFormatRGBA32Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32B32A32_SINT:
        out.pixel = MTLPixelFormatRGBA32Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        out.pixel = MTLPixelFormatRGBA32Float;
        out.attribute = true;
        break;

    /* -- 96-bit. Metal has no 96-bit pixel format. D3DMetal widens to the
     *    4-component format. Note it must not be narrowed to the 1-component R32
     *    instead: that would hand R32G32B32_UINT an IA_INDEX_BUFFER bit (only
     *    R16_UINT/R32_UINT are legal index formats) and R32's atomics. ------- */
    case DXGI_FORMAT_R32G32B32_TYPELESS:
        out.pixel = MTLPixelFormatRGBA32Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R32G32B32_UINT:
        out.pixel = MTLPixelFormatRGBA32Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32B32_SINT:
        out.pixel = MTLPixelFormatRGBA32Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32B32_FLOAT:
        out.pixel = MTLPixelFormatRGBA32Float;
        out.attribute = true;
        break;

    /* -- 64-bit ---------------------------------------------------------- */
    /* Every _TYPELESS resolves to the Uint flavour in D3DMetal's table.
     * Picking the Float/Unorm sibling here would over-report Blend and
     * Resolve. */
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        out.pixel = MTLPixelFormatRGBA16Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        out.pixel = MTLPixelFormatRGBA16Float;
        out.attribute = true;
        out.flags = kFmtBackbuffer;
        break;
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        out.pixel = MTLPixelFormatRGBA16Unorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16B16A16_UINT:
        out.pixel = MTLPixelFormatRGBA16Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16B16A16_SNORM:
        out.pixel = MTLPixelFormatRGBA16Snorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16B16A16_SINT:
        out.pixel = MTLPixelFormatRGBA16Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32_TYPELESS:
        out.pixel = MTLPixelFormatRG32Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R32G32_FLOAT:
        out.pixel = MTLPixelFormatRG32Float;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32_UINT:
        out.pixel = MTLPixelFormatRG32Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32G32_SINT:
        out.pixel = MTLPixelFormatRG32Sint;
        out.attribute = true;
        break;

    /* -- 32/8/24 depth-stencil group -------------------------------------- */
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        out.pixel = MTLPixelFormatDepth32Float_Stencil8;
        out.flags = kFmtTypeless | kFmtDepthPlane | kFmtStencilPlane;
        break;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        out.pixel = MTLPixelFormatDepth32Float_Stencil8;
        out.flags = kFmtDepthPlane | kFmtStencilPlane;
        break;
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        /* depth plane of Depth32Float_Stencil8, read as R */
        out.pixel = MTLPixelFormatDepth32Float_Stencil8;
        out.flags = kFmtTypeless | kFmtDepthPlane;
        break;
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        /* stencil plane only */
        out.pixel = MTLPixelFormatX32_Stencil8;
        out.flags = kFmtTypeless | kFmtStencilPlane;
        break;

    /* -- 32-bit ---------------------------------------------------------- */
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        out.pixel = MTLPixelFormatRGB10A2Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        out.pixel = MTLPixelFormatRGB10A2Unorm;
        out.attribute = true;
        out.flags = kFmtBackbuffer;
        break;
    case DXGI_FORMAT_R10G10B10A2_UINT:
        out.pixel = MTLPixelFormatRGB10A2Uint;
        break;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        out.pixel = MTLPixelFormatRG11B10Float;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        out.pixel = MTLPixelFormatRGBA8Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
        out.pixel = MTLPixelFormatRGBA8Unorm;
        out.attribute = true;
        out.flags = kFmtBackbuffer;
        break;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        out.pixel = MTLPixelFormatRGBA8Unorm_sRGB;
        out.flags = kFmtBackbuffer;
        break;
    case DXGI_FORMAT_R8G8B8A8_UINT:
        out.pixel = MTLPixelFormatRGBA8Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8G8B8A8_SNORM:
        out.pixel = MTLPixelFormatRGBA8Snorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8G8B8A8_SINT:
        out.pixel = MTLPixelFormatRGBA8Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16_TYPELESS:
        out.pixel = MTLPixelFormatRG16Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R16G16_FLOAT:
        out.pixel = MTLPixelFormatRG16Float;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16_UNORM:
        out.pixel = MTLPixelFormatRG16Unorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16_UINT:
        out.pixel = MTLPixelFormatRG16Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16_SNORM:
        out.pixel = MTLPixelFormatRG16Snorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16G16_SINT:
        out.pixel = MTLPixelFormatRG16Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32_TYPELESS:
        out.pixel = MTLPixelFormatR32Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_D32_FLOAT:
        out.pixel = MTLPixelFormatDepth32Float;
        out.flags = kFmtDepthPlane;
        break;
    case DXGI_FORMAT_R32_FLOAT:
        out.pixel = MTLPixelFormatR32Float;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32_UINT:
        out.pixel = MTLPixelFormatR32Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R32_SINT:
        out.pixel = MTLPixelFormatR32Sint;
        out.attribute = true;
        break;

    /* -- D24 group. THE ONE PLACE D3DMetal'S TABLE MUST NOT BE TAKEN
     *    LITERALLY. It maps these to Depth24Unorm_Stencil8 / X24_Stencil8,
     *    which are real formats on Intel Macs -- its mapping table is static
     *    and device-independent, so it names the Intel-correct format. On
     *    Apple silicon both are absent (isDepth24Stencil8PixelFormatSupported
     *    is 0; Metal rejects the descriptor as "invalid pixelFormat"), yet
     *    D24S8 textures and DSVs still create successfully through D3DMetal --
     *    so it substitutes at resource-creation time. The format the GPU
     *    actually gets is the 32-bit one, so that is what we report caps for.
     *    Taking the table at face value here would key the lookup on a format
     *    absent from the capability table and report D24S8 -- which every
     *    D3D11 game uses -- as having no depth-stencil support at all. ----- */
    case DXGI_FORMAT_R24G8_TYPELESS:
        out.pixel = MTLPixelFormatDepth32Float_Stencil8;
        out.flags = kFmtTypeless | kFmtDepthPlane | kFmtStencilPlane;
        break;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        out.pixel = MTLPixelFormatDepth32Float_Stencil8;
        out.flags = kFmtDepthPlane | kFmtStencilPlane;
        break;
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        out.pixel = MTLPixelFormatDepth32Float_Stencil8;
        out.flags = kFmtDepthPlane;
        break;
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        /* D3DMetal names X24_Stencil8; absent on Apple silicon, same as
         * above, so the 32-bit stencil view is the effective format. */
        out.pixel = MTLPixelFormatX32_Stencil8;
        out.flags = kFmtStencilPlane;
        break;

    /* -- 16-bit ---------------------------------------------------------- */
    case DXGI_FORMAT_R8G8_TYPELESS:
        out.pixel = MTLPixelFormatRG8Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R8G8_UNORM:
        out.pixel = MTLPixelFormatRG8Unorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8G8_UINT:
        out.pixel = MTLPixelFormatRG8Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8G8_SNORM:
        out.pixel = MTLPixelFormatRG8Snorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8G8_SINT:
        out.pixel = MTLPixelFormatRG8Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16_TYPELESS:
        out.pixel = MTLPixelFormatR16Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R16_FLOAT:
        out.pixel = MTLPixelFormatR16Float;
        out.attribute = true;
        break;
    case DXGI_FORMAT_D16_UNORM:
        out.pixel = MTLPixelFormatDepth16Unorm;
        out.flags = kFmtDepthPlane;
        break;
    case DXGI_FORMAT_R16_UNORM:
        out.pixel = MTLPixelFormatR16Unorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16_UINT:
        out.pixel = MTLPixelFormatR16Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16_SNORM:
        out.pixel = MTLPixelFormatR16Snorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R16_SINT:
        out.pixel = MTLPixelFormatR16Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_B5G6R5_UNORM:
        out.pixel = MTLPixelFormatB5G6R5Unorm;
        break;
    case DXGI_FORMAT_B5G5R5A1_UNORM:
        out.pixel = MTLPixelFormatBGR5A1Unorm;
        break;
    case DXGI_FORMAT_B4G4R4A4_UNORM:
        out.pixel = MTLPixelFormatABGR4Unorm;
        break;

    /* -- 8-bit ----------------------------------------------------------- */
    case DXGI_FORMAT_R8_TYPELESS:
        out.pixel = MTLPixelFormatR8Uint;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_R8_UNORM:
        out.pixel = MTLPixelFormatR8Unorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8_UINT:
        out.pixel = MTLPixelFormatR8Uint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8_SNORM:
        out.pixel = MTLPixelFormatR8Snorm;
        out.attribute = true;
        break;
    case DXGI_FORMAT_R8_SINT:
        out.pixel = MTLPixelFormatR8Sint;
        out.attribute = true;
        break;
    case DXGI_FORMAT_A8_UNORM:
        out.pixel = MTLPixelFormatA8Unorm;
        break;

    /* -- shared-exponent -------------------------------------------------- */
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        out.pixel = MTLPixelFormatRGB9E5Float;
        out.attribute = true;
        break;
    /* -- BGR ------------------------------------------------------------- */
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        out.pixel = MTLPixelFormatBGRA8Unorm;
        out.attribute = true;
        out.flags = kFmtBackbuffer;
        break;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        out.pixel = MTLPixelFormatBGRA8Unorm;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        out.pixel = MTLPixelFormatBGRA8Unorm_sRGB;
        out.flags = kFmtBackbuffer;
        break;
    /* B8G8R8X8 is BGRA8 with alpha forced to one — a swizzle, not a distinct
     * Metal format, so it inherits BGRA8's capabilities. */
    case DXGI_FORMAT_B8G8R8X8_UNORM:
        out.pixel = MTLPixelFormatBGRA8Unorm;
        break;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
        out.pixel = MTLPixelFormatBGRA8Unorm;
        out.flags = kFmtTypeless;
        break;
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        out.pixel = MTLPixelFormatBGRA8Unorm_sRGB;
        break;

    /* -- block compressed ------------------------------------------------- */
    case DXGI_FORMAT_BC1_TYPELESS:
        out.pixel = MTLPixelFormatBC1_RGBA;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC1_UNORM:
        out.pixel = MTLPixelFormatBC1_RGBA;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC1_UNORM_SRGB:
        out.pixel = MTLPixelFormatBC1_RGBA_sRGB;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC2_TYPELESS:
        out.pixel = MTLPixelFormatBC2_RGBA;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC2_UNORM:
        out.pixel = MTLPixelFormatBC2_RGBA;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC2_UNORM_SRGB:
        out.pixel = MTLPixelFormatBC2_RGBA_sRGB;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC3_TYPELESS:
        out.pixel = MTLPixelFormatBC3_RGBA;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC3_UNORM:
        out.pixel = MTLPixelFormatBC3_RGBA;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC3_UNORM_SRGB:
        out.pixel = MTLPixelFormatBC3_RGBA_sRGB;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC4_TYPELESS:
        out.pixel = MTLPixelFormatBC4_RUnorm;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC4_UNORM:
        out.pixel = MTLPixelFormatBC4_RUnorm;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC4_SNORM:
        out.pixel = MTLPixelFormatBC4_RSnorm;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC5_TYPELESS:
        out.pixel = MTLPixelFormatBC5_RGUnorm;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC5_UNORM:
        out.pixel = MTLPixelFormatBC5_RGUnorm;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC5_SNORM:
        out.pixel = MTLPixelFormatBC5_RGSnorm;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC6H_TYPELESS:
        out.pixel = MTLPixelFormatBC6H_RGBFloat;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC6H_UF16:
        out.pixel = MTLPixelFormatBC6H_RGBUfloat;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC6H_SF16:
        out.pixel = MTLPixelFormatBC6H_RGBFloat;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC7_TYPELESS:
        out.pixel = MTLPixelFormatBC7_RGBAUnorm;
        out.flags = kFmtTypeless | kFmtBC;
        break;
    case DXGI_FORMAT_BC7_UNORM:
        out.pixel = MTLPixelFormatBC7_RGBAUnorm;
        out.flags = kFmtBC;
        break;
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        out.pixel = MTLPixelFormatBC7_RGBAUnorm_sRGB;
        out.flags = kFmtBC;
        break;

    /* Everything else — the planar/packed video formats (NV12, YUY2, AYUV,
     * P010, ...), the palettized formats (P8, A8P8, AI44, IA44), R1_UNORM and
     * R10G10B10_XR_BIAS_A2_UNORM — has no Metal equivalent. DXGI_FORMAT_UNKNOWN
     * also lands here, but callers handle it before reaching describe(). */
    default:
        return false;
    }
    return true;
}

/* == Linear (buffer-backed) element size ================================== */
/* Deliberately a switch and not a table build: this is the stride oracle for
 * cross-process shared surfaces, so it must answer before any device exists,
 * cannot depend on GPU family, and must fail closed on anything it has not
 * been taught. Every colour format the capability table above knows is listed;
 * the omissions are the ones with no linear element size — block-compressed
 * (4x4 blocks), depth/stencil (opaque, tiled-only), and the 4:2:2 pair (two
 * pixels per element). */
uint32_t linear_bpp(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatA8Unorm:
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Unorm_sRGB:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
        return 1;

    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Unorm_sRGB:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatB5G6R5Unorm:
    case MTLPixelFormatA1BGR5Unorm:
    case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm:
        return 2;

    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatBGR10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatBGR10_XR:
    case MTLPixelFormatBGR10_XR_sRGB:
        return 4;

    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatBGRA10_XR:
    case MTLPixelFormatBGRA10_XR_sRGB:
        return 8;

    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
    case MTLPixelFormatRGBA32Float:
        return 16;

    default:
        return 0;
    }
}

} /* namespace */

/* == Public queries ======================================================= */

uint32_t dmn_format_linear_bpp(uint32_t mtl_pixel_format) {
    return linear_bpp((MTLPixelFormat)mtl_pixel_format);
}

bool dmn_format_support(uint32_t dxgi_format, uint32_t* out_support) {
    /* UNKNOWN is legal and means "structured/raw buffer": no texture caps. */
    if (dxgi_format == DXGI_FORMAT_UNKNOWN) {
        *out_support = DMN_FMT_SUPPORT_BUFFER | DMN_FMT_SUPPORT_CPU_LOCKABLE;
        return true;
    }

    DmnFormatDesc desc;
    if (!describe(dxgi_format, desc))
        return false;

    uint32_t support = 0;

    if (desc.pixel != MTLPixelFormatInvalid) {
        /* Every graphics and compute shader can read or sample a texture of
         * any pixel format, so these are unconditional. (Filtering is a
         * separate capability; SHADER_SAMPLE only promises point sampling.) */
        support |= DMN_FMT_SUPPORT_SHADER_LOAD |
                   DMN_FMT_SUPPORT_SHADER_SAMPLE |
                   DMN_FMT_SUPPORT_SHADER_GATHER |
                   DMN_FMT_SUPPORT_MULTISAMPLE_LOAD |
                   DMN_FMT_SUPPORT_CPU_LOCKABLE;

        /* UNCHECKED — inherited from dxmt, which flags these as asserted
         * rather than verified. */
        support |= DMN_FMT_SUPPORT_TEXTURE1D | DMN_FMT_SUPPORT_TEXTURE2D |
                   DMN_FMT_SUPPORT_TEXTURE3D | DMN_FMT_SUPPORT_TEXTURECUBE |
                   DMN_FMT_SUPPORT_MIP |
                   DMN_FMT_SUPPORT_CAST_WITHIN_BIT_LAYOUT;

        /* Compressed and depth/stencil-planar formats cannot back a typed
         * buffer, and neither can have mips generated. */
        if (!(desc.flags &
              (kFmtBC | kFmtDepthPlane | kFmtStencilPlane)))
            support |= DMN_FMT_SUPPORT_BUFFER | DMN_FMT_SUPPORT_MIP_AUTOGEN;

        if (desc.flags & kFmtBackbuffer)
            support |= DMN_FMT_SUPPORT_DISPLAY;
    }

    if (desc.attribute)
        support |= DMN_FMT_SUPPORT_IA_VERTEX_BUFFER;

    if (desc.pixel == MTLPixelFormatR32Uint ||
        desc.pixel == MTLPixelFormatR16Uint)
        support |= DMN_FMT_SUPPORT_IA_INDEX_BUFFER;

    const uint32_t caps = caps_for(desc.pixel);

    if (caps & kCapColor)
        support |= DMN_FMT_SUPPORT_RENDER_TARGET;
    if (caps & kCapBlend)
        support |= DMN_FMT_SUPPORT_BLENDABLE;
    if (caps & kCapDepthStencil)
        support |= DMN_FMT_SUPPORT_DEPTH_STENCIL |
                   DMN_FMT_SUPPORT_SHADER_SAMPLE_COMPARISON |
                   DMN_FMT_SUPPORT_SHADER_GATHER_COMPARISON;
    if (caps & kCapResolve)
        support |= DMN_FMT_SUPPORT_MULTISAMPLE_RESOLVE;
    if (caps & kCapMSAA)
        support |= DMN_FMT_SUPPORT_MULTISAMPLE_RENDERTARGET;
    if (caps & kCapWrite)
        support |= DMN_FMT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW;

    /* Stream-output takes 32-bit components only. */
    switch (dxgi_format) {
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        support |= DMN_FMT_SUPPORT_SO_BUFFER;
        break;
    default:
        break;
    }

    *out_support = support;
    return true;
}

bool dmn_format_support2(uint32_t dxgi_format, uint32_t* out_support2) {
    constexpr uint32_t kAtomicBits =
        DMN_FMT_SUPPORT2_UAV_ATOMIC_ADD |
        DMN_FMT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS |
        DMN_FMT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE |
        DMN_FMT_SUPPORT2_UAV_ATOMIC_EXCHANGE |
        DMN_FMT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX |
        DMN_FMT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX;

    /* UNKNOWN here is the raw/structured-buffer UAV, which supports the whole
     * atomic set and typed access. */
    if (dxgi_format == DXGI_FORMAT_UNKNOWN) {
        *out_support2 = kAtomicBits |
                        DMN_FMT_SUPPORT2_UAV_TYPED_LOAD |
                        DMN_FMT_SUPPORT2_UAV_TYPED_STORE |
                        DMN_FMT_SUPPORT2_SHAREABLE;
        return true;
    }

    DmnFormatDesc desc;
    if (!describe(dxgi_format, desc))
        return false;

    const uint32_t caps = caps_for(desc.pixel);
    uint32_t support2 = 0;

    if (caps & kCapBufferRead)
        support2 |= DMN_FMT_SUPPORT2_UAV_TYPED_LOAD;
    if (caps & kCapBufferWrite)
        support2 |= DMN_FMT_SUPPORT2_UAV_TYPED_STORE;
    if (caps & kCapBufferReadWrite)
        support2 |= DMN_FMT_SUPPORT2_UAV_TYPED_LOAD |
                    DMN_FMT_SUPPORT2_UAV_TYPED_STORE;
    if (caps & kCapAtomic)
        support2 |= kAtomicBits | DMN_FMT_SUPPORT2_SHAREABLE;
    if (caps & kCapBlend)
        support2 |= DMN_FMT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP; /* UNCHECKED */
    if (caps & kCapSparse)
        support2 |= DMN_FMT_SUPPORT2_TILED;

    *out_support2 = support2;
    return true;
}
