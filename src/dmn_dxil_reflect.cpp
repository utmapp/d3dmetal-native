/*
 * DXIL container reflection via the framework's own libdxcompiler.dylib.
 *
 * dmn_dxil_input_semantics (public, d3dmetal_native.h) names a DXIL vertex
 * shader's input registers: reflect the container with
 * IDxcContainerReflection -> ID3D12ShaderReflection.  The Neptune render
 * server uses it to resolve its guest's register-tagged input-layout names,
 * which a guest holding only a bare DXIL program part cannot know; D3DMetal
 * matches layout elements to the shader BY NAME (a mismatch silently
 * produces a no-op PSO).
 *
 * ABI: libdxcompiler.dylib is a NATIVE darwin build -- its DxcCreateInstance
 * and every interface it returns are SysV/Itanium, NOT the ms_abi the rest
 * of the D3D world here uses.  All interfaces are therefore declared locally
 * with default convention; only the plain data structs are reused from the
 * vendored d3d12shader.h.
 *
 * dmn_init already preloads libdxcompiler.dylib by absolute path, so the
 * leaf dlopen here resolves to the loaded image.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include <unknwn.h>
#include <d3d12shader.h> /* D3D12_SHADER_DESC / D3D12_SIGNATURE_PARAMETER_DESC */

#include "dmn_log.h"
#include "dmn_dxil_reflect.h"

namespace {

/* SysV/Itanium declarations of the DXC interfaces we touch.  Slot order
 * follows the shipped library: a plain 3-slot IUnknown with NO virtual
 * destructor, then the interface methods; Apple's IDxcContainerReflection
 * carries a GetPartContent between GetPartKind and FindFirstPartKind.
 * Single inheritance, no overloads, so clang lays the vtables out the same
 * way. */
struct SysUnknown {
    virtual int32_t QueryInterface(const GUID &iid, void **ppv) = 0;
    virtual uint32_t AddRef() = 0;
    virtual uint32_t Release() = 0;
};

struct SysDxcBlob : public SysUnknown {
    virtual void *GetBufferPointer() = 0;
    virtual size_t GetBufferSize() = 0;
    /* The consumer calls slots past GetBufferSize (it treats the blob as the
     * larger IDxcBlobEncoding/Utf8 shape); harmless zero-returning slots keep
     * those calls from landing in unrelated code. */
    virtual uint64_t Pad5() { return 0; }
    virtual uint64_t Pad6() { return 0; }
    virtual uint64_t Pad7() { return 0; }
    virtual uint64_t Pad8() { return 0; }
};

struct SysDxcContainerReflection : public SysUnknown {
    virtual int32_t Load(SysDxcBlob *pContainer) = 0;
    virtual int32_t GetPartCount(uint32_t *pResult) = 0;
    virtual int32_t GetPartKind(uint32_t idx, uint32_t *pResult) = 0;
    virtual int32_t GetPartContent(uint32_t idx, void **ppResult) = 0;
    virtual int32_t FindFirstPartKind(uint32_t kind, uint32_t *pResult) = 0;
    virtual int32_t GetPartReflection(uint32_t idx, const GUID &iid,
                                      void **ppvObject) = 0;
};

struct SysShaderReflection : public SysUnknown {
    virtual int32_t GetDesc(D3D12_SHADER_DESC *desc) = 0;
    virtual void *GetConstantBufferByIndex(uint32_t index) = 0;
    virtual void *GetConstantBufferByName(const char *name) = 0;
    virtual int32_t GetResourceBindingDesc(uint32_t index, void *desc) = 0;
    virtual int32_t GetInputParameterDesc(
        uint32_t index, D3D12_SIGNATURE_PARAMETER_DESC *desc) = 0;
    virtual int32_t GetOutputParameterDesc(
        uint32_t index, D3D12_SIGNATURE_PARAMETER_DESC *desc) = 0;
};

struct SysDxcOptimizer : public SysUnknown {
    virtual int32_t GetAvailablePassCount(uint32_t *pCount) = 0;
    virtual int32_t GetAvailablePass(uint32_t index, void **ppResult) = 0;
    virtual int32_t RunOptimizer(SysDxcBlob *pBlob, const wchar_t **ppOptions,
                                 uint32_t optionCount,
                                 SysDxcBlob **pOutputModule,
                                 SysDxcBlob **ppOutputText) = 0;
};

const GUID kClsidDxcOptimizer = {
    0xae2cd79f, 0xcc22, 0x453f,
    {0x9b, 0x6b, 0xb1, 0x24, 0xe7, 0xa5, 0x20, 0x4c}};
const GUID kIidDxcOptimizer = {
    0x25740e2e, 0x9cba, 0x401b,
    {0x91, 0x19, 0x4f, 0xb4, 0x2f, 0x39, 0xf2, 0x70}};

const GUID kClsidDxcContainerReflection = {
    0xb9f54489, 0x55b8, 0x400c,
    {0xba, 0x3a, 0x16, 0x75, 0xe4, 0x72, 0x8b, 0x91}};
const GUID kIidDxcContainerReflection = {
    0xd2c21b26, 0x8350, 0x4bdc,
    {0x97, 0x6a, 0x33, 0x1c, 0xe6, 0xf4, 0xc5, 0x4c}};
const GUID kIidShaderReflection = {
    0x5a58797d, 0xa72c, 0x478d,
    {0x8b, 0xa2, 0xef, 0xc6, 0xb0, 0xef, 0xe8, 0x8e}};

/* Non-owning SysDxcBlob view over caller memory. */
struct PinnedBlob final : public SysDxcBlob {
    const void *p;
    size_t n;
    PinnedBlob(const void *p_, size_t n_) : p(p_), n(n_) {}
    int32_t QueryInterface(const GUID &, void **ppv) override {
        *ppv = this;
        return 0;
    }
    uint32_t AddRef() override { return 2; }
    uint32_t Release() override { return 1; }
    void *GetBufferPointer() override { return const_cast<void *>(p); }
    size_t GetBufferSize() override { return n; }
};

typedef int32_t (*PFN_SysDxcCreateInstance)(const GUID *rclsid,
                                            const GUID *riid, void **ppv);

PFN_SysDxcCreateInstance
dxcCreateInstance()
{
    static PFN_SysDxcCreateInstance pfn = [] {
        void *h = dlopen("libdxcompiler.dylib", RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            DMN_WARN("dxil-reflect: libdxcompiler.dylib not loadable: %s",
                     dlerror());
            return (PFN_SysDxcCreateInstance) nullptr;
        }
        return (PFN_SysDxcCreateInstance)dlsym(h, "DxcCreateInstance");
    }();
    return pfn;
}

/* Sequential reader over an LLVM bitstream (the DXIL bitcode format). */
class BitCursor {
    const uint8_t *p_;
    uint64_t end_; /* in bits */
    uint64_t pos_ = 0;
    uint64_t steps_;
    bool bad_ = false;

public:
    BitCursor(const uint8_t *p, size_t bytes)
        : p_(p), end_((uint64_t)bytes * 8), steps_(end_ * 2) {}
    bool bad() const { return bad_; }
    /* Charged per decoded value: literal and zero-width fields take no
     * bits, so the stream length alone does not bound the work. */
    void step() {
        if (steps_)
            steps_--;
        else
            bad_ = true;
    }
    uint64_t left() const { return end_ - pos_; }
    uint64_t fixed(unsigned width) {
        if (width > 64 || width > left()) {
            bad_ = true;
            return 0;
        }
        uint64_t v = 0;
        for (unsigned got = 0; got < width;) {
            const unsigned off = pos_ & 7;
            const unsigned take = width - got < 8 - off ? width - got : 8 - off;
            v |= (uint64_t)((p_[pos_ >> 3] >> off) & ((1u << take) - 1)) << got;
            got += take;
            pos_ += take;
        }
        return v;
    }
    uint64_t vbr(unsigned width) {
        const uint64_t more = 1ull << (width - 1);
        uint64_t v = 0;
        for (unsigned shift = 0;; shift += width - 1) {
            const uint64_t chunk = fixed(width);
            if (bad_ || shift >= 64) {
                bad_ = true;
                return 0;
            }
            v |= (chunk & (more - 1)) << shift;
            if (!(chunk & more))
                return v;
        }
    }
    void skip(uint64_t bits) {
        if (bits > left())
            bad_ = true;
        else
            pos_ += bits;
    }
    void align32() { skip((32 - (pos_ & 31)) & 31); }
};

struct AbbrevOp {
    enum Kind : uint8_t { Literal, Fixed, Vbr, Array, Char6, Blob } kind;
    uint64_t value; /* the literal, or the field width */
};
typedef std::vector<AbbrevOp> Abbrev;

bool
readAbbrev(BitCursor &c, Abbrev &a)
{
    const uint64_t n = c.vbr(5);
    for (uint64_t i = 0; i < n && !c.bad(); i++) {
        if (c.fixed(1)) {
            a.push_back({AbbrevOp::Literal, c.vbr(8)});
            continue;
        }
        const uint64_t enc = c.fixed(3);
        if (enc < AbbrevOp::Fixed || enc > AbbrevOp::Blob)
            return false;
        const bool sized = enc == AbbrevOp::Fixed || enc == AbbrevOp::Vbr;
        const uint64_t width = sized ? c.vbr(5) : 0;
        if (enc == AbbrevOp::Vbr && (width < 2 || width > 32))
            return false;
        a.push_back({(AbbrevOp::Kind)enc, width});
    }
    return !c.bad();
}

/* Reads one record: its code, and its first `cap` operands into ops. */
bool
readRecord(BitCursor &c, const Abbrev *abbrev, uint64_t *code, uint64_t *ops,
           unsigned cap, uint64_t *nops)
{
    uint64_t n = 0;
    auto put = [&](uint64_t v) {
        c.step();
        if (n == 0)
            *code = v;
        else if (n - 1 < cap)
            ops[n - 1] = v;
        n++;
    };
    auto scalar = [&](const AbbrevOp &op) {
        switch (op.kind) {
        case AbbrevOp::Literal: return op.value;
        case AbbrevOp::Fixed: return c.fixed((unsigned)op.value);
        case AbbrevOp::Vbr: return c.vbr((unsigned)op.value);
        default: return c.fixed(6); /* Char6 */
        }
    };
    if (!abbrev) {
        put(c.vbr(6));
        const uint64_t count = c.vbr(6);
        if (count > c.left())
            return false;
        for (uint64_t i = 0; i < count && !c.bad(); i++)
            put(c.vbr(6));
    } else {
        for (size_t i = 0; i < abbrev->size() && !c.bad(); i++) {
            const AbbrevOp &op = (*abbrev)[i];
            if (op.kind == AbbrevOp::Array) {
                const uint64_t count = c.vbr(6);
                if (++i >= abbrev->size() || (*abbrev)[i].kind >= AbbrevOp::Array ||
                    count > c.left())
                    return false;
                for (uint64_t k = 0; k < count && !c.bad(); k++)
                    put(scalar((*abbrev)[i]));
            } else if (op.kind == AbbrevOp::Blob) {
                const uint64_t bytes = c.vbr(6);
                c.align32();
                c.skip(bytes <= c.left() / 8 ? bytes * 8 : ~0ull);
                c.align32();
                c.step();
            } else {
                put(scalar(op));
            }
        }
    }
    *nops = n ? n - 1 : 0;
    return n && !c.bad();
}

enum { kEndBlock = 0, kEnterSubblock = 1, kDefineAbbrev = 2, kUnabbrevRecord = 3 };
enum { kBlockInfo = 0, kModuleBlock = 8 };

/* Walks a BLOCKINFO block.  False if it is malformed or registers
 * abbreviations for the module block, which would renumber that block's
 * own. */
bool
skipBlockInfo(BitCursor &c, unsigned width)
{
    uint64_t target = ~0ull;
    while (!c.bad()) {
        switch (c.fixed(width)) {
        case kEndBlock:
            c.align32();
            return !c.bad();
        case kDefineAbbrev: {
            Abbrev a;
            if (target == kModuleBlock || !readAbbrev(c, a))
                return false;
            break;
        }
        case kUnabbrevRecord: {
            uint64_t code, op, n;
            if (!readRecord(c, nullptr, &code, &op, 1, &n))
                return false;
            if (code == 1 /* SETBID */ && n)
                target = op;
            break;
        }
        default:
            return false;
        }
    }
    return false;
}

enum class LocalFunctions { No, Yes, Unknown };

/* Whether a bitcode module defines a function with internal linkage, read
 * off the module block's own records: every nested block (function bodies,
 * constants, metadata) is stepped over by its length. */
LocalFunctions
scanLocalFunctions(const uint8_t *bc, size_t size)
{
    static const uint8_t kMagic[4] = {'B', 'C', 0xC0, 0xDE};
    if (size < 4 || memcmp(bc, kMagic, 4) != 0)
        return LocalFunctions::Unknown;
    BitCursor c(bc, size);
    c.skip(32);

    unsigned width = 2;
    bool inModule = false;
    std::vector<Abbrev> abbrevs;
    while (!c.bad() && c.left()) {
        const uint64_t id = c.fixed(width);
        if (id == kEnterSubblock) {
            const uint64_t block = c.vbr(8);
            const unsigned inner = (unsigned)c.vbr(4);
            c.align32();
            const uint64_t words = c.fixed(32);
            if (c.bad() || inner < 2 || inner > 32)
                break;
            if (!inModule && block == kModuleBlock) {
                inModule = true;
                width = inner;
            } else if (inModule && block == kBlockInfo) {
                if (!skipBlockInfo(c, inner))
                    break;
            } else if (block == kBlockInfo) {
                break;
            } else {
                c.skip(words * 32);
            }
            continue;
        }
        if (!inModule)
            break;
        if (id == kEndBlock)
            return LocalFunctions::No;
        if (id == kDefineAbbrev) {
            abbrevs.emplace_back();
            if (!readAbbrev(c, abbrevs.back()))
                break;
            continue;
        }
        const Abbrev *abbrev = nullptr;
        if (id != kUnabbrevRecord) {
            if (id - 4 >= abbrevs.size())
                break;
            abbrev = &abbrevs[id - 4];
        }
        /* MODULE_CODE_FUNCTION: [type, callingconv, isproto, linkage, ...] */
        uint64_t code, ops[4], n;
        if (!readRecord(c, abbrev, &code, ops, 4, &n))
            break;
        if (code == 8 && n >= 4 && ops[2] == 0 && ops[3] == 3 /* internal */)
            return LocalFunctions::Yes;
    }
    return LocalFunctions::Unknown;
}

} // namespace

extern "C" int
dmn_dxil_input_semantics(const void *container, size_t size,
                         struct dmn_dxil_semantic *out, int cap)
{
    PFN_SysDxcCreateInstance create = dxcCreateInstance();
    if (!create)
        return 0;

    PinnedBlob blob(container, size);
    SysDxcContainerReflection *refl = nullptr;
    int32_t hr = create(&kClsidDxcContainerReflection,
                        &kIidDxcContainerReflection, (void **)&refl);
    if (hr < 0 || !refl) {
        DMN_WARN("dxil-reflect: DxcCreateInstance failed 0x%08x",
                 (unsigned)hr);
        return 0;
    }

    int n = 0;
    SysShaderReflection *sr = nullptr;
    uint32_t partIdx = 0;
    hr = refl->Load(&blob);
    if (hr < 0) {
        DMN_WARN("dxil-reflect: Load failed 0x%08x", (unsigned)hr);
        goto out;
    }
    hr = refl->FindFirstPartKind(0x4C495844u /* 'DXIL' */, &partIdx);
    if (hr < 0) {
        DMN_WARN("dxil-reflect: no DXIL part (0x%08x)", (unsigned)hr);
        goto out;
    }
    hr = refl->GetPartReflection(partIdx, kIidShaderReflection, (void **)&sr);
    if (hr < 0 || !sr) {
        DMN_WARN("dxil-reflect: GetPartReflection failed 0x%08x",
                 (unsigned)hr);
        goto out;
    }
    {
        D3D12_SHADER_DESC sd = {};
        if (sr->GetDesc(&sd) < 0)
            goto out;
        for (uint32_t i = 0; i < sd.InputParameters && n < cap; i++) {
            D3D12_SIGNATURE_PARAMETER_DESC pd = {};
            if (sr->GetInputParameterDesc(i, &pd) < 0 || !pd.SemanticName)
                continue;
            out[n].reg = pd.Register;
            out[n].index = pd.SemanticIndex;
            snprintf(out[n].name, sizeof(out[n].name), "%s", pd.SemanticName);
            n++;
        }
    }

out:
    if (sr)
        sr->Release();
    refl->Release();
    return n;
}

/* The runtime's in-driver DXBC conversion (dxilconv ConvertInDriver) leaves
 * optimization to the driver: a shader's call/label subroutines arrive as
 * internal functions, and Metal Shader Converter dereferences a null entry
 * point on any module defining more than main (SIGSEGV in
 * DXILMDHelper::parseEntryPoint, taking the worker with it).  Inlining
 * restores the single-function form that DXIL from a full compile has. */
extern "C" int
dmn_dxil_inline_subroutines(const void *container, size_t size, void **out,
                            size_t *out_size)
{
    const uint8_t *c = static_cast<const uint8_t *>(container);
    if (!c || size < 68 || memcmp(c, "DXBC", 4) != 0)
        return 0;
    for (int i = 4; i < 20; i++)
        if (c[i])
            return 0;
    uint32_t total, parts, partOff, fourcc, partSize;
    memcpy(&total, c + 24, 4);
    memcpy(&parts, c + 28, 4);
    memcpy(&partOff, c + 32, 4);
    /* Bounds in a form that cannot wrap: the container is guest data. */
    if (total != size || parts != 1 || partOff < 36 || partOff > size - 32)
        return 0;
    memcpy(&fourcc, c + partOff, 4);
    memcpy(&partSize, c + partOff + 4, 4);
    if (fourcc != 0x4C495844u /* 'DXIL' */ || partSize > size - partOff - 8)
        return 0;
    /* DxilProgramHeader: version, size in dwords, then the bitcode header
     * {'DXIL', DXIL version, bitcode offset, bitcode size}. */
    const uint8_t *prog = c + partOff + 8;
    uint32_t hdr[6];
    memcpy(hdr, prog, sizeof(hdr));
    if (hdr[2] != 0x4C495844u || 8 + (uint64_t)hdr[4] + hdr[5] > partSize)
        return 0;
    const uint8_t *bitcode = prog + 8 + hdr[4];

    /* Every shader of every pipeline create comes through here, so the
     * common answer -- no internal function -- is read straight off the
     * bitcode's module block instead of printing the module. */
    LocalFunctions local = scanLocalFunctions(bitcode, hdr[5]);
    if (local == LocalFunctions::No)
        return 0;

    PFN_SysDxcCreateInstance create = dxcCreateInstance();
    if (!create)
        return 0;
    SysDxcOptimizer *opt = nullptr;
    if (create(&kClsidDxcOptimizer, &kIidDxcOptimizer, (void **)&opt) < 0 ||
        !opt)
        return 0;

    PinnedBlob input(bitcode, hdr[5]);
    SysDxcBlob *mod = nullptr, *text = nullptr;

    /* Only a module that defines more than its entry point: the passes are
     * not neutral on a full compile's output (D3D9On12's own conversion
     * shaders render wrong after them).  A module the scan cannot read is
     * printed and searched instead. */
    if (local == LocalFunctions::Unknown) {
        const wchar_t *print[] = {L"-print-module"};
        const int32_t phr = opt->RunOptimizer(&input, print, 1, &mod, &text);
        if (phr >= 0 && text) {
            const char *t = static_cast<const char *>(text->GetBufferPointer());
            const size_t n = text->GetBufferSize();
            static const char kNeedle[] = "define internal ";
            for (size_t i = 0; t && i + sizeof(kNeedle) - 1 <= n; i++)
                if (memcmp(t + i, kNeedle, sizeof(kNeedle) - 1) == 0) {
                    local = LocalFunctions::Yes;
                    break;
                }
        }
        if (mod)
            mod->Release();
        if (text)
            text->Release();
        mod = text = nullptr;
    }
    if (local != LocalFunctions::Yes) {
        opt->Release();
        return 0;
    }

    /* globalopt + mem2reg turn the globals the subroutines shared registers
     * through (dx.v32.rN, which the converter reads back as zero) into
     * locals of main. */
    const wchar_t *passes[] = {L"-inline", L"-globalopt", L"-mem2reg",
                               L"-globaldce"};
    int32_t hr = opt->RunOptimizer(&input, passes, 4, &mod, &text);
    if (text)
        text->Release();
    opt->Release();
    if (hr < 0 || !mod) {
        DMN_WARN("dxil-normalize: RunOptimizer failed 0x%08x", (unsigned)hr);
        if (mod)
            mod->Release();
        return 0;
    }

    const uint32_t bc = (uint32_t)mod->GetBufferSize();
    const uint32_t newPart = 24 + ((bc + 3) & ~3u);
    const uint32_t newTotal = 36 + 8 + newPart;
    uint8_t *o = static_cast<uint8_t *>(calloc(1, newTotal));
    if (!o) {
        mod->Release();
        return 0;
    }
    const uint32_t one = 1, partAt = 36, dxil = 0x4C495844u;
    memcpy(o, c, 24); /* magic, zero digest, version */
    memcpy(o + 24, &newTotal, 4);
    memcpy(o + 28, &one, 4);
    memcpy(o + 32, &partAt, 4);
    memcpy(o + 36, &dxil, 4);
    memcpy(o + 40, &newPart, 4);
    const uint32_t nh[6] = {hdr[0], newPart / 4, dxil, hdr[3], 16, bc};
    memcpy(o + 44, nh, sizeof(nh));
    memcpy(o + 68, mod->GetBufferPointer(), bc);
    mod->Release();
    *out = o;
    *out_size = newTotal;
    return 1;
}
