/*
 * DXIL container reflection via the framework's own libdxcompiler.dylib.
 *
 * The guest D3D12 UMD receives BARE DXIL program parts (no ISG1), so it
 * cannot know input-signature semantic names; it sends register-tagged
 * placeholder names ("NPTA<reg>") in the input layout instead.  D3DMetal
 * matches input-layout elements to the vertex shader BY NAME against the
 * module metadata (a mismatch silently produces a no-op PSO), so the
 * CreateGraphicsPipelineState hook resolves the placeholders here: reflect
 * the VS container with IDxcContainerReflection -> ID3D12ShaderReflection
 * and look the real name/index up by register.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <utility>
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
    uint64_t position() const { return pos_; }
    const uint8_t *data() const { return p_; }
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

/* Most DXIL abbreviations fit inline; unusually large records spill. */
struct Abbrev {
    std::array<AbbrevOp, 8> inlineOps{};
    std::vector<AbbrevOp> overflow;
    size_t count = 0;

    void push_back(AbbrevOp op) {
        if (count < inlineOps.size())
            inlineOps[count] = op;
        else
            overflow.push_back(op);
        count++;
    }
    size_t size() const { return count; }
    const AbbrevOp &operator[](size_t i) const {
        return i < inlineOps.size() ? inlineOps[i] : overflow[i - inlineOps.size()];
    }
};

/* Keep the common module and VST abbreviations inline, spilling only when
 * an input contains an unusually large abbreviation table. */
class AbbrevSet {
    static const size_t kInlineCount = 16;
    std::array<Abbrev, kInlineCount> inlineAbbrevs_;
    std::vector<Abbrev> overflow_;
    size_t count_ = 0;

public:
    void push_back(Abbrev &&a) {
        if (count_ < kInlineCount)
            inlineAbbrevs_[count_] = std::move(a);
        else
            overflow_.push_back(std::move(a));
        count_++;
    }
    size_t size() const { return count_; }
    const Abbrev &operator[](size_t i) const {
        return i < kInlineCount ? inlineAbbrevs_[i]
                                : overflow_[i - kInlineCount];
    }
};

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

char
decodeChar6(uint64_t value)
{
    if (value < 26)
        return (char)('a' + value);
    if (value < 52)
        return (char)('A' + value - 26);
    if (value < 62)
        return (char)('0' + value - 52);
    return value == 62 ? '.' : '_';
}

/* Reads one record: its code, and its first `cap` operands into ops. */
bool
readRecord(BitCursor &c, const Abbrev *abbrev, uint64_t *code, uint64_t *ops,
           unsigned cap, uint64_t *nops, uint8_t *symbolFlags = nullptr)
{
    static const char kQuadVote[] = "dx.op.quadVote.i1";
    uint64_t n = 0;
    bool valueSymtabEntry = false;
    bool voteNameMatches = true;
    size_t nameChars = 0;
    auto put = [&](uint64_t v, bool char6 = false) {
        c.step();
        if (n == 0) {
            *code = v;
            valueSymtabEntry = v == 1;
        } else {
            if (n - 1 < cap)
                ops[n - 1] = v;
            if (symbolFlags && valueSymtabEntry && n >= 2) {
                const char ch = char6 ? decodeChar6(v) :
                                v <= 0x7f ? (char)v : '\0';
                const size_t at = nameChars++;
                voteNameMatches &= at < sizeof(kQuadVote) - 1 &&
                                   ch == kQuadVote[at];
            }
        }
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
                if (++i >= abbrev->size() ||
                    (*abbrev)[i].kind == AbbrevOp::Array ||
                    (*abbrev)[i].kind == AbbrevOp::Blob || count > c.left())
                    return false;
                for (uint64_t k = 0; k < count && !c.bad(); k++)
                    put(scalar((*abbrev)[i]), (*abbrev)[i].kind == AbbrevOp::Char6);
            } else if (op.kind == AbbrevOp::Blob) {
                const uint64_t bytes = c.vbr(6);
                c.align32();
                c.skip(bytes <= c.left() / 8 ? bytes * 8 : ~0ull);
                c.align32();
                c.step();
            } else {
                put(scalar(op), op.kind == AbbrevOp::Char6);
            }
        }
    }
    if (symbolFlags && valueSymtabEntry) {
        if (voteNameMatches && nameChars == sizeof(kQuadVote) - 1)
            *symbolFlags |= 1;
    }
    *nops = n ? n - 1 : 0;
    return n && !c.bad();
}

enum { kEndBlock = 0, kEnterSubblock = 1, kDefineAbbrev = 2, kUnabbrevRecord = 3 };
enum { kBlockInfo = 0, kModuleBlock = 8, kValueSymtabBlock = 14 };

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

enum class QuadVoteSymbols { No, Yes, Unknown };

bool
readSymbolBlockInfo(BitCursor &c, unsigned width, AbbrevSet &moduleAbbrevs,
                    AbbrevSet &vstAbbrevs)
{
    uint64_t target = ~0ull;
    while (!c.bad()) {
        switch (c.fixed(width)) {
        case kEndBlock:
            c.align32();
            return !c.bad();
        case kDefineAbbrev: {
            Abbrev a;
            if (!readAbbrev(c, a))
                return false;
            if (target == kModuleBlock)
                moduleAbbrevs.push_back(std::move(a));
            else if (target == kValueSymtabBlock)
                vstAbbrevs.push_back(std::move(a));
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

QuadVoteSymbols
scanValueSymtab(BitCursor &c, unsigned width, const AbbrevSet &inherited,
                uint8_t *symbols)
{
    AbbrevSet local;
    while (!c.bad()) {
        const uint64_t id = c.fixed(width);
        if (c.bad())
            return QuadVoteSymbols::Unknown;
        if (id == kEndBlock) {
            c.align32();
            return c.bad() ? QuadVoteSymbols::Unknown :
                   (*symbols & 1) ? QuadVoteSymbols::Yes : QuadVoteSymbols::No;
        }
        if (id == kDefineAbbrev) {
            Abbrev a;
            if (!readAbbrev(c, a))
                return QuadVoteSymbols::Unknown;
            local.push_back(std::move(a));
            continue;
        }
        const Abbrev *abbrev = nullptr;
        if (id != kUnabbrevRecord) {
            const uint64_t index = id - 4;
            if (index < inherited.size())
                abbrev = &inherited[(size_t)index];
            else if (index - inherited.size() < local.size())
                abbrev = &local[(size_t)(index - inherited.size())];
            else
                return QuadVoteSymbols::Unknown;
        }
        uint64_t code, n;
        uint8_t found = 0;
        if (!readRecord(c, abbrev, &code, nullptr, 0, &n, &found))
            return QuadVoteSymbols::Unknown;
        (void)code;
        (void)n;
        *symbols |= found;
    }
    return QuadVoteSymbols::Unknown;
}

QuadVoteSymbols
scanModuleSymbols(BitCursor &c, unsigned width, AbbrevSet &moduleInherited,
                  AbbrevSet &vstInherited, uint8_t *symbols)
{
    AbbrevSet moduleLocal;
    while (!c.bad()) {
        const uint64_t id = c.fixed(width);
        if (c.bad())
            return QuadVoteSymbols::Unknown;
        if (id == kEndBlock) {
            c.align32();
            return c.bad() ? QuadVoteSymbols::Unknown : QuadVoteSymbols::No;
        }
        if (id == kEnterSubblock) {
            const uint64_t block = c.vbr(8);
            const uint64_t inner = c.vbr(4);
            c.align32();
            const uint64_t words = c.fixed(32);
            if (c.bad() || inner < 2 || inner > 32 || words > c.left() / 32 ||
                words > SIZE_MAX / 4)
                return QuadVoteSymbols::Unknown;
            const uint64_t offset = c.position() / 8;
            const size_t bytes = (size_t)(words * 4);
            if (offset > SIZE_MAX || (size_t)offset > SIZE_MAX - bytes)
                return QuadVoteSymbols::Unknown;
            if (block == kBlockInfo) {
                BitCursor info(c.data() + (size_t)offset, bytes);
                if (!readSymbolBlockInfo(info, (unsigned)inner,
                                         moduleInherited, vstInherited))
                    return QuadVoteSymbols::Unknown;
            }
            if (block == kValueSymtabBlock) {
                BitCursor vst(c.data() + (size_t)offset, bytes);
                QuadVoteSymbols result = scanValueSymtab(
                    vst, (unsigned)inner, vstInherited, symbols);
                if (result != QuadVoteSymbols::No)
                    return result;
            }
            c.skip(words * 32);
            if (c.bad())
                return QuadVoteSymbols::Unknown;
            continue;
        }
        if (id == kDefineAbbrev) {
            Abbrev a;
            if (!readAbbrev(c, a))
                return QuadVoteSymbols::Unknown;
            moduleLocal.push_back(std::move(a));
            continue;
        }
        const Abbrev *abbrev = nullptr;
        if (id != kUnabbrevRecord) {
            const uint64_t index = id - 4;
            if (index < moduleInherited.size())
                abbrev = &moduleInherited[(size_t)index];
            else if (index - moduleInherited.size() < moduleLocal.size())
                abbrev = &moduleLocal[(size_t)(index - moduleInherited.size())];
            else
                return QuadVoteSymbols::Unknown;
        }
        uint64_t code, n;
        if (!readRecord(c, abbrev, &code, nullptr, 0, &n))
            return QuadVoteSymbols::Unknown;
        (void)code;
        (void)n;
    }
    return QuadVoteSymbols::Unknown;
}

QuadVoteSymbols
scanQuadVoteSymbols(const uint8_t *bc, size_t size)
{
    static const uint8_t kMagic[4] = {'B', 'C', 0xC0, 0xDE};
    
    if (!bc || size < 4 || memcmp(bc, kMagic, 4) != 0)
        return QuadVoteSymbols::Unknown;

    BitCursor c(bc, size);
    c.skip(32);
    AbbrevSet moduleAbbrevs, vstAbbrevs;
    while (!c.bad() && c.left()) {
        if (c.fixed(2) != kEnterSubblock)
            return QuadVoteSymbols::Unknown;
        const uint64_t block = c.vbr(8);
        const uint64_t inner = c.vbr(4);
        c.align32();
        const uint64_t words = c.fixed(32);
        if (c.bad() || inner < 2 || inner > 32 || words > c.left() / 32 ||
            words > SIZE_MAX / 4)
            return QuadVoteSymbols::Unknown;
        const uint64_t offset = c.position() / 8;
        const size_t bytes = (size_t)(words * 4);
        if (offset > SIZE_MAX || (size_t)offset > SIZE_MAX - bytes)
            return QuadVoteSymbols::Unknown;
        BitCursor blockCursor(bc + (size_t)offset, bytes);
        if (block == kBlockInfo) {
            if (!readSymbolBlockInfo(blockCursor, (unsigned)inner,
                                     moduleAbbrevs, vstAbbrevs))
                return QuadVoteSymbols::Unknown;
        } else if (block == kModuleBlock) {
            uint8_t symbols = 0;
            QuadVoteSymbols result = scanModuleSymbols(
                blockCursor, (unsigned)inner, moduleAbbrevs, vstAbbrevs,
                &symbols);
            return result;
        }
        c.skip(words * 32);
        if (c.bad())
            return QuadVoteSymbols::Unknown;
    }
    return QuadVoteSymbols::Unknown;
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
/* Existing subroutine normalization, intentionally restricted to the guest's
 * original one-part, zero-digest container shape. */
static int
inlineSubroutines(const void *container, size_t size, void **out,
                  size_t *out_size)
{
    const uint8_t *c = static_cast<const uint8_t *>(container);
    if (!c || !out || !out_size || size < 68 ||
        memcmp(c, "DXBC", 4) != 0)
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
    if (fourcc != 0x4C495844u /* 'DXIL' */ || partSize < 24 ||
        partSize > size - partOff - 8)
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

    const size_t bcSize = mod->GetBufferSize();
    const void *optimized = mod->GetBufferPointer();
    if (!optimized || !bcSize || bcSize > UINT32_MAX - 3) {
        mod->Release();
        return 0;
    }
    const uint32_t bc = (uint32_t)bcSize;
    const uint64_t newPart64 = 24 + ((bcSize + 3) & ~(size_t)3);
    const uint64_t newTotal64 = 36 + 8 + newPart64;
    if (newPart64 > UINT32_MAX || newTotal64 > UINT32_MAX) {
        mod->Release();
        return 0;
    }
    const uint32_t newPart = (uint32_t)newPart64;
    const uint32_t newTotal = (uint32_t)newTotal64;
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
    memcpy(o + 68, optimized, bc);
    mod->Release();
    *out = o;
    *out_size = newTotal;
    return 1;
}
namespace {

const uint32_t kDxilFourcc = 0x4C495844u;

struct DxilProgramView {
    uint32_t header[6];
    const uint8_t *bitcode;
    size_t bitcodeSize;
};

bool
parseDxilProgram(const uint8_t *chunk, uint32_t partSize,
                 DxilProgramView *program)
{
    if (!chunk || !program || partSize < 24)
        return false;
    uint32_t fourcc;
    memcpy(&fourcc, chunk, 4);
    if (fourcc != kDxilFourcc)
        return false;
    memcpy(program->header, chunk + 8, sizeof(program->header));
    const uint32_t *h = program->header;
    if (h[2] != kDxilFourcc || h[4] < 16 || h[4] > partSize - 8 ||
        h[5] > partSize - 8 - h[4])
        return false;
    program->bitcode = chunk + 16 + h[4];
    program->bitcodeSize = h[5];
    return true;
}

bool
quadVoteStageAndVersion(const uint32_t header[6])
{
    const uint32_t kind = header[0] >> 16;
    switch (kind) {
    case 0:  /* pixel */
    case 5:  /* compute */
    case 6:  /* library */
    case 13: /* mesh */
    case 14: /* amplification */
    case 15: /* node */
        break;
    default:
        return false;
    }
    const uint32_t major = (header[0] >> 4) & 0xf;
    const uint32_t minor = header[0] & 0xf;
    return (major > 6 || (major == 6 && minor >= 7)) &&
           header[3] >= 0x107;
}

bool
containsBytes(const char *text, size_t size, const char *needle,
              size_t needleSize)
{
    if (!text || needleSize > size)
        return false;
    for (size_t i = 0; i <= size - needleSize; i++)
        if (memcmp(text + i, needle, needleSize) == 0)
            return true;
    return false;
}

bool
printModuleText(SysDxcOptimizer *opt, const uint8_t *bitcode, size_t size,
                std::string *ir)
{
    PinnedBlob input(bitcode, size);
    SysDxcBlob *mod = nullptr, *text = nullptr;
    const wchar_t *options[] = {L"-print-module"};
    const int32_t hr = opt->RunOptimizer(&input, options, 1, &mod, &text);
    if (mod)
        mod->Release();
    if (hr < 0 || !text) {
        if (text)
            text->Release();
        return false;
    }
    const char *bytes = static_cast<const char *>(text->GetBufferPointer());
    size_t n = text->GetBufferSize();
    while (n && bytes && bytes[n - 1] == '\0')
        n--;
    const bool valid = bytes && n && !memchr(bytes, '\0', n);
    if (valid)
        ir->assign(bytes, n);
    text->Release();
    return valid;
}

bool
printModuleHasQuadVote(const uint8_t *bitcode, size_t size)
{
    PFN_SysDxcCreateInstance create = dxcCreateInstance();
    if (!create)
        return false;
    SysDxcOptimizer *opt = nullptr;
    if (create(&kClsidDxcOptimizer, &kIidDxcOptimizer, (void **)&opt) < 0 ||
        !opt) {
        if (opt)
            opt->Release();
        return false;
    }
    std::string ir;
    const bool printed = printModuleText(opt, bitcode, size, &ir);
    opt->Release();
    static const char kSymbol[] = "@dx.op.quadVote.i1";
    return printed && ir.find(kSymbol) != std::string::npos;
}

} // namespace
namespace {

enum class DeclState { Absent, Present, Invalid };

bool
validDeclarationSuffix(const char *suffix, size_t size)
{
    size_t i = 0;
    while (i < size && (suffix[i] == ' ' || suffix[i] == '\t'))
        i++;
    if (i == size)
        return true;
    if (suffix[i++] != '#')
        return false;
    const size_t digits = i;
    while (i < size && suffix[i] >= '0' && suffix[i] <= '9')
        i++;
    if (i == digits)
        return false;
    while (i < size && (suffix[i] == ' ' || suffix[i] == '\t'))
        i++;
    if (i == size)
        return true;
    static const char kDbg[] = "!dbg !";
    if (size - i < sizeof(kDbg) - 1 ||
        memcmp(suffix + i, kDbg, sizeof(kDbg) - 1) != 0)
        return false;
    i += sizeof(kDbg) - 1;
    const size_t dbgDigits = i;
    while (i < size && suffix[i] >= '0' && suffix[i] <= '9')
        i++;
    return i > dbgDigits && i == size;
}

DeclState
inspectFunctionDecl(const std::string &ir, const char *signature,
                    const char *symbol, size_t *lineStart, size_t *lineSize)
{
    const size_t signatureSize = strlen(signature);
    const size_t symbolSize = strlen(symbol);
    size_t start = 0;
    size_t declarations = 0;
    bool symbolSeen = false;
    size_t foundStart = 0, foundSize = 0;
    while (start < ir.size()) {
        size_t end = ir.find('\n', start);
        if (end == std::string::npos)
            end = ir.size();
        const size_t length = end - start;
        if (containsBytes(ir.data() + start, length, symbol, symbolSize))
            symbolSeen = true;
        if (length >= signatureSize &&
            memcmp(ir.data() + start, signature, signatureSize) == 0 &&
            validDeclarationSuffix(ir.data() + start + signatureSize,
                                   length - signatureSize)) {
            declarations++;
            foundStart = start;
            foundSize = length;
        }
        if (end == ir.size())
            break;
        start = end + 1;
    }
    if (declarations > 1)
        return DeclState::Invalid;
    if (declarations == 1) {
        if (lineStart)
            *lineStart = foundStart;
        if (lineSize)
            *lineSize = foundSize;
        return DeclState::Present;
    }
    return symbolSeen ? DeclState::Invalid : DeclState::Absent;
}

const char kQuadVoteDefinition[] =
    "define internal i1 @dx.op.quadVote.i1(i32 %op, i1 %cond, i8 %kind) "
    "alwaysinline nounwind {\n"
    "entry:\n"
    "  %cond32 = sext i1 %cond to i32\n"
    "  %x = call i32 @dx.op.quadOp.i32(i32 123, i32 %cond32, i8 0)\n"
    "  %y = call i32 @dx.op.quadOp.i32(i32 123, i32 %cond32, i8 1)\n"
    "  %z = call i32 @dx.op.quadOp.i32(i32 123, i32 %cond32, i8 2)\n"
    "  %xy.all = and i32 %x, %y\n"
    "  %xyz.all = and i32 %xy.all, %z\n"
    "  %all = and i32 %xyz.all, %cond32\n"
    "  %xy.any = or i32 %x, %y\n"
    "  %xyz.any = or i32 %xy.any, %z\n"
    "  %any = or i32 %xyz.any, %cond32\n"
    "  %is.all = icmp eq i8 %kind, 1\n"
    "  %result32 = select i1 %is.all, i32 %all, i32 %any\n"
    "  %result = trunc i32 %result32 to i1\n"
    "  ret i1 %result\n"
    "}";

bool
lowerQuadVoteBitcode(const DxilProgramView &program, uint8_t **newChunk,
                     size_t *newChunkSize)
{
    *newChunk = nullptr;
    *newChunkSize = 0;
    PFN_SysDxcCreateInstance create = dxcCreateInstance();
    if (!create)
        return false;
    SysDxcOptimizer *opt = nullptr;
    if (create(&kClsidDxcOptimizer, &kIidDxcOptimizer, (void **)&opt) < 0 ||
        !opt) {
        if (opt)
            opt->Release();
        return false;
    }

    std::string ir;
    if (!printModuleText(opt, program.bitcode, program.bitcodeSize, &ir)) {
        opt->Release();
        return false;
    }
    static const char kVoteDecl[] =
        "declare i1 @dx.op.quadVote.i1(i32, i1, i8)";
    static const char kVoteSymbol[] = "@dx.op.quadVote.i1";
    static const char kOpDecl[] =
        "declare i32 @dx.op.quadOp.i32(i32, i32, i8)";
    static const char kOpSymbol[] = "@dx.op.quadOp.i32";
    size_t voteStart, voteSize;
    if (inspectFunctionDecl(ir, kVoteDecl, kVoteSymbol, &voteStart,
                            &voteSize) != DeclState::Present) {
        opt->Release();
        return false;
    }
    const DeclState opDecl = inspectFunctionDecl(ir, kOpDecl, kOpSymbol,
                                                  nullptr, nullptr);
    if (opDecl == DeclState::Invalid) {
        opt->Release();
        return false;
    }
    ir.replace(voteStart, voteSize, kQuadVoteDefinition);
    if (opDecl == DeclState::Absent) {
        if (!ir.empty() && ir.back() != '\n')
            ir.push_back('\n');
        ir += kOpDecl;
        ir += " nounwind\n";
    }

    PinnedBlob input(ir.data(), ir.size());
    SysDxcBlob *mod = nullptr, *text = nullptr;
    const wchar_t *options[] = {L"-always-inline"};
    const int32_t hr = opt->RunOptimizer(&input, options, 1, &mod, &text);
    if (text)
        text->Release();
    opt->Release();
    if (hr < 0 || !mod) {
        if (mod)
            mod->Release();
        return false;
    }
    const void *optimized = mod->GetBufferPointer();
    const size_t bitcodeSize = mod->GetBufferSize();
    const QuadVoteSymbols remaining =
        scanQuadVoteSymbols(static_cast<const uint8_t *>(optimized),
                            bitcodeSize);
    if (!optimized || remaining != QuadVoteSymbols::No ||
        bitcodeSize > UINT32_MAX - 3) {
        mod->Release();
        return false;
    }

    const uint64_t paddedBitcode = (bitcodeSize + 3) & ~(uint64_t)3;
    const uint64_t partSize64 = 24 + paddedBitcode;
    const uint64_t chunkSize64 = 8 + partSize64;
    if (partSize64 > UINT32_MAX || chunkSize64 > UINT32_MAX ||
        chunkSize64 > SIZE_MAX) {
        mod->Release();
        return false;
    }
    uint8_t *chunk = static_cast<uint8_t *>(
        calloc(1, (size_t)chunkSize64));
    if (!chunk) {
        mod->Release();
        return false;
    }
    const uint32_t partSize = (uint32_t)partSize64;
    memcpy(chunk, &kDxilFourcc, 4);
    memcpy(chunk + 4, &partSize, 4);
    const uint32_t outputHeader[6] = {
        program.header[0], partSize / 4, kDxilFourcc, program.header[3],
        16, (uint32_t)bitcodeSize};
    memcpy(chunk + 8, outputHeader, sizeof(outputHeader));
    memcpy(chunk + 32, optimized, bitcodeSize);
    mod->Release();
    *newChunk = chunk;
    *newChunkSize = (size_t)chunkSize64;
    return true;
}

} // namespace
namespace {

struct ContainerPart {
    const uint8_t *chunk;
    uint32_t offset;
    uint32_t partSize;
    uint32_t newOffset;
    uint32_t fourcc;
    uint8_t *replacement;
    size_t replacementSize;
};

bool
programHasQuadVote(const DxilProgramView &program)
{
    if (!quadVoteStageAndVersion(program.header))
        return false;
    const QuadVoteSymbols symbols =
        scanQuadVoteSymbols(program.bitcode, program.bitcodeSize);
    if (symbols == QuadVoteSymbols::Yes)
        return true;
    if (symbols == QuadVoteSymbols::No)
        return false;
    return printModuleHasQuadVote(program.bitcode, program.bitcodeSize);
}

void
freeParts(ContainerPart *parts, size_t count, ContainerPart **sorted)
{
    if (parts) {
        for (size_t i = 0; i < count; i++)
            free(parts[i].replacement);
    }
    free(sorted);
    free(parts);
}

int
normalizeQuadVoteContainer(const void *container, size_t size, void **out,
                           size_t *outSize)
{
    const uint8_t *c = static_cast<const uint8_t *>(container);
    if (!c || !out || !outSize || size < 36 || memcmp(c, "DXBC", 4) != 0)
        return 0;
    uint32_t total, count;
    memcpy(&total, c + 24, 4);
    memcpy(&count, c + 28, 4);
    /* The fixed DXBC header is 32 bytes; DWORD part offsets follow it. */
    if (total != size || count > (total - 32) / 4)
        return 0;
    const uint64_t tableEnd = 32 + (uint64_t)count * 4;

    /* Common shaders return here without allocating or printing IR. */
    bool affected = false;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t offset;
        memcpy(&offset, c + 32 + (size_t)i * 4, 4);
        if ((offset & 3) || offset < tableEnd || offset > total - 8)
            return 0;
        uint32_t fourcc, partSize;
        memcpy(&fourcc, c + offset, 4);
        memcpy(&partSize, c + offset + 4, 4);
        if (partSize > total - offset - 8)
            return 0;
        if (fourcc != kDxilFourcc)
            continue;
        DxilProgramView program;
        if (!parseDxilProgram(c + offset, partSize, &program))
            return 0;
        if (programHasQuadVote(program))
            affected = true;
    }
    if (!affected || count == 0 ||
        (uint64_t)count > SIZE_MAX / sizeof(ContainerPart) ||
        (uint64_t)count > SIZE_MAX / sizeof(ContainerPart *))
        return 0;

    const size_t partCount = count;
    ContainerPart *parts = static_cast<ContainerPart *>(
        calloc(partCount, sizeof(ContainerPart)));
    ContainerPart **sorted = static_cast<ContainerPart **>(
        malloc(partCount * sizeof(ContainerPart *)));
    if (!parts || !sorted) {
        free(parts);
        free(sorted);
        return 0;
    }
    for (size_t i = 0; i < partCount; i++) {
        uint32_t offset;
        memcpy(&offset, c + 32 + i * 4, 4);
        uint32_t partSize, fourcc;
        memcpy(&fourcc, c + offset, 4);
        memcpy(&partSize, c + offset + 4, 4);
        parts[i].chunk = c + offset;
        parts[i].offset = offset;
        parts[i].partSize = partSize;
        parts[i].fourcc = fourcc;
        sorted[i] = &parts[i];
    }
    std::sort(sorted, sorted + partCount,
              [](const ContainerPart *a, const ContainerPart *b) {
                  return a->offset < b->offset;
              });
    uint64_t previousEnd = tableEnd;
    for (size_t i = 0; i < partCount; i++) {
        const ContainerPart &part = *sorted[i];
        if (part.offset < previousEnd) {
            freeParts(parts, partCount, sorted);
            return 0;
        }
        previousEnd = (uint64_t)part.offset + 8 + part.partSize;
    }

    bool changed = false;
    for (size_t i = 0; i < partCount; i++) {
        ContainerPart &part = parts[i];
        if (part.fourcc != kDxilFourcc)
            continue;
        DxilProgramView program;
        if (!parseDxilProgram(part.chunk, part.partSize, &program)) {
            freeParts(parts, partCount, sorted);
            return 0;
        }
        if (!programHasQuadVote(program))
            continue;
        if (!lowerQuadVoteBitcode(program, &part.replacement,
                                  &part.replacementSize)) {
            freeParts(parts, partCount, sorted);
            return 0;
        }
        changed = true;
    }
    if (!changed) {
        freeParts(parts, partCount, sorted);
        return 0;
    }

    uint64_t outputSize = tableEnd;
    for (size_t i = 0; i < partCount; i++) {
        if (outputSize > UINT32_MAX - 3) {
            freeParts(parts, partCount, sorted);
            return 0;
        }
        outputSize = (outputSize + 3) & ~3ull;
        const uint64_t chunkSize = parts[i].replacement
                                       ? parts[i].replacementSize
                                       : 8ull + parts[i].partSize;
        if (outputSize > UINT32_MAX ||
            chunkSize > UINT32_MAX - outputSize) {
            freeParts(parts, partCount, sorted);
            return 0;
        }
        parts[i].newOffset = (uint32_t)outputSize;
        outputSize += chunkSize;
    }
    if (outputSize > UINT32_MAX - 3) {
        freeParts(parts, partCount, sorted);
        return 0;
    }
    outputSize = (outputSize + 3) & ~3ull;
    if (outputSize > SIZE_MAX || outputSize > UINT32_MAX) {
        freeParts(parts, partCount, sorted);
        return 0;
    }
    uint8_t *normalized =
        static_cast<uint8_t *>(calloc(1, (size_t)outputSize));
    if (!normalized) {
        freeParts(parts, partCount, sorted);
        return 0;
    }
    memcpy(normalized, c, 24);
    memset(normalized + 4, 0, 16);
    const uint32_t newTotal = (uint32_t)outputSize;
    memcpy(normalized + 24, &newTotal, 4);
    memcpy(normalized + 28, &count, 4);
    for (size_t i = 0; i < partCount; i++) {
        const ContainerPart &part = parts[i];
        memcpy(normalized + 32 + i * 4, &part.newOffset, 4);
        const void *chunk = part.replacement
                                ? part.replacement
                                : static_cast<const void *>(part.chunk);
        const size_t chunkSize = part.replacement
                                     ? part.replacementSize
                                     : (size_t)part.partSize + 8;
        memcpy(normalized + part.newOffset, chunk, chunkSize);
    }
    freeParts(parts, partCount, sorted);
    *out = normalized;
    *outSize = (size_t)outputSize;
    return 1;
}

} // namespace

extern "C" int
dmn_dxil_normalize(const void *container, size_t size, void **out,
                   size_t *out_size)
{
    if (!out || !out_size)
        return 0;
    *out = nullptr;
    *out_size = 0;

    void *inlined = nullptr;
    size_t inlinedSize = 0;
    const bool didInline =
        inlineSubroutines(container, size, &inlined, &inlinedSize) != 0;
    const void *working = didInline ? inlined : container;
    const size_t workingSize = didInline ? inlinedSize : size;

    void *quadVote = nullptr;
    size_t quadVoteSize = 0;
    if (normalizeQuadVoteContainer(working, workingSize, &quadVote,
                                  &quadVoteSize)) {
        free(inlined);
        *out = quadVote;
        *out_size = quadVoteSize;
        return 1;
    }
    if (didInline) {
        *out = inlined;
        *out_size = inlinedSize;
        return 1;
    }
    return 0;
}
