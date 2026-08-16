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

#include <cstdio>
#include <cstring>
#include <dlfcn.h>

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
