// format_caps_test: does CheckFormatSupport report real per-format
// capabilities rather than D3DMetal's all-bits-or-nothing answer?  D3DMetal
// natively returns 0xffffffff for any format it recognizes and 0 for any it
// does not, so every assertion below that distinguishes two bits within one
// format fails against the unhooked path.  Also checks that CheckFeatureSupport
// agrees with CheckFormatSupport (they are the same query in D3D11) and that
// features we do not own still reach D3DMetal.
#include <d3d11_1.h>
#include <cstdio>

template<class T> struct Com {
    T* p=nullptr; ~Com(){ if(p)p->Release(); }
    T** operator&(){ return &p; } T* ptr(){ return p; } T* operator->(){ return p; }
};

static int g_fail=0;

static void expect(bool ok,const char* what){
    if(!ok){ printf("FAIL %s\n",what); g_fail++; }
}

// Every bit in `want` set, every bit in `deny` clear.
static void expect_bits(ID3D11Device* dev,DXGI_FORMAT f,const char* name,
                        UINT want,UINT deny){
    UINT sup=0;
    HRESULT hr=dev->CheckFormatSupport(f,&sup);
    if(FAILED(hr)){ printf("FAIL %s: CheckFormatSupport hr=0x%08x\n",name,hr); g_fail++; return; }
    if((sup&want)!=want){ printf("FAIL %s: 0x%08x missing 0x%08x\n",name,sup,want&~sup); g_fail++; }
    if(sup&deny){ printf("FAIL %s: 0x%08x has forbidden 0x%08x\n",name,sup,sup&deny); g_fail++; }
}

static void expect_invalid(ID3D11Device* dev,DXGI_FORMAT f,const char* name){
    UINT sup=0xdeadbeef;
    HRESULT hr=dev->CheckFormatSupport(f,&sup);
    if(hr!=E_INVALIDARG){ printf("FAIL %s: expected E_INVALIDARG, got 0x%08x\n",name,hr); g_fail++; }
}

int main(){
    Com<ID3D11Device> dev; Com<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl=D3D_FEATURE_LEVEL_11_0;
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,&fl,1,
              D3D11_SDK_VERSION,&dev,nullptr,&ctx))){fprintf(stderr,"dev fail\n");return 2;}

    // UNKNOWN is legal and means a raw/structured buffer: no texture caps.
    {
        UINT sup=0;
        HRESULT hr=dev->CheckFormatSupport(DXGI_FORMAT_UNKNOWN,&sup);
        expect(hr==S_OK,"UNKNOWN returns S_OK");
        expect(sup==(D3D11_FORMAT_SUPPORT_BUFFER|D3D11_FORMAT_SUPPORT_CPU_LOCKABLE),
               "UNKNOWN == BUFFER|CPU_LOCKABLE");
        if(sup!=(D3D11_FORMAT_SUPPORT_BUFFER|D3D11_FORMAT_SUPPORT_CPU_LOCKABLE))
            printf("     (got 0x%08x)\n",sup);
    }

    // No Metal representation at all.
    expect_invalid(dev.ptr(),DXGI_FORMAT_NV12,"NV12");
    expect_invalid(dev.ptr(),DXGI_FORMAT_P8,"P8");
    expect_invalid(dev.ptr(),DXGI_FORMAT_R1_UNORM,"R1_UNORM");

    // A plain color format: renderable, blendable, displayable, mip-genable.
    expect_bits(dev.ptr(),DXGI_FORMAT_R8G8B8A8_UNORM,"RGBA8_UNORM",
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BLENDABLE|
        D3D11_FORMAT_SUPPORT_DISPLAY|D3D11_FORMAT_SUPPORT_MIP_AUTOGEN|
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE|D3D11_FORMAT_SUPPORT_BUFFER,
        D3D11_FORMAT_SUPPORT_SO_BUFFER|D3D11_FORMAT_SUPPORT_DEPTH_STENCIL);

    // Block compressed: sampleable only. Not a render target, not a buffer,
    // and no mip autogen.
    expect_bits(dev.ptr(),DXGI_FORMAT_BC1_UNORM,"BC1_UNORM",
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE|D3D11_FORMAT_SUPPORT_TEXTURE2D,
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BUFFER|
        D3D11_FORMAT_SUPPORT_MIP_AUTOGEN|D3D11_FORMAT_SUPPORT_BLENDABLE);

    // Depth-stencil: comparison sampling, never a color target or a buffer.
    expect_bits(dev.ptr(),DXGI_FORMAT_D24_UNORM_S8_UINT,"D24S8",
        D3D11_FORMAT_SUPPORT_DEPTH_STENCIL|
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE_COMPARISON,
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BUFFER|
        D3D11_FORMAT_SUPPORT_BLENDABLE);
    expect_bits(dev.ptr(),DXGI_FORMAT_D32_FLOAT,"D32_FLOAT",
        D3D11_FORMAT_SUPPORT_DEPTH_STENCIL,
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BUFFER);

    // A8_UNORM is fully capable on Apple3+; filter-only applies to Apple2
    // alone. a8_test.cpp exercises the render path end to end.
    expect_bits(dev.ptr(),DXGI_FORMAT_A8_UNORM,"A8_UNORM",
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE|D3D11_FORMAT_SUPPORT_RENDER_TARGET|
        D3D11_FORMAT_SUPPORT_BLENDABLE,0);

    // Packed 16-bit formats are resolvable but not shader-writable.
    expect_bits(dev.ptr(),DXGI_FORMAT_B5G6R5_UNORM,"B5G6R5_UNORM",
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BLENDABLE|
        D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE,
        D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW);

    // Integer formats: writable and MSAA-able, never blendable or resolvable.
    expect_bits(dev.ptr(),DXGI_FORMAT_R32G32B32A32_UINT,"RGBA32_UINT",
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|
        D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET|
        D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW,
        D3D11_FORMAT_SUPPORT_BLENDABLE|D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE);

    // 16-bit Unorm/Snorm formats are MSAA-resolvable per Apple's tables.
    expect_bits(dev.ptr(),DXGI_FORMAT_R16_UNORM,"R16_UNORM",
        D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE|
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BLENDABLE,0);
    expect_bits(dev.ptr(),DXGI_FORMAT_R16G16B16A16_UNORM,"RGBA16_UNORM",
        D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE,0);

    // Depth formats are sparse/tiled from Apple7, which surfaces as
    // FORMAT_SUPPORT2_TILED.
    {
        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 f2={DXGI_FORMAT_D32_FLOAT,0};
        dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2,&f2,sizeof(f2));
        expect((f2.OutFormatSupport2&D3D11_FORMAT_SUPPORT2_TILED)!=0,
               "D32_FLOAT is TILED");
    }

    // Stream output takes 32-bit components only.
    expect_bits(dev.ptr(),DXGI_FORMAT_R32G32B32A32_FLOAT,"RGBA32_FLOAT",
        D3D11_FORMAT_SUPPORT_SO_BUFFER|D3D11_FORMAT_SUPPORT_IA_VERTEX_BUFFER,0);

    // Index buffers are R16_UINT / R32_UINT and nothing else.
    expect_bits(dev.ptr(),DXGI_FORMAT_R32_UINT,"R32_UINT",
        D3D11_FORMAT_SUPPORT_IA_INDEX_BUFFER,0);
    expect_bits(dev.ptr(),DXGI_FORMAT_R32_SINT,"R32_SINT",
        0,D3D11_FORMAT_SUPPORT_IA_INDEX_BUFFER);

    // CheckFeatureSupport(FORMAT_SUPPORT) is the same query; it must agree.
    static const DXGI_FORMAT kSpread[]={
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_A8_UNORM,
        DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R32G32B32A32_UINT,
    };
    for(DXGI_FORMAT f : kSpread){
        UINT direct=0;
        dev->CheckFormatSupport(f,&direct);
        D3D11_FEATURE_DATA_FORMAT_SUPPORT fs={f,0};
        HRESULT hr=dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT,&fs,sizeof(fs));
        if(hr!=S_OK||fs.OutFormatSupport!=direct){
            printf("FAIL format %d: CheckFeatureSupport hr=0x%08x 0x%08x != 0x%08x\n",
                   (int)f,hr,fs.OutFormatSupport,direct);
            g_fail++;
        }
    }

    // FORMAT_SUPPORT2 must differentiate formats rather than answering
    // uniformly. R32_UINT carries UAV atomics on every Apple GPU; BC3 carries
    // nothing (it is not even UAV-typed-accessible).
    {
        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 a={DXGI_FORMAT_R32_UINT,0};
        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 b={DXGI_FORMAT_BC3_UNORM,0};
        dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2,&a,sizeof(a));
        dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2,&b,sizeof(b));
        expect((a.OutFormatSupport2&D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_ADD)!=0,
               "R32_UINT has UAV_ATOMIC_ADD");
        expect((b.OutFormatSupport2&D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE)==0,
               "BC3_UNORM lacks UAV_TYPED_STORE");
        printf("FORMAT_SUPPORT2: R32_UINT=0x%08x BC3_UNORM=0x%08x\n",
               a.OutFormatSupport2,b.OutFormatSupport2);
    }

    // A wrong-sized in-out struct is rejected, not read.
    {
        D3D11_FEATURE_DATA_FORMAT_SUPPORT fs={DXGI_FORMAT_R8G8B8A8_UNORM,0};
        expect(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT,&fs,
                                        sizeof(fs)-1)==E_INVALIDARG,
               "FORMAT_SUPPORT rejects wrong size");
    }

    // A feature we do not own must still reach D3DMetal.
    {
        D3D11_FEATURE_DATA_D3D11_OPTIONS o={};
        HRESULT hr=dev->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS,&o,sizeof(o));
        expect(hr==S_OK,"D3D11_OPTIONS forwards to D3DMetal");
    }

    // A null out pointer is an error, not a crash.
    expect(dev->CheckFormatSupport(DXGI_FORMAT_R8G8B8A8_UNORM,nullptr)==E_INVALIDARG,
           "null pFormatSupport rejected");

    printf(g_fail?"format-caps: %d FAILED\n":"format-caps: OK\n",g_fail);
    return g_fail?1:0;
}
