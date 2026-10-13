// format_caps_test: does CheckFormatSupport report real per-format
// capabilities rather than D3DMetal's all-bits-or-nothing answer?  D3DMetal
// natively returns 0xffffffff for any format it recognizes and 0 for any it
// does not, so every assertion below that distinguishes two bits within one
// format fails against the unhooked path.  Also checks that CheckFeatureSupport
// agrees with CheckFormatSupport (they are the same query in D3D11) and that
// features we do not own still reach D3DMetal.
#include <d3d11_2.h>
#include <d3d12.h>
#include "dmn_d3d12_up.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <initializer_list>

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

// MSDN: CheckFormatSupport returns E_FAIL for a format the device does
// not support.
static void expect_unsupported(ID3D11Device* dev,DXGI_FORMAT f,const char* name){
    UINT sup=0xdeadbeef;
    HRESULT hr=dev->CheckFormatSupport(f,&sup);
    if(hr!=E_FAIL||sup){ printf("FAIL %s: expected E_FAIL and no bits, got 0x%08x 0x%08x\n",name,hr,sup); g_fail++; }
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

    // CheckMultisampleQualityLevels: D3DMetal's answer, less the combinations
    // Metal cannot make.  An unsupported pair may answer a failure or zero
    // levels; both are documented.
    {
        struct { DXGI_FORMAT f; UINT n; bool want; const char* name; } cases[] = {
            { DXGI_FORMAT_R8G8B8A8_UNORM, 4, true,  "msaa R8G8B8A8_UNORM x4" },
            { DXGI_FORMAT_R8G8B8A8_UNORM, 3, false, "msaa R8G8B8A8_UNORM x3" },
            { DXGI_FORMAT_R8G8B8A8_UNORM, 5, false, "msaa R8G8B8A8_UNORM x5" },
            { DXGI_FORMAT_BC1_UNORM,      4, false, "msaa BC1_UNORM x4" },
            { DXGI_FORMAT_R1_UNORM,       4, false, "msaa R1_UNORM x4" },
            { DXGI_FORMAT_YUY2,           4, false, "msaa YUY2 x4" },
            { DXGI_FORMAT_D32_FLOAT,      4, true,  "msaa D32_FLOAT x4" },
            { DXGI_FORMAT_R24G8_TYPELESS, 4, true,  "msaa R24G8_TYPELESS x4" },
        };
        for (auto& c : cases) {
            UINT levels = 0xdeadbeef;
            HRESULT hr = dev->CheckMultisampleQualityLevels(c.f, c.n, &levels);
            bool ok = c.want ? (SUCCEEDED(hr) && levels > 0)
                             : (FAILED(hr) || levels == 0);
            expect(ok, c.name);
            if (!ok)
                printf("     (hr=0x%08x levels=%u)\n", (unsigned)hr, levels);
        }
    }

    // ID3D11Device2::CheckMultisampleQualityLevels1 gets the same filter.
    // D3DMetal <= 3.0 does not implement it (E_NOTIMPL), which stands.
    {
        Com<ID3D11Device2> d2;
        if(SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device2),(void**)&d2))){
            UINT levels=0xdeadbeef;
            HRESULT hr=d2->CheckMultisampleQualityLevels1(DXGI_FORMAT_BC1_UNORM,4,0,&levels);
            expect(FAILED(hr)||levels==0,"msaa1 BC1_UNORM x4");
            hr=d2->CheckMultisampleQualityLevels1(DXGI_FORMAT_R8G8B8A8_UNORM,4,0,&levels);
            expect(hr==E_NOTIMPL||(SUCCEEDED(hr)&&levels>0),"msaa1 R8G8B8A8_UNORM x4");
        }
    }

    // The typeless parent of a depth family is not itself a depth-stencil
    // format; only its D* view is (DXGI format support tables, FL 11.0).
    expect_bits(dev.ptr(),DXGI_FORMAT_R32_TYPELESS,"R32_TYPELESS",
        0,D3D11_FORMAT_SUPPORT_DEPTH_STENCIL);
    expect_bits(dev.ptr(),DXGI_FORMAT_R24G8_TYPELESS,"R24G8_TYPELESS",
        0,D3D11_FORMAT_SUPPORT_DEPTH_STENCIL);

    // No Metal representation at all.
    expect_unsupported(dev.ptr(),DXGI_FORMAT_NV12,"NV12");
    expect_unsupported(dev.ptr(),DXGI_FORMAT_P8,"P8");
    expect_unsupported(dev.ptr(),DXGI_FORMAT_R1_UNORM,"R1_UNORM");

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

    // Depth-stencil: a target, never a color target or a buffer.  Comparison
    // sampling belongs to the single-plane view (R24_UNORM_X8_TYPELESS), not
    // the D* format, which is not shader-readable at all.
    expect_bits(dev.ptr(),DXGI_FORMAT_D24_UNORM_S8_UINT,"D24S8",
        D3D11_FORMAT_SUPPORT_DEPTH_STENCIL,
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_BUFFER|
        D3D11_FORMAT_SUPPORT_BLENDABLE|D3D11_FORMAT_SUPPORT_SHADER_LOAD|
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE_COMPARISON|D3D11_FORMAT_SUPPORT_TEXTURE3D);
    expect_bits(dev.ptr(),DXGI_FORMAT_R24_UNORM_X8_TYPELESS,"R24_UNORM_X8_TYPELESS",
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE_COMPARISON,
        D3D11_FORMAT_SUPPORT_DEPTH_STENCIL);

    // Capabilities MSDN's DXGI format support tables disallow outright.
    expect_bits(dev.ptr(),DXGI_FORMAT_R8G8B8A8_TYPELESS,"R8G8B8A8_TYPELESS",
        D3D11_FORMAT_SUPPORT_TEXTURE2D,
        D3D11_FORMAT_SUPPORT_RENDER_TARGET|D3D11_FORMAT_SUPPORT_SHADER_SAMPLE|
        D3D11_FORMAT_SUPPORT_SHADER_LOAD|D3D11_FORMAT_SUPPORT_BUFFER);
    expect_bits(dev.ptr(),DXGI_FORMAT_R32_UINT,"R32_UINT (no filtering)",
        D3D11_FORMAT_SUPPORT_SHADER_LOAD,
        D3D11_FORMAT_SUPPORT_SHADER_SAMPLE|D3D11_FORMAT_SUPPORT_MIP_AUTOGEN);
    expect_bits(dev.ptr(),DXGI_FORMAT_BC1_UNORM,"BC1_UNORM (no 1D, no MSAA load)",
        D3D11_FORMAT_SUPPORT_TEXTURE2D,
        D3D11_FORMAT_SUPPORT_TEXTURE1D|D3D11_FORMAT_SUPPORT_MULTISAMPLE_LOAD);
    {
        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 f={DXGI_FORMAT_R32_FLOAT,0};
        dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2,&f,sizeof(f));
        expect(!(f.OutFormatSupport2&D3D11_FORMAT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP),
               "R32_FLOAT has no logic op");
        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 g={DXGI_FORMAT_R32G32_UINT,0};
        dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2,&g,sizeof(g));
        expect(!(g.OutFormatSupport2&D3D11_FORMAT_SUPPORT2_UAV_ATOMIC_ADD),
               "R32G32_UINT has no typed UAV atomics");
        D3D11_FEATURE_DATA_FORMAT_SUPPORT2 n={DXGI_FORMAT_NV12,0xdeadbeef};
        HRESULT hr=dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2,&n,sizeof(n));
        expect(hr==E_FAIL&&!n.OutFormatSupport2,"NV12 FORMAT_SUPPORT2 is E_FAIL");
    }
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

    // D3D12 FORMAT_SUPPORT is D3DMetal's answer less what the D3D11 table
    // says Metal lacks, so it never claims more than CheckFormatSupport, and
    // never a D3D11-only bit (MIP_AUTOGEN, CPU_LOCKABLE are not in
    // D3D12_FORMAT_SUPPORT1).  D3D9On12 sizes its D3D9 format list from it.
    {
        Com<ID3D12Device> d12;
        if(FAILED(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_11_0,
                                    __uuidof(ID3D12Device),(void**)&d12))){
            printf("FAIL D3D12CreateDevice\n"); g_fail++;
        } else {
            const UINT kD3D11Only=D3D11_FORMAT_SUPPORT_MIP_AUTOGEN|
                                  D3D11_FORMAT_SUPPORT_CPU_LOCKABLE;
            for(DXGI_FORMAT f : kSpread){
                UINT direct=0;
                dev->CheckFormatSupport(f,&direct);
                D3D12_FEATURE_DATA_FORMAT_SUPPORT fs={f,D3D12_FORMAT_SUPPORT1_NONE,
                                                     D3D12_FORMAT_SUPPORT2_NONE};
                HRESULT hr=d12->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,&fs,sizeof(fs));
                UINT s1=(UINT)fs.Support1;
                if(hr!=S_OK||!(s1&D3D12_FORMAT_SUPPORT1_TEXTURE2D)||(s1&~direct)||
                   (s1&kD3D11Only)){
                    printf("FAIL D3D12 format %d: FORMAT_SUPPORT hr=0x%08x 0x%08x vs D3D11 0x%08x\n",
                           (int)f,hr,s1,direct);
                    g_fail++;
                }
            }
            auto s12=[&](DXGI_FORMAT f,UINT* s1,UINT* s2)->HRESULT{
                D3D12_FEATURE_DATA_FORMAT_SUPPORT fs={f,D3D12_FORMAT_SUPPORT1_NONE,
                                                     D3D12_FORMAT_SUPPORT2_NONE};
                HRESULT hr=d12->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,&fs,sizeof(fs));
                *s1=(UINT)fs.Support1; if(s2)*s2=(UINT)fs.Support2;
                return hr;
            };
            UINT s1=0,s2=0;
            s12(DXGI_FORMAT_R8G8B8A8_UNORM,&s1,nullptr);
            expect((s1&(D3D12_FORMAT_SUPPORT1_RENDER_TARGET|D3D12_FORMAT_SUPPORT1_BLENDABLE))==
                   (D3D12_FORMAT_SUPPORT1_RENDER_TARGET|D3D12_FORMAT_SUPPORT1_BLENDABLE),
                   "D3D12 R8G8B8A8_UNORM renders and blends");
            expect(!(s1&D3D12_FORMAT_SUPPORT1_IA_INDEX_BUFFER),
                   "D3D12 R8G8B8A8_UNORM is not an index format");
            s12(DXGI_FORMAT_R32_UINT,&s1,nullptr);
            expect(s1&D3D12_FORMAT_SUPPORT1_IA_INDEX_BUFFER,"D3D12 R32_UINT is an index format");
            s12(DXGI_FORMAT_BC1_UNORM,&s1,nullptr);
            expect(!(s1&D3D12_FORMAT_SUPPORT1_RENDER_TARGET),"D3D12 BC1_UNORM is not a render target");
            s12(DXGI_FORMAT_R32_TYPELESS,&s1,nullptr);
            expect(!(s1&D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL),"D3D12 R32_TYPELESS is not depth-stencil");
            s12(DXGI_FORMAT_R32_UINT,&s1,&s2);
            expect(!(s2&D3D11_FORMAT_SUPPORT2_SHAREABLE),"D3D12 Support2 has no D3D11-only SHAREABLE");

            // A format D3DMetal cannot make is unsupported: E_FAIL, no bits.
            for(DXGI_FORMAT f : {DXGI_FORMAT_R8G8_B8G8_UNORM,DXGI_FORMAT_YUY2,DXGI_FORMAT_NV12}){
                HRESULT hr=s12(f,&s1,&s2);
                if(!(FAILED(hr)&&s1==0&&s2==0)){
                    printf("FAIL D3D12 format %d: expected unsupported, hr=0x%08x 0x%08x/0x%08x\n",
                           (int)f,hr,s1,s2);
                    g_fail++;
                }
            }

            // D3DMetal claims quality levels for BC and video formats here.
            struct { DXGI_FORMAT f; UINT n; bool want; const char* name; } mcases[] = {
                { DXGI_FORMAT_R8G8B8A8_UNORM, 4, true,  "D3D12 msaa R8G8B8A8_UNORM x4" },
                { DXGI_FORMAT_BC1_UNORM,      4, false, "D3D12 msaa BC1_UNORM x4" },
                { DXGI_FORMAT_YUY2,           4, false, "D3D12 msaa YUY2 x4" },
            };
            for(auto& c : mcases){
                D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS q={c.f,c.n,
                    D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,0xdead};
                HRESULT hr=d12->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,&q,sizeof(q));
                bool ok=c.want?(SUCCEEDED(hr)&&q.NumQualityLevels>0)
                              :(FAILED(hr)||q.NumQualityLevels==0);
                expect(ok,c.name);
                if(!ok) printf("     (hr=0x%08x levels=%u)\n",(unsigned)hr,q.NumQualityLevels);
            }

            // A texture in such a format is refused, not handed to Metal
            // (which aborts the process on it): GetResourceAllocationInfo
            // reports an invalid description, the creates E_INVALIDARG.
            D3D12_RESOURCE_DESC rd={};
            rd.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width=64; rd.Height=64; rd.DepthOrArraySize=1; rd.MipLevels=1;
            rd.Format=DXGI_FORMAT_YUY2; rd.SampleDesc.Count=1;
            D3D12_RESOURCE_ALLOCATION_INFO ai=d12->GetResourceAllocationInfo(0,1,&rd);
            expect(ai.SizeInBytes==UINT64_MAX,"D3D12 YUY2 allocation info is invalid");
            D3D12_HEAP_PROPERTIES hp={}; hp.Type=D3D12_HEAP_TYPE_DEFAULT;
            ID3D12Resource* res=nullptr;
            HRESULT hr=d12->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,
                D3D12_RESOURCE_STATE_COMMON,nullptr,__uuidof(ID3D12Resource),(void**)&res);
            expect(hr==E_INVALIDARG&&!res,"D3D12 YUY2 committed create is E_INVALIDARG");
            if(res) res->Release();

            // Every CreateReservedResource variant refuses it too.
            {
                D3D12_RESOURCE_DESC rr=rd;
                rr.Layout=D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
                res=nullptr;
                hr=d12->CreateReservedResource(&rr,D3D12_RESOURCE_STATE_COMMON,nullptr,
                    __uuidof(ID3D12Resource),(void**)&res);
                expect(hr==E_INVALIDARG&&!res,"D3D12 YUY2 reserved create is E_INVALIDARG");
                if(res) res->Release();
                Com<ID3D12Device4> r4;
                if(SUCCEEDED(d12->QueryInterface(__uuidof(ID3D12Device4),(void**)&r4))){
                    res=nullptr;
                    hr=r4->CreateReservedResource1(&rr,D3D12_RESOURCE_STATE_COMMON,nullptr,
                        nullptr,__uuidof(ID3D12Resource),(void**)&res);
                    expect(hr==E_INVALIDARG&&!res,"D3D12 YUY2 reserved create 1 is E_INVALIDARG");
                    if(res) res->Release();
                }
                Com<ID3D12Device10> r10;
                if(SUCCEEDED(d12->QueryInterface(__uuidof(ID3D12Device10),(void**)&r10))){
                    res=nullptr;
                    hr=r10->CreateReservedResource2(&rr,0 /* D3D12_BARRIER_LAYOUT_COMMON */,
                        nullptr,nullptr,0,nullptr,__uuidof(ID3D12Resource),(void**)&res);
                    expect(hr==E_INVALIDARG&&!res,"D3D12 YUY2 reserved create 2 is E_INVALIDARG");
                    if(res) res->Release();
                } else {
                    expect(false,"ID3D12Device10 exposed");
                }
            }

            // The newer allocation-info entry points get the same refusal,
            // and still answer a supported description.
            D3D12_RESOURCE_DESC1 rd1={};
            memcpy(&rd1,&rd,sizeof(rd));
            D3D12_RESOURCE_DESC ok=rd; ok.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
            D3D12_RESOURCE_DESC1 ok1={}; memcpy(&ok1,&ok,sizeof(ok));
            Com<ID3D12Device4> d4;
            if(SUCCEEDED(d12->QueryInterface(__uuidof(ID3D12Device4),(void**)&d4))){
                d4->GetResourceAllocationInfo1(&ai,0,1,&rd,nullptr);
                expect(ai.SizeInBytes==UINT64_MAX,"D3D12 YUY2 allocation info 1 is invalid");
                d4->GetResourceAllocationInfo1(&ai,0,1,&ok,nullptr);
                expect(ai.SizeInBytes&&ai.SizeInBytes!=UINT64_MAX,"D3D12 RGBA8 allocation info 1");
            }
            Com<ID3D12Device8> d8;
            if(SUCCEEDED(d12->QueryInterface(__uuidof(ID3D12Device8),(void**)&d8))){
                d8->GetResourceAllocationInfo2(&ai,0,1,&rd1,nullptr);
                expect(ai.SizeInBytes==UINT64_MAX,"D3D12 YUY2 allocation info 2 is invalid");
                d8->GetResourceAllocationInfo2(&ai,0,1,&ok1,nullptr);
                expect(ai.SizeInBytes&&ai.SizeInBytes!=UINT64_MAX,"D3D12 RGBA8 allocation info 2");
            }
            Com<ID3D12Device12> dv12;
            if(SUCCEEDED(d12->QueryInterface(__uuidof(ID3D12Device12),(void**)&dv12))){
                UINT32 nc=0; const DXGI_FORMAT* cf=nullptr;
                dv12->GetResourceAllocationInfo3(&ai,0,1,&rd1,&nc,&cf,nullptr);
                expect(ai.SizeInBytes==UINT64_MAX,"D3D12 YUY2 allocation info 3 is invalid");
            }
        }
    }

    printf(g_fail?"format-caps: %d FAILED\n":"format-caps: OK\n",g_fail);
    return g_fail?1:0;
}
