// Min-precision draw shaders (fxc, vs_5_0/ps_5_0).  When the driver claims
// D3D12DDI_SHADER_MIN_PRECISION_16_BIT the runtime lets these through, and
// tritonBuildDxbc then emits ISG1 (not ISGN) signature chunks -- exercising
// the same reader path DXIL containers need.

struct VSIn  { min16float3 pos : POSITION; min16float3 col : COLOR0; };
struct VSOut { float4 pos : SV_Position; min16float3 col : COLOR0; };

VSOut VSMain(VSIn i)
{
    VSOut o;
    o.pos = float4(i.pos, 1.0);
    o.col = i.col;
    return o;
}

float4 PSMain(VSOut i) : SV_Target0
{
    return float4(i.col, 1.0);
}
