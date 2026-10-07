// Fullscreen triangle from SV_VertexID with the TEXCOORD0 the converted
// ps_2_x shader in ps_label_calls.h declares as input.
struct VSOut { float4 pos : SV_Position; float2 tc : TEXCOORD0; };

VSOut VSMain(uint id : SV_VertexID)
{
    VSOut o;
    float2 p = float2(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0);
    o.pos = float4(p, 0.0, 1.0);
    o.tc = p;
    return o;
}
