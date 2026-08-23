// SV_Barycentrics draw (SM 6.1).  Fullscreen triangle from SV_VertexID; the
// pixel shader writes the barycentric weights as colour, so the centre of
// the render target -- (0,0) in NDC, weights (0.5, 0.25, 0.25) for the
// (-1,-1),(3,-1),(-1,3) triangle -- must read back as (128, 64, 64).
struct VSOut { float4 pos : SV_Position; };

VSOut VSMain(uint id : SV_VertexID)
{
    VSOut o;
    float2 p = float2(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0);
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

float4 PSMain(VSOut i, float3 b : SV_Barycentrics) : SV_Target0
{
    return float4(b, 1.0);
}
