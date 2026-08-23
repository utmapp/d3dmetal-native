// Native 16-bit ops (SM 6.2, -enable-16bit-types).  Each thread does
// float16_t arithmetic that only rounds this way at half precision:
// (1.0h + 1.0h/1024.0h) == 1.0h exactly at 11-bit mantissa... use the
// classic probe instead: half(2049) rounds to 2048, float keeps 2049.
// out[i] = asuint16(float16_t(2049.0 + i)) as uint, plus a real half
// multiply whose rounding differs from fp32: h(1.001h * 3.0h).
RWStructuredBuffer<uint> Out : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    float16_t a = (float16_t)(2049.0 + (float)id.x);   // 2049..2112 -> even numbers only at half
    float16_t b = (float16_t)1.001 * (float16_t)3.0;    // rounds at half
    Out[id.x * 2 + 0] = (uint)asuint16(a);
    Out[id.x * 2 + 1] = (uint)asuint16(b);
}
