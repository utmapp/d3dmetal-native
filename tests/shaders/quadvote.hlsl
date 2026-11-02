// All 16 boolean patterns, one per quad. The 2D variant also checks that
// quadRead partners follow D3D's 2x2 layout rather than linear thread IDs.
RWStructuredBuffer<uint> output : register(u0);
#ifndef TWO_D
#define TWO_D 0
#endif
#if TWO_D
[numthreads(8, 8, 1)]
#else
[numthreads(64, 1, 1)]
#endif
void main(uint3 tid : SV_GroupThreadID, uint index : SV_GroupIndex)
{
#if TWO_D
    uint quad = (tid.y >> 1) * 4 + (tid.x >> 1);
    uint lane = (tid.y & 1) * 2 + (tid.x & 1);
#else
    uint quad = index >> 2;
    uint lane = index & 3;
#endif
    bool condition = ((quad >> lane) & 1) != 0;
    output[index * 4] = uint(QuadAny(condition)) | (uint(QuadAll(condition)) << 1);
    output[index * 4 + 1] = QuadReadAcrossX(index);
    output[index * 4 + 2] = QuadReadAcrossY(index);
    output[index * 4 + 3] = QuadReadAcrossDiagonal(index);
}
