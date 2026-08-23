// 64-bit integer atomics (SM 6.6).  One group of 64 threads:
//   typed      RWBuffer<uint64_t> (R32G32_UINT view), InterlockedAdd/Max
//   groupshared uint64_t, InterlockedAdd
//   raw        RWByteAddressBuffer, InterlockedAdd64
// Every accumulation crosses 2^32 so a 32-bit fallback cannot fake it.
RWBuffer<uint64_t>        T : register(u0);   // [0]=add target, [1]=max target
RWStructuredBuffer<uint64_t> G : register(u1);   // [0]=groupshared result
RWByteAddressBuffer       B : register(u2);   // [0..7]=raw add target

groupshared uint64_t gsum;

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0) gsum = 0;
    GroupMemoryBarrierWithGroupSync();

    uint64_t one = 0x100000001ull;
    InterlockedAdd(T[0], one);
    InterlockedMax(T[1], ((uint64_t)(id.x + 1)) << 33);
    InterlockedAdd(gsum, one);
    B.InterlockedAdd64(0, one);

    GroupMemoryBarrierWithGroupSync();
    if (id.x == 0) G[0] = gsum;
}
