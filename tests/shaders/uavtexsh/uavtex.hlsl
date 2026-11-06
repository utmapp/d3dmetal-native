// sparse-uav-test kernels: fill a 2D R32_UINT UAV with a per-texel value,
// by a plain store (STORE) or by an atomic max onto a zeroed texel (ATOMIC).
// Compiled with dxc -T cs_6_0 -Fh (-D ATOMIC for the atomic variant).
RWTexture2D<uint> tex : register(u0);
[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
    uint v = (id.x << 16) | id.y | 0x80000000u;
#ifdef ATOMIC
    uint old;
    InterlockedMax(tex[id.xy], v, old);
#else
    tex[id.xy] = v;
#endif
}
