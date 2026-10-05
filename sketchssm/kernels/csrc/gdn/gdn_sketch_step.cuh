// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
// SketchSSM decode kernels (https://arxiv.org/abs/2609.33051).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#if SKETCH_BF16
using sketch_t = __nv_bfloat16;
#else
using sketch_t = float;
#endif

#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

// SketchSSM window W (the build passes it): any multiple of 16
#ifndef WMAX
#define WMAX 16
#endif
static_assert(WMAX >= 16 && WMAX % 16 == 0, "window must be a multiple of 16");
#ifndef NF_ABLATE
#define NF_ABLATE 0
#endif
#ifndef NF_SMEM_PAD
#define NF_SMEM_PAD 0
#endif
#define FULL 0xffffffffu

template <typename T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }

template <int CODE> __device__ __forceinline__ float ldx(const void* p, long i) {
    if (CODE == 0) return ((const float*)p)[i];
    if (CODE == 1) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    return __half2float(((const __half*)p)[i]);
}
__device__ __forceinline__ float round_to(float x, int code) {
    if (code == 1) return __bfloat162float(__float2bfloat16(x));
    if (code == 2) return __half2float(__float2half(x));
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}

template <typename T> __device__ __forceinline__ float4 ld4(const T* p);
template <> __device__ __forceinline__ float4 ld4<float>(const float* p) { return *((const float4*)p); }
template <> __device__ __forceinline__ float4 ld4<__nv_bfloat16>(const __nv_bfloat16* p) {
    const uint2 u = *((const uint2*)p);
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
template <typename T> __device__ __forceinline__ void st4(T* p, float4 v);
template <> __device__ __forceinline__ void st4<float>(float* p, float4 v) { *((float4*)p) = v; }
template <> __device__ __forceinline__ void st4<__nv_bfloat16>(__nv_bfloat16* p, float4 v) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    uint2 u; u.x = *reinterpret_cast<unsigned*>(&a); u.y = *reinterpret_cast<unsigned*>(&b);
    *((uint2*)p) = u;
}
// four sketch elements (bf16 or f32) -> float4, from global or shared
__device__ __forceinline__ float4 ld_sk4(const sketch_t* p) {
#if SKETCH_BF16
    return ld4<__nv_bfloat16>(p);
#else
    return *((const float4*)p);
#endif
}
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
#if NF_FFMA2
    unsigned long long x = *reinterpret_cast<unsigned long long*>(&a), y = *reinterpret_cast<unsigned long long*>(&b), z = *reinterpret_cast<unsigned long long*>(&c);
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(z) : "l"(x), "l"(y), "l"(z));
    return *reinterpret_cast<float2*>(&z);
#else
    return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}
__device__ __forceinline__ float bf_lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_u32(smem)), "l"(gmem) : "memory");
}
// 16-byte chunk with source size 16 or 0 (zero fill): unconditional instruction, no divergence
__device__ __forceinline__ void cp_async16_z(void* smem, const void* gmem, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(smem)), "l"(gmem), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void pf_l2_if(const void* p, bool pred) {
    asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" :: "l"(p), "r"((int)pred));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(smem_u32(smem)), "l"(gmem) : "memory");
}
// contiguous global bytes -> shared: chunk c = lane + 32 j (warp-coalesced 512-byte spans), predicated per chunk;
// the global side uses two lane bases (chunks 0..3 and 4..7) so every offset is a small immediate
template <int NCH> __device__ __forceinline__ void stage16(unsigned char* sm, const unsigned char* gm, int bytes, int lane) {
    const int n = (bytes + 15) >> 4;
    unsigned char* s0 = sm + 16 * lane;
    const unsigned char* g0 = gm + 16 * lane;
    const unsigned char* g1 = g0 + 4 * 512;
    #pragma unroll
    for (int j = 0; j < NCH; ++j) {
        if (lane + 32 * j < n) cp_async16(s0 + 512 * j, (j < 4) ? g0 + 512 * j : g1 + 512 * (j - 4));
    }
}
// same as stage16 but chunks beyond `bytes` are written as zeros (src-size 0)
template <int NCH> __device__ __forceinline__ void stage16z(unsigned char* sm, const unsigned char* gm, int bytes, int lane) {
    const int n = (bytes + 15) >> 4;
    unsigned char* s0 = sm + 16 * lane;
    const unsigned char* g0 = gm + 16 * lane;
    const unsigned char* g1 = g0 + 4 * 512;
    #pragma unroll
    for (int j = 0; j < NCH; ++j) cp_async16_z(s0 + 512 * j, (j < 4) ? g0 + 512 * j : g1 + 512 * (j - 4), lane + 32 * j < n);
}
template <int NCH> __device__ __forceinline__ void stage8(unsigned char* sm, const unsigned char* gm, int bytes, int lane) {
    const int n = (bytes + 7) >> 3;
    unsigned char* s0 = sm + 8 * lane;
    const unsigned char* g0 = gm + 8 * lane;
    #pragma unroll
    for (int j = 0; j < NCH; ++j) {
        if (lane + 32 * j < n) cp_async8(s0 + 256 * j, g0 + 256 * j);
    }
}

template <int N> __device__ __forceinline__ float xposeN(float (&v)[N], int lane) {
    #pragma unroll
    for (int off = N / 2; off >= 1; off >>= 1) {
        const bool up = (lane & off) != 0;
        #pragma unroll
        for (int i = 0; i < off; ++i) {
            const float send = up ? v[i] : v[i + off];
            const float keep = up ? v[i + off] : v[i];
            v[i] = keep + __shfl_xor_sync(FULL, send, off);
        }
    }
    float r = v[0];
    #pragma unroll
    for (int o = N; o < 32; o <<= 1) r += __shfl_xor_sync(FULL, r, o);
    return r;
}

template <int N> struct Pow2 { static constexpr int v = (N <= 1) ? 1 : (N <= 2) ? 2 : (N <= 4) ? 4 : (N <= 8) ? 8 : (N <= 16) ? 16 : 32; };

template <int K, int V, int HPG> struct Smem {
    static constexpr int NR = 8;
    // per-warp staging (bytes): d-ring rows 0..RS-1 (bf16) | first NF_UROWS sketch rows | 4 pivot rows | erase history.
    // RS = 15 ring rows are staged (all of them at W = 16); rows RS..W-2 stream from global in 16-row tiles.
    static constexpr int RS = 15;
    static constexpr int RING_B = RS * V * 2;
    static constexpr int U_B = NF_UROWS * V * (int)sizeof(sketch_t);
    static constexpr int AG_FG = NF_AG_FG;                                       // coefficient block: 4 pivot rows + 5 rows of FG (FG <= AG_FG)
    static constexpr int PV_B = (4 * K + 5 * AG_FG) * (int)sizeof(sketch_t);
    static constexpr int FS_FG = NF_FS_FG, FS_B = RS * FS_FG * (int)sizeof(sketch_t);   // erase rows 0..RS-1 at pitch FG (FG <= FS_FG)
    static_assert(4 * K * (int)sizeof(sketch_t) >= K * (int)sizeof(float2), "pivot rows hold the (q, k) pairs");
    static constexpr int PV_OFF = RING_B + U_B;
    static constexpr int FS_OFF = PV_OFF + PV_B;
    static constexpr int WBYTES = FS_OFF + FS_B;
    static constexpr int BYTES = HPG * WBYTES + NF_SMEM_PAD;
};

template <int K, int V, int GT, int HPG, typename TIO>
__global__ void __launch_bounds__(32 * HPG, NF_MINB)
gdn_step_kernel(
    const TIO* __restrict__ mixed_qkv, const void* __restrict__ a, const void* __restrict__ b, int ab_code,
    const void* __restrict__ A_log, const void* __restrict__ dt_bias, int p_code, int bias_code,
    TIO* __restrict__ out, const float* __restrict__ h0,
    __nv_bfloat16* __restrict__ d_cache, __nv_bfloat16* __restrict__ k_cache, float* __restrict__ g_cache,
    const int* __restrict__ ssm_state_indices, const int* __restrict__ write_pos, float scale,
    const sketch_t* __restrict__ sk_ubar, const sketch_t* __restrict__ sk_phi,
    const int* __restrict__ sk_mh,
    sketch_t* __restrict__ sk_fs, const int* __restrict__ sk_meta, float* __restrict__ beta_ring, float* current_d, float* current_k, const int4* __restrict__ layout,
    int s_mix, int s_a, int s_b, int s_u_slot, int s_phi_slot, int s_fs_slot,
    long S_H0_SLOT, long S_D_SLOT, long S_K_SLOT, long S_G_SLOT)
{
    // compile-time geometry: H = NF_H key heads, HV = NF_HV = HPG * H value heads, W = WMAX, K = V = 128;
    // one warp per value head of the key head (CTA).
    // Slot strides are runtime: vLLM packs conv/state/ring buffers of a layer in one cache page, so a slot's
    // state stride is the page, not HV*K*V (the raw fixture uses dense tensors).  Inner strides stay dense.
    constexpr int H = NF_H, HV = NF_HV, W = WMAX;
    using SM = Smem<K, V, HPG>;
    constexpr int NW = HPG;
    constexpr int CK = K / 32;
    constexpr int VL = V / 32;
    constexpr int RS = SM::RS;                           // ring rows of the head tile (staged); W-1 = RS + 16 (W/16 - 1)
    constexpr int NSW = (RS + NW - 1) / NW;
    constexpr int NGC = (W + 31) / 32;                   // window rows per lane: gates, replay weights
    constexpr bool NORM_X = 2 * NSW + 3 <= 32;           // norm dots ride the ring-key transpose (HPG >= 2)
    constexpr int NKV = Pow2<NORM_X ? 2 * NSW + 3 : 2 * NSW>::v;
    constexpr int NG = (GT + 31) / 32;
    static_assert(K == 128 && V == 128, "K == V == 128");
    static_assert(GT % 4 == 0 && HPG >= 1 && HPG <= 8, "GT/HPG");
    static_assert(VL == 4 && CK == 4, "layout");
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int i_n = blockIdx.x, i_h = blockIdx.y, i_hv = i_h * HPG + warp;
    extern __shared__ __align__(128) unsigned char dsm[];
    unsigned char* wsm = dsm + warp * SM::WBYTES;           // this warp's staging buffers

    __shared__ __align__(16) float sKQK[2][WMAX];        // [0] = <q, k_s>, [1] = <k, k_s> (raw, unnormalized)
    __shared__ __align__(16) float2 sC[NW][WMAX];
    __shared__ __align__(16) float2 sKK[NW][WMAX];       // (kq, kk) pairs of the key head, per warp
    __shared__ __align__(16) float sA[NW][GT];

    // ---- row-indexed operands: issued before the slot chain resolves ----
    const int sidx = ssm_state_indices[i_n];
    const int wp = write_pos[i_n];
    const int mh = sk_mh[i_hv];
    const int4 lay = layout[i_hv];                        // (u_off, phi_off, fs_off, FG); dense heads: u_off -1 = no rows
    float qc[CK], kc[CK];
    {
        const TIO* pq = mixed_qkv + (long)i_n * s_mix + (i_h * K + lane * CK);
        const TIO* pk = pq + H * K;
        const float4 q4 = ld4<TIO>(pq), k4 = ld4<TIO>(pk);
        qc[0] = q4.x; qc[1] = q4.y; qc[2] = q4.z; qc[3] = q4.w;
        kc[0] = k4.x; kc[1] = k4.y; kc[2] = k4.z; kc[3] = k4.w;
    }
    const float4 vv = ld4<TIO>(mixed_qkv + (long)i_n * s_mix + (2 * H * K + i_hv * V + lane * VL));
    const float a_val = ldx<NF_AB_CODE>(a, (long)i_n * s_a + i_hv);
    const float b_val = ldx<NF_AB_CODE>(b, (long)i_n * s_b + i_hv);
    const float Al = ldx<NF_P_CODE>(A_log, i_hv);
    const float dtb = ldx<NF_BIAS_CODE>(dt_bias, i_hv);
    const int FG = lay.w;
    TIO* p_o = out + (i_n * HV + i_hv) * V + lane * VL;
    if (sidx <= 0) { st4<TIO>(p_o, make_float4(0.f, 0.f, 0.f, 0.f)); return; }
    const int cidx = sk_meta[i_n];
    const bool latch = mh > 0;

    const float* pst = h0 + (long)sidx * S_H0_SLOT + i_hv * (K * V);
    __nv_bfloat16* bd = d_cache + (long)sidx * S_D_SLOT;
    const __nv_bfloat16* pdr = bd + i_hv * (W * V);
    __nv_bfloat16* bk = k_cache + (long)sidx * S_K_SLOT;
    float* bg = g_cache + (long)sidx * S_G_SLOT;
    const sketch_t* bu = sk_ubar + (long)cidx * s_u_slot + lay.x;
    const sketch_t* bp = sk_phi + (long)cidx * s_phi_slot + lay.y;
    sketch_t* pf = sk_fs + (long)cidx * s_fs_slot + lay.z;

    // ---- slot-indexed operands: all issued now ----
    float gs[NGC];                                       // gates of rows lane + 32 j
    gs[0] = (lane < wp) ? bg[i_hv * W + lane] : 0.f;
    #pragma unroll
    for (int j = 1; j < NGC; ++j) gs[j] = (lane + 32 * j < wp) ? bg[i_hv * W + lane + 32 * j] : 0.f;
    const int wph = (W > 16 && wp > RS) ? RS : wp;       // ring rows < wp in the head tile
    const int s0k = warp * NSW;
    const int nrk = (wph - s0k < NSW) ? (wph - s0k) : NSW;
    float kr[NSW][CK];
    {
        const uint2* kb = (const uint2*)(bk + (i_h * W + s0k) * K + lane * CK);
        #pragma unroll
        for (int i = 0; i < NSW; ++i) {
            const uint2 r = (i < nrk) ? kb[i * (K / 4)] : make_uint2(0u, 0u);   // rows >= nrk never reach a stored dot
            kr[i][0] = bf_lo(r.x); kr[i][1] = bf_hi(r.x); kr[i][2] = bf_lo(r.y); kr[i][3] = bf_hi(r.y);
        }
    }
    const bool fs_in_smem = FG <= SM::FS_FG, ag_in_smem = FG <= SM::AG_FG;
    const int pv_bytes = (mh >= K) ? 0 : (mh <= 4 ? mh * K : (ag_in_smem ? 4 * K + 5 * FG : 4 * K)) * (int)sizeof(sketch_t);
    const int np = (mh < 4) ? mh : 4;
    const int urows = (mh < NF_UROWS) ? mh : NF_UROWS;
    // shared staging (cp.async), straight-line: d-ring rows < wp; first sketch rows; the coefficient block; erase history.
    // Dense warps (mh == 0) stage nothing but the ring; their rows (if any) stream from global. Coefficient/erase blocks use 8-byte chunks (phi blocks are 8-byte aligned).
    static_assert(RS * V * 2 / 16 <= 32 * 8 && NF_UROWS * V * (int)sizeof(sketch_t) / 16 <= 32 * 4 && SM::FS_B / 8 <= 32 * 4 && SM::PV_B / 8 <= 32 * 7, "chunk counts");
    static_assert(SM::RING_B % 16 == 0 && SM::U_B % 16 == 0 && SM::PV_OFF % 16 == 0 && SM::FS_OFF % 16 == 0, "alignment");
#if !(NF_ABLATE & 32)
    stage16<8>(wsm, (const unsigned char*)pdr, wph * V * 2, lane);
    stage16z<4>(wsm + SM::RING_B, (const unsigned char*)bu, urows * V * (int)sizeof(sketch_t), lane);   // rows >= mh zero-filled
    stage8<7>(wsm + SM::PV_OFF, (const unsigned char*)bp, pv_bytes, lane);                                   // pivot rows + diagonal + gains, one block
    stage8<4>(wsm + SM::FS_OFF, (const unsigned char*)pf, (fs_in_smem && mh > 0) ? wph * FG * (int)sizeof(sketch_t) : 0, lane);   // erase rows 0..wph-1, pitch FG (dense heads have none: their FG pads no fs rows)
#endif
    cp_async_commit();

    // ---- gates, normalization ----
    const float xg = a_val + dtb;
    const float sp = (xg <= 20.f) ? log1pf(expf(xg)) : xg;   // log1pf: not a fast-math intrinsic (the gate enters the state)
    const float g_val = -expf(Al) * sp;
    const float alpha = expf(g_val);
    const float beta = 1.f / (1.f + expf(-b_val));      // FP32, as the reference decode
    if (beta_ring && lane == 0) beta_ring[(cidx * HV + i_hv) * W + wp] = beta;
    // ---- raw dots (unnormalized q/k): ring keys (10 slots) + norms <q,q>, <k,k>, <q,k> (slots 10..12), one transpose-reduction ----
    float kv[NKV];
    #pragma unroll
    for (int i = 0; i < NKV; ++i) kv[i] = 0.f;
    float2 qk2[CK];
    #pragma unroll
    for (int c = 0; c < CK; ++c) qk2[c] = make_float2(qc[c], kc[c]);
#if !(NF_ABLATE & 16)
    #pragma unroll
    for (int i = 0; i < NSW; ++i) {
        float2 pp = make_float2(0.f, 0.f);           // (q dot, k dot)
        #pragma unroll
        for (int c = 0; c < CK; ++c) pp = ffma2(make_float2(kr[i][c], kr[i][c]), qk2[c], pp);
        kv[i] = pp.x; kv[NSW + i] = pp.y;
    }
#endif
    static_assert(!NORM_X || 2 * NSW + 3 <= NKV, "norm slots");
    float2 nn = make_float2(0.f, 0.f); float qkd = 0.f;
    #pragma unroll
    for (int c = 0; c < CK; ++c) { nn = ffma2(qk2[c], qk2[c], nn); qkd = fmaf(qc[c], kc[c], qkd); }
    if constexpr (NORM_X) { kv[2 * NSW] = nn.x; kv[2 * NSW + 1] = nn.y; kv[2 * NSW + 2] = qkd; }
    const float kval = xposeN<NKV>(kv, lane);         // lane e holds slot e (e < NKV)
    float sq, sk, qk;
    if constexpr (NORM_X) { sq = __shfl_sync(FULL, kval, 2 * NSW); sk = __shfl_sync(FULL, kval, 2 * NSW + 1); qk = __shfl_sync(FULL, kval, 2 * NSW + 2); }
    else { sq = warp_sum(nn.x); sk = warp_sum(nn.y); qk = warp_sum(qkd); }   // HPG == 1: 30 ring slots fill the warp
    const float q_sc = (1.f / sqrtf(sq + 1e-6f)) * scale;
    const float k_rn = 1.f / sqrtf(sk + 1e-6f);
    #pragma unroll
    for (int c = 0; c < CK; ++c) { qc[c] *= q_sc; kc[c] *= k_rn; qk2[c] = make_float2(qc[c], kc[c]); }
    const float cur_kq = qk * q_sc * k_rn;
    if (warp == 0) {
        if (wp == W - 1) st4<float>(current_k + (cidx * H + i_h) * K + lane * CK, make_float4(kc[0], kc[1], kc[2], kc[3]));
        else st4<__nv_bfloat16>(bk + (i_h * W + wp) * K + lane * CK, make_float4(kc[0], kc[1], kc[2], kc[3]));
    }
    {   // slot e < 2*NSW: dot (e < NSW ? q : k) with ring key s0k + (e mod NSW); scaled by the matching norm
        const int e = lane & (NKV - 1);
        const int i = (e < NSW) ? e : e - NSW;
        const float scl = (e < NSW) ? q_sc : k_rn;
        if (lane < 2 * NSW && i < nrk) sKQK[e < NSW ? 0 : 1][s0k + i] = kval * scl;
    }
    if constexpr (W > 16) {
        // ---- ring keys RS .. wp-1 in 16-row tiles (normalized q, k): one transpose-reduction per tile ----
        constexpr int NSX = (16 + NW - 1) / NW, NKX = Pow2<2 * NSX>::v;
        const __nv_bfloat16* kb = bk + i_h * W * K + lane * CK;
        #pragma unroll 1
        for (int r0 = RS; r0 < wp; r0 += 16) {
            const int s0 = r0 + warp * NSX;
            const int nr = min(min(wp, r0 + 16) - s0, NSX);
            float kx[NKX];
            #pragma unroll
            for (int i = 0; i < NKX; ++i) kx[i] = 0.f;
            #pragma unroll
            for (int i = 0; i < NSX; ++i) {
                const uint2 r = (i < nr) ? *((const uint2*)(kb + (s0 + i) * K)) : make_uint2(0u, 0u);
                const float kk4[4] = {bf_lo(r.x), bf_hi(r.x), bf_lo(r.y), bf_hi(r.y)};
                float2 pp = make_float2(0.f, 0.f);
                #pragma unroll
                for (int c = 0; c < CK; ++c) pp = ffma2(make_float2(kk4[c], kk4[c]), qk2[c], pp);
                kx[i] = pp.x; kx[NSX + i] = pp.y;
            }
            const float xv = xposeN<NKX>(kx, lane);
            const int e = lane & (NKX - 1);
            const int i = (e < NSX) ? e : e - NSX;
            if (lane < 2 * NSX && i < nr) sKQK[e < NSX ? 0 : 1][s0 + i] = xv;
        }
    }

    // gate prefix over the window rows lane + 32 j; rep_s = exp(gtot - pre_s) for s < wp
    constexpr int SCN = W < 32 ? W : 32;
    float pre = gs[0];
    #pragma unroll
    for (int o = 1; o < SCN; o <<= 1) { const float n = __shfl_up_sync(FULL, pre, o); if (lane >= o) pre += n; }
    float gtot = __shfl_sync(FULL, pre, SCN - 1);
    float prex[NGC];
    #pragma unroll
    for (int j = 1; j < NGC; ++j) {
        float p = gs[j];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const float n = __shfl_up_sync(FULL, p, o); if (lane >= o) p += n; }
        p += gtot;
        prex[j] = p;
        gtot = __shfl_sync(FULL, p, 31);
    }
    float rep[NGC];
    rep[0] = (lane < wp) ? expf(gtot - pre) : 0.f;
    #pragma unroll
    for (int j = 1; j < NGC; ++j) rep[j] = (lane + 32 * j < wp) ? expf(gtot - prex[j]) : 0.f;
    const float tot = expf(gtot);

#if (NF_ABLATE & 16)
    #pragma unroll
    for (int j = 0; j < NGC; ++j) { const int r = lane + 32 * j; if (r < WMAX) { sKQK[0][r] = 0.f; sKQK[1][r] = 0.f; } }
#endif
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < NGC; ++j) {
        const int r = lane + 32 * j;
        if (r < WMAX) {
            const float kq_s = (r < wp) ? sKQK[0][r] : 0.f, kk_s = (r < wp) ? sKQK[1][r] : 0.f;
            sC[warp][r] = make_float2(kq_s * rep[j], kk_s * rep[j]); sKK[warp][r] = make_float2(kq_s, kk_s);
        }
    }
    __syncwarp();

    float4 hq = make_float4(0.f, 0.f, 0.f, 0.f), hk = hq;
    float4 s_q = hq, s_k = hq;
    cp_async_wait_all();
    __syncwarp();
    {
#if !(NF_ABLATE & 8)
        // ---- coefficient dots: tq[p] = <phi_p, q>, tk[p] = <phi_p, k> (pivot rows from shared); no work for mh == 0 ----
        const sketch_t* pv_sm = (const sketch_t*)(wsm + SM::PV_OFF);
        float tt[8];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {                    // unconditional: rows beyond np hold stale data whose dots are never used
            const float4 z = ld_sk4(pv_sm + p * K + lane * CK);
            float2 pp = make_float2(0.f, 0.f);
            pp = ffma2(make_float2(z.x, z.x), qk2[0], pp); pp = ffma2(make_float2(z.y, z.y), qk2[1], pp);
            pp = ffma2(make_float2(z.z, z.z), qk2[2], pp); pp = ffma2(make_float2(z.w, z.w), qk2[3], pp);
            tt[p] = pp.x; tt[4 + p] = pp.y;
        }
        __syncwarp();                                    // every lane is done with the pivot rows: park the (q, k) pairs there
        float2* qk_sm = (float2*)pv_sm;
        *((float4*)(qk_sm + lane * CK)) = make_float4(qk2[0].x, qk2[0].y, qk2[1].x, qk2[1].y);
        *((float4*)(qk_sm + lane * CK + 2)) = make_float4(qk2[2].x, qk2[2].y, qk2[3].x, qk2[3].y);
        const float tred = xposeN<8>(tt, lane);
        float tq[4], tk[4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) { tq[p] = __shfl_sync(FULL, tred, p); tk[p] = __shfl_sync(FULL, tred, 4 + p); }
        const float xq4 = __shfl_sync(FULL, tred, lane & 3), xk4 = __shfl_sync(FULL, tred, 4 + (lane & 3));   // mh <= 4: x[g] = t[g]
        // ---- x[g] with the projected erase history ----
        const sketch_t* fs_sm = (const sketch_t*)(wsm + SM::FS_OFF);
        if (lane >= mh && lane < NF_UROWS) sA[warp][lane] = 0.f;       // staged rows beyond mh contribute zero
        #pragma unroll
        for (int i = 0; i < NG; ++i) {
            const int g = lane + 32 * i;
            if (g < mh) {
                float xq, xk;
                if (mh <= 4) {
                    xq = xq4; xk = xk4;
                } else if (mh < K) {
                    float ad, gn[4];
                    if (ag_in_smem) {
                        const sketch_t* acol = pv_sm + 4 * K + g;
                        ad = to_f(acol[0]);
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) { acol += FG; gn[p] = to_f(acol[0]); }
                    } else {
                        const sketch_t* acol = bp + 4 * K + g;
                        ad = to_f(acol[0]);
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) { acol += FG; gn[p] = to_f(acol[0]); }
                    }
                    const float2 qkg = qk_sm[g];
                    xq = ad * qkg.x; xk = ad * qkg.y;
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) { xq = fmaf(gn[p], tq[p], xq); xk = fmaf(gn[p], tk[p], xk); }
                } else {
                    const float2 qkg = qk_sm[g];
                    xq = qkg.x; xk = qkg.y;
                }
                float2 er = make_float2(0.f, 0.f);          // (eq, ek)
#if !(NF_ABLATE & 1)
                if (fs_in_smem) {
                    if constexpr (W > 16) {
                        const sketch_t* fcol = pf + g + (wp - 1) * FG;     // rows wp-1 .. RS (global)
                        for (int s = wp - 1; s >= RS; --s) {
                            const float fv = to_f(fcol[0]); fcol -= FG;
                            er = ffma2(make_float2(fv, fv), sKK[warp][s], er);
                        }
                    }
                    const sketch_t* fcol = fs_sm + g + (wph - 1) * FG;     // rows wph-1 .. 0 (shared), column pointer steps by FG
                    for (int s = wph - 1; s >= 0; --s) {
                        const float fv = to_f(fcol[0]); fcol -= FG;
                        er = ffma2(make_float2(fv, fv), sKK[warp][s], er);
                    }
                } else {
                    const sketch_t* fcol = pf + g + (wp - 1) * FG;
                    for (int s = wp - 1; s >= 0; --s) {
                        const float fv = to_f(fcol[0]); fcol -= FG;
                        er = ffma2(make_float2(fv, fv), sKK[warp][s], er);
                    }
                }
#endif
                const float eq = er.x, ek = er.y;
                const float fcur = beta * (xk - ek);
                sA[warp][g] = xq - eq - fcur * cur_kq;
#if !(NF_ABLATE & 64)
                pf[wp * FG + g] = from_f<sketch_t>(fcur);
#endif
            }
        }
        __syncwarp();
#else
    if (lane < GT) sA[warp][lane] = 1.f;
    __syncwarp();
#endif
#if !(NF_ABLATE & 4)
        // ---- sketch readout: staged rows, then streamed tail ----
        const sketch_t* u_sm = (const sketch_t*)(wsm + SM::RING_B);
        float2 hq01 = make_float2(0.f, 0.f), hq23 = make_float2(0.f, 0.f);
        if (latch) {
            #pragma unroll
            for (int g = 0; g < NF_UROWS; ++g) {           // unconditional: rows >= mh are zero-filled and x[g] = 0
                const float x = sA[warp][g];
                const float4 u = ld_sk4(u_sm + g * V + lane * VL);
                const float2 x2 = make_float2(x, x);
                hq01 = ffma2(make_float2(u.x, u.y), x2, hq01);
                hq23 = ffma2(make_float2(u.z, u.w), x2, hq23);
            }
            const sketch_t* ub = bu + NF_UROWS * V + lane * VL;
            for (int g = NF_UROWS; g < mh; g += 2) {       // streamed tail, two rows per iteration
                const float4 u0 = ld_sk4(ub);
                const float x0 = sA[warp][g];
                const bool two = g + 1 < mh;
                const float4 u1 = two ? ld_sk4(ub + V) : make_float4(0.f, 0.f, 0.f, 0.f);
                const float x1 = two ? sA[warp][g + 1] : 0.f;
                ub += 2 * V;
                hq01 = ffma2(make_float2(u0.x, u0.y), make_float2(x0, x0), hq01);
                hq23 = ffma2(make_float2(u0.z, u0.w), make_float2(x0, x0), hq23);
                hq01 = ffma2(make_float2(u1.x, u1.y), make_float2(x1, x1), hq01);
                hq23 = ffma2(make_float2(u1.z, u1.w), make_float2(x1, x1), hq23);
            }
        }
        hq = make_float4(hq01.x, hq01.y, hq23.x, hq23.y);
    }
    if (!latch && lay.x >= 0) {
        // ---- dense head with BF16 rows U[k][v] of the window-start state (SketchSSM): hq = U^T q, hk = U^T k,
        //      key rows streamed from global (lane owns values 4l..4l+3), (q, k) pairs from the parked shared copy ----
        const float2* qk_sm = (const float2*)(wsm + SM::PV_OFF);
        const sketch_t* ub = bu + lane * VL;
        float2 q01 = make_float2(0.f, 0.f), q23 = q01, k01 = q01, k23 = q01;
        #pragma unroll 8
        for (int g = 0; g < K; ++g) {
            const float4 u = ld_sk4(ub + g * V);
            const float2 p = qk_sm[g];
            const float2 xq = make_float2(p.x, p.x), xk = make_float2(p.y, p.y);
            q01 = ffma2(make_float2(u.x, u.y), xq, q01); q23 = ffma2(make_float2(u.z, u.w), xq, q23);
            k01 = ffma2(make_float2(u.x, u.y), xk, k01); k23 = ffma2(make_float2(u.z, u.w), xk, k23);
        }
        hq = make_float4(q01.x, q01.y, q23.x, q23.y);
        hk = make_float4(k01.x, k01.y, k23.x, k23.y);
    } else if (!latch) {
        // ---- dense head without rows (ReplaySSM): hq = S^T q, hk = S^T k streamed from global (rows of 128 keys;
        //      lane owns keys 4l..4l+3), warp transpose-reductions over 16-row groups, results gathered to the lane's
        //      four values ----
        float oq[4] = {0.f, 0.f, 0.f, 0.f}, ok_[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 1
        for (int gi = 0; gi < V / 16; ++gi) {
            float pq[16], pk[16];
            #pragma unroll
            for (int r0 = 0; r0 < 16; r0 += 4) {
                float4 x[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) x[j] = *((const float4*)(pst + (gi * 16 + r0 + j) * K + lane * CK));
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    pq[r0 + j] = x[j].x * qc[0] + x[j].y * qc[1] + x[j].z * qc[2] + x[j].w * qc[3];
                    pk[r0 + j] = x[j].x * kc[0] + x[j].y * kc[1] + x[j].z * kc[2] + x[j].w * kc[3];
                }
            }
            const float rq = xposeN<16>(pq, lane), rk = xposeN<16>(pk, lane);   // lane e (and e+16) holds row gi*16 + (e & 15)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float vq = __shfl_sync(FULL, rq, (4 * (lane & 3) + j) & 15), vk = __shfl_sync(FULL, rk, (4 * (lane & 3) + j) & 15);
                if ((lane >> 2) == gi) { oq[j] = vq; ok_[j] = vk; }
            }
        }
        hq = make_float4(oq[0], oq[1], oq[2], oq[3]);
        hk = make_float4(ok_[0], ok_[1], ok_[2], ok_[3]);
#endif
    }
    {
        // ---- ring replay from shared (bf16 rows), both paths; rows wp-1 .. 0 through a jump table ----
        const unsigned char* ring_sm = wsm + lane * 8;
#define NF_REPLAY_ROW(s) { \
            const float2 cc = sC[warp][s]; \
            const uint2 r = *((const uint2*)(ring_sm + (s) * V * 2)); \
            const float2 d01 = make_float2(bf_lo(r.x), bf_hi(r.x)), d23 = make_float2(bf_lo(r.y), bf_hi(r.y)); \
            const float2 cq2 = make_float2(cc.x, cc.x), ck2 = make_float2(cc.y, cc.y); \
            float2 t; \
            t = ffma2(d01, cq2, make_float2(s_q.x, s_q.y)); s_q.x = t.x; s_q.y = t.y; \
            t = ffma2(d23, cq2, make_float2(s_q.z, s_q.w)); s_q.z = t.x; s_q.w = t.y; \
            t = ffma2(d01, ck2, make_float2(s_k.x, s_k.y)); s_k.x = t.x; s_k.y = t.y; \
            t = ffma2(d23, ck2, make_float2(s_k.z, s_k.w)); s_k.z = t.x; s_k.w = t.y; }
#if !(NF_ABLATE & 2)
        if constexpr (W > 16) {                          // rows RS .. wp-1 from global, 16-row tiles
            const unsigned char* ring_g = (const unsigned char*)pdr + lane * 8;
            #pragma unroll 1
            for (int r0 = RS; r0 < wp; r0 += 16) {
                uint2 rr[16];
                #pragma unroll
                for (int i = 0; i < 16; ++i) rr[i] = (r0 + i < wp) ? *((const uint2*)(ring_g + (r0 + i) * V * 2)) : make_uint2(0u, 0u);
                #pragma unroll
                for (int i = 0; i < 16; ++i) {           // rows >= wp: zero row, zero weight
                    const float2 cc = sC[warp][r0 + i];
                    const float2 d01 = make_float2(bf_lo(rr[i].x), bf_hi(rr[i].x)), d23 = make_float2(bf_lo(rr[i].y), bf_hi(rr[i].y));
                    const float2 cq2 = make_float2(cc.x, cc.x), ck2 = make_float2(cc.y, cc.y);
                    float2 t;
                    t = ffma2(d01, cq2, make_float2(s_q.x, s_q.y)); s_q.x = t.x; s_q.y = t.y;
                    t = ffma2(d23, cq2, make_float2(s_q.z, s_q.w)); s_q.z = t.x; s_q.w = t.y;
                    t = ffma2(d01, ck2, make_float2(s_k.x, s_k.y)); s_k.x = t.x; s_k.y = t.y;
                    t = ffma2(d23, ck2, make_float2(s_k.z, s_k.w)); s_k.z = t.x; s_k.w = t.y;
                }
            }
        }
        int s = wph - 1;
        for (; s >= 3; s -= 4) { NF_REPLAY_ROW(s) NF_REPLAY_ROW(s - 1) NF_REPLAY_ROW(s - 2) NF_REPLAY_ROW(s - 3) }
        for (; s >= 0; --s) NF_REPLAY_ROW(s)
#endif
#undef NF_REPLAY_ROW
    }
    {
        float o[4], dc[4];
        const float hqv[4] = {hq.x, hq.y, hq.z, hq.w}, hkv[4] = {hk.x, hk.y, hk.z, hk.w};
        const float sqv[4] = {s_q.x, s_q.y, s_q.z, s_q.w}, skv[4] = {s_k.x, s_k.y, s_k.z, s_k.w};
        const float vvv[4] = {vv.x, vv.y, vv.z, vv.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float stq = alpha * (hqv[c] * tot + sqv[c]);
            const float stk = alpha * (hkv[c] * tot + skv[c]);
            dc[c] = latch ? beta * (vvv[c] - alpha * skv[c])
                          : beta * (vvv[c] - stk);
            o[c] = stq + dc[c] * cur_kq;
        }
        st4<TIO>(p_o, make_float4(o[0], o[1], o[2], o[3]));
#if !(NF_ABLATE & 64)
        if (wp == W - 1) st4<float>(current_d + (cidx * HV + i_hv) * V + lane * VL, make_float4(dc[0], dc[1], dc[2], dc[3]));
        else st4<__nv_bfloat16>(bd + (i_hv * W + wp) * V + lane * VL, make_float4(dc[0], dc[1], dc[2], dc[3]));
        if (lane == 0) bg[i_hv * W + wp] = g_val;
#endif
    }
}

