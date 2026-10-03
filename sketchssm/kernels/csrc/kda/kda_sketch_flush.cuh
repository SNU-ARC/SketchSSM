// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
// SketchSSM decode kernels (https://arxiv.org/abs/2609.33051).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>


#define FULL 0xffffffffu
#define KPB 136                 // bf16 row pitch of the key-indexed tables (sA, sC, sO): keys contiguous, conflict-free ldmatrix rows
#define TP 24                   // bf16 row pitch of sT [16 i][TP] (j contiguous)
#define NWARPS 4
#define NTHREADS (32 * NWARPS)
#ifndef K1_MINB
#define K1_MINB 3
#endif
// Per rank bucket (MMAX) minimum CTAs per SM of the main kernel (register cap and grid size); K1_MINB16 is the
// a fixed 4, the others default to K1_MINB. K2_MINB: the finish. Scheduling only (vLLM tuning knobs).
#ifndef K1_MINB16
#define K1_MINB16 4
#endif
#ifndef K1_MINB32
#define K1_MINB32 K1_MINB
#endif
#ifndef K1_MINB64
#define K1_MINB64 K1_MINB
#endif
#ifndef K1_MINB128
#define K1_MINB128 K1_MINB
#endif
#define K1_MINB_OF(M) ((M) == 16 ? K1_MINB16 : (M) == 32 ? K1_MINB32 : (M) == 64 ? K1_MINB64 : K1_MINB128)
#ifndef K2_MINB
#define K2_MINB 6
#endif
#ifndef KDA_W
#define KDA_W 16        // window rows W (a multiple of 16): ring rows per head, f history columns
#endif
static_assert(KDA_W % 16 == 0 && KDA_W >= 16, "kda flush: the window must be a positive multiple of 16");
#define NBLK (KDA_W / 16)       // 16-row blocks of the window (the blocked WY flush)
#ifndef K1_ABLATE
#define K1_ABLATE 0     // timing only: 1 = no WY prologue (a/c/L/T), 2 = no block WY/update, 4 = no statistics, 8 = no finish
#endif
#define TILE_B 8192
#define O4P 136                 // f32 row pitch of sO4 [4 g][O4P]: rows 8 banks apart, so the (g & 3, c) float2 loads are conflict-free
#define SCRATCH_F 1792          // per-item finish scratch (floats): U4 [4][128], Y4 [4][128], energy [128], sumsq, W = U4^T U [4][128]

__device__ __forceinline__ float ex2_ftz(float t) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(t)); return y; }   // no denormal fix-up path
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}
__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
    const __nv_bfloat162 h2 = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h2);
}
// x = hi + lo, hi = bf16(x) (round to nearest), lo = bf16(x - hi): 17 significant bits, fp32 exponent
// (the dropped lo x lo term of a three-term product is then <= 2^-18 relative)
__device__ __forceinline__ void split2(float x0, float x1, unsigned& h, unsigned& l) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm(
        "{ .reg .b16 h0,h1,bn; .reg .f32 r0,r1;"
        " mov.b16 bn,0xbf80;"
        " cvt.rn.bf16x2.f32 %0,%3,%2;"
        " mov.b32 {h0,h1},%0;"
        " fma.rn.f32.bf16 r0,h0,bn,%2;"
        " fma.rn.f32.bf16 r1,h1,bn,%3;"
        " cvt.rn.bf16x2.f32 %1,r1,r0; }"
        : "=&r"(h),"=r"(l) : "f"(x0),"f"(x1));
#else
    // x - hi is exact-product fma(hi, -1, x): one rounding, the same bits
    h = pack_bf16(x0, x1);
    l = pack_bf16(x0 - __uint_as_float(h << 16), x1 - __uint_as_float(h & 0xffff0000u));
#endif
}
// bf16 hi/lo A fragment (m16n8k16) from the fp32 C fragments of two consecutive n8 tiles (same layout)
__device__ __forceinline__ void c2a_bf16(const float* c0, const float* c1, unsigned* ah, unsigned* al) {
    split2(c0[0], c0[1], ah[0], al[0]); split2(c0[2], c0[3], ah[1], al[1]);
    split2(c1[0], c1[1], ah[2], al[2]); split2(c1[2], c1[3], ah[3], al[3]);
}
__device__ __forceinline__ void mma16816_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma3(float* c, const unsigned* ah, const unsigned* al, unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma16816_bf16(c, ah, bh0, bh1); mma16816_bf16(c, ah, bl0, bl1); mma16816_bf16(c, al, bh0, bh1);
}
// tf32 m16n8k8 with the C-fragment-as-A trick (virtual k permutation c <-> 2c, c+4 <-> 2c+1; B follows)
__device__ __forceinline__ float tf32_hi(float x) { return __uint_as_float(__float_as_uint(x) & 0xffffe000u); }
__device__ __forceinline__ void mma_tf32(float* c, float a0, float a1, float a2, float a3, float b0, float b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)), "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)),
                   "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1)));
}
__device__ __forceinline__ void mma_tf32_c(float* c, const float* a, float b0, float b1) { mma_tf32(c, a[0], a[2], a[1], a[3], b0, b1); }
__device__ __forceinline__ unsigned movtrans(unsigned a) {
    unsigned d;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
    return d;
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(unsigned* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm2(unsigned* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(unsigned* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
// Key order of the key-indexed tables (sA, sC, sO) and of the accumulator n-tiles: within each 16-key group, position
// p holds key KEY(p): positions 0..7 = keys 0,1,4,5,8,9,12,13 and 8..15 = keys 2,3,6,7,10,11,14,15.  Then n-tile
// pair (2 p, 2 p + 1) gives lane c the four consecutive keys 16 p + 4 c .. + 3, so the state tile moves between
// registers and memory with 16-byte accesses (half the L1 wavefronts of the fragment-order float2 stores).
__device__ __forceinline__ int KEY(int p) { return 4 * ((p & 7) >> 1) + (p & 1) + ((p & 8) ? 2 : 0); }
// state tile in shared memory: natural key order, 16-byte chunks XOR-swizzled by (row & 1) (the (g, c) float4 accesses
// of a quarter-warp cover two rows)
__device__ __forceinline__ int tile_at(int row, int col) { return row * 128 + (col ^ ((row & 1) << 4)); }

template <typename T> __device__ __forceinline__ void st_f(T* p, float v);
template <> __device__ __forceinline__ void st_f<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void st_f<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
template <typename T> __device__ __forceinline__ void st4_f(T* p, float a, float b, float c, float d);
template <> __device__ __forceinline__ void st4_f<float>(float* p, float a, float b, float c, float d) { *(float4*)p = make_float4(a, b, c, d); }
template <> __device__ __forceinline__ void st4_f<__nv_bfloat16>(__nv_bfloat16* p, float a, float b, float c, float d) { *(uint2*)p = make_uint2(pack_bf16(a, b), pack_bf16(c, d)); }


__device__ __forceinline__ int tma_at(int row,int col,unsigned tile_u) {
    const int linear=row*128+col;
    return linear ^ ((((linear>>5)+((tile_u>>7)&7))&7)<<2);
}
__device__ __forceinline__ void tma_load_tile(const CUtensorMap* map,unsigned tile_u,unsigned bar,int y) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 8192;" :: "r"(bar) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {0, %2}], [%3];"
        :: "r"(tile_u), "l"(map), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_wait_tile(unsigned bar,int phase) {
    asm volatile("{ .reg .pred p; wait_tma: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra wait_tma; }"
        :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void load_tile(float (&acc)[16][4], const float* tile, int g, int c) {
    #pragma unroll
    for (int p = 0; p < 8; ++p)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float4 v = *(const float4*)(tile + tma_at(g + 8 * hh, 16 * p + 4 * c, smem_u32(tile)));
            acc[2 * p][2 * hh] = v.x; acc[2 * p][2 * hh + 1] = v.y; acc[2 * p + 1][2 * hh] = v.z; acc[2 * p + 1][2 * hh + 1] = v.w;
        }
}
// the inverse of load_tile: the 16-value tile back to its (swizzled) shared-memory slots
__device__ __forceinline__ void store_tile_smem(const float (&acc)[16][4], float* tile, int g, int c) {
    #pragma unroll
    for (int p = 0; p < 8; ++p)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
            *(float4*)(tile + tma_at(g + 8 * hh, 16 * p + 4 * c, smem_u32(tile))) =
                make_float4(acc[2 * p][2 * hh], acc[2 * p][2 * hh + 1], acc[2 * p + 1][2 * hh], acc[2 * p + 1][2 * hh + 1]);
}
__device__ __forceinline__ void store_tile(const float (&acc)[16][4], float* dst, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        float* pr = dst + (long)(g + 8 * hh) * 128 + 4 * c;
        #pragma unroll
        for (int p = 0; p < 8; ++p)
            *(float4*)(pr + 16 * p) = make_float4(acc[2 * p][2 * hh], acc[2 * p][2 * hh + 1], acc[2 * p + 1][2 * hh], acc[2 * p + 1][2 * hh + 1]);
    }
}
__device__ __forceinline__ void store16_hilo(__nv_bfloat16* ph, __nv_bfloat16* pl, const float* x) {   // x[k]: 16 keys -> positions
    unsigned h[8], l[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) split2(x[KEY(2 * i)], x[KEY(2 * i + 1)], h[i], l[i]);
    *(uint4*)ph = make_uint4(h[0], h[1], h[2], h[3]); *(uint4*)(ph + 8) = make_uint4(h[4], h[5], h[6], h[7]);
    *(uint4*)pl = make_uint4(l[0], l[1], l[2], l[3]); *(uint4*)(pl + 8) = make_uint4(l[4], l[5], l[6], l[7]);
}


__device__ __forceinline__ int value_row(int g) { return ((g&1)<<1)|((g&2)>>1)|(g&4); }
__device__ __forceinline__ void tma_load_half(const CUtensorMap* map,unsigned tile_u,unsigned bar,int row,int half_key) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;" :: "r"(bar) : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {0, %2, %3}], [%4];"
        :: "r"(tile_u), "l"(map), "r"(half_key), "r"(row), "r"(bar) : "memory");
}
template<int HALF>
__device__ __forceinline__ void load_half(float (&acc)[16][4],const float* tile,int vg,int c) {
    const unsigned tile_u=smem_u32(tile);
    #pragma unroll
    for(int p=0;p<4;++p) {
        #pragma unroll
        for(int hh=0;hh<2;++hh) {
            const int linear=(vg+8*hh)*64+16*p+4*c;
            const int at=linear ^ ((((linear>>5)+((tile_u>>7)&7))&7)<<2);
            const float4 v=*(const float4*)(tile+at);
            acc[8*HALF+2*p][2*hh]=v.x; acc[8*HALF+2*p][2*hh+1]=v.y;
            acc[8*HALF+2*p+1][2*hh]=v.z; acc[8*HALF+2*p+1][2*hh+1]=v.w;
        }
    }
}
__device__ __forceinline__ void spill_y4(float* p,float a,float b,float c,float d) {
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(smem_u32(p)),"f"(a),"f"(b),"f"(c),"f"(d) : "memory");
}
__device__ __forceinline__ float4 reload_y4(const float* p) {
    float4 r;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x),"=f"(r.y),"=f"(r.z),"=f"(r.w) : "r"(smem_u32(p)) : "memory");
    return r;
}

// BLOCKED (W > 16, flush phase): the whole state stays in shared memory over the window's blocks, 16 KB per warp (its
// two 16-value tiles); the operand tables hold one 16-row block at a time
template <int MMAX, bool BLOCKED = false> struct Smem {
    static constexpr int A_OFF = 0;                                  // sAh, sAl [16 w][KPB] bf16
    static constexpr int A_B = 16 * KPB * 2;
    static constexpr int C_OFF = A_OFF + 2 * A_B;                    // sCh, sCl [16 w][KPB] bf16
    static constexpr int T_OFF = C_OFF + 2 * A_B;                    // sTh, sTl [16 i][TP] bf16
    static constexpr int STP = MMAX == 16 ? 16 : TP;
    static constexpr int T_B = 16 * STP * 2;
    static constexpr int O_OFF = T_OFF + 2 * T_B;                    // sOh, sOl [MMAX][KPB] bf16
    static constexpr int O_B = (MMAX == 16 ? 8 : MMAX) * KPB * 2;
    static constexpr int V_OFF = O_OFF + 2 * O_B;                    // sV [16 j][KPB] bf16 (v ring; pitch 136: conflict-free ldmatrix.trans)
    static constexpr int Q_OFF = V_OFF + 16 * KPB * 2;                       // sQ [128] f32
    static constexpr int PW_OFF = Q_OFF + 512;                       // sPW [128] f32
    static constexpr int O4_OFF = PW_OFF + 512;                      // sO4h, sO4l [4 g][O4P] f32: tf32 hi/lo pivot columns of Omega
    static constexpr int U4_OFF = O4_OFF + 2 * 4 * O4P * 4;                     // sU4 [4][128] f32
    static constexpr int L_OFF = U4_OFF + (MMAX == 16 ? 0 : 2048);                      // sL [16][16] f32
    static constexpr int TILE_OFF = L_OFF + (MMAX == 16 ? 8192 : 1024);                    // tiles [NWARPS][8 KB]
    static constexpr int TILE_BYTES = BLOCKED ? 2 * TILE_B : MMAX == 16 ? 4096 : TILE_B;
    static constexpr int BYTES = TILE_OFF + NWARPS * TILE_BYTES;
};

// W > 16: the CTAs per SM the blocked main kernel's shared memory allows (228 KB per SM, 1 KB reserved per CTA). Its
// minimum CTAs per SM (register cap) is the knob's at most this: more CTAs could not be resident, the cap would only
// spill.
template <int MMAX> constexpr int blocked_fit() {
    return (228 * 1024) / (Smem<MMAX, true>::BYTES + 1024) > 1 ? (228 * 1024) / (Smem<MMAX, true>::BYTES + 1024) : 1;
}
#define K1_MINB_LB(M, BL) (((BL) && blocked_fit<M>() < K1_MINB_OF(M)) ? blocked_fit<M>() : K1_MINB_OF(M))

// the rings of a state page: every ring shares the page stride RS (bytes), rp = page * RS
template <typename T> __device__ __forceinline__ const T* page_ring(const T* base, long rp) { return (const T*)((const char*)base + rp); }

// W > 16, one window block b < NBLK - 1 (operand tables of block b in sA / sC / sV / sT / sPW): this warp's two 16-value
// tiles of the shared-memory state S_b (decayed to ref_b) become S_{b+1} = (S_b + X C) * P_W with X = V T'' - P T'',
// P = S_b A^T: the blocked forward substitution of the W x W WY system, its off-diagonal blocks applied through the
// state (P of the next block sees this block's X C). The first block waits for the state tiles (TMA) and issues the
// second one.
template <int MMAX>
__device__ __forceinline__ void prepass_block(bool first, int warp, int lane, float* tile, unsigned tile_u,
    const CUtensorMap& input_map, unsigned state_bar, long page, long S0, long S1, int h,
    const __nv_bfloat16* sAh, const __nv_bfloat16* sAl, const __nv_bfloat16* sCh, const __nv_bfloat16* sCl,
    const __nv_bfloat16* sTh, const __nv_bfloat16* sTl, const __nv_bfloat16* sV, const float* sPW)
{
    constexpr int STP = Smem<MMAX, true>::STP;
    const int g = lane >> 2, c = lane & 3, lr = lane & 7, lt = lane >> 3;
    const int ld_row = lr + 8 * (lt >> 1), ld_col = 8 * (lt & 1);
    const unsigned pa_h = smem_u32(sAh + ld_row * KPB + ld_col), pa_l = smem_u32(sAl + ld_row * KPB + ld_col);
    const unsigned pc = smem_u32(((lt & 2) ? sCl : sCh) + (lr + 8 * (lt & 1)) * KPB);
    const unsigned pv = smem_u32(sV + (lr + 8 * (lt >> 1)) * KPB + 8 * (lt & 1));
    #pragma unroll 1
    for (int bi = 0; bi < 2; ++bi) {
        const int v0 = 16 * (2 * warp + bi);
        float* vt = tile + bi * (TILE_B / 4);
        if (first) {
            tma_wait_tile(state_bar, bi); __syncwarp();
            if (bi == 0 && lane == 0) tma_load_tile(&input_map, tile_u + TILE_B, state_bar, (int)((page * S0 + (long)h * S1 + (long)(2 * warp + 1) * 2048) / 32));
        }
        if (K1_ABLATE & 2) continue;
        float acc[16][4];
        load_tile(acc, vt, g, c);
        float P[2][4] = {};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned ah[4], al[4], bh[4], bl[4];
            c2a_bf16(acc[2 * j], acc[2 * j + 1], ah, al);
            ldsm4(bh, pa_h + 32 * j); ldsm4(bl, pa_l + 32 * j);
            mma3(P[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
            mma3(P[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
        float X[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        {
            unsigned av[4], ph[4], pl[4];
            ldsm4t(av, pv + 2 * v0);
            c2a_bf16(P[0], P[1], ph, pl);
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                const int i = 8 * nt2 + g;
                const unsigned th0 = *(const unsigned*)(sTh + i * STP + 2 * c), th1 = *(const unsigned*)(sTh + i * STP + 2 * c + 8);
                const unsigned tl0 = *(const unsigned*)(sTl + i * STP + 2 * c), tl1 = *(const unsigned*)(sTl + i * STP + 2 * c + 8);
                mma16816_bf16(X[nt2], av, th0, th1); mma16816_bf16(X[nt2], av, tl0, tl1);
                float Y[4] = {0.f, 0.f, 0.f, 0.f};
                mma3(Y, ph, pl, th0, th1, tl0, tl1);
                #pragma unroll
                for (int e = 0; e < 4; ++e) X[nt2][e] -= Y[e];
            }
        }
        unsigned xh[4], xl[4];
        c2a_bf16(X[0], X[1], xh, xl);
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            unsigned bq[4];
            ldsm4t(bq, pc + 16 * nt);
            mma3(acc[nt], xh, xl, bq[0], bq[1], bq[2], bq[3]);
            const float2 pw = *(const float2*)(sPW + 16 * (nt >> 1) + 4 * c + 2 * (nt & 1));
            acc[nt][0] *= pw.x; acc[nt][1] *= pw.y; acc[nt][2] *= pw.x; acc[nt][3] *= pw.y;
        }
        store_tile_smem(acc, vt, g, c);
    }
}

template <int MMAX, typename ST, bool FLUSH>
__device__ __forceinline__ void kda_flush_item(int row, int h, int m, long sidx, bool stage_omega, const CUtensorMap& input_map, unsigned state_bar,
    float* __restrict__ state, long S0, long S1,
    const int* __restrict__ slots, const int* __restrict__ page_indices, int H, int G,
    const float* __restrict__ frame, const int* __restrict__ widths,
    ST* __restrict__ latch,
    const __nv_bfloat16* __restrict__ kr, const __nv_bfloat16* __restrict__ vr,
    const float* __restrict__ prefix_r, const float* __restrict__ beta_r, long RS,
    const __nv_bfloat16* __restrict__ q_in, long QS, __nv_bfloat16* __restrict__ out, float scale,
    float* __restrict__ scratch)
{
    constexpr bool BLOCKED = FLUSH && (NBLK > 1);                  // W > 16: blocked WY over the shared-memory state
    using SM = Smem<MMAX, BLOCKED>;
    constexpr int STP = SM::STP;
    extern __shared__ __align__(128) unsigned char k1sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    if (m <= 0) return;
    const int vg = MMAX == 16 ? value_row(g) : g;
    const int slot = slots[row];
    const long page = page_indices[row];
    float* sp = state + page * S0 + (long)h * S1;
    __nv_bfloat16* sAh = (__nv_bfloat16*)(k1sm + SM::A_OFF); __nv_bfloat16* sAl = sAh + 16 * KPB;
    __nv_bfloat16* sCh = (__nv_bfloat16*)(k1sm + SM::C_OFF); __nv_bfloat16* sCl = sCh + 16 * KPB;
    __nv_bfloat16* sTh = (__nv_bfloat16*)(k1sm + SM::T_OFF); __nv_bfloat16* sTl = sTh + 16 * STP;
    __nv_bfloat16* sOh = (__nv_bfloat16*)(k1sm + SM::O_OFF); __nv_bfloat16* sOl = sOh + (MMAX == 16 ? 8 : MMAX) * KPB;
    __nv_bfloat16* sV = (__nv_bfloat16*)(k1sm + SM::V_OFF);
    float* sQ = (float*)(k1sm + SM::Q_OFF);
    float* sPW = (float*)(k1sm + SM::PW_OFF);
    float* sO8 = (float*)(k1sm + SM::O4_OFF);
    float* sU4 = MMAX == 16 ? scratch + sidx*SCRATCH_F : (float*)(k1sm + SM::U4_OFF);
    float* sL = (float*)(k1sm + SM::L_OFF);
    float* tiles = (float*)(k1sm + SM::TILE_OFF);
    float* tile = tiles + warp * (SM::TILE_BYTES / 4);
    const unsigned tile_u = smem_u32(tile);
    const long ring = (long)h * KDA_W, rp = page * RS;          // ring row 0 of head h in the page's rings
    const int mt_n = (m + 7) >> 3;
    const float* om = frame + (long)h * 34880;                  // Omega rows [g][k]

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if(lane==0) {
        if constexpr(MMAX==16 && !BLOCKED) tma_load_half(&input_map,tile_u,state_bar,(int)((page*S0+(long)h*S1)/128+32*warp),0);
        else tma_load_tile(&input_map,tile_u,state_bar,(int)((page*S0+(long)h*S1+(long)(2*warp)*2048)/32));
    }

    // ── prologue ──
    // Omega rows (bf16 hi/lo), 8 keys per thread-iteration; rows m..8 mt_n zero-filled (once per head: items are head-major)
    const int rows16 = ((mt_n + 1) & ~1) * 8;                 // Omega rows staged (zero-filled up to a multiple of 16)
    if (stage_omega) {
        const auto* packed=reinterpret_cast<const __nv_bfloat16*>(om+16384);
        if constexpr(MMAX==16) {
            if(mt_n>1) for(int i=t;i<8*KPB/8;i+=NTHREADS) {
                cp16(smem_u32(sOh+8*i),packed+8*KPB+8*i);
                cp16(smem_u32(sOl+8*i),packed+128*KPB+8*KPB+8*i);
            }
        } else {
        for(int i=t; i<rows16*KPB/8; i+=NTHREADS) {
            cp16(smem_u32(sOh+8*i),packed+8*i);
            cp16(smem_u32(sOl+8*i),packed+128*KPB+8*i);
        }
        }
        const float* piv=om+33792;
        for(int i=t; i<8*O4P/4; i+=NTHREADS)
            cp16(smem_u32(sO8+4*i),piv+4*i);
        cp_commit();
    }
    // W > 16 (BLOCKED): the window's 16-row blocks b in order; each block's operand tables (sA, sC, sV, sT, sPW) are built
    // in the same places, in coordinates rebased to ref_b = prefix row 16 b - 1 (0 for b = 0), and every block but the
    // last updates the shared-memory state (prepass); the last block runs the fused update / output / statistics below
    int b = 0;
    do {
    const long rb = ring + 16 * b;                              // ring row 0 of block b
    if (FLUSH && !(K1_ABLATE & 1)) {
        // v ring -> sV (cp.async, 4 KB); q (warp 0); P_W (warp 1)
        {
            const __nv_bfloat16* pv = page_ring(vr, rp) + rb * 128;
            for (int i = t; i < 256; i += NTHREADS) cp16(smem_u32(sV) + ((i >> 4) * KPB + 8 * (i & 15)) * 2, pv + 8 * i);
            cp_commit();
        }
        if (warp == 0 && b == 0) {
            const __nv_bfloat16* pq = q_in + (long)row * QS + h * 128 + 4 * lane;   // QS: q's token row stride
            const uint2 u = *(const uint2*)pq;
            const float4 v = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
            const float qf = rsqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) + 1.e-6f) * scale;
            *(float4*)(sQ + 4 * lane) = make_float4(v.x * qf, v.y * qf, v.z * qf, v.w * qf);
        } else if (warp == 1) {
            const float4 p = *(const float4*)(page_ring(prefix_r, rp) + (rb + 15) * 128 + 4 * lane);
            if constexpr (BLOCKED) {                               // P_W of block b: exp(prefix_{16 b + 15} - ref_b)
                float4 r = make_float4(0.f, 0.f, 0.f, 0.f);
                if (b > 0) r = *(const float4*)(page_ring(prefix_r, rp) + (rb - 1) * 128 + 4 * lane);
                *(float4*)(sPW + 4 * lane) = make_float4(ex2_ftz((p.x - r.x) * 1.4426950408889634f), ex2_ftz((p.y - r.y) * 1.4426950408889634f),
                                                         ex2_ftz((p.z - r.z) * 1.4426950408889634f), ex2_ftz((p.w - r.w) * 1.4426950408889634f));
            } else
            *(float4*)(sPW + 4 * lane) = make_float4(ex2_ftz(p.x * 1.4426950408889634f), ex2_ftz(p.y * 1.4426950408889634f), ex2_ftz(p.z * 1.4426950408889634f), ex2_ftz(p.w * 1.4426950408889634f));
        }
        // decayed keys: thread (tr = ring row, c8 = 16-key chunk): a = k^ exp(prefix), c = k^ exp(-prefix) -> sA, sC (hi/lo, 16-B stores)
        {
            // warp = 4 ring rows x 8 chunks; a quarter-warp = 4 consecutive rows x chunks {a, a + 2}: its eight 16-B
            // stores land on distinct bank slots (row pitch 272 B); the row sum reduces over lanes xor 4, 8, 16
            const int kq = lane >> 2;
            const int tr = 4 * warp + (lane & 3), c8 = ((kq >> 1) & 1) + 4 * (kq >> 2) + 2 * (kq & 1);
            const __nv_bfloat16* pk = page_ring(kr, rp) + (rb + tr) * 128 + 16 * c8;
            const float* pp = page_ring(prefix_r, rp) + (rb + tr) * 128 + 16 * c8;
            float kk[16], pr[16];
            #pragma unroll
            for (int i = 0; i < 16; i += 4) {
                const uint2 u = *(const uint2*)(pk + i);
                kk[i] = __uint_as_float(u.x << 16); kk[i + 1] = __uint_as_float(u.x & 0xffff0000u);
                kk[i + 2] = __uint_as_float(u.y << 16); kk[i + 3] = __uint_as_float(u.y & 0xffff0000u);
                const float4 p4 = *(const float4*)(pp + i);
                pr[i] = p4.x; pr[i + 1] = p4.y; pr[i + 2] = p4.z; pr[i + 3] = p4.w;
            }
            if constexpr (BLOCKED) {                               // block b: a = k^ exp(prefix - ref_b), c = k^ exp(ref_b - prefix)
                if (b > 0) {
                    const float* pf = page_ring(prefix_r, rp) + (rb - 1) * 128 + 16 * c8;
                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        const float4 r4 = *(const float4*)(pf + i);
                        pr[i] -= r4.x; pr[i + 1] -= r4.y; pr[i + 2] -= r4.z; pr[i + 3] -= r4.w;
                    }
                }
            }
            float ss = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) ss = fmaf(kk[i], kk[i], ss);
            ss += __shfl_xor_sync(FULL, ss, 4); ss += __shfl_xor_sync(FULL, ss, 8); ss += __shfl_xor_sync(FULL, ss, 16);
            const float inv = rsqrtf(ss + 1.e-6f);
            float av[16], cv[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) { const float kn = kk[i] * inv, t = pr[i] * 1.4426950408889634f; av[i] = kn * ex2_ftz(t); cv[i] = kn * ex2_ftz(-t); }
            store16_hilo(sAh + tr * KPB + 16 * c8, sAl + tr * KPB + 16 * c8, av);
            store16_hilo(sCh + tr * KPB + 16 * c8, sCl + tr * KPB + 16 * c8, cv);
        }
        __syncthreads();
        // L = A C^T (16 x 16, three-term bf16 MMAs, warp 0): L[i][j] = beta_i <a_i, c_j> for j < i
        if (warp == 0) {
            float L0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, L1[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};   // k-step parity chains
            // ldmatrix lane addresses: tiles (rows 0-7, +0), (rows 8-15, +0), (rows 0-7, +8), (rows 8-15, +8) for A;
            // for B (non-trans, rows = j): (j 0-7, +0), (j 0-7, +8), (j 8-15, +0), (j 8-15, +8)
            const int lr = lane & 7, lt = lane >> 3;
            const unsigned a_h = smem_u32(sAh + (lr + 8 * (lt & 1)) * KPB + 8 * (lt >> 1)), a_l = smem_u32(sAl + (lr + 8 * (lt & 1)) * KPB + 8 * (lt >> 1));
            const unsigned b_h = smem_u32(sCh + (lr + 8 * (lt >> 1)) * KPB + 8 * (lt & 1)), b_l = smem_u32(sCl + (lr + 8 * (lt >> 1)) * KPB + 8 * (lt & 1));
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                unsigned ah[4], al[4], bh[4], bl[4];
                ldsm4(ah, a_h + 32 * q); ldsm4(al, a_l + 32 * q); ldsm4(bh, b_h + 32 * q); ldsm4(bl, b_l + 32 * q);
                float (*Lq)[4] = (q & 1) ? L1 : L0;
                mma3(Lq[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
                mma3(Lq[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) { L0[0][e] += L1[0][e]; L0[1][e] += L1[1][e]; }
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = g + 8 * (e >> 1), j = 8 * nt2 + 2 * c + (e & 1);
                    sL[i * 16 + j] = (j < i) ? page_ring(beta_r, rp)[rb + i] * L0[nt2][e] : 0.f;
                }
        }
        __syncthreads();
        // T = (I + L)^-1: lane l solves (I + L) t = e_l (column l, forward substitution); T''[l][i] = beta_l T[i][l] = beta_l t_i
        if (warp == 0) {
            float s[16];
            const int l = lane & 15;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                float v = (i == l) ? 1.f : 0.f, w = 0.f;
                #pragma unroll
                for (int j = 0; j < i; j += 2) v = fmaf(-sL[i * 16 + j], s[j], v);
                #pragma unroll
                for (int j = 1; j < i; j += 2) w = fmaf(-sL[i * 16 + j], s[j], w);
                s[i] = v + w;
            }
            if (lane < 16) {
                const float bl = page_ring(beta_r, rp)[rb + l];
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float x = bl * s[i];
                    const __nv_bfloat16 hh = __float2bfloat16_rn(x);
                    sTh[i * STP + l] = hh; sTl[i * STP + l] = __float2bfloat16_rn(x - __bfloat162float(hh));
                }
            }
        }
        cp_wait_all();
    }
    if constexpr (BLOCKED) {
        if (b + 1 < NBLK) {
            cp_wait_all();
            __syncthreads();
            prepass_block<MMAX>(b == 0, warp, lane, tile, tile_u, input_map, state_bar, page, S0, S1, h,
                                sAh, sAl, sCh, sCl, sTh, sTl, sV, sPW);
            __syncthreads();
        }
    }
    } while (BLOCKED && ++b < NBLK);
    cp_wait_all();
    __syncthreads();

    // ── block loop: two 16-value blocks per warp ──
    float y4_full[MMAX == 16 ? 1 : 8][4] = {};
    constexpr bool USE_W = (MMAX == 16);                         // W^T = U^T U4 (g x 4) for the finish's z_j[g] = sum_i C[j][i] W[i][g]
    float wt[USE_W ? 1 : 1][4] = {{0.f, 0.f, 0.f, 0.f}};          // (larger buckets keep the finish's Omega . zr loop: registers)
    float en[MMAX / 8][2];
    #pragma unroll
    for (int i = 0; i < MMAX / 8; ++i) { en[i][0] = 0.f; en[i][1] = 0.f; }
    float sumsq = 0.f;
    // ldmatrix lane addresses (this lane): P / U B fragments: tiles (rows 0-7, +0), (rows 0-7, +8), (rows 8-15, +0), (rows 8-15, +8)
    const int lr = lane & 7, lt = lane >> 3;
    const int ld_row = lr + 8 * (lt >> 1), ld_col = 8 * (lt & 1);
    const unsigned pa_h = smem_u32(sAh + ld_row * KPB + ld_col), pa_l = smem_u32(sAl + ld_row * KPB + ld_col);
    const unsigned po_h = smem_u32(sOh + ld_row * KPB + ld_col), po_l = smem_u32(sOl + ld_row * KPB + ld_col);
    // update B fragments (trans): tiles hi (w 0-7), hi (w 8-15), lo (w 0-7), lo (w 8-15) at column 8 nt
    const unsigned pc = smem_u32(((lt & 2) ? sCl : sCh) + (lr + 8 * (lt & 1)) * KPB);
    // v ring A fragments (trans): tiles (j 0-7, v0), (j 0-7, v0 + 8), (j 8-15, v0), (j 8-15, v0 + 8)
    const unsigned pv = smem_u32(sV + (lr + 8 * (lt >> 1)) * KPB + 8 * (lt & 1));
    if constexpr (!BLOCKED) {
    pf_l2((const char*)(sp + (2 * warp + 1) * 16 * 128) + lane * 256); pf_l2((const char*)(sp + (2 * warp + 1) * 16 * 128) + lane * 256 + 128);
    }
    #pragma unroll 1
    for (int bi = 0; bi < 2; ++bi) {

        const int vb = 2 * warp + bi, v0 = 16 * vb;
        float acc[16][4];
        float P[2][4] = {};
        float y4_local[MMAX == 16 ? 8 : 1][4] = {};
        float (*y4)[4] = MMAX == 16 ? y4_local : y4_full;
        if constexpr(BLOCKED) {                                     // W > 16: the state tile after blocks 0 .. NBLK - 2
            load_tile(acc, tile + bi * (TILE_B / 4), vg, c);
            if constexpr(!(K1_ABLATE&2)) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned ah[4], al[4], bh[4], bl[4];
                c2a_bf16(acc[2 * j], acc[2 * j + 1], ah, al);
                ldsm4(bh, pa_h + 32 * j); ldsm4(bl, pa_l + 32 * j);
                mma3(P[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
                mma3(P[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            }
            }
        } else if constexpr(MMAX==16) {
            tma_wait_tile(state_bar,0); __syncwarp();
            load_half<0>(acc,tile,vg,c);
            // Order generic shared reads before the async proxy overwrites this
            // tile. A warp barrier alone does not provide cross-proxy ordering.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if(lane==0) tma_load_half(&input_map,tile_u,state_bar,(int)((page*S0+(long)h*S1)/128+v0),2);
            if constexpr(FLUSH && !(K1_ABLATE&2)) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned ah[4], al[4], bh[4], bl[4];
                c2a_bf16(acc[2 * j], acc[2 * j + 1], ah, al);
                ldsm4(bh, pa_h + 32 * j); ldsm4(bl, pa_l + 32 * j);
                mma3(P[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
                mma3(P[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            }

            }
            tma_wait_tile(state_bar,1); __syncwarp();
            load_half<1>(acc,tile,vg,c);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if(bi==0 && lane==0) tma_load_half(&input_map,tile_u,state_bar,(int)((page*S0+(long)h*S1)/128+v0+16),0);
            if constexpr(FLUSH && !(K1_ABLATE&2)) {
            #pragma unroll
            for (int j = 4; j < 8; ++j) {
                unsigned ah[4], al[4], bh[4], bl[4];
                c2a_bf16(acc[2 * j], acc[2 * j + 1], ah, al);
                ldsm4(bh, pa_h + 32 * j); ldsm4(bl, pa_l + 32 * j);
                mma3(P[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
                mma3(P[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            }

            }
        } else {
            tma_wait_tile(state_bar,bi); __syncwarp();
            load_tile(acc,tile,g,c);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if(bi==0 && lane==0) tma_load_tile(&input_map,tile_u,state_bar,(int)((page*S0+(long)h*S1+(long)(vb+1)*2048)/32));
            if constexpr(FLUSH && !(K1_ABLATE&2)) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned ah[4], al[4], bh[4], bl[4];
                c2a_bf16(acc[2 * j], acc[2 * j + 1], ah, al);
                ldsm4(bh, pa_h + 32 * j); ldsm4(bl, pa_l + 32 * j);
                mma3(P[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
                mma3(P[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            }

            }
        }
        float u4[4]={0.f,0.f,0.f,0.f};
        if (FLUSH && !(K1_ABLATE & 2)) {
            // X = V T'' - P T''  (v x i)
            float X[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            {
                unsigned av[4], ph[4], pl[4];
                ldsm4t(av, pv + 2 * v0);
                if constexpr(MMAX==16) {
                    #pragma unroll
                    for(int e=0;e<4;++e) av[e]=__shfl_sync(FULL,av[e],4*vg+c);
                }
                c2a_bf16(P[0], P[1], ph, pl);
                #pragma unroll
                for (int nt2 = 0; nt2 < 2; ++nt2) {
                    const int i = 8 * nt2 + g;
                    const unsigned th0 = *(const unsigned*)(sTh + i * STP + 2 * c), th1 = *(const unsigned*)(sTh + i * STP + 2 * c + 8);
                    const unsigned tl0 = *(const unsigned*)(sTl + i * STP + 2 * c), tl1 = *(const unsigned*)(sTl + i * STP + 2 * c + 8);
                    mma16816_bf16(X[nt2], av, th0, th1); mma16816_bf16(X[nt2], av, tl0, tl1);
                    float Y[4] = {0.f, 0.f, 0.f, 0.f};
                    mma3(Y, ph, pl, th0, th1, tl0, tl1);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) X[nt2][e] -= Y[e];
                }
            }

            // Delta -> exact output/norm/U4 -> BF16 working representation.
            if constexpr(MMAX==16) {
                unsigned xh[4],xl[4];c2a_bf16(X[0],X[1],xh,xl);
                float so0=0.f,so1=0.f;
                #pragma unroll
                for(int j=0;j<8;++j) {
                    #pragma unroll
                    for(int p=0;p<2;++p) {
                        const int nt=2*j+p;
                        unsigned b[4];ldsm4t(b,pc+16*nt);
                        mma3(acc[nt],xh,xl,b[0],b[1],b[2],b[3]);
                        const float2 pw=*(const float2*)(sPW+16*j+4*c+2*p);
                        acc[nt][0]*=pw.x;acc[nt][1]*=pw.y;acc[nt][2]*=pw.x;acc[nt][3]*=pw.y;
                        const float2 qq=*(const float2*)(sQ+16*j+4*c+2*p);
                        so0=fmaf(acc[nt][0],qq.x,fmaf(acc[nt][1],qq.y,so0));
                        so1=fmaf(acc[nt][2],qq.x,fmaf(acc[nt][3],qq.y,so1));
                        if constexpr(!(K1_ABLATE&4)) {

                    #pragma unroll
                    for(int e=0;e<4;++e) sumsq=fmaf(acc[nt][e],acc[nt][e],sumsq);
                    float ah[4],al[4];
                    #pragma unroll
                    for(int e=0;e<4;++e) {ah[e]=tf32_hi(acc[nt][e]);al[e]=acc[nt][e]-ah[e];}
                    const float2 b=*(const float2*)(sO8+g*O4P+8*nt+2*c);
                    const float2 bh=make_float2(tf32_hi(b.x),tf32_hi(b.y));
                    const float2 bl=make_float2(b.x-bh.x,b.y-bh.y);
                    mma_tf32_c(u4,ah,bh.x,bh.y);mma_tf32_c(u4,ah,bl.x,bl.y);mma_tf32_c(u4,al,bh.x,bh.y);

                        }
                    }
                    #pragma unroll
                    for(int hh=0;hh<2;++hh) {
                        float* dst=sp+(long)(v0+vg+8*hh)*128+16*j+4*c;
                        *(float4*)dst=make_float4(acc[2*j][2*hh],acc[2*j][2*hh+1],acc[2*j+1][2*hh],acc[2*j+1][2*hh+1]);
                    }

            #pragma unroll
            for(int p=0;p<2;++p) {
                const int nt=2*j+p;
                unsigned h0,l0,h1,l1;
                split2(acc[nt][0],acc[nt][1],h0,l0);
                split2(acc[nt][2],acc[nt][3],h1,l1);
                acc[nt][0]=__uint_as_float(h0);acc[nt][1]=__uint_as_float(l0);
                acc[nt][2]=__uint_as_float(h1);acc[nt][3]=__uint_as_float(l1);
            }

                }
                so0+=__shfl_xor_sync(FULL,so0,1);so0+=__shfl_xor_sync(FULL,so0,2);
                so1+=__shfl_xor_sync(FULL,so1,1);so1+=__shfl_xor_sync(FULL,so1,2);
                if(c==0) {
                    auto* po=out+((long)row*H+h)*128+v0+vg;
                    po[0]=__float2bfloat16_rn(so0);po[8]=__float2bfloat16_rn(so1);
                }
            } else {
            // S' = (S + X C) * P_W
            unsigned xh[4], xl[4];
            c2a_bf16(X[0], X[1], xh, xl);
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                unsigned b[4];
                ldsm4t(b, pc + 16 * nt);
                mma3(acc[nt], xh, xl, b[0], b[1], b[2], b[3]);
                const float2 pw = *(const float2*)(sPW + 16 * (nt >> 1) + 4 * c + 2 * (nt & 1));
                acc[nt][0] *= pw.x; acc[nt][1] *= pw.y; acc[nt][2] *= pw.x; acc[nt][3] *= pw.y;
            }
            store_tile(acc, sp + (long)v0 * 128, vg, c);
            {   // exact output
                float s0 = 0.f, s1 = 0.f;
                #pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    const float2 qq = *(const float2*)(sQ + 16 * (nt >> 1) + 4 * c + 2 * (nt & 1));
                    s0 = fmaf(acc[nt][0], qq.x, fmaf(acc[nt][1], qq.y, s0));
                    s1 = fmaf(acc[nt][2], qq.x, fmaf(acc[nt][3], qq.y, s1));
                }
                s0 += __shfl_xor_sync(FULL, s0, 1); s0 += __shfl_xor_sync(FULL, s0, 2);
                s1 += __shfl_xor_sync(FULL, s1, 1); s1 += __shfl_xor_sync(FULL, s1, 2);
                if (c == 0) {
                    __nv_bfloat16* po = out + ((long)row * H + h) * 128 + v0 + vg;
                    po[0] = __float2bfloat16_rn(s0); po[8] = __float2bfloat16_rn(s1);
                }
            }
            }
        }
        if (K1_ABLATE & 4) continue;
        // Fused low-rank flush already performed these exact reductions.
        if constexpr(MMAX!=16 || !FLUSH || (K1_ABLATE&2)) {
        // ── statistics ──
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) sumsq = fmaf(acc[nt][e], acc[nt][e], sumsq);
        // pivot columns U4 = S' Omega[:, :4] on tf32 hi/lo MMAs (three terms): fp32-accurate for the dependence test

        #pragma unroll
        for (int q = 0; q < 16; ++q) {
            float ah[4], al[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) { ah[e] = tf32_hi(acc[q][e]); al[e] = acc[q][e] - ah[e]; }
            const float2 b = *(const float2*)(sO8 + g * O4P + 8 * q + 2 * c);
            const float2 bh=make_float2(tf32_hi(b.x),tf32_hi(b.y));
            const float2 bl=make_float2(b.x-bh.x,b.y-bh.y);
            mma_tf32_c(u4, ah, bh.x, bh.y); mma_tf32_c(u4, ah, bl.x, bl.y); mma_tf32_c(u4, al, bh.x, bh.y);
        }

            if constexpr(MMAX==16) {
                #pragma unroll
                for(int j=0;j<8;++j) {

            #pragma unroll
            for(int p=0;p<2;++p) {
                const int nt=2*j+p;
                unsigned h0,l0,h1,l1;
                split2(acc[nt][0],acc[nt][1],h0,l0);
                split2(acc[nt][2],acc[nt][3],h1,l1);
                acc[nt][0]=__uint_as_float(h0);acc[nt][1]=__uint_as_float(l0);
                acc[nt][2]=__uint_as_float(h1);acc[nt][3]=__uint_as_float(l1);
            }

                }
            }
        }
        // u4[e]: U4[v = g + 8 (e >> 1)][j = 2c + (e & 1)] (lanes c < 2 hold the four pivot columns)
        unsigned u4h[2], u4l[2];
        {
            unsigned h0, l0, h1, l1;
            split2(u4[0], u4[1], h0, l0); split2(u4[2], u4[3], h1, l1);
            u4h[0] = movtrans(h0); u4l[0] = movtrans(l0); u4h[1] = movtrans(h1); u4l[1] = movtrans(l1);
            if (c < 2) {
                sU4[(2 * c) * 128 + v0 + vg] = u4[0]; sU4[(2 * c + 1) * 128 + v0 + vg] = u4[1];
                sU4[(2 * c) * 128 + v0 + vg + 8] = u4[2]; sU4[(2 * c + 1) * 128 + v0 + vg + 8] = u4[3];
            }
        }
        // U = S' Omega (bf16 hi/lo, three terms) in chunks of <= 4 n-tiles; Y4^T += S'^T U4 merged into the first chunk's k loop
        ST* pu = latch + ((long)slot * H + h) * (long)G * 128 + v0 + vg;
        constexpr int NTC = (MMAX / 8 < 4) ? MMAX / 8 : 4;      // n-tiles per chunk (even: one ldmatrix.x4 covers two n-tiles)
        #pragma unroll 1
        for (int nc = 0; nc < MMAX / 8; nc += NTC) {
            if (nc >= mt_n) break;
            float U[NTC][4];
            #pragma unroll
            for (int i = 0; i < NTC; ++i) { U[i][0] = 0.f; U[i][1] = 0.f; U[i][2] = 0.f; U[i][3] = 0.f; }
            if(nc==0) {
                #pragma unroll
                for(int e=0;e<4;++e) U[0][e]=u4[e];
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned sh[4], sl[4];
                if constexpr(MMAX==16) {
                    sh[0]=__float_as_uint(acc[2*j][0]);sh[1]=__float_as_uint(acc[2*j][2]);
                    sh[2]=__float_as_uint(acc[2*j+1][0]);sh[3]=__float_as_uint(acc[2*j+1][2]);
                    sl[0]=__float_as_uint(acc[2*j][1]);sl[1]=__float_as_uint(acc[2*j][3]);
                    sl[2]=__float_as_uint(acc[2*j+1][1]);sl[3]=__float_as_uint(acc[2*j+1][3]);
                } else {
                c2a_bf16(acc[2 * j], acc[2 * j + 1], sh, sl);
                }
                #pragma unroll
                for (int i = 0; i < NTC; i += 2) {
                    const int nt2 = nc + i;                      // rows 8 nt2 .. 8 nt2 + 15 of Omega (two n-tiles)
                    if (nt2 < mt_n) {
                        unsigned bh[4], bl[4];
                        const unsigned off = (unsigned)(nt2 * 8 * KPB * 2 + 32 * j);
                        if(nt2==0) {
                            if(mt_n>1) {
                                ldsm2(bh,po_h+off+(MMAX==16?0:8*KPB*2)); ldsm2(bl,po_l+off+(MMAX==16?0:8*KPB*2));
                                mma3(U[i+1],sh,sl,bh[0],bh[1],bl[0],bl[1]);
                            }
                        } else {
                            if (nt2 + 1 < mt_n) { ldsm4(bh, po_h + off); ldsm4(bl, po_l + off); }
                            else { ldsm2(bh, po_h + off); ldsm2(bl, po_l + off); }
                            mma3(U[i], sh, sl, bh[0], bh[1], bl[0], bl[1]);
                            if (nt2 + 1 < mt_n) mma3(U[i + 1], sh, sl, bh[2], bh[3], bl[2], bl[3]);
                        }
                    }
                }
                if (nc == 0) {                                   // Y4^T (k x j) += S'^T (k x v) U4 (v x j), m-tile j
                    unsigned ah[4], al[4];
                    ah[0] = movtrans(sh[0]); ah[1] = movtrans(sh[2]); ah[2] = movtrans(sh[1]); ah[3] = movtrans(sh[3]);
                    al[0] = movtrans(sl[0]); al[1] = movtrans(sl[2]); al[2] = movtrans(sl[1]); al[3] = movtrans(sl[3]);
                    mma3(y4[j], ah, al, u4h[0], u4h[1], u4l[0], u4l[1]);
                }
            }
            #pragma unroll
            for (int i = 0; i < NTC; ++i) {
                const int nt2 = nc + i;
                if (nt2 >= mt_n) break;
                if (nt2 == 0 && c < 2) { U[i][0] = u4[0]; U[i][1] = u4[1]; U[i][2] = u4[2]; U[i][3] = u4[3]; }
                en[nt2][0] = fmaf(U[i][0], U[i][0], fmaf(U[i][2], U[i][2], en[nt2][0]));
                en[nt2][1] = fmaf(U[i][1], U[i][1], fmaf(U[i][3], U[i][3], en[nt2][1]));
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int gcol = 8 * nt2 + 2 * c + (e & 1);
                    if (gcol < m) st_f<ST>(pu + (long)gcol * 128 + 8 * (e >> 1), U[i][e]);
                }
            }
            if (USE_W && nc == 0) {                              // W^T (g x i') += U^T (g x v) U4 (v x i'), ranks 0..15
                unsigned sh[4], sl[4], ah[4], al[4];
                c2a_bf16(U[0], U[1 < NTC ? 1 : 0], sh, sl);      // U[1] is zero when its n-tile is beyond m
                ah[0] = movtrans(sh[0]); ah[1] = movtrans(sh[2]); ah[2] = movtrans(sh[1]); ah[3] = movtrans(sh[3]);
                al[0] = movtrans(sl[0]); al[1] = movtrans(sl[2]); al[2] = movtrans(sl[1]); al[3] = movtrans(sl[3]);
                mma3(wt[0], ah, al, u4h[0], u4h[1], u4l[0], u4l[1]);
            }
        }
        if constexpr(MMAX==16 && !(K1_ABLATE&4)) {
            if(c<2) {
                #pragma unroll
                for(int mt=0;mt<8;++mt) {
                    float* tmp=sL+warp*512+(mt*16+g*2+c)*4;
                    if(bi==0) spill_y4(tmp,y4[mt][0],y4[mt][1],y4[mt][2],y4[mt][3]);
                    else {
                        const float4 old=reload_y4(tmp);
                        y4[mt][0]+=old.x; y4[mt][1]+=old.y; y4[mt][2]+=old.z; y4[mt][3]+=old.w;
                        #pragma unroll
                        for(int e=0;e<4;++e) tile[(2*c+(e&1))*128+16*mt+KEY(g+8*(e>>1))]=y4[mt][e];
                    }
                }
            }
        }
    }
    if (K1_ABLATE & 4) return;
    // ── warp partials -> this warp's tile, summed in warp order (deterministic), then the per-item scratch ──
    {
        float* part_y = tile; float* part_e = tile + 512;
        const float ws = warp_sum(sumsq);
        if (lane == 0) tile[640] = ws;
        for (int i = lane; i < 128; i += 32) part_e[i] = 0.f;
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < MMAX / 8; ++i)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = en[i][e];
                v += __shfl_xor_sync(FULL, v, 4); v += __shfl_xor_sync(FULL, v, 8); v += __shfl_xor_sync(FULL, v, 16);
                if (g == 0) part_e[8 * i + 2 * c + e] = v;
            }
        if (c < 2) {
            #pragma unroll
            if constexpr(MMAX!=16) {
                #pragma unroll
                for (int mt = 0; mt < 8; ++mt)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) part_y[(2 * c + (e & 1)) * 128 + 16 * mt + KEY(g + 8 * (e >> 1))] = y4_full[mt][e];
            }
            if (USE_W) {
                float* part_w = tile + 704;                     // W [4 i'][128 g]
                #pragma unroll
                for (int e = 0; e < 4; ++e) part_w[(2 * c + (e & 1)) * 16 + g + 8 * (e >> 1)] = wt[0][e];
            }
        }
    }
    __syncthreads();
    float* sc = scratch + sidx * SCRATCH_F;
    for (int i = t; i < 4 * 128; i += NTHREADS) {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) v += tiles[w * (SM::TILE_BYTES / 4) + i];
        sc[512 + i] = v;
        if constexpr(MMAX!=16) sc[i] = sU4[i];
        if (USE_W && (i & 127) < 16) {
            float w4 = 0.f;
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) w4 += tiles[w * (SM::TILE_BYTES / 4) + 704 + 16*(i>>7) + (i&127)];
            sc[1280 + i] = w4;
        }
    }
    {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) v += tiles[w * (SM::TILE_BYTES / 4) + 512 + t];
        sc[1024 + t] = (t < m) ? v : 0.f;
        if (t == 0) { float ms = 0.f; for (int w = 0; w < NWARPS; ++w) ms += tiles[w * (SM::TILE_BYTES / 4) + 640]; sc[1152] = ms; }
    }
}

struct HeadTable { int packed[128]; };
inline HeadTable host_head_table(torch::Tensor heads) {
    TORCH_CHECK(heads.device().is_cpu() && heads.scalar_type()==torch::kInt32 && heads.is_contiguous() && heads.numel()<=128,
                "head metadata must be an immutable CPU int32 vector of length<=128");
    HeadTable result{};
    for(int i=0;i<heads.numel();++i) result.packed[i]=heads.data_ptr<int>()[i];
    return result;
}

template <int MMAX, typename ST, bool FLUSH>
__global__ void __launch_bounds__(NTHREADS, K1_MINB_LB(MMAX, FLUSH && (NBLK > 1)))
kda_flush_main(int NH, int head_base, int NHall, const __grid_constant__ CUtensorMap input_map,
    float* __restrict__ state, long S0, long S1,
    const int* __restrict__ rows, int n_rows,
    const int* __restrict__ slots, const int* __restrict__ meta,
    const __grid_constant__ HeadTable head_table, int H, int G,
    const float* __restrict__ frame, const int* __restrict__ widths,
    ST* __restrict__ latch,
    const __nv_bfloat16* __restrict__ kr, const __nv_bfloat16* __restrict__ vr,
    const float* __restrict__ prefix_r, const float* __restrict__ beta_r, long RS,
    const __nv_bfloat16* __restrict__ q_in, long QS, __nv_bfloat16* __restrict__ out, float scale,
    float* __restrict__ scratch)
{
    __shared__ __align__(8) unsigned long long state_bars[NWARPS];
    if(threadIdx.x<NWARPS) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&state_bars[threadIdx.x])) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    // this launch's phase: FLUSH -> the rows at position 15, else the rows of a cold build; `rows` lists them, then
    // -1 padding (a CTA's items only increase, so it stops at the first -1); scratch index = the finish's item order
    const int total = n_rows * NH;
    // interleaved (item, head) order: at any time the grid works on a few consecutive rows (consecutive state pages) and
    // every CTA sees the same head mix (head-major orders lost 3-5%: scattered pages / per-head load imbalance)
    for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
        __syncthreads();                                         // shared memory of the previous item is free
        const int item = idx / NH, hy = idx % NH;
        const int info = head_table.packed[hy], h = info & 65535, m = info >> 16;
        const int row = rows[item];
        if (row < 0) break;
        const long sidx = (long)item * NHall + head_base + hy;
        // the item's sketch row is meta[row], its state page and rings slots[row]
        kda_flush_item<MMAX, ST, FLUSH>(row, h, m, sidx, true, input_map, smem_u32(&state_bars[threadIdx.x>>5]), state, S0, S1, meta, slots, H, G, frame, widths,
                                        latch, kr, vr, prefix_r, beta_r, RS, q_in, QS, out, scale, scratch);
    }
}

// ── finish: one warp per item; lane holds keys 4 lane .. 4 lane + 3 of every 128-vector ──
template <typename ST>
__global__ void __launch_bounds__(128, K2_MINB)
kda_flush_finish(
    const int* __restrict__ rows, int n_rows,
    const int* __restrict__ meta, const __grid_constant__ HeadTable head_table, int NH, int H, int G,
    const float* __restrict__ frame, const int* __restrict__ widths,
    ST* __restrict__ phi, ST* __restrict__ fr, const float* __restrict__ scratch)
{
    constexpr int MMAX = 128;
    __shared__ float sZR[4][4][128];                             // per warp: exact-Z rows
    __shared__ float sTab[4][6][MMAX];                           // per warp: z_j (4), factor - 1, a
    __shared__ float sGm[4][4][MMAX];                            // per warp: g_j
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int total = n_rows * NH;
    for (int idx = blockIdx.x * 4 + warp; idx < total; idx += gridDim.x * 4) {
    __syncwarp();
    const int item = idx / NH, hy = idx % NH;                    // scratch index = idx
    const int row = rows[item];
    if (row < 0) break;
    const int info = head_table.packed[hy], h = info & 65535, m = info >> 16;
    if (m <= 0) continue;
    const int slot = meta[row];
    {   // the window restarts: zero the (slot, head) f history so the step's entries t >= pos are exact zeros
        uint4* f4 = (uint4*)(fr + ((long)slot * H + h) * (long)G * KDA_W);
        for (int i = lane; i < G * KDA_W * (int)sizeof(ST) / 16; i += 32) f4[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    const float* sc = scratch + (long)idx * SCRATCH_F;
    const float* om = frame + (long)h * 34880;
    float u[4][4], v[4][4], C[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 x = *(const float4*)(sc + j * 128 + 4 * lane);
        u[j][0] = x.x; u[j][1] = x.y; u[j][2] = x.z; u[j][3] = x.w;
        #pragma unroll
        for (int i = 0; i < 4; ++i) { C[j][i] = 0.f; v[j][i] = 0.f; }
    }
    const float mean = sc[1152] / 128.f;
    const float safe = (mean > 0.f) ? mean : 1.f;
    const float inv_s = rsqrtf(safe);
    // pivots: modified Gram-Schmidt (two passes) on the explicit U4 columns, recording v_j = sum_i C[j][i] u_i
    const int np = (m < 4) ? m : 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j < np) {
            float w[4] = {u[j][0], u[j][1], u[j][2], u[j][3]};
            float cw[4] = {0.f, 0.f, 0.f, 0.f}; cw[j] = 1.f;
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass)
                #pragma unroll
                for (int i = 0; i < j; ++i) {
                    const float d = warp_sum(v[i][0] * w[0] + v[i][1] * w[1] + v[i][2] * w[2] + v[i][3] * w[3]);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) { w[e] = fmaf(-v[i][e], d, w[e]); cw[e] = fmaf(-C[i][e], d, cw[e]); }
                }
            const float e2 = warp_sum(w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
            const float uu = warp_sum(u[j][0] * u[j][0] + u[j][1] * u[j][1] + u[j][2] * u[j][2] + u[j][3] * u[j][3]);
            const bool keep = (j == 0) ? (e2 > 0.f) : (e2 > 1.e-12f * uu);
            const float inv = keep ? rsqrtf(e2) : 0.f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { v[j][e] = w[e] * inv; C[j][e] = cw[e] * inv; }
        }
    }
    // zr_j = sum_i C[j][i] Y4[i] / sqrt(safe)  (lane's four keys) -> shared
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float z4[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 y = *(const float4*)(sc + 512 + i * 128 + 4 * lane);
            z4[0] = fmaf(C[j][i], y.x, z4[0]); z4[1] = fmaf(C[j][i], y.y, z4[1]); z4[2] = fmaf(C[j][i], y.z, z4[2]); z4[3] = fmaf(C[j][i], y.w, z4[3]);
        }
        *(float4*)(&sZR[warp][j][4 * lane]) = make_float4(z4[0] * inv_s, z4[1] * inv_s, z4[2] * inv_s, z4[3] * inv_s);
    }
    __syncwarp();
    // per g (lane-owned): z_j[g] = Omega[:, g] . zr_j, residual, ridge terms, partial sums for the Cholesky scalars
    float part[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) part[i] = 0.f;
    const int m32 = (m + 31) & ~31;
    for (int gg = lane; gg < m32; gg += 32) {
        float z[4] = {0.f, 0.f, 0.f, 0.f};
        if (gg < m && m <= 16) {                                 // z_j[g] = Omega[:, g] . zr_j = sum_i C[j][i] W[i][g] / sqrt(safe) (W from the main kernel)
            const float w0 = sc[1280 + gg], w1 = sc[1280 + 128 + gg], w2 = sc[1280 + 256 + gg], w3 = sc[1280 + 384 + gg];
            #pragma unroll
            for (int j = 0; j < 4; ++j) z[j] = inv_s * fmaf(C[j][0], w0, fmaf(C[j][1], w1, fmaf(C[j][2], w2, C[j][3] * w3)));
        } else if (gg < m) {
            const float* orow = om + gg * 128;
            #pragma unroll 4
            for (int k = 0; k < 128; k += 4) {
                const float4 w = *(const float4*)(orow + k);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float4 zr = *(const float4*)(&sZR[warp][j][k]);
                    z[j] = fmaf(w.x, zr.x, fmaf(w.y, zr.y, fmaf(w.z, zr.z, fmaf(w.w, zr.w, z[j]))));
                }
            }
        }
        float res = (gg < m) ? sc[1024 + gg] / safe - (z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3]) : 0.f;
        res = fmaxf(res, 0.f);
        if (gg < np) res = 0.f;
        const float den = res + 0.1f;
        const float inv_den = __frcp_rn(den);
        float b[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { b[j] = (gg < m) ? z[j] * inv_den : 0.f; sTab[warp][j][gg] = z[j]; sGm[warp][j][gg] = b[j]; }
        sTab[warp][4][gg] = (gg < m) ? (0.1f * inv_den - 1.f) : 0.f;
        sTab[warp][5][gg] = res * inv_den;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int i = 0; i <= j; ++i) { const int q = j * (j + 1) / 2 + i; part[q] = fmaf(z[j], b[i], part[q]); }
    }
    float S[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) S[i] = warp_sum(part[i]);
    __syncwarp();
    {
        const float l00 = sqrtf(1.f + S[0]);
        const float r00 = __frcp_rn(l00);
        const float l10 = S[1] * r00;
        const float l11 = sqrtf(1.f + S[2] - l10 * l10);
        const float r11 = __frcp_rn(l11);
        const float l20 = S[3] * r00;
        const float l21 = (S[4] - l20 * l10) * r11;
        const float l22 = sqrtf(1.f + S[5] - l20 * l20 - l21 * l21);
        const float r22 = __frcp_rn(l22);
        const float l30 = S[6] * r00;
        const float l31 = (S[7] - l30 * l10) * r11;
        const float l32 = (S[8] - l30 * l20 - l31 * l21) * r22;
        const float l33 = sqrtf(1.f + S[9] - l30 * l30 - l31 * l31 - l32 * l32);
        const float r33 = __frcp_rn(l33);
        for (int gg = lane; gg < m32; gg += 32) {
            const float b0 = sGm[warp][0][gg], b1 = sGm[warp][1][gg], b2 = sGm[warp][2][gg], b3 = sGm[warp][3][gg];
            const float y0 = b0 * r00;
            const float y1 = (b1 - l10 * y0) * r11;
            const float y2 = (b2 - l20 * y0 - l21 * y1) * r22;
            const float y3 = (b3 - l30 * y0 - l31 * y1 - l32 * y2) * r33;
            const float g3 = y3 * r33;
            const float g2 = (y2 - l32 * g3) * r22;
            const float g1 = (y1 - l21 * g2 - l31 * g3) * r11;
            const float g0 = (y0 - l10 * g1 - l20 * g2 - l30 * g3) * r00;
            sGm[warp][0][gg] = g0; sGm[warp][1][gg] = g1; sGm[warp][2][gg] = g2; sGm[warp][3][gg] = g3;
        }
    }
    __syncwarp();
    // native_j[k] = zr_j[k] + sum_g Omega[k][g] z_j[g] (factor[g] - 1);  phi[k][g] = Omega[k][g] a[g] + sum_j native_j[k] g_j[g]
    float nat[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 z = *(const float4*)(&sZR[warp][j][4 * lane]);
        nat[j][0] = z.x; nat[j][1] = z.y; nat[j][2] = z.z; nat[j][3] = z.w;
    }
    for (int gg = 0; gg < m; ++gg) {
        const float4 w = *(const float4*)(om + gg * 128 + 4 * lane);
        const float f = sTab[warp][4][gg];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float zf = sTab[warp][j][gg] * f;
            nat[j][0] = fmaf(w.x, zf, nat[j][0]); nat[j][1] = fmaf(w.y, zf, nat[j][1]); nat[j][2] = fmaf(w.z, zf, nat[j][2]); nat[j][3] = fmaf(w.w, zf, nat[j][3]);
        }
    }
    ST* pp = phi + ((long)slot * H + h) * (long)G * 128 + 4 * lane;
    for (int gg = 0; gg < m; ++gg) {
        const float4 w = *(const float4*)(om + gg * 128 + 4 * lane);
        const float a = sTab[warp][5][gg];
        float r[4] = {w.x * a, w.y * a, w.z * a, w.w * a};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gj = sGm[warp][j][gg];
            r[0] = fmaf(nat[j][0], gj, r[0]); r[1] = fmaf(nat[j][1], gj, r[1]); r[2] = fmaf(nat[j][2], gj, r[2]); r[3] = fmaf(nat[j][3], gj, r[3]);
        }
        st4_f<ST>(pp + (long)gg * 128, r[0], r[1], r[2], r[3]);
    }
    }
}

// ── dense heads: one CTA (128 threads) per item (row, dense head); thread v owns value row v of the state, which sits
//    in shared memory (16-byte chunks XOR-swizzled by row & 7).  FOLD (flush): the window's raw rows are applied in
//    16-row blocks b in the WY form (FP32), rebased to ref_b = prefix row 16 b - 1:
//      a_i = k^_i exp(p_i - ref_b), c_i = k^_i exp(ref_b - p_i), L[i][j] = beta_i <a_i, c_j> (j < i),
//      X = (V - S A^T) T'', T''[w][i] = beta_w (I + L)^-1[i][w],  S' = (S + X C) diag(exp(p_{16 b + 15} - ref_b)),
//    then out = S' q^.  Both phases then write the BF16 state rows Dense[meta[row]][j] the next window's steps read. ──
#define DN_S 0                                  // state [128 v][32 chunks] float4, chunk c at c ^ (v & 7)
#define DN_A (DN_S + 128 * 128 * 4)             // a [16][128] f32
#define DN_C (DN_A + 16 * 128 * 4)              // c [16][128] f32
#define DN_T (DN_C + 16 * 128 * 4)              // T'' [16 w][16 i] f32
#define DN_L (DN_T + 16 * 16 * 4)               // L [16][16] f32
#define DN_PW (DN_L + 16 * 16 * 4)              // exp(p_{16 b + 15} - ref_b) [128]
#define DN_Q (DN_PW + 128 * 4)                  // q^ [128]
#define DN_B (DN_Q + 128 * 4)                   // beta [16]
#define DN_BYTES (DN_B + 16 * 4)
__device__ __forceinline__ float4* dn_chunk(unsigned char* sm, int v, int c) { return (float4*)(sm + DN_S) + v * 32 + (c ^ (v & 7)); }

template <bool FOLD>
__global__ void __launch_bounds__(128)
kda_dense_flush(const int* __restrict__ rows, int n_rows, const int* __restrict__ slots, const int* __restrict__ meta,
    const int* __restrict__ heads, int NHd, float* __restrict__ state, long S0, long S1,
    __nv_bfloat16* __restrict__ dense, long DS,
    const __nv_bfloat16* __restrict__ kr, const __nv_bfloat16* __restrict__ vr,
    const float* __restrict__ prefix_r, const float* __restrict__ beta_r, long RS,
    const __nv_bfloat16* __restrict__ q_in, long QS, __nv_bfloat16* __restrict__ out, int H, float scale)
{
    extern __shared__ __align__(16) unsigned char dsm[];
    float* sA = (float*)(dsm + DN_A); float* sC = (float*)(dsm + DN_C); float* sT = (float*)(dsm + DN_T);
    float* sL = (float*)(dsm + DN_L); float* sPW = (float*)(dsm + DN_PW); float* sQ = (float*)(dsm + DN_Q);
    float* sB = (float*)(dsm + DN_B);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int total = n_rows * NHd;
    for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
        const int row = rows[idx / NHd], j = idx % NHd;
        if (row < 0) break;
        const long page = slots[row];
        if (page <= 0) continue;
        const int h = heads[j];
        float* sp = state + page * S0 + (long)h * S1;
        __syncthreads();                                         // shared memory of the previous item is free
        #pragma unroll 4
        for (int i = 0; i < 32; ++i) {                           // rows 4 i + warp, chunk lane: coalesced
            const int v = 4 * i + warp;
            *dn_chunk(dsm, v, lane) = *(const float4*)(sp + v * 128 + 4 * lane);
        }
        if constexpr (FOLD) {
            const long rp = page * RS, ring = (long)h * KDA_W;
            for (int b = 0; b < NBLK; ++b) {
                const long rb = ring + 16 * b;
                float4 ref = make_float4(0.f, 0.f, 0.f, 0.f);
                if (b > 0) ref = *(const float4*)(page_ring(prefix_r, rp) + (rb - 1) * 128 + 4 * lane);
                #pragma unroll
                for (int r = 0; r < 4; ++r) {                    // warp: rows 4 warp + r; lane: keys 4 lane .. + 3
                    const int i = 4 * warp + r;
                    const uint2 u = *(const uint2*)(page_ring(kr, rp) + (rb + i) * 128 + 4 * lane);
                    float4 k = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
                    const float kf = rsqrtf(warp_sum(k.x * k.x + k.y * k.y + k.z * k.z + k.w * k.w) + 1.e-6f);
                    const float4 p = *(const float4*)(page_ring(prefix_r, rp) + (rb + i) * 128 + 4 * lane);
                    const float4 d = make_float4(p.x - ref.x, p.y - ref.y, p.z - ref.z, p.w - ref.w);
                    k.x *= kf; k.y *= kf; k.z *= kf; k.w *= kf;
                    *(float4*)(sA + i * 128 + 4 * lane) = make_float4(k.x * expf(d.x), k.y * expf(d.y), k.z * expf(d.z), k.w * expf(d.w));
                    *(float4*)(sC + i * 128 + 4 * lane) = make_float4(k.x * expf(-d.x), k.y * expf(-d.y), k.z * expf(-d.z), k.w * expf(-d.w));
                    if (i == 15) *(float4*)(sPW + 4 * lane) = make_float4(expf(d.x), expf(d.y), expf(d.z), expf(d.w));
                }
                if (t < 16) sB[t] = page_ring(beta_r, rp)[rb + t];
                if (b == 0) {                                    // q^ (warp 0)
                    if (warp == 0) {
                        const uint2 u = *(const uint2*)(q_in + (long)row * QS + h * 128 + 4 * lane);
                        float4 q = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
                        const float qf = rsqrtf(warp_sum(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) + 1.e-6f) * scale;
                        *(float4*)(sQ + 4 * lane) = make_float4(q.x * qf, q.y * qf, q.z * qf, q.w * qf);
                    }
                }
                __syncthreads();
                // L[i][jj] = beta_i <a_i, c_jj> for jj < i: thread t -> pairs t, t + 128 of the 16 x 16 grid
                for (int e = t; e < 256; e += 128) {
                    const int i = e >> 4, jj = e & 15;
                    float acc = 0.f;
                    if (jj < i) {
                        const float4* a4 = (const float4*)(sA + i * 128);
                        const float4* c4 = (const float4*)(sC + jj * 128);
                        #pragma unroll 8
                        for (int k = 0; k < 32; ++k) {
                            const float4 x = a4[(k + t) & 31], y = c4[(k + t) & 31];
                            acc = fmaf(x.x, y.x, fmaf(x.y, y.y, fmaf(x.z, y.z, fmaf(x.w, y.w, acc))));
                        }
                        acc *= sB[i];
                    }
                    sL[e] = acc;
                }
                __syncthreads();
                if (t < 16) {                                    // column t of (I + L)^-1; T''[t][i] = beta_t x_i
                    float x[16];
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        float v = (i == t) ? 1.f : 0.f;
                        #pragma unroll
                        for (int jj = 0; jj < i; ++jj) v = fmaf(-sL[i * 16 + jj], x[jj], v);
                        x[i] = v;
                    }
                    const float bt = sB[t];
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) sT[t * 16 + i] = bt * x[i];
                }
                __syncthreads();
                // thread v: P = S_v A^T, X = (V_v - P) T'', S_v = (S_v + X C) diag(PW) (+ out on the last block)
                const int v = t;
                float P[16];
                #pragma unroll
                for (int i = 0; i < 16; ++i) P[i] = 0.f;
                #pragma unroll 2
                for (int c = 0; c < 32; ++c) {
                    const float4 s4 = *dn_chunk(dsm, v, c);
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float4 a4 = *(const float4*)(sA + i * 128 + 4 * c);
                        P[i] = fmaf(s4.x, a4.x, fmaf(s4.y, a4.y, fmaf(s4.z, a4.z, fmaf(s4.w, a4.w, P[i]))));
                    }
                }
                #pragma unroll
                for (int w = 0; w < 16; ++w) P[w] = __bfloat162float(page_ring(vr, rp)[(rb + w) * 128 + v]) - P[w];
                float X[16];
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float acc = 0.f;
                    #pragma unroll
                    for (int w = 0; w < 16; ++w) acc = fmaf(P[w], sT[w * 16 + i], acc);
                    X[i] = acc;
                }
                float o = 0.f;
                #pragma unroll 2
                for (int c = 0; c < 32; ++c) {
                    float4 s4 = *dn_chunk(dsm, v, c);
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float4 c4 = *(const float4*)(sC + i * 128 + 4 * c);
                        s4.x = fmaf(X[i], c4.x, s4.x); s4.y = fmaf(X[i], c4.y, s4.y); s4.z = fmaf(X[i], c4.z, s4.z); s4.w = fmaf(X[i], c4.w, s4.w);
                    }
                    const float4 pw = *(const float4*)(sPW + 4 * c);
                    s4.x *= pw.x; s4.y *= pw.y; s4.z *= pw.z; s4.w *= pw.w;
                    *dn_chunk(dsm, v, c) = s4;
                    const float4 q4 = *(const float4*)(sQ + 4 * c);
                    o = fmaf(s4.x, q4.x, fmaf(s4.y, q4.y, fmaf(s4.z, q4.z, fmaf(s4.w, q4.w, o))));
                }
                if (b == NBLK - 1) out[((long)row * H + h) * 128 + v] = __float2bfloat16_rn(o);
                __syncthreads();
            }
        }
        __nv_bfloat16* pd = dense + (long)meta[row] * DS + (long)j * 16384;
        #pragma unroll 4
        for (int i = 0; i < 32; ++i) {
            const int v = 4 * i + warp;
            const float4 s4 = *dn_chunk(dsm, v, lane);
            if constexpr (FOLD) *(float4*)(sp + v * 128 + 4 * lane) = s4;
            *(uint2*)(pd + v * 128 + 4 * lane) = make_uint2(pack_bf16(s4.x, s4.y), pack_bf16(s4.z, s4.w));
        }
    }
}
