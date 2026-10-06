// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
// SketchSSM decode kernels (https://arxiv.org/abs/2609.33051).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <unordered_map>
#include <ATen/cuda/CUDAContext.h>

#define FULL 0xffffffffu
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}


#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Ridge of the coefficient solve (coefficients in the orthonormal sketch basis).
#ifndef SK_RIDGE
#define SK_RIDGE .003f
#endif

#if SKETCH_BF16
using sketch_t = __nv_bfloat16;
#else
using sketch_t = float;
#endif


__device__ __forceinline__ sketch_t sketch_round(float x) {
#if SKETCH_BF16
    return __float2bfloat16_rn(x);
#else
    return x;
#endif
}
// x minus its stored sketch value (exact): the rounding residual of a dense head's rows
__device__ __forceinline__ float sketch_residual(float x) {
#if SKETCH_BF16
    return x - __bfloat162float(__float2bfloat16_rn(x));
#else
    return 0.f;
#endif
}
#define SK 128
#define SV 128
// SketchSSM window W (the build passes it): any multiple of 16, folded in 16-row tiles
#ifndef WMAX
#define WMAX 16
#endif
static_assert(WMAX >= 16 && WMAX % 16 == 0, "window must be a multiple of 16");
#define NTS 256
#define LDD 132     // d ring [W][LDD] f32 (x2 buffers); the consumed one holds Phi partial sums
#define LDK 128
#define KSS (WMAX * SK) // K_r split [W][SK] f16 (hi | lo), 16 B chunks XOR-swizzled by (row&7)
#define LDT2 16     // T'' split transposed [16 c][LDT2] f16
#define SEXP 10
#define SPHI_MAX 2048         // Phi partial-sum chunk (floats): sPhiA, then the store staging
#define TBL 64      // item table entries (a CTA's work list, index chains resolved up front)
#define RING_H (2 * KSS + 2 * 16 * LDT2)
#define TE 8        // entry: sidx, cidx, hv, i_h, rh, rt, m, kK | kT << 16
#define RING_F (WMAX * LDD + WMAX)                             // d rows | gates (f32)
#define SMEM_STREAM ((SV * SK + 2 * RING_F + SPHI_MAX + TBL * TE) * 4 + 2 * RING_H * 2)
// Key hi/lo records per request/key head; T records per request/value head.
#define PREP_K_BYTES (2 * WMAX * SK * 2)
#define PREP_T_BYTES (2 * 16 * 16 * 2)

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x = fmaxf(x, __shfl_xor_sync(FULL, x, o));
    return x;
}
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ int lds_i32(unsigned a) {
    int v;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
    return v;
}

__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
    const __nv_bfloat162 h2 = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h2);
}
__device__ __forceinline__ float2 bf16x2_f2(const __nv_bfloat16* p) {   // two consecutive bf16 -> (lo, hi) as fp32 (exact)
    const unsigned u = *(const unsigned*)p;
    return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ int scale_exp(float mx) { return mx > 0.f ? min(40, SEXP - ilogbf(mx)) : 0; }
__device__ __forceinline__ float amax4(float4 x) {
    return fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
}
__device__ __forceinline__ float block_max(float v, float* s_red, int t) {
    v = warp_max(v);
    if ((t & 31) == 0) s_red[t >> 5] = v;
    __syncthreads();
    float m = s_red[0];
    #pragma unroll
    for (int i = 1; i < NTS / 32; ++i) m = fmaxf(m, s_red[i]);
    return m;
}
// (a, b) * sc as packed fp16 hi/lo pairs: hi + lo carries 22 significant bits.
__device__ __forceinline__ void split_u(float a, float b, float sc, unsigned& h, unsigned& l) {
    const float as = a * sc, bs = b * sc;
    const __half2 hh = __floats2half2_rn(as, bs);
    const float2 hf = __half22float2(hh);
    const __half2 ll = __floats2half2_rn(as - hf.x, bs - hf.y);
    h = *reinterpret_cast<const unsigned*>(&hh);
    l = *reinterpret_cast<const unsigned*>(&ll);
}
__device__ __forceinline__ __half2 as_h2(unsigned u) { return *reinterpret_cast<const __half2*>(&u); }
// C fragments of n-tiles (2q, 2q+1) -> A fragment for k-block q (logical order).
__device__ __forceinline__ void c2a(const float* c0, const float* c1, float sc, unsigned* ah, unsigned* al) {
    split_u(c0[0], c0[1], sc, ah[0], al[0]);
    split_u(c0[2], c0[3], sc, ah[1], al[1]);
    split_u(c1[0], c1[1], sc, ah[2], al[2]);
    split_u(c1[2], c1[3], sc, ah[3], al[3]);
}
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// tf32 operands straight from fp32 registers: hi = the 13 low mantissa bits cleared, lo = x - hi; the MMA
// reads its operands as tf32 (hi exact, lo truncated to 11 bits), hi + lo = 22 significant bits.
__device__ __forceinline__ float tf32_hi(float x) { return __int_as_float(__float_as_int(x) & 0xffffe000); }
// x rounded to the nearest tf32 (the MMA truncates fp32 operands; rounding halves the error and removes its bias)
__device__ __forceinline__ float tf32_rn(float x) { unsigned r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
__device__ __forceinline__ void mma_tf32(float* c, float a0, float a1, float a2, float a3, float b0, float b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(__float_as_int(a0)), "r"(__float_as_int(a1)), "r"(__float_as_int(a2)), "r"(__float_as_int(a3)),
                   "r"(__float_as_int(b0)), "r"(__float_as_int(b1)));
}
// C fragment of an n8 tile (c0: (g, 2c), c1: (g, 2c+1), c2: (g+8, 2c), c3: (g+8, 2c+1)) used as the A fragment of a
// k8 step with the k index permuted (virtual k = c <-> actual 2c, c+4 <-> 2c+1); B fragments follow the same
// permutation (b0: actual k 2c, b1: 2c+1), so the reduction is exact and no data moves.
__device__ __forceinline__ void mma_tf32_c(float* c, const float* a, float b0, float b1) {
    mma_tf32(c, a[0], a[2], a[1], a[3], b0, b1);
}
__device__ __forceinline__ void mma3(float* c, const unsigned* ah, const unsigned* al,
                                     unsigned bh0, unsigned bh1, unsigned bl0, unsigned bl1) {
    mma16816(c, ah, bl0, bl1);
    mma16816(c, al, bh0, bh1);
    mma16816(c, ah, bh0, bh1);
}
__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm2t(unsigned* r, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}

__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
// 8x8 b16 transpose within the warp (fragment layout in == out).
__device__ __forceinline__ unsigned movtrans(unsigned a) {
    unsigned d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
    return d;
}

#define ROW_OF(hh) (16 * warp + g + 8 * (hh))

// acc[nt][e]: rows ROW_OF(e>>1), cols 8nt + 2c + (e&1).  The warp's 16 state rows land in its
// own 8 KB block of sS by four tensor copies on a per-warp mbarrier (issued an item ahead, right
// after the previous fragments were read); S_W goes back to h0 straight from the fragments.
// Block layout [4 col-chunks][16 rows][32] f32 with the TMA 128 B swizzle on a 1 KB-aligned base:
// element (row, col) sits at (col>>5)*512 + row*32 + ((((col&31)>>2) ^ (row&7))<<2) + (col&3).
// For the fragment (row = g + 8hh, col = 8nt + 2c) the XOR term is (2(nt&3)) ^ ((c>>1) ^ g), so
// every address is a lane constant plus an immediate.
__device__ __forceinline__ void load_state(float (&acc)[16][4], const float* sSw, int g, int c) {
    const int x = (c >> 1) ^ g;
    const float* base = sSw + (g << 5) + ((c & 1) << 1);
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int xo = ((2 * (nt & 3)) ^ x) << 2;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float2 v = *(const float2*)(base + (nt >> 2) * 512 + hh * 256 + xo);
            acc[nt][2 * hh] = v.x; acc[nt][2 * hh + 1] = v.y;
        }
    }
}
__device__ __forceinline__ void store_state(const float (&acc)[16][4], float* ph, int warp, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        float* pr = ph + (long)ROW_OF(hh) * SK + 2 * c;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) *(float2*)(pr + 8 * nt) = make_float2(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
    }
}
// K split element (row, 8-column chunk) -> swizzled half offset (ldmatrix rows hit distinct banks);
// the prep records are written in this smem image order
__device__ __forceinline__ int ks_off(int row, int chunk) { return row * SK + 8 * (chunk ^ (row & 7)); }

__device__ __forceinline__ float4 ld4_bf16(const __nv_bfloat16* p) {
    const uint2 u = *((const uint2*)p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

// ───────────────────────── flush w1: one warp per (row, value head) ─────────────────────────
#ifndef W1_WARPS
#define W1_WARPS 3      // value heads per key head (HV / H): one warp each, sharing the key head's CTA tables
#endif
#define W1_THREADS (32 * W1_WARPS)
#ifndef W1_MINB
#define W1_MINB 5
#endif
#ifndef W1_KBF16
#define W1_KBF16 1      // 1: the current key is applied at bf16 precision (no K_lo MMAs), like the 15 bf16 ring keys; 0: fp32 key (hi + lo)
#endif
#ifndef W1_NOSKETCH
#define W1_NOSKETCH 0   // 1: timing-only control arm, the same flush body without sketch construction (no U rows, statistics, coefficient finish)
#endif
#ifndef W1_ABLATE
#define W1_ABLATE 0     // timing-only bits: 1 = no WY (X = D rep), 2 = no state update MMAs, 4 = no state store, 8 = no prologue (key Gram + WY solve)
#endif
#ifndef W1_NOSK
#define W1_NOSK 0       // timing-only bits: 1 = no U rows, 2 = no energy, 4 = no Gram rows, 8 = no coefficient finish
#endif
#ifndef W1_SMEM_PAD
#define W1_SMEM_PAD 0   // extra dynamic shared memory per CTA (occupancy probe, timing only)
#endif
#define TILE_F (16 * 128)                                     // staged state block, XOR-swizzled 16-byte groups
#define STATS_F (5 * 128)                                     // energy + 4 Gram rows (aliases the tile after the loop)
#define W1_NT (WMAX / 16)                                     // 16-row window tiles
#define W1_NB (W1_NT * (W1_NT + 1) / 2)                       // upper 16 x 16 blocks of T (a <= b)
#define DT_BYTES ((WMAX - 1) * 32 + 96)                       // d block: W-1 bf16 rows of 16 values + one f32 row (576 B at W = 16)
#define KPB 136                                               // sKr [W w][KPB] bf16: keys contiguous (P = S K, B rows)
#define WPB (WMAX + 8)                                        // sKt [128 k][WPB] bf16: ring rows contiguous (S' += X K^T)
#define TP 20                                                 // W = 16: sT [16 n][TP] f32 hi, then lo: T''(k, n) = x_n[k]
// W > 16: T'' as W1_NB packed 16 x 16 f32 blocks (block (a, b) at b (b + 1) / 2 + a), element (k, n) at tb_at(k, n)
#define T_BYTES (WMAX == 16 ? 2 * 16 * TP * 4 : W1_NB * 256 * 4)
#define CTA_BYTES (WMAX * KPB * 2 + 128 * WPB * 2 + 128 * 4 + 256 * 4 + 128 * 4)   // sKr, sKt (bf16), sKlo, sG, sq: 12.5 KB at W = 16
#define W1_SMEM_WARP (TILE_F * 4 + STATS_F * 4 + T_BYTES + DT_BYTES)   // 13,888 B at W = 16; CTA 12.5 + W1_WARPS x 13.6 KB (54.2 KB at 3)
#define W1_SMEM (CTA_BYTES + W1_WARPS * W1_SMEM_WARP)
// tile element (row, col) sits at row * 128 + (col ^ (8 * (row & 7))): the swizzle acts on 16-byte groups
__device__ __forceinline__ int tile_at(int row, int col) { return row * 128 + (col ^ ((row & 7) << 3)); }
// T'' block element (row r, column n) of a 16 x 16 block, stored by column; the row index of columns with bit 1 set
// is XORed with 8 so the (k, k + 1) pairs of the Y = P T'' B fragments hit distinct banks
__device__ __forceinline__ int tb_at(int r, int n) { return n * 16 + (r ^ ((n & 2) << 2)); }
__device__ __forceinline__ int tb_blk(int a, int b) { return (b * (b + 1) / 2 + a) * 256; }
#undef ROW_OF
#define ROW_OF(hh) (g + 8 * (hh))                              // block-local value row of accumulator half hh

__device__ __forceinline__ void w1_cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void w1_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void w1_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void w1_pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
// value block vb (16 rows x 128 f32, contiguous 8 KB) -> warp tile [16][TILE_PITCH]
__device__ __forceinline__ void w1_issue_tile(unsigned tile_u, const float* sp, int vb, int lane) {
    const float* src = sp + vb * (16 * SK);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int ch = lane + 32 * i;                          // 16-byte chunk: row ch >> 5, floats 4 (ch & 31)
        w1_cp16(tile_u + tile_at(ch >> 5, 4 * (ch & 31)) * 4, src + ch * 4);
    }
}
// d block of value rows v0..v0+15: slots 0..W-2 (bf16, 32 B each) then slot W-1 (f32, 64 B); 2 (W - 1) + 4 chunks
__device__ __forceinline__ void w1_issue_dblock(unsigned dt_u, const __nv_bfloat16* dsrc, const float* dlast, int v0, int lane) {
    if constexpr (WMAX == 16) {                                // 34 chunks: lanes 0..29 bf16, lanes 30, 31 and 0, 1 f32
        if (lane < 30) w1_cp16(dt_u + 16 * lane, dsrc + (lane >> 1) * SV + v0 + 8 * (lane & 1));
        else w1_cp16(dt_u + 480 + 16 * (lane - 30), dlast + v0 + 4 * (lane - 30));   // lanes 30,31: floats 0..7
        if (lane < 2) w1_cp16(dt_u + 512 + 16 * lane, dlast + v0 + 8 + 4 * lane);    // floats 8..15
    } else {                                                   // chunk lane + 32 j: bf16 rows, then the f32 row
        constexpr int NBF = 2 * (WMAX - 1), NCH = NBF + 4;
        #pragma unroll
        for (int j = 0; j < (NCH + 31) / 32; ++j) {
            const int ch = lane + 32 * j;
            if (ch < NBF) w1_cp16(dt_u + 16 * ch, dsrc + (ch >> 1) * SV + v0 + 8 * (ch & 1));
            else if (ch < NCH) w1_cp16(dt_u + 16 * ch, dlast + v0 + 4 * (ch - NBF));
        }
    }
}
// 3xTF32 m16n8k8 (hi x lo, lo x hi, hi x hi): FP32-level products for the off-diagonal T'' blocks
__device__ __forceinline__ void mma_3xtf32(float* c, const float (&a)[4], const float (&b)[2]) {
    float ah[4], al[4], bh[2], bl[2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { ah[i] = tf32_hi(a[i]); al[i] = a[i] - ah[i]; }
    #pragma unroll
    for (int i = 0; i < 2; ++i) { bh[i] = tf32_hi(b[i]); bl[i] = b[i] - bh[i]; }
    mma_tf32(c, al[0], al[1], al[2], al[3], bh[0], bh[1]);
    mma_tf32(c, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
    mma_tf32(c, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
}
// fragments of 16 x 16 T'' blocks (row-major A: rows g, g + 8, k-step ks; B: k-step ks, n-tile nt)
__device__ __forceinline__ void tb_frag_a(float (&a)[4], const float* blk, int ks, int g, int c) {
    a[0] = blk[tb_at(g, 8 * ks + c)]; a[1] = blk[tb_at(g + 8, 8 * ks + c)];
    a[2] = blk[tb_at(g, 8 * ks + c + 4)]; a[3] = blk[tb_at(g + 8, 8 * ks + c + 4)];
}
__device__ __forceinline__ void tb_frag_b(float (&b)[2], const float* blk, int ks, int nt, int g, int c) {
    b[0] = blk[tb_at(8 * ks + c, 8 * nt + g)]; b[1] = blk[tb_at(8 * ks + c + 4, 8 * nt + g)];
}
__device__ __forceinline__ void tb_store_c(float* blk, const float (&acc)[2][4], float sgn, int g, int c) {
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) blk[tb_at(g + 8 * (e >> 1), 8 * nt + 2 * c + (e & 1))] = sgn * acc[nt][e];
}
__device__ __forceinline__ void w1_load_tile(float (&acc)[16][4], const float* tile, int g, int c) {
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float2 v = *(const float2*)(tile + tile_at(ROW_OF(hh), 8 * nt + 2 * c));
            acc[nt][2 * hh] = v.x; acc[nt][2 * hh + 1] = v.y;
        }
}
#ifndef W1_STCS
#define W1_STCS 0       // 1: state stores streaming (st.global.cs): the written state does not displace the reads in L2
#endif
#ifndef W1_PF
#define W1_PF 1         // L2 prefetch of the next state blocks (0: none, cp.async one block ahead only)
#endif
__device__ __forceinline__ void w1_store_tile(const float (&acc)[16][4], float* dst, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        float* pr = dst + (long)ROW_OF(hh) * SK + 2 * c;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            if constexpr (W1_STCS) asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};" :: "l"(pr + 8 * nt), "f"(acc[nt][2 * hh]), "f"(acc[nt][2 * hh + 1]) : "memory");
            else *(float2*)(pr + 8 * nt) = make_float2(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
        }
    }
}
// Coefficient finish from the accumulated statistics (energy E[k], Gram rows G4[j][k]).  Directions:
// q_p = sum_j R[p][j] U_j with L L^T = G4[:4,:4] (Cholesky; a pivot whose residual energy is not above
// 1e-12 x its own energy is dropped exactly like the two-pass Gram-Schmidt), R = L^-1, so
// z_p[k] = sum_j R[p][j] G4[j][k].  The rest is the pivot_stats/inline_finish arithmetic (ridge SK_RIDGE = 0.003).
__device__ __forceinline__ void w1_finish(const float* stats, sketch_t* out, int m, int G, int lane) {
    constexpr int P = 4;
    const int n = min(m, P);
    float mean;
    {
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) sum += stats[lane + 32 * i];
        sum = warp_sum(sum);
        mean = sum > 0.f ? sum * (1.f / 128.f) : 1.f;
    }
    const float inv_mean = 1.f / mean, inv_rs = rsqrtf(mean);
    float Bm[P][P];
    #pragma unroll
    for (int j = 0; j < P; ++j)
        #pragma unroll
        for (int jj = 0; jj < P; ++jj) Bm[j][jj] = stats[(j + 1) * 128 + jj];
    float L[P][P], R[P][P];
    #pragma unroll
    for (int p = 0; p < P; ++p) {
        #pragma unroll
        for (int j = 0; j < P; ++j) { L[p][j] = 0.f; R[p][j] = 0.f; }
        if (p < n) {
            float res = Bm[p][p];
            #pragma unroll
            for (int j = 0; j < p; ++j) {
                float v = Bm[p][j];
                #pragma unroll
                for (int i = 0; i < j; ++i) v -= L[p][i] * L[j][i];
                L[p][j] = (L[j][j] > 0.f) ? v / L[j][j] : 0.f;
                res -= L[p][j] * L[p][j];
            }
            const bool keep = res > (p == 0 ? 0.f : 1.e-12f * Bm[p][p]);
            L[p][p] = keep ? sqrtf(res) : 0.f;
        }
    }
    #pragma unroll
    for (int p = 0; p < P; ++p) {                              // R = L^-1 (lower triangular), dropped pivots -> zero rows
        const float inv = (L[p][p] > 0.f) ? 1.f / L[p][p] : 0.f;
        #pragma unroll
        for (int j = 0; j < P; ++j) {
            if (j > p) continue;
            float v = (j == p) ? 1.f : 0.f;
            #pragma unroll
            for (int i = 0; i < P; ++i) if (i >= j && i < p) v -= L[p][i] * R[i][j];
            R[p][j] = v * inv;
        }
    }
    float z[P][4], b[P][4], a[4], factor[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        float res = stats[k] * inv_mean;
        float gk[P];
        #pragma unroll
        for (int j = 0; j < P; ++j) gk[j] = stats[(j + 1) * 128 + k];
        #pragma unroll
        for (int p = 0; p < P; ++p) {
            float v = 0.f;
            #pragma unroll
            for (int j = 0; j < P; ++j) v += R[p][j] * gk[j];
            z[p][i] = v * inv_rs; res -= z[p][i] * z[p][i];
        }
        res = k < n ? 0.f : fmaxf(res, 0.f);
        const float den = res + SK_RIDGE, inv_den = 1.f / den;
        a[i] = res * inv_den; factor[i] = k < m ? SK_RIDGE * inv_den : 1.f;
        #pragma unroll
        for (int p = 0; p < P; ++p) b[p][i] = k < m ? z[p][i] * inv_den : 0.f;
    }
    float T[P][P];
    #pragma unroll
    for (int i = 0; i < P; ++i)
        #pragma unroll
        for (int j = 0; j <= i; ++j) {
            float v = 0.f;
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) v += z[i][kk] * b[j][kk];
            T[i][j] = warp_sum(v);
        }
    float invd[P];
    #pragma unroll
    for (int i = 0; i < P; ++i) {
        #pragma unroll
        for (int j = 0; j <= i; ++j) {
            float v = (i == j ? 1.f : 0.f) + T[i][j];
            #pragma unroll
            for (int kk = 0; kk < j; ++kk) v -= T[i][kk] * T[j][kk];
            T[i][j] = (i == j) ? sqrtf(v) : v * invd[j];
        }
        invd[i] = 1.f / T[i][i];
    }
    float x[P][4];
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        #pragma unroll
        for (int i = 0; i < P; ++i) {
            float v = b[i][kk];
            #pragma unroll
            for (int l = 0; l < i; ++l) v -= T[i][l] * x[l][kk];
            x[i][kk] = (i < n) ? v * invd[i] : 0.f;
        }
        #pragma unroll
        for (int i = P - 1; i >= 0; --i) {
            float v = x[i][kk];
            #pragma unroll
            for (int l = i + 1; l < P; ++l) v -= (l < n ? T[l][i] * x[l][kk] : 0.f);
            x[i][kk] = (i < n) ? v * invd[i] : 0.f;
        }
    }
    float xrow[P][P];
    #pragma unroll
    for (int row = 0; row < P; ++row)
        #pragma unroll
        for (int j = 0; j < P; ++j) xrow[row][j] = __shfl_sync(FULL, x[j][0], row);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        if (m <= P) {
            #pragma unroll
            for (int row = 0; row < P; ++row) {
                if (row < m) {
                    float v = 0.f;
                    #pragma unroll
                    for (int j = 0; j < P; ++j) v += xrow[row][j] * z[j][i];
                    out[row * 128 + k] = sketch_round(v * factor[i]);
                }
            }
        } else {
            if (k < m) {
                out[P * 128 + k] = sketch_round(a[i]);
                #pragma unroll
                for (int j = 0; j < P; ++j) if (j < n) out[P * 128 + (j + 1) * G + k] = sketch_round(x[j][i]);
            }
            #pragma unroll
            for (int j = 0; j < P; ++j) if (j < n) out[j * 128 + k] = sketch_round(z[j][i] * factor[i]);
        }
    }
}

__global__ void __launch_bounds__(W1_THREADS, W1_MINB)
gdn_flush_warp_kernel(
    float* __restrict__ h0, const __nv_bfloat16* __restrict__ d_cache, const float* current_d,
    const float* __restrict__ g_cache, const int* __restrict__ rows, int n_rows,
    const int* __restrict__ slots, const int* __restrict__ meta, const int* __restrict__ sk_mh,
    sketch_t* __restrict__ sk_ubar, sketch_t* __restrict__ packed, const int* layout, long s_phi_slot,
    const __nv_bfloat16* __restrict__ k_cache, const float* __restrict__ current_k, const float* __restrict__ sk_beta,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_g_slot, long s_u_slot, long s_k_slot, long s_beta_slot, int H, int HV, int G,
    const void* query, void* output, long qs,
    bool query_bf16, bool output_bf16, float output_scale, bool emit_output)
{
    extern __shared__ __align__(1024) unsigned char w1sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    const int i_h = blockIdx.y;
    // A fixed number of CTAs walk the flush rows, then -1 padding.
    for (int it = blockIdx.x; it < n_rows; it += gridDim.x) {
    const int row = rows[it];
    if (row < 0) break;
    const int hv = i_h * W1_WARPS + warp;
    const int sidx = slots[row];
    const int cidx = meta[row];
    const int m = sk_mh[hv];
    // Dense heads with BF16 rows (u_off >= 0) stored full updates computed from those rows S~ (S rounded to bf16 by
    // the build and the flush): the WY form erases only the rounding residual, X = D rep - ((S - S~) K) T''. Dense
    // heads without rows (ReplaySSM) stored exact updates.
    const bool dense_rows = m == 0 && layout[(long)hv * 4] >= 0;
    const bool exact = m > 0 || dense_rows;
    const int urows = dense_rows ? SK : m;                     // U rows written from S
    __nv_bfloat16* sKr = (__nv_bfloat16*)w1sm;                 // [W][KPB]
    __nv_bfloat16* sKt = sKr + WMAX * KPB;                     // [128][WPB]
    float* sKlo = (float*)(sKt + 128 * WPB);                   // [128]  remainder of the fp32 current key (row W-1) after bf16 rounding
    float* sG = sKlo + 128;                                    // [16][16] key Gram
    float* sq = sG + 256;                                      // [128] query row of this key head
    unsigned char* wsm = w1sm + CTA_BYTES + warp * W1_SMEM_WARP;
    float* tile = (float*)wsm;
    float* stats = tile + TILE_F;                              // [5][128]: energy accumulated per block, Gram rows at the end
    float* sT = stats + STATS_F;                               // W = 16: [16][TP] hi; W > 16: the T'' blocks
    float* sTl = sT + 16 * TP;                                 // W = 16: [16][TP] lo
    unsigned char* dtile = wsm + (TILE_F + STATS_F) * 4 + T_BYTES;
    float* sp = h0 + (long)sidx * s_h0_slot + (long)hv * s_h0_h;          // this head's state [128 v][128 k]
    const unsigned tile_u = smem_u32(tile), dt_u = smem_u32(dtile);
    const __nv_bfloat16* dsrc = d_cache + (long)sidx * s_d_slot + (long)hv * (WMAX * SV);
    const float* dlast = current_d + ((long)cidx * HV + hv) * SV;
    w1_issue_tile(tile_u, sp, 0, lane);
    w1_issue_dblock(dt_u, dsrc, dlast, 0, lane);
    w1_commit();
    if (W1_PF) {
        w1_pf_l2((const char*)(sp + 16 * SK) + lane * 256); w1_pf_l2((const char*)(sp + 16 * SK) + lane * 256 + 128);
        w1_pf_l2((const char*)(sp + 32 * SK) + lane * 256); w1_pf_l2((const char*)(sp + 32 * SK) + lane * 256 + 128);
    }
    // ── exact output: query row of this key head (CTA) and its normalization ──
    const int orow = emit_output ? row : -1;
    float qfac = 0.f;
    if (orow >= 0) {
        const long qoff = (long)orow * qs + (long)i_h * SK;
        float4 v;
        if (query_bf16) v = ld4_bf16((const __nv_bfloat16*)query + qoff + 4 * lane);
        else v = *(const float4*)((const float*)query + qoff + 4 * lane);
        if (warp == 0) *(float4*)(sq + 4 * lane) = v;
        qfac = rsqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) + 1.e-6f) * output_scale;
    }
    // ── folded prep: key ring staged as fp32 (two layouts, CTA), key Gram, per-warp WY solve into T'' ──
    {
        const __nv_bfloat16* pk = k_cache + (long)sidx * s_k_slot + (long)i_h * (WMAX * SK);
        const float* pk15 = current_k + ((long)cidx * H + i_h) * SK;
        #pragma unroll
        for (int q = 0; q < (32 * WMAX + W1_THREADS - 1) / W1_THREADS; ++q) {
            const int i = t + q * W1_THREADS;                  // 16-byte chunk: ring row i >> 5, keys 4 (i & 31)
            if (i < 32 * WMAX) {
                const int sl = i >> 5, c4 = (i & 31) * 4;
                float4 kv;
                if (sl == WMAX - 1) {                          // fp32 current key: bf16 hi (stored) + fp32 lo vector
                    const float4 kf = *((const float4*)(pk15 + c4));
                    kv = make_float4(__bfloat162float(__float2bfloat16_rn(kf.x)), __bfloat162float(__float2bfloat16_rn(kf.y)),
                                     __bfloat162float(__float2bfloat16_rn(kf.z)), __bfloat162float(__float2bfloat16_rn(kf.w)));
                    *(float4*)(sKlo + c4) = make_float4(kf.x - kv.x, kf.y - kv.y, kf.z - kv.z, kf.w - kv.w);
                } else kv = ld4_bf16(pk + sl * SK + c4);       // bf16 ring rows are exact tf32 operands
                *(uint2*)(sKr + sl * KPB + c4) = make_uint2(pack_bf16(kv.x, kv.y), pack_bf16(kv.z, kv.w));
                sKt[(c4 + 0) * WPB + sl] = __float2bfloat16_rn(kv.x); sKt[(c4 + 1) * WPB + sl] = __float2bfloat16_rn(kv.y);
                sKt[(c4 + 2) * WPB + sl] = __float2bfloat16_rn(kv.z); sKt[(c4 + 3) * WPB + sl] = __float2bfloat16_rn(kv.w);
            }
        }
        __syncthreads();
        float* gam = sT;                                       // prologue scratch (T'' is written after gam is consumed)
        if (W1_ABLATE & 8) goto prologue_done;                 // timing only: skip the key Gram + WY solve
        if constexpr (WMAX == 16) {
        if (warp == 0) {
            // key Gram <k_j, k_s>: A = K (w x key), B = K^T (key x w'); lo only in row 15 (lane g == 7 / n-tile 1)
            float G0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int q = 0; q < 16; ++q) {
                const float2 k0 = bf16x2_f2(sKr + g * KPB + 8 * q + 2 * c);
                const float2 k1 = bf16x2_f2(sKr + (g + 8) * KPB + 8 * q + 2 * c);
                const float2 kl = *(const float2*)(sKlo + 8 * q + 2 * c);
                const float l0 = (g == 7) ? kl.x : 0.f, l1 = (g == 7) ? kl.y : 0.f;
                // n-tile 0 (w' = g): B = row g; n-tile 1 (w' = g + 8): B = row g + 8
                mma_tf32(G0[0], k0.x, k1.x, k0.y, k1.y, k0.x, k0.y);
                mma_tf32(G0[1], k0.x, k1.x, k0.y, k1.y, k1.x, k1.y);
                if (!W1_KBF16) {
                mma_tf32(G0[1], k0.x, k1.x, k0.y, k1.y, l0, l1);            // K K_lo^T (column w' = 15)
                mma_tf32(G0[0], 0.f, l0, 0.f, l1, k0.x, k0.y);              // K_lo K^T (row w = 15)
                mma_tf32(G0[1], 0.f, l0, 0.f, l1, k1.x, k1.y);
                }
            }
            #pragma unroll
            for (int st = 0; st < 2; ++st)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j = g + 8 * (e >> 1), sl = 8 * st + 2 * c + (e & 1);
                    sG[j * 16 + sl] = G0[st][e];
                }
        }
        __syncthreads();
        // per warp: Gamma~[j][s] = beta_s <k_j,k_s> exp(pre_s - pre_j) (j < s); x = row `lane` of T''
        const float prew = (lane < WMAX) ? g_cache[(long)sidx * s_g_slot + (long)hv * WMAX + lane] : 0.f;
        const float beta = (lane < WMAX) ? sk_beta[(long)cidx * s_beta_slot + (long)hv * WMAX + lane] : 0.f;
        float prs = prew;
        #pragma unroll
        for (int o = 1; o < WMAX; o <<= 1) { const float y = __shfl_up_sync(FULL, prs, o); if (lane >= o) prs += y; }
        const float gtw = __shfl_sync(FULL, prs, WMAX - 1);
        const float repw = expf(gtw - prs);
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int idx = lane + 32 * q, j = idx >> 4, sl = idx & 15;
            const float bs = __shfl_sync(FULL, beta, sl), ps = __shfl_sync(FULL, prs, sl), pj = __shfl_sync(FULL, prs, j);
            gam[idx] = (j < sl) ? sG[idx] * bs * expf(ps - pj) : 0.f;
        }
        __syncwarp();
        float x[16];
        #pragma unroll
        for (int sl = 15; sl >= 0; --sl) {
            float v = (sl == lane) ? 1.f : 0.f;
            #pragma unroll
            for (int j = sl + 1; j < 16; ++j) v = fmaf(-gam[sl * 16 + j], x[j], v);
            x[sl] = v;
        }
        #pragma unroll
        for (int sl = 0; sl < 16; ++sl) {
            const float bs = __shfl_sync(FULL, beta, sl), ps = __shfl_sync(FULL, prs, sl);
            x[sl] *= bs * expf(ps) * repw;
        }
        __syncwarp();                                          // every lane is done with gam before T'' overwrites it
        if (lane < 16) {
            #pragma unroll
            for (int sl = 0; sl < 16; ++sl) {
                const float h_ = tf32_hi(x[sl]);
                sT[lane * TP + sl] = h_; sTl[lane * TP + sl] = x[sl] - h_;
            }
        }
        __syncwarp();
        } else {
        // ── W > 16: blocked UT transform in 16 x 16 tiles. (1) key Gram blocks (a <= b), spread over the warps and
        //    written to every warp's T'' blocks; (2) per warp, Gamma~ in place; (3) the diagonal blocks inverted as at
        //    W = 16 (M_bb = (I + Gamma~_bb)^-1); (4) off-diagonal blocks, columns right to left, rows bottom up:
        //    M_ab = -M_aa sum_{a < c <= b} Gamma~_ac M_cb (3xTF32); (5) T''(k, n) = M(k, n) beta_k exp(pre_k) exp(gt - pre_n).
        for (int bi = warp; bi < W1_NB; bi += W1_WARPS) {
            int b = 0;
            while ((b + 1) * (b + 2) / 2 <= bi) ++b;
            const int a = bi - b * (b + 1) / 2;
            float G0[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll 4
            for (int q = 0; q < 16; ++q) {
                const float2 k0 = bf16x2_f2(sKr + (16 * a + g) * KPB + 8 * q + 2 * c);
                const float2 k1 = bf16x2_f2(sKr + (16 * a + g + 8) * KPB + 8 * q + 2 * c);
                const float2 j0 = bf16x2_f2(sKr + (16 * b + g) * KPB + 8 * q + 2 * c);
                const float2 j1 = bf16x2_f2(sKr + (16 * b + g + 8) * KPB + 8 * q + 2 * c);
                mma_tf32(G0[0], k0.x, k1.x, k0.y, k1.y, j0.x, j0.y);
                mma_tf32(G0[1], k0.x, k1.x, k0.y, k1.y, j1.x, j1.y);
                if (!W1_KBF16) {                               // lo of the current key: row / column W-1 (g == 7, second half)
                    const float2 kl = *(const float2*)(sKlo + 8 * q + 2 * c);
                    const float l0 = (g == 7) ? kl.x : 0.f, l1 = (g == 7) ? kl.y : 0.f;
                    if (b == W1_NT - 1) mma_tf32(G0[1], k0.x, k1.x, k0.y, k1.y, l0, l1);
                    if (a == W1_NT - 1) { mma_tf32(G0[0], 0.f, l0, 0.f, l1, j0.x, j0.y); mma_tf32(G0[1], 0.f, l0, 0.f, l1, j1.x, j1.y); }
                }
            }
            for (int w = 0; w < W1_WARPS; ++w) {
                float* blk = (float*)(w1sm + CTA_BYTES + w * W1_SMEM_WARP) + TILE_F + STATS_F + tb_blk(a, b);
                tb_store_c(blk, G0, 1.f, g, c);
            }
        }
        __syncthreads();
        float* spre = stats;                                   // [W] gate prefix, then [W] beta (scratch until the stats reset)
        float* sbeta = stats + WMAX;
        float gtw = 0.f;
        #pragma unroll
        for (int j = 0; j < (WMAX + 31) / 32; ++j) {
            const int r = lane + 32 * j;
            float p = (r < WMAX) ? g_cache[(long)sidx * s_g_slot + (long)hv * WMAX + r] : 0.f;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const float y = __shfl_up_sync(FULL, p, o); if (lane >= o) p += y; }
            p += gtw;
            gtw = __shfl_sync(FULL, p, 31);
            if (r < WMAX) { spre[r] = p; sbeta[r] = sk_beta[(long)cidx * s_beta_slot + (long)hv * WMAX + r]; }
        }
        __syncwarp();
        #pragma unroll 1
        for (int bi = 0; bi < W1_NB; ++bi) {                   // Gamma~[j][s] = beta_s G[j][s] exp(pre_s - pre_j) (j < s)
            int b = 0;
            while ((b + 1) * (b + 2) / 2 <= bi) ++b;
            const int a = bi - b * (b + 1) / 2;
            float* blk = sT + bi * 256;
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int e = lane + 32 * q, n = e >> 4, r = e & 15, j = 16 * a + r, s_ = 16 * b + n;
                const int at = tb_at(r, n);
                blk[at] = (j < s_) ? blk[at] * sbeta[s_] * expf(spre[s_] - spre[j]) : 0.f;
            }
        }
        __syncwarp();
        #pragma unroll 1
        for (int bb = 0; bb < W1_NT; ++bb) {                   // diagonal blocks: column `lane` of M_bb
            float* blk = sT + tb_blk(bb, bb);
            float x[16];
            #pragma unroll
            for (int sl = 15; sl >= 0; --sl) {
                float v = (sl == lane) ? 1.f : 0.f;
                #pragma unroll
                for (int j = sl + 1; j < 16; ++j) v = fmaf(-blk[tb_at(sl, j)], x[j], v);
                x[sl] = v;
            }
            __syncwarp();
            if (lane < 16) {
                #pragma unroll
                for (int sl = 0; sl < 16; ++sl) blk[tb_at(sl, lane)] = x[sl];
            }
            __syncwarp();
        }
        #pragma unroll 1
        for (int b = W1_NT - 1; b >= 1; --b) {
            #pragma unroll 1
            for (int a = b - 1; a >= 0; --a) {
                float* blk = sT + tb_blk(a, b);
                float U[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                #pragma unroll 1
                for (int cc = a + 1; cc <= b; ++cc) {
                    const float* ga = sT + tb_blk(a, cc);
                    const float* mc = sT + tb_blk(cc, b);
                    #pragma unroll
                    for (int ks = 0; ks < 2; ++ks) {
                        float fa[4], fb[2][2];
                        tb_frag_a(fa, ga, ks, g, c);
                        tb_frag_b(fb[0], mc, ks, 0, g, c);
                        tb_frag_b(fb[1], mc, ks, 1, g, c);
                        mma_3xtf32(U[0], fa, fb[0]);
                        mma_3xtf32(U[1], fa, fb[1]);
                    }
                }
                __syncwarp();                                  // Gamma~_ab consumed: the block holds U, then M_ab
                tb_store_c(blk, U, 1.f, g, c);
                __syncwarp();
                const float* ma = sT + tb_blk(a, a);
                float R[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    float fa[4], fb[2][2];
                    tb_frag_a(fa, ma, ks, g, c);
                    tb_frag_b(fb[0], blk, ks, 0, g, c);
                    tb_frag_b(fb[1], blk, ks, 1, g, c);
                    mma_3xtf32(R[0], fa, fb[0]);
                    mma_3xtf32(R[1], fa, fb[1]);
                }
                __syncwarp();
                tb_store_c(blk, R, -1.f, g, c);
                __syncwarp();
            }
        }
        #pragma unroll 1
        for (int bi = 0; bi < W1_NB; ++bi) {
            int b = 0;
            while ((b + 1) * (b + 2) / 2 <= bi) ++b;
            const int a = bi - b * (b + 1) / 2;
            float* blk = sT + bi * 256;
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int e = lane + 32 * q, n = e >> 4, r = e & 15, k = 16 * a + r, nn = 16 * b + n;
                const int at = tb_at(r, n);
                blk[at] = tf32_hi(blk[at] * (sbeta[k] * expf(spre[k]) * expf(gtw - spre[nn])));
            }
        }
        __syncwarp();
        }
    }
    prologue_done:
    // ── gate scan: rep_s = exp(gt - pre_s), tot = exp(gt) ──
    constexpr int NGC = (WMAX + 31) / 32, SCN = WMAX < 32 ? WMAX : 32;
    float rep_l[NGC], gt = 0.f;
    #pragma unroll
    for (int j = 0; j < NGC; ++j) {
        const int r = lane + 32 * j;
        float pre = (r < WMAX) ? g_cache[(long)sidx * s_g_slot + (long)hv * WMAX + r] : 0.f;
        #pragma unroll
        for (int o = 1; o < SCN; o <<= 1) { const float y = __shfl_up_sync(FULL, pre, o); if (lane >= o) pre += y; }
        if (j > 0) pre += gt;
        rep_l[j] = pre;
        gt = __shfl_sync(FULL, pre, SCN - 1);
    }
    #pragma unroll
    for (int j = 0; j < NGC; ++j) rep_l[j] = expf(gt - rep_l[j]);
    const float tot = expf(gt);
    float rep[2][2];                                           // replay weights of window tile 0 (all of W = 16)
    #pragma unroll
    for (int nt2 = 0; nt2 < 2; ++nt2)
        #pragma unroll
        for (int e1 = 0; e1 < 2; ++e1) rep[nt2][e1] = __shfl_sync(FULL, rep_l[0], 8 * nt2 + 2 * c + e1);
    sketch_t* pu = sk_ubar + (long)cidx * s_u_slot + layout[(long)hv * 4];
    sketch_t* pu_l = pu + (2 * c) * SV + g;                     // + (8 nt + (e & 1)) * SV + 8 (e >> 1)
    float acc[16][4];
    float gram[8][4];                                          // Gram rows G4^T tiles (k = 16 mt + g (+8), j = 2 c + (e & 1)), all blocks
    #pragma unroll
    for (int mt = 0; mt < 8; ++mt) { gram[mt][0] = 0.f; gram[mt][1] = 0.f; gram[mt][2] = 0.f; gram[mt][3] = 0.f; }
    for (int i = lane; i < 128; i += 32) stats[i] = 0.f;
    __syncwarp();
    for (int vb = 0; vb < 8; ++vb) {
        const int v0 = 16 * vb;
        w1_wait_all(); __syncwarp();
        w1_load_tile(acc, tile, g, c);
        // d values of window tile b (slot 16 b + 8 nt2 + 2 c + (e & 1), value row g + 8 (e >> 1)), from the d block
        auto d_of = [&](float (&dv)[2][4], int b) {
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int sl = 16 * b + 8 * nt2 + 2 * c + (e & 1), vl = g + 8 * (e >> 1);
                    dv[nt2][e] = (sl == WMAX - 1) ? ((const float*)(dtile + 32 * (WMAX - 1)))[vl] : __bfloat162float(((const __nv_bfloat16*)dtile)[sl * 16 + vl]);
                }
        };
        auto rep_of = [&](float (&rp)[2][2], int b) {
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e1 = 0; e1 < 2; ++e1)
                    rp[nt2][e1] = (b == 0) ? rep[nt2][e1] : __shfl_sync(FULL, rep_l[(16 * b) >> 5], ((16 * b) & 31) + 8 * nt2 + 2 * c + e1);
        };
        float dv[2][4];
        if constexpr (W1_NT == 1) d_of(dv, 0);
        if constexpr (W1_NT == 1) {
            __syncwarp();
            if (vb + 1 < 8) { w1_issue_tile(tile_u, sp, vb + 1, lane); w1_issue_dblock(dt_u, dsrc, dlast, v0 + 16, lane); w1_commit(); }
        } else {
            __syncwarp();
            if (vb + 1 < 8) { w1_issue_tile(tile_u, sp, vb + 1, lane); w1_commit(); }
        }
        if (W1_PF && vb + 3 < 8) { w1_pf_l2((const char*)(sp + (vb + 3) * 16 * SK) + lane * 256); w1_pf_l2((const char*)(sp + (vb + 3) * 16 * SK) + lane * 256 + 128); }
        // ── P = S K (tf32: S hi/lo from the fragments, K exact + the current row's lo), Y = P T'', X = D rep - Y ──
        float X[W1_NT][2][4];
        if (exact && !(W1_ABLATE & 1)) {
            // W = 16: two accumulator chains per n-tile (even / odd k-steps) so consecutive HMMAs are independent;
            // W > 16: the 2 W / 16 n-tiles are independent already
            constexpr int NCHN = (W1_NT == 1) ? 2 : 1;
            float P[NCHN][2 * W1_NT][4];                       // one tf32 term: the fp32 fragments as tf32 A operands (w17)
            #pragma unroll
            for (int ch = 0; ch < NCHN; ++ch)
                #pragma unroll
                for (int nt = 0; nt < 2 * W1_NT; ++nt) { P[ch][nt][0] = 0.f; P[ch][nt][1] = 0.f; P[ch][nt][2] = 0.f; P[ch][nt][3] = 0.f; }
            #pragma unroll
            for (int q = 0; q < 16; ++q) {
                float2 bq[2 * W1_NT];
                #pragma unroll
                for (int nt = 0; nt < 2 * W1_NT; ++nt) bq[nt] = bf16x2_f2(sKr + (8 * nt + g) * KPB + 8 * q + 2 * c);
                float (*Pq)[4] = P[q % NCHN];
                float sq4[4];                                  // S, or S - S~ for dense rows
                #pragma unroll
                for (int e = 0; e < 4; ++e) sq4[e] = dense_rows ? sketch_residual(acc[q][e]) : acc[q][e];
                #pragma unroll
                for (int nt = 0; nt < 2 * W1_NT; ++nt) mma_tf32_c(Pq[nt], sq4, bq[nt].x, bq[nt].y);
                if (!W1_KBF16) { const float2 bl = *(const float2*)(sKlo + 8 * q + 2 * c); mma_tf32_c(P[NCHN - 1][2 * W1_NT - 1], sq4, (g == 7) ? bl.x : 0.f, (g == 7) ? bl.y : 0.f); }
            }
            if constexpr (NCHN == 2) {
                #pragma unroll
                for (int st = 0; st < 2; ++st)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) P[0][st][e] += P[1][st][e];
            }
            #pragma unroll
            for (int b = W1_NT - 1; b >= 0; --b) {             // descending: P tiles > b are dead after tile b
                float Y[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                #pragma unroll
                for (int a = 0; a <= b; ++a)
                    #pragma unroll
                    for (int ks = 0; ks < 2; ++ks) {             // one tf32 term (T'' hi) (w17)
                        #pragma unroll
                        for (int nt2 = 0; nt2 < 2; ++nt2) {
                            const float2 bh = (W1_NT == 1) ? *(const float2*)(sT + (8 * nt2 + g) * TP + 8 * ks + 2 * c)
                                                           : *(const float2*)(sT + tb_blk(a, b) + tb_at(8 * ks + 2 * c, 8 * nt2 + g));
                            mma_tf32_c(Y[nt2], P[0][2 * a + ks], bh.x, bh.y);
                        }
                    }
                float rp[2][2];
                rep_of(rp, b);
                if constexpr (W1_NT > 1) d_of(dv, b);
                #pragma unroll
                for (int nt2 = 0; nt2 < 2; ++nt2)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) X[b][nt2][e] = fmaf(dv[nt2][e], rp[nt2][e & 1], -Y[nt2][e]);
            }
        } else {
            #pragma unroll
            for (int b = 0; b < W1_NT; ++b) {
                float rp[2][2];
                rep_of(rp, b);
                if constexpr (W1_NT > 1) d_of(dv, b);
                #pragma unroll
                for (int nt2 = 0; nt2 < 2; ++nt2)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) X[b][nt2][e] = dv[nt2][e] * rp[nt2][e & 1];
            }
        }
        if constexpr (W1_NT > 1) {                             // every lane is done with the d block: fetch the next one
            __syncwarp();
            if (vb + 1 < 8) { w1_issue_dblock(dt_u, dsrc, dlast, v0 + 16, lane); w1_commit(); }
        }
        // ── S' = tot S + X K^T: the MMAs accumulate straight into the scaled fragments ──
        float xa[W1_NT][2][4];                                   // X as tf32 A fragments, one term (w17); rounded for W > 16
        #pragma unroll
        for (int b = 0; b < W1_NT; ++b)
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                xa[b][ks][0] = X[b][ks][0]; xa[b][ks][1] = X[b][ks][2]; xa[b][ks][2] = X[b][ks][1]; xa[b][ks][3] = X[b][ks][3];
                if constexpr (W1_NT > 1) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) xa[b][ks][i] = tf32_rn(xa[b][ks][i]);
                }
            }
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[nt][e] *= tot;
            if (W1_ABLATE & 2) continue;
            #pragma unroll
            for (int b = 0; b < W1_NT; ++b)
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    const float2 bk = bf16x2_f2(sKt + (8 * nt + g) * WPB + 16 * b + 8 * ks + 2 * c);
                    mma_tf32(acc[nt], xa[b][ks][0], xa[b][ks][1], xa[b][ks][2], xa[b][ks][3], bk.x, bk.y);
                }
            if (!W1_KBF16) {
            const float bl = (c == 3) ? sKlo[8 * nt + g] : 0.f;                 // K_lo^T row w = W-1 (last tile, k-step 1, b1 of lanes c == 3)
            mma_tf32(acc[nt], xa[W1_NT - 1][1][0], xa[W1_NT - 1][1][1], xa[W1_NT - 1][1][2], xa[W1_NT - 1][1][3], 0.f, bl);
            }
        }
        // ── writes: state block, sketch rows U[L][v] = S'[v][L] (L < m; all K for dense rows), exact-output partial ──
        if (!(W1_ABLATE & 4)) w1_store_tile(acc, sp + (long)v0 * SK, g, c);   // timing only: no state store
        if (!W1_NOSKETCH && !(W1_NOSK & 1) && exact) {
            sketch_t* pv = pu_l + v0;
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                if (8 * nt >= urows) break;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int L = 8 * nt + 2 * c + (e & 1);
                    if (L < urows) pv[(8 * nt + (e & 1)) * SV + 8 * (e >> 1)] = sketch_round(acc[nt][e]);
                }
            }
        }
        if (orow >= 0) {
            float sum0 = 0.f, sum1 = 0.f;
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const float2 qq = *(const float2*)(sq + 8 * nt + 2 * c);
                sum0 = fmaf(acc[nt][0], qq.x, fmaf(acc[nt][1], qq.y, sum0));
                sum1 = fmaf(acc[nt][2], qq.x, fmaf(acc[nt][3], qq.y, sum1));
            }
            sum0 += __shfl_xor_sync(FULL, sum0, 1); sum0 += __shfl_xor_sync(FULL, sum0, 2);
            sum1 += __shfl_xor_sync(FULL, sum1, 1); sum1 += __shfl_xor_sync(FULL, sum1, 2);
            if (c == 0) {
                const long off = ((long)orow * HV + hv) * SV + v0 + g;
                if (output_bf16) { ((__nv_bfloat16*)output)[off] = __float2bfloat16_rn(sum0 * qfac); ((__nv_bfloat16*)output)[off + 8] = __float2bfloat16_rn(sum1 * qfac); }
                else { ((float*)output)[off] = sum0 * qfac; ((float*)output)[off + 8] = sum1 * qfac; }
            }
        }
        // ── statistics of this block: energy partials (registers), Gram rows G4^T += S'^T S'[:, :4] (bf16 tiles) ──
        if (!W1_NOSKETCH) {
            if (!(W1_NOSK & 2)) {                              // energy partials of this block, reduced over the eight g lanes, added in place
                float v[32];
                #pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    v[2 * nt] = fmaf(acc[nt][0], acc[nt][0], acc[nt][2] * acc[nt][2]);
                    v[2 * nt + 1] = fmaf(acc[nt][1], acc[nt][1], acc[nt][3] * acc[nt][3]);
                }
                #pragma unroll
                for (int off = 16; off >= 4; off >>= 1) {
                    const bool up = (lane & off) != 0;
                    #pragma unroll
                    for (int i = 0; i < off; ++i) {
                        const float send = up ? v[i] : v[i + off];
                        const float keep = up ? v[i + off] : v[i];
                        v[i] = keep + __shfl_xor_sync(FULL, send, off);
                    }
                }
                float* srow = stats + 16 * g + 2 * c;
                srow[0] += v[0]; srow[1] += v[1]; srow[8] += v[2]; srow[9] += v[3];
            }
            if (!(W1_NOSK & 4)) {
            unsigned bh[2];
            {
                float sv[4];
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                    #pragma unroll
                    for (int e1 = 0; e1 < 2; ++e1) {
                        const int src = ((2 * c + e1) << 2) | (g >> 1);
                        const float x0 = __shfl_sync(FULL, acc[0][2 * hh], src), x1 = __shfl_sync(FULL, acc[0][2 * hh + 1], src);
                        sv[2 * hh + e1] = (g < 4) ? ((g & 1) ? x1 : x0) : 0.f;
                    }
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const __nv_bfloat162 h2 = __floats2bfloat162_rn(sv[2 * q], sv[2 * q + 1]);
                    bh[q] = *reinterpret_cast<const unsigned*>(&h2);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < 8; ++mt) {
                unsigned ah[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {                 // a0 (2mt, hh0)  a1 (2mt+1, hh0)  a2 (2mt, hh1)  a3 (2mt+1, hh1)
                    const int nt = 2 * mt + (q & 1), hh = q >> 1;
                    const __nv_bfloat162 h2 = __floats2bfloat162_rn(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
                    ah[q] = movtrans(*reinterpret_cast<const unsigned*>(&h2));
                }
                mma16816_bf16(gram[mt], ah, bh[0], bh[1]);
            }
            }
        }
    }
    if (!(W1_NOSKETCH || (W1_NOSK & 8) || m == 0)) {
    // ── Gram rows into the stats region ──
    __syncwarp();
    if (c < 2) {                                               // rows k = 16 mt + g (+8), cols j = 2 c + (e & 1)
        #pragma unroll
        for (int mt = 0; mt < 8; ++mt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) stats[(2 * c + (e & 1) + 1) * 128 + 16 * mt + g + 8 * (e >> 1)] = gram[mt][e];
    }
    __syncwarp();
    if (m < SK) w1_finish(stats, packed + (long)cidx * s_phi_slot + layout[(long)hv * 4 + 1], m, (int)layout[(long)hv * 4 + 3], lane);
    }
    __syncthreads();                                           // the CTA-shared key tables and query are reused by the next row
    }
}

