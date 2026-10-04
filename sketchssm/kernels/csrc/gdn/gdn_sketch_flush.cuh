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

// The flush row list: the flush rows, then padding. Padding -2 - n carries the number n of flush rows (any entry,
// the last one is read); plain -1 padding leaves it unknown (-1 returned). With all rows flushing there is none.
__device__ __forceinline__ int flush_count(const int* rows, int n_rows) {
    const int last = rows[n_rows - 1];
    return last >= 0 ? n_rows : (last <= -2 ? -2 - last : -1);
}
// Value split of the flush: each of the n rows is folded by vs CTAs (value blocks 8 p / vs .. 8 (p + 1) / vs), up to
// W1_VSMAX, so that the n vs work units per key head reach about W1_VSUNITS (and the G CTAs) when few rows flush;
// vs = 1 with an unknown count.
#ifndef W1_VSMAX
#define W1_VSMAX 4
#endif
#ifndef W1_VSUNITS
#define W1_VSUNITS 32
#endif
__device__ __forceinline__ int value_split(int n, int G) {
    return n > 0 ? max(1, min(W1_VSMAX, min(G, W1_VSUNITS) / n)) : 1;
}

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
#ifndef W1_NOSKETCH
#define W1_NOSKETCH 0   // 1: timing-only control arm, the same flush body without sketch construction (no U rows, statistics, coefficient finish)
#endif
#ifndef W1_ABLATE
#define W1_ABLATE 0     // timing-only bits: 1 = no P_c, 2 = no state update MMAs, 4 = no state store, 8 = no prologue (key Gram
                        // + WY solve), 32 = no unit-key loads, 64 = no d residual loads
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
#define DB_BYTES (WMAX * 64)                                  // d block: W bf16 hi rows of 16 values, then W fp16 lo rows (32 B each)
#define ZB_BYTES (2 * WMAX * 4)                               // per window row: beta exp(pre), rep
#define KPH 136                                               // sKh, sKl [W w][KPH] f16: KSC x unit keys in the state's frame, hi + lo
#define KSC 1024.f                                            // (the step writes them: fp16 hi + lo of k x 2^10, |k_i| <= 1)
#define TP 20                                                 // W = 16: sT [16 n][TP] f32: Mr(k, n) = x_n[k]
// W > 16: Mr as W1_NB packed 16 x 16 f32 blocks (block (a, b) at b (b + 1) / 2 + a), element (k, n) at tb_at(k, n)
#define T_BYTES (WMAX == 16 ? 16 * TP * 4 : W1_NB * 256 * 4)
#define CTA_BYTES (2 * WMAX * KPH * 2 + 256 * 4 + 128 * 4)     // sKh, sKl, sG, sq: 10 KB at W = 16
#define W1_SMEM_WARP (TILE_F * 4 + STATS_F * 4 + T_BYTES + DB_BYTES + ZB_BYTES)   // 13,248 B at W = 16; 49.8 KB per CTA at 3 warps
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
// d block of value rows v0..v0+15: W bf16 hi rows (d ring of the slot) then W fp16 lo rows (per request), 32 B each
__device__ __forceinline__ void w1_issue_dblock(unsigned db_u, const __nv_bfloat16* dhi, const __half* dlo, int v0, int lane) {
    constexpr int NCH = 4 * WMAX;
    #pragma unroll
    for (int j = 0; j < (NCH + 31) / 32; ++j) {
        const int ch = lane + 32 * j;
        if (ch < 2 * WMAX) w1_cp16(db_u + 16 * ch, dhi + (ch >> 1) * SV + v0 + 8 * (ch & 1));
        else if (ch < NCH && !(W1_ABLATE & 64)) w1_cp16(db_u + 16 * ch, dlo + ((ch - 2 * WMAX) >> 1) * SV + v0 + 8 * (ch & 1));
    }
}
// x as hi + lo tf32 operands (hi = x rounded to tf32, lo = x - hi exactly, read as tf32 by the MMA: |error| <= 2^-23 |x|):
// 3xTF32 products reach FP32 accuracy
__device__ __forceinline__ void tf32_split(float x, float& h, float& l) { h = tf32_rn(x); l = x - h; }
// 3xTF32 m16n8k8 with A already split (hi, lo) and fp32 B
__device__ __forceinline__ void mma3_tf32_s(float* c, const float (&ah)[4], const float (&al)[4], float b0, float b1) {
    float bh[2], bl[2];
    tf32_split(b0, bh[0], bl[0]); tf32_split(b1, bh[1], bl[1]);
    mma_tf32(c, al[0], al[1], al[2], al[3], bh[0], bh[1]);
    mma_tf32(c, ah[0], ah[1], ah[2], ah[3], bl[0], bl[1]);
    mma_tf32(c, ah[0], ah[1], ah[2], ah[3], bh[0], bh[1]);
}
// Per-row power-of-two scales (rows g and g + 8 of a warp tile) that bring the row maxima to [2^SEXP, 2^(SEXP+1)),
// so that fp16 hi + lo keeps 22 bits relative to each row's largest element.
__device__ __forceinline__ float2 row_scales(float m0, float m1) {
    m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2));
    return make_float2(ldexpf(1.f, scale_exp(m0)), ldexpf(1.f, scale_exp(m1)));
}
// (a, b) * sc as packed fp16 hi/lo pairs (no residual scaling): hi + lo carries 22 significant bits.
__device__ __forceinline__ void split2(float a, float b, float sc, unsigned& h, unsigned& l) {
    const float as = a * sc, bs = b * sc;
    const __half2 hh = __floats2half2_rn(as, bs);
    const float2 hf = __half22float2(hh);
    const __half2 ll = __floats2half2_rn(as - hf.x, bs - hf.y);
    h = *reinterpret_cast<const unsigned*>(&hh);
    l = *reinterpret_cast<const unsigned*>(&ll);
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
__device__ __forceinline__ void w1_store_tile(const float (&acc)[16][4], float* dst, int g, int c) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        float* pr = dst + (long)ROW_OF(hh) * SK + 2 * c;
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) *(float2*)(pr + 8 * nt) = make_float2(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
    }
}
// Coefficient finish from the accumulated statistics (energy E[k], Gram rows G4[j][k]).  Directions:
// q_p = sum_j R[p][j] U_j with L L^T = G4[:4,:4] (Cholesky; a pivot whose residual energy is not above
// 1e-12 x its own energy is dropped exactly like the two-pass Gram-Schmidt), R = L^-1, so
// z_p[k] = sum_j R[p][j] G4[j][k].  The rest is the pivot_stats/inline_finish arithmetic (ridge 0.1).
// Approximate (MUFU) reciprocal, square root and division for the coefficient finish: its few dependent scalar
// operations set the finish latency, and its inputs are statistics (2^-22 relative error is far below their noise).
__device__ __forceinline__ float frcp_a(float x) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float fsqrt_a(float x) { float r; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float fdiv_a(float a, float b) { return a * frcp_a(b); }
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
    const float inv_mean = frcp_a(mean), inv_rs = rsqrtf(mean);
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
                L[p][j] = (L[j][j] > 0.f) ? fdiv_a(v, L[j][j]) : 0.f;
                res -= L[p][j] * L[p][j];
            }
            const bool keep = res > (p == 0 ? 0.f : 1.e-12f * Bm[p][p]);
            L[p][p] = keep ? fsqrt_a(res) : 0.f;
        }
    }
    #pragma unroll
    for (int p = 0; p < P; ++p) {                              // R = L^-1 (lower triangular), dropped pivots -> zero rows
        const float inv = (L[p][p] > 0.f) ? frcp_a(L[p][p]) : 0.f;
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
            T[i][j] = (i == j) ? fsqrt_a(v) : v * invd[j];
        }
        invd[i] = frcp_a(T[i][i]);
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

// Gram block G[j][s] = <k_j, k_s> (rows j = 16 a + .., columns s = 16 b + ..) of the unit keys, from the fp16 hi/lo
// of KSC k (three m16n8k16 terms, FP32 accuracy), as two n8 C fragments
__device__ __forceinline__ void w1_gram_block(float (&G0)[2][4], const __half* sKh, const __half* sKl, int a, int b, int lane) {
    const int lr = lane & 7, lm = lane >> 3;
    #pragma unroll
    for (int st = 0; st < 2; ++st) { G0[st][0] = 0.f; G0[st][1] = 0.f; G0[st][2] = 0.f; G0[st][3] = 0.f; }
    #pragma unroll 2
    for (int q = 0; q < 8; ++q) {
        const int oa = (16 * a + 8 * (lm & 1) + lr) * KPH + 16 * q + 8 * (lm >> 1);
        const int ob = (16 * b + 8 * (lm >> 1) + lr) * KPH + 16 * q + 8 * (lm & 1);
        unsigned ah[4], al[4], bh[4], bl[4];
        ldsm4(ah, sKh + oa); ldsm4(al, sKl + oa);
        ldsm4(bh, sKh + ob); ldsm4(bl, sKl + ob);
        #pragma unroll
        for (int st = 0; st < 2; ++st) {
            mma16816(G0[st], ah, bl[2 * st], bl[2 * st + 1]);
            mma16816(G0[st], al, bh[2 * st], bh[2 * st + 1]);
            mma16816(G0[st], ah, bh[2 * st], bh[2 * st + 1]);
        }
    }
    constexpr float inv = 1.f / (KSC * KSC);
    #pragma unroll
    for (int st = 0; st < 2; ++st) { G0[st][0] *= inv; G0[st][1] *= inv; G0[st][2] *= inv; G0[st][3] *= inv; }
}

// (a, b) * sc split as packed fp16 pairs h = bf16(x sc) (exact in fp16) and l = fp16(x sc - h): h is the BF16 row
// value S~ (scaled), l its residual (22 bits in all).
__device__ __forceinline__ void split_bf(float a, float b, float sc, unsigned& h, unsigned& l) {
    const float as = a * sc, bs = b * sc;
    const __nv_bfloat162 hb = __floats2bfloat162_rn(as, bs);
    const unsigned u = *reinterpret_cast<const unsigned*>(&hb);
    const float h0 = __uint_as_float(u << 16), h1 = __uint_as_float(u & 0xffff0000u);
    const __half2 hh = __floats2half2_rn(h0, h1), ll = __floats2half2_rn(as - h0, bs - h1);
    h = *reinterpret_cast<const unsigned*>(&hh);
    l = *reinterpret_cast<const unsigned*>(&ll);
}
// P_c = S K (res: (S - bf16(S)) K) of a warp's 16 x 128 state block as C fragments of 2 W1_NT window n-tiles:
// fp16 m16n8k16, A = the row-scaled block from the fragments split as bf16(S) + residual, B = KSC K (hi + lo) by
// ldmatrix from sKh/sKl ([W][KPH]); three terms for S (FP32 accuracy), the residual's one term for res.
template <int NCHN>
__device__ __forceinline__ void w1_pc(const float (&acc)[16][4], float (&P)[NCHN][2 * W1_NT][4], const __half* sKh,
                                      const __half* sKl, bool res, int lane) {
    const int lr = lane & 7, lm = lane >> 3;
    float m0 = 0.f, m1 = 0.f;
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        m0 = fmaxf(m0, fmaxf(fabsf(acc[nt][0]), fabsf(acc[nt][1])));
        m1 = fmaxf(m1, fmaxf(fabsf(acc[nt][2]), fabsf(acc[nt][3])));
    }
    const float2 ss = row_scales(m0, m1);
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        unsigned ah[4], al[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                          // a0 (2q, row g), a1 (2q, g + 8), a2 (2q + 1, g), a3 (2q + 1, g + 8)
            const float* f = acc[2 * q + (i >> 1)] + 2 * (i & 1);
            split_bf(f[0], f[1], (i & 1) ? ss.y : ss.x, ah[i], al[i]);
        }
        float (*Pq)[4] = P[q % NCHN];
        #pragma unroll
        for (int np = 0; np < W1_NT; ++np) {                   // n-tiles 2 np, 2 np + 1 (16 window rows)
            const int off = (16 * np + 8 * (lm >> 1) + lr) * KPH + 16 * q + 8 * (lm & 1);
            unsigned bh[4], bl[4];
            ldsm4(bh, sKh + off);
            ldsm4(bl, sKl + off);
            #pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                float* pc = Pq[2 * np + h2];
                mma16816(pc, al, bh[2 * h2], bh[2 * h2 + 1]);
                if (!res) {
                    mma16816(pc, ah, bl[2 * h2], bl[2 * h2 + 1]);
                    mma16816(pc, ah, bh[2 * h2], bh[2 * h2 + 1]);
                }
            }
        }
    }
    const float i0 = 1.f / (ss.x * KSC), i1 = 1.f / (ss.y * KSC);
    #pragma unroll
    for (int nt = 0; nt < 2 * W1_NT; ++nt) {
        float s_[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) s_[e] = (NCHN == 2) ? P[0][nt][e] + P[NCHN - 1][nt][e] : P[0][nt][e];
        P[0][nt][0] = s_[0] * i0; P[0][nt][1] = s_[1] * i0; P[0][nt][2] = s_[2] * i1; P[0][nt][3] = s_[3] * i1;
    }
}

// Flush of the rows in `rows` (-1 padded): every head folds the window exactly, as the reference recurrence does in
// FP32. The steps stored d^_t = BF16 hi + lo residual (rows 0..W-1 and W..2W-1 of the slot's d ring); they computed it with the exact unit keys K and the BF16 ring
// rows of the earlier updates, and with the window-start state S as far as they read it exactly (S itself for dense
// heads without rows (ReplaySSM), its BF16 rows S~ for dense heads with rows, nothing for sketch heads). With the WY
// solve M = (I + Gamma~)^-1 of the window and P_c = (S - S_read) K, the exact updates are
//   D = H + (R - P_c diag(beta exp(pre))) M        (H, R: hi and lo of d^),
// and S' = tot S + D diag(rep) K^T, i.e. X = H diag(rep) + Z Mr with Z = R - P_c diag(beta exp(pre)), Mr = M diag(rep).
// The unit keys come from the steps in the state's frame (fp16 hi/lo of KSC k, FP32 rotation); P_c (full for sketch
// heads, the BF16 residual for dense heads with rows, none without rows) and the update S' are fp16 hi/lo
// m16n8k16 MMAs of row-scaled operands (FP32 accuracy), Z Mr is 3xTF32.
__global__ void __launch_bounds__(W1_THREADS, W1_MINB)
gdn_flush_warp_kernel(
    float* __restrict__ h0, const __nv_bfloat16* __restrict__ d_cache, const __nv_bfloat16* __restrict__ k_cache,
    const float* __restrict__ g_cache,
    const int* __restrict__ rows, int n_rows,
    const int* __restrict__ slots, const int* __restrict__ meta, const int* __restrict__ sk_mh,
    sketch_t* __restrict__ sk_ubar, const int* layout, float* __restrict__ fin_stats,
    const float* __restrict__ sk_beta,
    long s_h0_slot, long s_h0_h, long s_d_slot, long s_k_slot, long s_g_slot, long s_u_slot,
    long s_beta_slot, int H, int HV, int G,
    const void* query, void* output, long qs,
    bool query_bf16, bool output_bf16, float output_scale, bool emit_output)
{
    extern __shared__ __align__(1024) unsigned char w1sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, c = lane & 3;
    const int lr = lane & 7, lm = lane >> 3;                  // ldmatrix: row lr of tile lm
    const int i_h = blockIdx.x;                                // key heads first: the busy CTAs of all heads dispatch first
    // The G = gridDim.y CTAs of a key head walk the work units u = (flush row position it, value part p): with vs
    // parts per row (value_split), CTA u folds value blocks [vb0, vb1) of its row (the prologue is repeated per part;
    // the sketch statistics go out per unit as partial sums).
    const int nf = flush_count(rows, n_rows);
    const int vs = value_split(nf, gridDim.y);
    const int units = nf >= 0 ? nf * vs : n_rows;
    for (int u = blockIdx.y; u < units; u += gridDim.y) {
    const int it = u / vs, part = u - it * vs;
    const int vb0 = 8 * part / vs, vb1 = 8 * (part + 1) / vs;
    const int row = rows[it];
    if (row < 0) break;
    const int hv = i_h * W1_WARPS + warp;
    const int sidx = slots[row];
    const int cidx = meta[row];
    const int m = sk_mh[hv];
    const bool dense_rows = m == 0 && layout[(long)hv * 4] >= 0;
    const bool write_rows = m > 0 || dense_rows;
    const int urows = dense_rows ? SK : m;                     // U rows written from S
    __half* sKh = (__half*)w1sm;                               // [W][KPH] KSC x unit keys: fp16 hi
    __half* sKl = sKh + WMAX * KPH;                            //                            and lo
    float* sG = (float*)(sKl + WMAX * KPH);                    // [16][16] key Gram
    float* sq = sG + 256;                                      // [128] query row of this key head
    unsigned char* wsm = w1sm + CTA_BYTES + warp * W1_SMEM_WARP;
    float* tile = (float*)wsm;
    float* stats = tile + TILE_F;                              // [5][128]: energy accumulated per block, Gram rows at the end
    float* sT = stats + STATS_F;                               // W = 16: [16][TP]; W > 16: the Mr blocks
    unsigned char* dtile = wsm + (TILE_F + STATS_F) * 4 + T_BYTES;
    float* zb = (float*)(dtile + DB_BYTES);                    // [W] beta_w exp(pre_w), then [W] rep_w
    float* sp = h0 + (long)sidx * s_h0_slot + (long)hv * s_h0_h;          // this head's state [128 v][128 k]
    const unsigned tile_u = smem_u32(tile), dt_u = smem_u32(dtile);
    const __nv_bfloat16* dhi = d_cache + (long)sidx * s_d_slot + (long)hv * (2 * WMAX * SV);
    const __half* dlo = (const __half*)(dhi + WMAX * SV);
    {                                                          // the window's unit keys (CTA): 2 W rows of 256 B
        const __half* pk = (const __half*)(k_cache + (long)sidx * s_k_slot + (long)i_h * (3 * WMAX * SK) + WMAX * SK);
        for (int ch = t; ch < ((W1_ABLATE & 32) ? 0 : 2 * WMAX * 16); ch += W1_THREADS) {
            const int r = ch >> 4, part = ch & 15;             // r < W: hi rows, then lo rows
            w1_cp16(smem_u32((r < WMAX ? sKh + r * KPH : sKl + (r - WMAX) * KPH) + 8 * part), pk + r * SK + 8 * part);
        }
    }
    w1_issue_tile(tile_u, sp, vb0, lane);
    w1_issue_dblock(dt_u, dhi, dlo, 16 * vb0, lane);
    w1_commit();
    #pragma unroll
    for (int i = 0; i < WMAX / 16; ++i) {                      // the head's d ring rows (hi and lo, 2 W x 256 B) into L2
        w1_pf_l2((const char*)dhi + 4096 * i + lane * 128);
        w1_pf_l2((const char*)dlo + 4096 * i + lane * 128);
    }
    if (vb0 + 1 < vb1) { w1_pf_l2((const char*)(sp + (vb0 + 1) * 16 * SK) + lane * 256); w1_pf_l2((const char*)(sp + (vb0 + 1) * 16 * SK) + lane * 256 + 128); }
    if (vb0 + 2 < vb1) { w1_pf_l2((const char*)(sp + (vb0 + 2) * 16 * SK) + lane * 256); w1_pf_l2((const char*)(sp + (vb0 + 2) * 16 * SK) + lane * 256 + 128); }
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
    w1_wait_all();
    __syncthreads();
    // ── prologue: key Gram, per-warp WY solve into Mr ──
    {
        float* gam = sT;                                       // prologue scratch (Mr is written after gam is consumed)
        if (W1_ABLATE & 8) goto prologue_done;                 // timing only: skip the key Gram + WY solve
        if constexpr (WMAX == 16) {
        if (warp == 0) {
            float G0[2][4];
            w1_gram_block(G0, sKh, sKl, 0, 0, lane);
            #pragma unroll
            for (int st = 0; st < 2; ++st)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j = g + 8 * (e >> 1), sl = 8 * st + 2 * c + (e & 1);
                    sG[j * 16 + sl] = G0[st][e];
                }
        }
        __syncthreads();
        // per warp: Gamma~[j][s] = beta_s <k_j,k_s> exp(pre_s - pre_j) (j < s); x = column `lane` of M
        const float prew = (lane < WMAX) ? g_cache[(long)sidx * s_g_slot + (long)hv * WMAX + lane] : 0.f;
        const float beta = (lane < WMAX) ? sk_beta[(long)cidx * s_beta_slot + (long)hv * WMAX + lane] : 0.f;
        float prs = prew;
        #pragma unroll
        for (int o = 1; o < WMAX; o <<= 1) { const float y = __shfl_up_sync(FULL, prs, o); if (lane >= o) prs += y; }
        const float gtw = __shfl_sync(FULL, prs, WMAX - 1);
        const float repw = expf(gtw - prs);
        if (lane < WMAX) { zb[lane] = beta * expf(prs); zb[WMAX + lane] = repw; }
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
        __syncwarp();                                          // every lane is done with gam before Mr overwrites it
        if (lane < 16) {
            #pragma unroll
            for (int sl = 0; sl < 16; ++sl) sT[lane * TP + sl] = x[sl] * repw;
        }
        __syncwarp();
        } else {
        // ── W > 16: blocked UT transform in 16 x 16 tiles. (1) key Gram blocks (a <= b), spread over the warps and
        //    written to every warp's Mr blocks; (2) per warp, Gamma~ in place; (3) the diagonal blocks inverted as at
        //    W = 16 (M_bb = (I + Gamma~_bb)^-1); (4) off-diagonal blocks, columns right to left, rows bottom up:
        //    M_ab = -M_aa sum_{a < c <= b} Gamma~_ac M_cb (3xTF32); (5) Mr(k, n) = M(k, n) exp(gt - pre_n).
        for (int bi = warp; bi < W1_NB; bi += W1_WARPS) {
            int b = 0;
            while ((b + 1) * (b + 2) / 2 <= bi) ++b;
            const int a = bi - b * (b + 1) / 2;
            float G0[2][4];
            w1_gram_block(G0, sKh, sKl, a, b, lane);
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
            if (r < WMAX) {
                const float bt = sk_beta[(long)cidx * s_beta_slot + (long)hv * WMAX + r];
                spre[r] = p; sbeta[r] = bt; zb[r] = bt * expf(p);
            }
        }
        __syncwarp();
        for (int r = lane; r < WMAX; r += 32) zb[WMAX + r] = expf(gtw - spre[r]);
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
                const int e = lane + 32 * q, n = e >> 4, r = e & 15, nn = 16 * b + n;
                const int at = tb_at(r, n);
                blk[at] = blk[at] * expf(gtw - spre[nn]);
            }
        }
        __syncwarp();
        }
    }
    prologue_done:
    // ── tot = exp(gt) ──
    float gt = 0.f;
    #pragma unroll
    for (int j = 0; j < (WMAX + 31) / 32; ++j) {
        const int r = lane + 32 * j;
        gt += warp_sum((r < WMAX) ? g_cache[(long)sidx * s_g_slot + (long)hv * WMAX + r] : 0.f);
    }
    const float tot = expf(gt);
    // P_c: 0 none (ReplaySSM: the steps read S exactly), 1 the BF16 residual S - S~ (dense heads with rows), 2 all of S
    const int pmode = (W1_ABLATE & 1) ? 0 : (m > 0 ? 2 : (dense_rows ? 1 : 0));
    sketch_t* pu = sk_ubar + (long)cidx * s_u_slot + layout[(long)hv * 4];
    sketch_t* pu_l = pu + (2 * c) * SV + g;                     // + (8 nt + (e & 1)) * SV + 8 (e >> 1)
    float acc[16][4];
    float gram[8][4];                                          // Gram rows G4^T tiles (k = 16 mt + g (+8), j = 2 c + (e & 1)), all blocks
    #pragma unroll
    for (int mt = 0; mt < 8; ++mt) { gram[mt][0] = 0.f; gram[mt][1] = 0.f; gram[mt][2] = 0.f; gram[mt][3] = 0.f; }
    __syncwarp();
    for (int i = lane; i < 128; i += 32) stats[i] = 0.f;
    __syncwarp();
    for (int vb = vb0; vb < vb1; ++vb) {
        const int v0 = 16 * vb;
        w1_wait_all(); __syncwarp();
        w1_load_tile(acc, tile, g, c);
        __syncwarp();
        if (vb + 1 < vb1) { w1_issue_tile(tile_u, sp, vb + 1, lane); w1_commit(); }
        if (vb + 3 < vb1) { w1_pf_l2((const char*)(sp + (vb + 3) * 16 * SK) + lane * 256); w1_pf_l2((const char*)(sp + (vb + 3) * 16 * SK) + lane * 256 + 128); }
        // ── P_c K (fp16 m16n8k16: A = the row-scaled S (or its BF16 residual) from the fragments, B = KSC K by ldmatrix
        //    from sKh/sKl; three terms for S, one for the residual (2^-9 of S)), two chains per n-tile at W = 16 ──
        constexpr int NCHN = (W1_NT == 1) ? 2 : 1;
        float P[NCHN][2 * W1_NT][4];
        #pragma unroll
        for (int ch = 0; ch < NCHN; ++ch)
            #pragma unroll
            for (int nt = 0; nt < 2 * W1_NT; ++nt) { P[ch][nt][0] = 0.f; P[ch][nt][1] = 0.f; P[ch][nt][2] = 0.f; P[ch][nt][3] = 0.f; }
        if (pmode) w1_pc<NCHN>(acc, P, sKh, sKl, pmode == 1, lane);
        // ── Z = R - beta exp(pre) P_c (in place of P) and H diag(rep), from the d block (window rows w = 8 nt + 2 c + (e & 1),
        //    value rows g + 8 (e >> 1)); R = lo x ulp(hi) ──
        float HR[2 * W1_NT][4];
        #pragma unroll
        for (int nt = 0; nt < 2 * W1_NT; ++nt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int w = 8 * nt + 2 * c + (e & 1), vl = g + 8 * (e >> 1);
                const float hi = __bfloat162float(((const __nv_bfloat16*)dtile)[w * 16 + vl]);
                const float lo = __half2float(((const __half*)(dtile + 32 * WMAX))[w * 16 + vl]);
                const int ex = (__float_as_uint(hi) >> 23) & 0xff;
                const float r = ex >= 8 ? lo * __uint_as_float((unsigned)(ex - 7) << 23) : 0.f;
                P[0][nt][e] = fmaf(-zb[w], P[0][nt][e], r);
                HR[nt][e] = hi * zb[WMAX + w];
            }
        __syncwarp();                                          // every lane is done with the d block: fetch the next one
        if (vb + 1 < vb1) { w1_issue_dblock(dt_u, dhi, dlo, v0 + 16, lane); w1_commit(); }
        // ── X = H diag(rep) + Z Mr (3xTF32) ──
        float X[W1_NT][2][4];
        #pragma unroll
        for (int b = 0; b < W1_NT; ++b) {
            float Y[2][4];
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) Y[nt2][e] = HR[2 * b + nt2][e];
            #pragma unroll
            for (int a = 0; a <= b; ++a)
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    const float* z = P[0][2 * a + ks];
                    float ah[4], al[4];
                    tf32_split(z[0], ah[0], al[0]); tf32_split(z[2], ah[1], al[1]);
                    tf32_split(z[1], ah[2], al[2]); tf32_split(z[3], ah[3], al[3]);
                    #pragma unroll
                    for (int nt2 = 0; nt2 < 2; ++nt2) {
                        const float2 bm = (W1_NT == 1) ? *(const float2*)(sT + (8 * nt2 + g) * TP + 8 * ks + 2 * c)
                                                       : *(const float2*)(sT + tb_blk(a, b) + tb_at(8 * ks + 2 * c, 8 * nt2 + g));
                        mma3_tf32_s(Y[nt2], ah, al, bm.x, bm.y);
                    }
                }
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2)
                #pragma unroll
                for (int e = 0; e < 4; ++e) X[b][nt2][e] = Y[nt2][e];
        }
        // ── S' = tot S + X K^T: fp16 hi/lo m16n8k16 MMAs of the row-scaled X and KSC K (B by transposed ldmatrix),
        //    accumulated straight into the fragments scaled by tot and the (power-of-two) row scales, removed after ──
        float m0 = 0.f, m1 = 0.f;
        #pragma unroll
        for (int b = 0; b < W1_NT; ++b)
            #pragma unroll
            for (int nt2 = 0; nt2 < 2; ++nt2) {
                m0 = fmaxf(m0, fmaxf(fabsf(X[b][nt2][0]), fabsf(X[b][nt2][1])));
                m1 = fmaxf(m1, fmaxf(fabsf(X[b][nt2][2]), fabsf(X[b][nt2][3])));
            }
        const float2 sx = row_scales(m0, m1);
        unsigned xh[W1_NT][4], xl[W1_NT][4];                     // X as fp16 A fragments of window tile b
        #pragma unroll
        for (int b = 0; b < W1_NT; ++b) {
            split2(X[b][0][0], X[b][0][1], sx.x, xh[b][0], xl[b][0]);
            split2(X[b][0][2], X[b][0][3], sx.y, xh[b][1], xl[b][1]);
            split2(X[b][1][0], X[b][1][1], sx.x, xh[b][2], xl[b][2]);
            split2(X[b][1][2], X[b][1][3], sx.y, xh[b][3], xl[b][3]);
        }
        const float u0 = tot * sx.x * KSC, u1 = tot * sx.y * KSC, r0 = 1.f / (sx.x * KSC), r1 = 1.f / (sx.y * KSC);
        #pragma unroll
        for (int np = 0; np < 8; ++np) {                       // key n-tiles 2 np, 2 np + 1
            if (W1_ABLATE & 2) {
                #pragma unroll
                for (int h2 = 0; h2 < 2; ++h2)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) acc[2 * np + h2][e] *= tot;
                continue;
            }
            #pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                float* a_ = acc[2 * np + h2];
                a_[0] *= u0; a_[1] *= u0; a_[2] *= u1; a_[3] *= u1;
            }
            #pragma unroll
            for (int b = 0; b < W1_NT; ++b) {
                const int off = (16 * b + 8 * (lm & 1) + lr) * KPH + 16 * np + 8 * (lm >> 1);
                unsigned bh[4], bl[4];
                ldsm4t(bh, sKh + off);
                ldsm4t(bl, sKl + off);
                #pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    float* a_ = acc[2 * np + h2];
                    mma16816(a_, xh[b], bl[2 * h2], bl[2 * h2 + 1]);
                    mma16816(a_, xl[b], bh[2 * h2], bh[2 * h2 + 1]);
                    mma16816(a_, xh[b], bh[2 * h2], bh[2 * h2 + 1]);
                }
            }
            #pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                float* a_ = acc[2 * np + h2];
                a_[0] *= r0; a_[1] *= r0; a_[2] *= r1; a_[3] *= r1;
            }
        }
        // ── writes: state block, sketch rows U[L][v] = S'[v][L] (L < m; all K for dense rows), exact-output partial ──
        if (!(W1_ABLATE & 4)) w1_store_tile(acc, sp + (long)v0 * SK, g, c);   // timing only: no state store
        if (!W1_NOSKETCH && !(W1_NOSK & 1) && write_rows) {
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
        if (!W1_NOSKETCH && m > 0 && m < SK) {               // the statistics serve the coefficient finish only
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
    if (!(W1_NOSKETCH || (W1_NOSK & 8) || m == 0 || m >= SK)) {
    // ── Gram rows into the stats region, then energy and Gram rows to fin_stats[it][hv] for the coefficient finish
    //    (gdn_finish_kernel, a separate launch) ──
    __syncwarp();
    if (c < 2) {                                               // rows k = 16 mt + g (+8), cols j = 2 c + (e & 1)
        #pragma unroll
        for (int mt = 0; mt < 8; ++mt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) stats[(2 * c + (e & 1) + 1) * 128 + 16 * mt + g + 8 * (e >> 1)] = gram[mt][e];
    }
    __syncwarp();
    float4* dst = (float4*)(fin_stats + ((long)u * HV + hv) * STATS_F);
    #pragma unroll
    for (int i = 0; i < STATS_F / 128; ++i) dst[lane + 32 * i] = ((const float4*)stats)[lane + 32 * i];
    }
    __syncthreads();                                           // the CTA-shared key tables and query are reused by the next row
    }
}

// Coefficient finish of the flushed sketch heads (0 < m < K), one warp per (flush row position, value head), from the
// statistics the flush left in fin_stats[it][hv] (energy, then the 4 Gram rows).
__global__ void __launch_bounds__(256)
gdn_finish_kernel(const float* __restrict__ fin_stats, const int* __restrict__ rows, int n_rows,
                  const int* __restrict__ meta, const int* __restrict__ sk_mh, const int* __restrict__ layout,
                  sketch_t* __restrict__ packed, long s_phi_slot, int HV, int flush_ctas)
{
    __shared__ __align__(16) float sst[8][STATS_F];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hv = blockIdx.y * 8 + warp;
    if (hv >= HV) return;
    const int m = sk_mh[hv];
    if (m <= 0 || m >= SK) return;
    const int vs = value_split(flush_count(rows, n_rows), flush_ctas);   // as the flush launch split the rows
    for (int it = blockIdx.x; it < n_rows; it += gridDim.x) {   // the flush rows, then padding
        const int row = rows[it];
        if (row < 0) break;
        const int cidx = meta[row];
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < STATS_F / 128; ++i) {              // the value-split partial sums
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int z = 0; z < vs; ++z) {
                const float4 b = ((const float4*)(fin_stats + (((long)it * vs + z) * HV + hv) * STATS_F))[lane + 32 * i];
                a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
            }
            ((float4*)sst[warp])[lane + 32 * i] = a;
        }
        __syncwarp();
        w1_finish(sst[warp], packed + (long)cidx * s_phi_slot + layout[(long)hv * 4 + 1], m, layout[(long)hv * 4 + 3], lane);
    }
}
