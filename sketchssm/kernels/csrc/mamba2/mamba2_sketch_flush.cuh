// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
// SketchSSM decode kernels (https://arxiv.org/abs/2609.33051).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define FULL 0xffffffffu
#ifndef SK_NHEADS
#define SK_NHEADS 128   // state heads
#endif
#ifndef SK_HPG
#define SK_HPG 16       // heads per B/C group
#endif
#ifndef SK_P
#define SK_P 64         // head dim (multiple of 16)
#endif
#define SK_NP (SK_P / 16)   // 16-value column blocks per head
static_assert(SK_P % 16 == 0 && SK_P <= 128, "head dim must be a multiple of 16, at most 128");
#ifndef SK_N
#define SK_N 128        // state size
#endif
#define SK_NKB (SK_N / 16)  // 16-key blocks of the state
#define SK_KF (SK_N / 32)   // keys per lane of the statistics finish
static_assert(SK_N == 64 || SK_N == 128 || SK_N == 256, "state size must be 64, 128 or 256");
#ifndef SK_W
#define SK_W 16         // window length
#endif
static_assert(SK_W % 16 == 0 && SK_W >= 16, "window must be a multiple of 16");
#define SK_WT (SK_W / 16)   // 16-step window tiles: the k dimension of the window MMA
// FL_WSMEM (windows longer than 16): the x window is staged in shared memory as MMA B tiles and
// the step scales as a vector, both read per window tile, instead of one tile of fragments and
// four scales in registers. The kernel then uses dynamic shared memory.
#define FL_WSMEM (SK_W > 16)
#define XW_PITCH (SK_P + 8)
typedef __nv_bfloat16 bf16;
// SUPER_NF_BF16_UW=1: packed sketch rows (U) and coefficient maps (W) are stored as bf16.
// Both are regenerated from the fp32 state at every flush, so the rounding does not
// accumulate across steps; the recurrent state itself stays fp32.
#ifndef UW_BF16
#define UW_BF16 0
#endif
#if UW_BF16
typedef bf16 uw_t;
#else
typedef float uw_t;
#endif
__device__ __forceinline__ unsigned pack_uw2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}
// Predicated 4-element store of a sketch row chunk in the U/W storage type.
__device__ __forceinline__ void st_pred_uw(uw_t* p, float4 v, bool pred) {
#if UW_BF16
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n @p st.global.v2.b32 [%0], {%1,%2};\n}\n"
               :: "l"(p), "r"(pack_uw2(v.x, v.y)), "r"(pack_uw2(v.z, v.w)), "r"((int)pred) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %5, 0;\n @p st.global.v4.f32 [%0], {%1,%2,%3,%4};\n}\n"
               :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"((int)pred) : "memory");
#endif
}
__device__ __forceinline__ void st_uw4(uw_t* p, float4 v) {
#if UW_BF16
  *reinterpret_cast<uint2*>(p) = make_uint2(pack_uw2(v.x, v.y), pack_uw2(v.z, v.w));
#else
  *reinterpret_cast<float4*>(p) = v;
#endif
}


struct FlushStrides {
  long ss0, ss1, ss2, ss3;      // state: slot, head, value, key
  long xs0, xs1, xs2;           // x: row, head, value
  long dts0, dts1;              // dt: row, head
  long biass0, as0;             // dt bias, A: head
  long bs0, bs1, bs2;           // B: row, group, key
  long cs0, cs1, cs2;           // C: row, group, key
  long ds0, ds1;                // D: head, value
  long os0, os1, os2;           // out: row, head, value
  long xrs0, xrs1, xrs2, xrs3;  // x ring: slot, head, t, value
  long drs0, drs1, drs2;        // dt ring: slot, head, t
  long brs0, brs1, brs2, brs3;  // B ring: slot, group, t, key
  long slots;
};

// Same PTX math as the Gluon reference body: div.full.f32, sqrt.approx.ftz.f32,
// ex2/lg2.approx (Triton lowers fp32 /, sqrt, exp and log this way).
__device__ __forceinline__ float2 fl_ffma2(float2 a, float2 b, float2 c) {
#if FL_STATS_FFMA2
  unsigned long long x = *reinterpret_cast<unsigned long long*>(&a), y = *reinterpret_cast<unsigned long long*>(&b), z = *reinterpret_cast<unsigned long long*>(&c);
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(z) : "l"(x), "l"(y), "l"(z));
  return *reinterpret_cast<float2*>(&z);
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}
__device__ __forceinline__ float fdiv(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fsqrt(float a) { float r; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float fexp(float a) { return __expf(a); }
__device__ __forceinline__ float flog(float a) { return __logf(a); }
__device__ __forceinline__ float wsum(float v) {
  v += __shfl_xor_sync(FULL, v, 16); v += __shfl_xor_sync(FULL, v, 8);
  v += __shfl_xor_sync(FULL, v, 4);  v += __shfl_xor_sync(FULL, v, 2);
  v += __shfl_xor_sync(FULL, v, 1);  return v;
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t v) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v));
}
__device__ __forceinline__ uint32_t pack_bf(bf16 lo, bf16 hi) {
  __nv_bfloat162 v; v.x = lo; v.y = hi;
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void fl_cp_async16(void* smem, const void* gmem) {
  unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(gmem));
}
__device__ __forceinline__ void fl_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void fl_cp_async8(void* smem, const void* gmem) {
  unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(a), "l"(gmem));
}
__device__ __forceinline__ void fl_pf_if(const void* p, bool pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" :: "l"(p), "r"((int)pred));
}
template <int N> __device__ __forceinline__ void fl_cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p) {
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// Fragment column n of MMA n-block j maps to logical value index
//   V(j, n) = 16*(j/2) + 4*(n/2) + 2*(j%2) + (n%2)
// so lane (g = lane/4, c = lane%4) owns values 16p + 4c + {0,1,2,3}, p = 0..3,
// of rows g and g+8 of each 16-row key block: accumulators
// {acc[2p][0], acc[2p][1], acc[2p+1][0], acc[2p+1][1]} for row g and
// {acc[2p][2], acc[2p][3], acc[2p+1][2], acc[2p+1][3]} for row g+8.
__device__ __forceinline__ int vmap(int j, int n) {
  return 16 * (j >> 1) + 4 * (n >> 1) + 2 * (j & 1) + (n & 1);
}

#ifndef WARPS
#define WARPS 1
#endif
#ifndef MINB
#define MINB 16
#endif
#ifndef QREG
#define QREG 1
#endif
#ifndef PREFETCH
#define PREFETCH 1
#endif
#ifndef UNROLL
#define UNROLL 1
#endif
#define RING_PITCH (SK_N + 8)
// FL_STAGES=n (>= 2): the state tiles are streamed through an n-deep shared-memory cp.async
// pipeline instead of register prefetch (frees the 32 prefetch registers and their 32 moves
// per block and keeps n-1 tiles in flight per warp).
#ifndef FL_STAGES
#define FL_STAGES 0
#endif
// FL_HALF=1: the pipeline unit is a half block (8 keys x 64 values = 2 KB) so that FL_STAGES
// half tiles fit in the same shared memory as FL_STAGES/2 full tiles and the copy for the
// first half of a block is reissued as soon as that half has been consumed.
#ifndef FL_HALF
#define FL_HALF 0
#endif
// FL_PF_ROWS=d (> 0): each warp prefetches the prologue inputs of row + d for its own head into
// L2 (window keys, x window, current x/B/C rows, dt ring, first state block).
#ifndef FL_PF_ROWS
#define FL_PF_ROWS 0
#endif
// FL_GRID_HF=1: heads run fastest in the grid (blockIdx.x = head block, blockIdx.y = row) so
// that the concurrently resident CTAs cover a few rows for every head: the group's window keys
// are shared in L2 by its 16 heads and a row + FL_PF_ROWS prefetch targets a row that is not
// yet resident.
#ifndef FL_GRID_HF
#define FL_GRID_HF 0
#endif
// FL_ROW_LIST=1: the grid's row dimension is a fixed number of CTAs that walk Rows, the flush
// rows of the batch followed by -1 padding, instead of one CTA per batch row that exits unless
// the row flushes. Early exits of non-flush rows cost more than the flush itself when few rows
// flush, as in a batch spread over the window.
#ifndef FL_ROW_LIST
#define FL_ROW_LIST 0
#endif
#if FL_ROW_LIST && FL_GRID_HF
#error "FL_ROW_LIST walks rows along blockIdx.x"
#endif
// FL_PF_BLOCKS=d (> 0, with FL_STAGES): while block kb is consumed, key block kb + d of the
// same state is prefetched into L2 (one 128-byte line per lane) so the later cp.async stage
// copies hit L2; this deepens the effective pipeline without shared memory.
#ifndef FL_PF_BLOCKS
#define FL_PF_BLOCKS 0
#endif
// FL_EARLY=1: the first stage copies and block prefetches are issued before the window-key,
// dt and x loads of the prologue instead of after them.
#ifndef FL_EARLY
#define FL_EARLY 0
#endif
// FL_CSMEM=1: the group's C row (128 bf16) is staged in shared memory once instead of being
// loaded from global memory (two values per lane) in every key block.
#ifndef FL_CSMEM
#define FL_CSMEM 0
#endif
// FL_XSMEM=1: the x window (rows t <= wp, 64 values) is staged in shared memory with 16-byte
// loads (4 per lane) and the MMA B fragments are built from there, instead of 32 scalar
// 2-byte global loads per lane.
#ifndef FL_XSMEM
#define FL_XSMEM 0
#endif
#define XS_PITCH 72
// FL_PF_HEADS=d (> 0): the inputs of head h + d of the same row (x window, current x, dt ring
// row, first state block) are prefetched into L2. Rows run fastest in the grid, so that CTA
// is dispatched roughly one CTA lifetime later, when this one retires.
#ifndef FL_PF_HEADS
#define FL_PF_HEADS 0
#endif
// FL_DENSE=1: the ReplaySSM (dense) flush. Same staged-tile / L2-prefetch / three-part MMA state
// update and output as the P4 flush, but on vLLM's dense state layout [value][key] (keys
// contiguous, ss3 == 1) and without sketch rows, statistics, coefficient maps or the flush-step
// ring append (the original Replay16 kernel does not append on flush). Needs FL_STAGES > 0 and
// FL_HALF = 1: a unit is 8 keys x 64 values (64 rows of 32 B) staged with 8-byte cp.async into a
// [64][DENSE_PITCH] float tile (pitch 40 B: 8-byte aligned, conflict-free for the (g, c) reads).
#ifndef FL_DENSE
#define FL_DENSE 0
#endif
// Register diet (both aim at 16 warps/SM, i.e. <= 128 registers, like the dense flush build):
// FL_Q0SMEM=1: block_stats reads direction 0 from shared memory like directions 1..3 (no q0[16] array).
// FL_OUTSMEM=1: the readout partials are reduced over the eight row groups every block and accumulated
// in shared memory (no out[16] array live across the block loop).
#ifndef FL_Q0SMEM
#define FL_Q0SMEM 0
#endif
#ifndef FL_OUTSMEM
#define FL_OUTSMEM 0
#endif
// FL_STATS_FFMA2=1: the block statistics accumulate the two key rows of a lane as one packed
// fma.rn.f32x2 (Blackwell), halving the statistics FMA count; same per-element arithmetic.
#ifndef FL_STATS_FFMA2
#define FL_STATS_FFMA2 0
#endif
// FL_DENSE_KEYMAJOR=1: the dense state already has the P4 layout [key][value] (value contiguous; the
// campaign's NS_STATE_NP=1 allocation), so the standard staged path is used and only the sketch work,
// the U/W/AG writes and the flush-step ring append are compiled out.
#ifndef FL_DENSE_KEYMAJOR
#define FL_DENSE_KEYMAJOR 0
#endif
#define FL_DENSE_T (FL_DENSE && !FL_DENSE_KEYMAJOR)
#if SK_P != 64 && (!FL_STAGES || FL_XSMEM || FL_DENSE_T || FL_PF_HEADS)
#error "head dims other than 64 support the staged, non-XSMEM, key-major build only"
#endif
#if SK_N != 128 && (FL_DENSE_T || FL_PF_ROWS || FL_PF_HEADS)
#error "state sizes other than 128 support the key-major build without row/head prefetch only"
#endif
#if SK_W != 16 && (FL_XSMEM || FL_DENSE || FL_PF_ROWS || FL_PF_HEADS)
#error "windows other than 16 support the key-major sketch build without XSMEM, row or head prefetch only"
#endif
#define DENSE_PITCH 10
#if FL_DENSE_T && !(FL_STAGES > 0 && FL_HALF)
#error "FL_DENSE (t-major state) needs FL_STAGES > 0 and FL_HALF = 1"
#endif

struct __align__(16) WarpShared {
#if FL_STAGES && FL_DENSE_T
  float dstile[FL_STAGES][64 * DENSE_PITCH];             // dense half tiles: 64 values x 8 keys (+ pad)
#elif FL_STAGES
  float4 stile[FL_STAGES][FL_HALF ? 8 * (SK_P / 4) : 16 * (SK_P / 4)];   // state tiles (or half tiles) in flight
#endif
  bf16 ring[SK_W * RING_PITCH]; // window keys (t, key), zero beyond wp
#if FL_WSMEM
  bf16 xw[SK_W * XW_PITCH];     // x window (t, permuted value column), zero beyond wp
  float sc[SK_W];               // per-step scales
#endif
  float q[4 * SK_P];            // u0 raw, then orthonormal directions 1..3
#if FL_OUTSMEM
  float o[SK_P];                // readout accumulator (FL_OUTSMEM)
#endif
  float e[SK_N];                // energy
  float z[4 * SK_N];            // cross statistics
#if FL_XSMEM
  bf16 xs[16 * XS_PITCH];       // x window (t, value), zero beyond wp
#endif
#if FL_CSMEM
  bf16 crow[SK_N];              // the group's C row
#endif
};

// Element n of the SK_N-vector distributed as SK_KF consecutive values per lane.
__device__ __forceinline__ float elem(float v0, float v1, int n) {
  return __shfl_sync(FULL, (n & 1) == 0 ? v0 : v1, n >> 1);
}
__device__ __forceinline__ float elem(float v0, float v1, float v2, float v3, int n) {
  float v = (n & 3) == 0 ? v0 : (n & 3) == 1 ? v1 : (n & 3) == 2 ? v2 : v3;
  return __shfl_sync(FULL, v, n >> 2);
}
__device__ __forceinline__ float elem(float v0, float v1, float v2, float v3, float v4, float v5, float v6,
                                      float v7, int n) {
  const int i = n & 7;
  float v = i == 0 ? v0 : i == 1 ? v1 : i == 2 ? v2 : i == 3 ? v3 : i == 4 ? v4 : i == 5 ? v5 : i == 6 ? v6 : v7;
  return __shfl_sync(FULL, v, n >> 3);
}
#if SK_KF == 2
#define SK_KF_ELEMS(v) v[0], v[1]
#elif SK_KF == 4
#define SK_KF_ELEMS(v) v[0], v[1], v[2], v[3]
#else
#define SK_KF_ELEMS(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
#endif
// A lane's SK_KF consecutive entries of a map row in the U/W storage type.
__device__ __forceinline__ void st_uwk(uw_t* p, const float (&v)[SK_KF]) {
#if SK_KF == 2
#if UW_BF16
  *reinterpret_cast<unsigned*>(p) = pack_uw2(v[0], v[1]);
#else
  *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
#endif
#else
  st_uw4(p, make_float4(v[0], v[1], v[2], v[3]));
#if SK_KF == 8
  st_uw4(p + 4, make_float4(v[4], v[5], v[6], v[7]));
#endif
#endif
}

__device__ void finish(int lane, int m, long meta_row, int woff, int aoff, const WarpShared& sh,
                       uw_t* __restrict__ Tail, float* __restrict__ AG, int SM, int WROWS) {
  const int j0 = lane * SK_KF;
  float e[SK_KF], z0[SK_KF], z1[SK_KF], z2[SK_KF], z3[SK_KF];
  #pragma unroll
  for (int i = 0; i < SK_KF; ++i) {
    e[i] = sh.e[j0 + i]; z0[i] = sh.z[j0 + i];
    z1[i] = m > 1 ? sh.z[SK_N + j0 + i] : 0.f;
    z2[i] = m > 2 ? sh.z[2 * SK_N + j0 + i] : 0.f;
    z3[i] = m > 3 ? sh.z[3 * SK_N + j0 + i] : 0.f;
  }
  float esum = e[0];
  #pragma unroll
  for (int i = 1; i < SK_KF; ++i) esum += e[i];
  const float mean = fdiv(wsum(esum), (float)SK_N);
  const float safe_mean = mean > 0.f ? mean : 1.f;
  const float pivot_energy = __shfl_sync(FULL, e[0], 0);
  const float rs = fsqrt(safe_mean);
  const float rp = fsqrt(pivot_energy > 0.f ? pivot_energy : 1.f);
  float en[SK_KF], res[SK_KF], den[SK_KF], a[SK_KF], b0[SK_KF], b1[SK_KF], b2[SK_KF], b3[SK_KF];
  float g0[SK_KF], g1[SK_KF], g2[SK_KF], g3[SK_KF], y0[SK_KF], y1[SK_KF], y2[SK_KF], fac[SK_KF];
  const int mp = m < 4 ? m : 4;
  #pragma unroll
  for (int i = 0; i < SK_KF; ++i) {
    const int j = j0 + i;
    en[i] = fdiv(e[i], safe_mean);
    z0[i] = pivot_energy > 0.f ? fdiv(fdiv(z0[i], rp), rs) : 0.f;
    z1[i] = fdiv(z1[i], rs); z2[i] = fdiv(z2[i], rs); z3[i] = fdiv(z3[i], rs);
    float r = en[i] - z0[i] * z0[i] - z1[i] * z1[i] - z2[i] * z2[i] - z3[i] * z3[i];
    r = r > 0.f ? r : 0.f;
    res[i] = j < mp ? 0.f : r;
    den[i] = res[i] + .1f;
    a[i] = fdiv(res[i], den[i]);
    b0[i] = j < m ? fdiv(z0[i], den[i]) : 0.f;
    b1[i] = b2[i] = b3[i] = 0.f;
    g1[i] = g2[i] = g3[i] = 0.f;
    fac[i] = j < m ? fdiv(.1f, den[i]) : 1.f;
  }
  float zb00 = 0.f;
  #pragma unroll
  for (int i = 0; i < SK_KF; ++i) zb00 += z0[i] * b0[i];
  zb00 = wsum(zb00);
  #pragma unroll
  for (int i = 0; i < SK_KF; ++i) g0[i] = fdiv(b0[i], 1.f + zb00);
  if (m > 1) {
    float s10 = 0.f, s11 = 0.f;
    #pragma unroll
    for (int i = 0; i < SK_KF; ++i) { b1[i] = (j0 + i) < m ? fdiv(z1[i], den[i]) : 0.f; s10 += z1[i] * b0[i]; s11 += z1[i] * b1[i]; }
    s10 = wsum(s10); s11 = wsum(s11);
    const float l0_0 = fsqrt(1.f + zb00);
    const float l1_0 = fdiv(s10, l0_0);
    const float l1_1 = fsqrt(1.f + s11 - l1_0 * l1_0);
    #pragma unroll
    for (int i = 0; i < SK_KF; ++i) {
      y0[i] = fdiv(b0[i], l0_0); y1[i] = fdiv(b1[i] - l1_0 * y0[i], l1_1);
      g1[i] = fdiv(y1[i], l1_1); g0[i] = fdiv(y0[i] - l1_0 * g1[i], l0_0);
    }
    if (m > 2) {
      float s20 = 0.f, s21 = 0.f, s22 = 0.f;
      #pragma unroll
      for (int i = 0; i < SK_KF; ++i) { b2[i] = (j0 + i) < m ? fdiv(z2[i], den[i]) : 0.f; s20 += z2[i] * b0[i]; s21 += z2[i] * b1[i]; s22 += z2[i] * b2[i]; }
      s20 = wsum(s20); s21 = wsum(s21); s22 = wsum(s22);
      const float l2_0 = fdiv(s20, l0_0);
      const float l2_1 = fdiv(s21 - l2_0 * l1_0, l1_1);
      const float l2_2 = fsqrt(1.f + s22 - l2_0 * l2_0 - l2_1 * l2_1);
      #pragma unroll
      for (int i = 0; i < SK_KF; ++i) {
        y2[i] = fdiv(b2[i] - l2_0 * y0[i] - l2_1 * y1[i], l2_2);
        g2[i] = fdiv(y2[i], l2_2); g1[i] = fdiv(y1[i] - l2_1 * g2[i], l1_1);
        g0[i] = fdiv(y0[i] - l1_0 * g1[i] - l2_0 * g2[i], l0_0);
      }
      if (m > 3) {
        float s30 = 0.f, s31 = 0.f, s32 = 0.f, s33 = 0.f;
        #pragma unroll
        for (int i = 0; i < SK_KF; ++i) { b3[i] = (j0 + i) < m ? fdiv(z3[i], den[i]) : 0.f; s30 += z3[i] * b0[i]; s31 += z3[i] * b1[i]; s32 += z3[i] * b2[i]; s33 += z3[i] * b3[i]; }
        s30 = wsum(s30); s31 = wsum(s31); s32 = wsum(s32); s33 = wsum(s33);
        const float l3_0 = fdiv(s30, l0_0);
        const float l3_1 = fdiv(s31 - l3_0 * l1_0, l1_1);
        const float l3_2 = fdiv(s32 - l3_0 * l2_0 - l3_1 * l2_1, l2_2);
        const float l3_3 = fsqrt(1.f + s33 - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2);
        #pragma unroll
        for (int i = 0; i < SK_KF; ++i) {
          const float y3 = fdiv(b3[i] - l3_0 * y0[i] - l3_1 * y1[i] - l3_2 * y2[i], l3_3);
          g3[i] = fdiv(y3, l3_3); g2[i] = fdiv(y2[i] - l3_2 * g3[i], l2_2);
          g1[i] = fdiv(y1[i] - l2_1 * g2[i] - l3_1 * g3[i], l1_1);
          g0[i] = fdiv(y0[i] - l1_0 * g1[i] - l2_0 * g2[i] - l3_0 * g3[i], l0_0);
        }
      }
    }
  }
  uw_t* tbase = Tail + (meta_row * WROWS + woff) * SK_N + j0;
  const long index = meta_row * (5L * SM) + aoff + j0;
  if (m <= 4) {
    for (int n = 0; n < m; ++n) {
      const float f0 = elem(SK_KF_ELEMS(g0), n);
      const float f1 = elem(SK_KF_ELEMS(g1), n);
      const float f2 = elem(SK_KF_ELEMS(g2), n);
      const float f3 = elem(SK_KF_ELEMS(g3), n);
      float mrg[SK_KF];
      #pragma unroll
      for (int i = 0; i < SK_KF; ++i) mrg[i] = (f0 * z0[i] + f1 * z1[i] + f2 * z2[i] + f3 * z3[i]) * fac[i];
      st_uwk(tbase + n * SK_N, mrg);
    }
  } else {
    auto st_row = [&](uw_t* p, const float (&z)[SK_KF]) {
      float t[SK_KF];
      #pragma unroll
      for (int i = 0; i < SK_KF; ++i) t[i] = z[i] * fac[i];
      st_uwk(p, t);
    };
    st_row(tbase, z0);
    st_row(tbase + SK_N, z1);
    st_row(tbase + 2 * SK_N, z2);
    st_row(tbase + 3 * SK_N, z3);
    #pragma unroll
    for (int i = 0; i < SK_KF; ++i) {
      if (j0 + i < m) {
        AG[index + i] = a[i]; AG[index + SM + i] = g0[i]; AG[index + 2L * SM + i] = g1[i];
        AG[index + 3L * SM + i] = g2[i]; AG[index + 4L * SM + i] = g3[i];
      }
    }
  }
}

__device__ __forceinline__ void st_pred(float* p, float4 v, bool pred) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %5, 0;\n @p st.global.v4.f32 [%0], {%1,%2,%3,%4};\n}\n"
               :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"((int)pred) : "memory");
}
// Rows 0..3 of the updated state (already in sh.q[0..3]) -> orthonormal
// directions 1..3 written back to sh.q[1..3]. Kept out of line so the hot
// loop stays small for the instruction cache.
__device__ __noinline__ void build_directions(int lane, int m, float* q) {
  // Lane owns values 2 * lane + 64 * i (i < NV); pairs beyond the head dim are zero.
  constexpr int NV = (SK_P + 63) / 64;
  float2 U0[NV], U1[NV], U2[NV], U3[NV];
  #pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int v = 2 * lane + 64 * i;
    const bool in = v < SK_P;
    U0[i] = in ? *reinterpret_cast<const float2*>(q + v) : make_float2(0.f, 0.f);
    U1[i] = in ? *reinterpret_cast<const float2*>(q + SK_P + v) : make_float2(0.f, 0.f);
    U2[i] = in ? *reinterpret_cast<const float2*>(q + 2 * SK_P + v) : make_float2(0.f, 0.f);
    U3[i] = in ? *reinterpret_cast<const float2*>(q + 3 * SK_P + v) : make_float2(0.f, 0.f);
  }
  auto dot = [&](const float2 (&a)[NV], const float2 (&b)[NV]) {
    float t = 0.f;
    #pragma unroll
    for (int i = 0; i < NV; ++i) t += a[i].x * b[i].x + a[i].y * b[i].y;
    return wsum(t);
  };
  auto axpy = [&](float2 (&w)[NV], const float2 (&v)[NV], float d) {
    #pragma unroll
    for (int i = 0; i < NV; ++i) { w[i].x -= v[i].x * d; w[i].y -= v[i].y * d; }
  };
  auto unit = [&](float2 (&w)[NV], float n) {
    #pragma unroll
    for (int i = 0; i < NV; ++i) w[i] = make_float2(fdiv(w[i].x, n), fdiv(w[i].y, n));
  };
  auto zero = [&](float2 (&w)[NV]) {
    #pragma unroll
    for (int i = 0; i < NV; ++i) w[i] = make_float2(0.f, 0.f);
  };
  const float e0 = dot(U0, U0);
  float2 v0[NV], v1[NV], v2[NV], v3[NV];
  #pragma unroll
  for (int i = 0; i < NV; ++i) v0[i] = U0[i];
  unit(v0, fsqrt(e0 > 0.f ? e0 : 1.f));
  zero(v1); zero(v2); zero(v3);
  if (m > 1) {
    float2 w1[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i) w1[i] = U1[i];
    axpy(w1, v0, dot(v0, U1));
    axpy(w1, v0, dot(v0, w1));
    const float e1 = dot(w1, w1);
    const bool keep1 = e1 > 1.e-12f * dot(U1, U1);
    unit(w1, fsqrt(keep1 ? e1 : 1.f));
    if (keep1) {
      #pragma unroll
      for (int i = 0; i < NV; ++i) v1[i] = w1[i];
    }
  }
  if (m > 2) {
    float2 w2[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i) w2[i] = U2[i];
    axpy(w2, v0, dot(v0, U2));
    axpy(w2, v1, dot(v1, w2));
    axpy(w2, v0, dot(v0, w2));
    axpy(w2, v1, dot(v1, w2));
    const float e2 = dot(w2, w2);
    const bool keep2 = e2 > 1.e-12f * dot(U2, U2);
    unit(w2, fsqrt(keep2 ? e2 : 1.f));
    if (keep2) {
      #pragma unroll
      for (int i = 0; i < NV; ++i) v2[i] = w2[i];
    }
  }
  if (m > 3) {
    float2 w3[NV];
    #pragma unroll
    for (int i = 0; i < NV; ++i) w3[i] = U3[i];
    axpy(w3, v0, dot(v0, U3));
    axpy(w3, v1, dot(v1, w3));
    axpy(w3, v2, dot(v2, w3));
    axpy(w3, v0, dot(v0, w3));
    axpy(w3, v1, dot(v1, w3));
    axpy(w3, v2, dot(v2, w3));
    const float e3 = dot(w3, w3);
    const bool keep3 = e3 > 1.e-12f * dot(U3, U3);
    unit(w3, fsqrt(keep3 ? e3 : 1.f));
    if (keep3) {
      #pragma unroll
      for (int i = 0; i < NV; ++i) v3[i] = w3[i];
    }
  }
  __syncwarp();
  #pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int v = 2 * lane + 64 * i;
    if (v < SK_P) {
      *reinterpret_cast<float2*>(q + SK_P + v) = v1[i];
      *reinterpret_cast<float2*>(q + 2 * SK_P + v) = v2[i];
      *reinterpret_cast<float2*>(q + 3 * SK_P + v) = v3[i];
    }
  }
  __syncwarp();
}

// Energy and NQ cross statistics of the 16-row block held in acc; one
// straight-line instantiation per rank bucket so no lane pays for unused ranks.
template <int NQ>
__device__ __forceinline__ void block_stats(const float (&acc)[2 * SK_NP][4], const float* q0,
#if QREG
                                            const float (&q1)[4 * SK_NP], const float (&q2)[4 * SK_NP], const float (&q3)[4 * SK_NP],
#else
                                            const float* qs,
#endif
                                            int k0, int g, int c, float* se, float* sz) {
  float en[2] = {0.f, 0.f}, c0[2] = {0.f, 0.f}, c1[2] = {0.f, 0.f}, c2[2] = {0.f, 0.f}, c3[2] = {0.f, 0.f};
  #pragma unroll
  for (int p = 0; p < SK_NP; ++p) {
    float t1[4], t2[4], t3[4];
#if QREG
    #pragma unroll
    for (int e = 0; e < 4; ++e) { t1[e] = q1[4 * p + e]; t2[e] = q2[4 * p + e]; t3[e] = q3[4 * p + e]; }
#else
    if (NQ > 1) { const float4 v = *reinterpret_cast<const float4*>(qs + SK_P + 16 * p + 4 * c); t1[0] = v.x; t1[1] = v.y; t1[2] = v.z; t1[3] = v.w; }
    if (NQ > 2) { const float4 v = *reinterpret_cast<const float4*>(qs + 2 * SK_P + 16 * p + 4 * c); t2[0] = v.x; t2[1] = v.y; t2[2] = v.z; t2[3] = v.w; }
    if (NQ > 3) { const float4 v = *reinterpret_cast<const float4*>(qs + 3 * SK_P + 16 * p + 4 * c); t3[0] = v.x; t3[1] = v.y; t3[2] = v.z; t3[3] = v.w; }
#endif
#if FL_Q0SMEM
    float t0[4];
    { const float4 v = *reinterpret_cast<const float4*>(qs + 16 * p + 4 * c); t0[0] = v.x; t0[1] = v.y; t0[2] = v.z; t0[3] = v.w; }
#else
    const float* t0 = q0 + 4 * p;
#endif
#if FL_STATS_FFMA2
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 u2 = make_float2(acc[2 * p + (e >> 1)][e & 1], acc[2 * p + (e >> 1)][2 + (e & 1)]);   // rows g and g + 8
      float2 en2 = make_float2(en[0], en[1]), c02 = make_float2(c0[0], c0[1]);
      en2 = fl_ffma2(u2, u2, en2); c02 = fl_ffma2(u2, make_float2(t0[e], t0[e]), c02);
      en[0] = en2.x; en[1] = en2.y; c0[0] = c02.x; c0[1] = c02.y;
      if (NQ > 1) { float2 v = fl_ffma2(u2, make_float2(t1[e], t1[e]), make_float2(c1[0], c1[1])); c1[0] = v.x; c1[1] = v.y; }
      if (NQ > 2) { float2 v = fl_ffma2(u2, make_float2(t2[e], t2[e]), make_float2(c2[0], c2[1])); c2[0] = v.x; c2[1] = v.y; }
      if (NQ > 3) { float2 v = fl_ffma2(u2, make_float2(t3[e], t3[e]), make_float2(c3[0], c3[1])); c3[0] = v.x; c3[1] = v.y; }
    }
#else
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float u = acc[2 * p + (e >> 1)][2 * rr + (e & 1)];
        en[rr] = fmaf(u, u, en[rr]);
        c0[rr] = fmaf(u, t0[e], c0[rr]);
        if (NQ > 1) c1[rr] = fmaf(u, t1[e], c1[rr]);
        if (NQ > 2) c2[rr] = fmaf(u, t2[e], c2[rr]);
        if (NQ > 3) c3[rr] = fmaf(u, t3[e], c3[rr]);
      }
    }
#endif
  }
  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int k = k0 + g + 8 * rr;
    float e0 = en[rr], x0 = c0[rr], x1 = c1[rr], x2 = c2[rr], x3 = c3[rr];
    e0 += __shfl_xor_sync(FULL, e0, 1); e0 += __shfl_xor_sync(FULL, e0, 2);
    x0 += __shfl_xor_sync(FULL, x0, 1); x0 += __shfl_xor_sync(FULL, x0, 2);
    if (NQ > 1) { x1 += __shfl_xor_sync(FULL, x1, 1); x1 += __shfl_xor_sync(FULL, x1, 2); }
    if (NQ > 2) { x2 += __shfl_xor_sync(FULL, x2, 1); x2 += __shfl_xor_sync(FULL, x2, 2); }
    if (NQ > 3) { x3 += __shfl_xor_sync(FULL, x3, 1); x3 += __shfl_xor_sync(FULL, x3, 2); }
    if (c == 0) {
      se[k] = e0; sz[k] = x0;
      if (NQ > 1) sz[SK_N + k] = x1;
      if (NQ > 2) sz[2 * SK_N + k] = x2;
      if (NQ > 3) sz[3 * SK_N + k] = x3;
    }
  }
}

extern "C" __global__ void __launch_bounds__(32 * WARPS, MINB)
flush_kernel(float* __restrict__ S, const bf16* __restrict__ X, const bf16* __restrict__ DT,
             const bf16* __restrict__ Bias, const float* __restrict__ A, const bf16* __restrict__ B,
             const bf16* __restrict__ C, const bf16* __restrict__ D, void* __restrict__ O,
             bf16* __restrict__ XR, float* __restrict__ DR, bf16* __restrict__ BR,
             const int* __restrict__ WP, const unsigned char* __restrict__ FL,
             const int* __restrict__ Slots, const int* __restrict__ Width, const int* __restrict__ Map,
             uw_t* __restrict__ Sketch, const int* __restrict__ SkOff, uw_t* __restrict__ Tail,
             float* __restrict__ AG, const int* __restrict__ Offsets, const int* __restrict__ WOff,
             FlushStrides st, int null_slot, int softplus, int out_f32, int SKROWS, int SM, int WROWS,
             const int* __restrict__ Rows, int batch) {
#if FL_WSMEM
  extern __shared__ __align__(16) unsigned char fl_smem[];
  WarpShared* shared = reinterpret_cast<WarpShared*>(fl_smem);
#else
  __shared__ WarpShared shared[WARPS];
#endif
#if FL_ROW_LIST
  const int hb = blockIdx.y;
  const int nrows = batch;
  for (int it = blockIdx.x; it < batch; it += gridDim.x) {
  const int row = Rows[it];
  if (row < 0) break;
  const int slot = Slots[row * st.slots];
  if (slot == null_slot) continue;
#else
#if FL_GRID_HF
  const int row = blockIdx.y;
  const int hb = blockIdx.x;
  const int nrows = gridDim.y;
#else
  const int row = blockIdx.x;
  const int hb = blockIdx.y;
  const int nrows = gridDim.x;
#endif
  if (FL[row] == 0) return;
  const int slot = Slots[row * st.slots];
  if (slot == null_slot) return;
#endif
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = hb * WARPS + warp;
#if FL_PF_ROWS > 0
  // Row + FL_PF_ROWS bookkeeping, loaded with the prologue and consumed after block 0.
  const int rp = row + FL_PF_ROWS;
  const bool pf_row = rp < nrows && FL[rp] != 0;
  const int sp2 = pf_row ? Slots[rp * st.slots] : null_slot;
  const int wp2 = pf_row ? WP[rp] : 0;
#endif
  const int group = h / SK_HPG;
  const int g = lane >> 2, c = lane & 3;
  const int wp = WP[row];
#if FL_DENSE
  const int m = 0;                                  // no sketch: the rank branches compile out
  const long meta_row = 0;
  const int skoff = 0;
#else
  const int m = Width[h];
  const long meta_row = Map[row];
  const int skoff = SkOff[h];
#endif
  const int keep = m == 0 ? SK_N : m;
  WarpShared& sh = shared[warp];
  float* sp = S + slot * st.ss0 + h * st.ss1;
#if FL_STAGES
  // Key block kb is 16 rows x 64 values = 4 KB, contiguous when ss3 == 64 (checked by the
  // wrapper). Unit u (a block, or a half block with FL_HALF) is copied by lane l in 16-byte
  // chunks l + 32 i into stage buffer u % FL_STAGES.
  constexpr int UNIT_F4 = FL_HALF ? 8 * (SK_P / 4) : 16 * (SK_P / 4);   // float4 per unit
  constexpr int UNITS = FL_HALF ? 2 * SK_NKB : SK_NKB;
#if FL_DENSE_T
  auto issue_tile = [&](int u) {
    // Unit u = keys 8u..8u+7 of all 64 value rows: 4 chunks of 8 B per row, 8 chunks per lane.
    const float* src = sp + u * 8;
    float* dst = sh.dstile[u % FL_STAGES];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int ch = lane + 32 * i, v = ch >> 2, part = ch & 3;
      fl_cp_async8(dst + v * DENSE_PITCH + 2 * part, src + v * st.ss2 + 2 * part);
    }
    fl_cp_commit();
  };
#else
  auto issue_tile = [&](int u) {
    const float* src = sp + u * (UNIT_F4 * 4);
    float4* dst = sh.stile[u % FL_STAGES];
    #pragma unroll
    for (int i = 0; i < UNIT_F4 / 32; ++i) fl_cp_async16(dst + lane + 32 * i, src + 4 * (lane + 32 * i));
    fl_cp_commit();
  };
#endif
  auto wait_unit = [&](int u) {
    // Unit u must have landed; the units issued after it (at most FL_STAGES-1, fewer at the end)
    // may still be in flight.
    const int pending = min(FL_STAGES - 1, UNITS - 1 - u);
    if (pending >= 5) fl_cp_wait<(FL_STAGES > 5 ? 5 : 0)>();
    else if (pending == 4) fl_cp_wait<(FL_STAGES > 4 ? 4 : 0)>();
    else if (pending == 3) fl_cp_wait<(FL_STAGES > 3 ? 3 : 0)>();
    else if (pending == 2) fl_cp_wait<(FL_STAGES > 2 ? 2 : 0)>();
    else if (pending == 1) fl_cp_wait<(FL_STAGES > 1 ? 1 : 0)>();
    else fl_cp_wait<0>();
    __syncwarp();
  };
#if FL_PF_BLOCKS > 0
  constexpr int PF_FIRST = FL_HALF ? (FL_STAGES + 1) / 2 : FL_STAGES;   // first block not staged
#if FL_DENSE_T
  // Block kbp spans 64 B of every value row; lanes cover rows 2*lane and 2*lane+1 (the 128-byte
  // line also holds the neighbouring block, so odd blocks mostly hit).
  auto pf_block = [&](int kbp) {
    fl_pf_if(sp + kbp * 16 + (2 * lane) * st.ss2, kbp < 8);
    fl_pf_if(sp + kbp * 16 + (2 * lane + 1) * st.ss2, kbp < 8);
  };
#else
  auto pf_block = [&](int kbp) {
    #pragma unroll
    for (int i = 0; i < (SK_P + 63) / 64; ++i)
      fl_pf_if(reinterpret_cast<const char*>(sp + kbp * 16 * SK_P) + (lane + 32 * i) * 128, kbp < SK_NKB && lane + 32 * i < SK_P / 2);
  };
#endif
#endif
  auto fl_issue_prologue = [&]() {
    #pragma unroll
    for (int s0i = 0; s0i < FL_STAGES; ++s0i) issue_tile(s0i);
#if FL_PF_BLOCKS > 0
    #pragma unroll
    for (int kbp = PF_FIRST; kbp < PF_FIRST + FL_PF_BLOCKS; ++kbp) pf_block(kbp);
#endif
  };
#if FL_EARLY
  fl_issue_prologue();
#endif
#endif

  // Window decay and per-step scales: lane l holds steps l + 32 i (lanes 0..15 for a window
  // of 16). Scans of SCAN lanes per register, carried across registers.
  float dt_cur = __bfloat162float(DT[row * st.dts0 + h * st.dts1]) + __bfloat162float(Bias[h * st.biass0]);
  if (softplus) dt_cur = dt_cur <= 20.f ? flog(fexp(dt_cur) + 1.f) : dt_cur;
  const float decay_a = A[h * st.as0];
  constexpr int SK_WR = (SK_W + 31) / 32, SCAN = SK_W < 32 ? 16 : 32;
  float dtv[SK_WR], cum[SK_WR], run = 0.f;
  #pragma unroll
  for (int i = 0; i < SK_WR; ++i) {
    const int t = lane + 32 * i;
    dtv[i] = 0.f;
    if (t < wp) dtv[i] = DR[slot * st.drs0 + h * st.drs1 + t * st.drs2];
    else if (t == wp) dtv[i] = dt_cur;
    cum[i] = dtv[i];
    #pragma unroll
    for (int o = 1; o < SCAN; o <<= 1) { float u = __shfl_up_sync(FULL, cum[i], o); if (lane >= o) cum[i] += u; }
    if (i > 0) cum[i] += run;
    run = __shfl_sync(FULL, cum[i], SCAN - 1);
  }
  const float total = decay_a * run;
  const float decay = fexp(total);
  float scale[SK_WR];
  scale[0] = (lane <= wp && lane < SK_W) ? dtv[0] * fexp(total - decay_a * cum[0]) : 0.f;
  #pragma unroll
  for (int i = 1; i < SK_WR; ++i) {
    const int t = lane + 32 * i;
    scale[i] = (t <= wp && t < SK_W) ? dtv[i] * fexp(total - decay_a * cum[i]) : 0.f;
  }
#if FL_WSMEM
  #pragma unroll
  for (int i = 0; i < SK_WR; ++i)
    if (lane + 32 * i < SK_W) sh.sc[lane + 32 * i] = scale[i];
#else
  const float s0 = __shfl_sync(FULL, scale[0], 2 * c), s1 = __shfl_sync(FULL, scale[0], 2 * c + 1);
  const float s2 = __shfl_sync(FULL, scale[0], 2 * c + 8), s3 = __shfl_sync(FULL, scale[0], 2 * c + 9);
#endif

  // Stage window keys for the group: rows t < wp from the ring, row wp from B.
  const bf16* brp = BR + slot * st.brs0 + group * st.brs1;
  const bf16* bp = B + row * st.bs0 + group * st.bs1;
  #pragma unroll
  for (int i = 0; i < SK_NKB * SK_WT; ++i) {
    const int ch = lane + 32 * i, t = ch / (SK_N / 8), part = ch % (SK_N / 8);
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (t < wp) v = *reinterpret_cast<const uint4*>(brp + t * st.brs2 + part * 8);
    else if (t == wp) v = *reinterpret_cast<const uint4*>(bp + part * 8);
    *reinterpret_cast<uint4*>(sh.ring + t * RING_PITCH + part * 8) = v;
  }
  // Window values as MMA B fragments (t along k, permuted value columns).
  // Value stride is one element for both the ring and x (checked by the wrapper).
  const bf16* xrp = XR + slot * st.xrs0 + h * st.xrs1;
  const bf16* xp = X + row * st.xs0 + h * st.xs1;
#if FL_PF_HEADS > 0
  {
    const int h2 = h + FL_PF_HEADS;
    if (h2 < 128) {
      fl_pf_if(XR + slot * st.xrs0 + h2 * st.xrs1 + lane * st.xrs2, lane < wp);
      fl_pf_if(X + row * st.xs0 + h2 * st.xs1, lane == 16);
      fl_pf_if(DR + slot * st.drs0 + h2 * st.drs1, lane == 17);
      fl_pf_if(reinterpret_cast<const char*>(S + slot * st.ss0 + h2 * st.ss1) + lane * 128, true);
    }
  }
#endif
#if FL_CSMEM
  #pragma unroll
  for (int i = 0; i < (SK_N + 255) / 256; ++i) {
    const int ch = lane + 32 * i;
    if (ch < SK_N / 8) *reinterpret_cast<uint4*>(sh.crow + ch * 8) = *reinterpret_cast<const uint4*>(C + row * st.cs0 + group * st.cs1 + ch * 8);
  }
#endif
#if FL_XSMEM
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int ch = lane + 32 * i, t = ch >> 3, part = ch & 7;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (t < wp) v = *reinterpret_cast<const uint4*>(xrp + t * st.xrs2 + part * 8);
    else if (t == wp) v = *reinterpret_cast<const uint4*>(xp + part * 8);
    *reinterpret_cast<uint4*>(sh.xs + t * XS_PITCH + part * 8) = v;
  }
  __syncwarp();
  uint32_t xf[8][2];
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int v = vmap(j, g);
    bf16 val[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) val[q] = sh.xs[(2 * c + (q & 1) + 8 * (q >> 1)) * XS_PITCH + v];
    xf[j][0] = pack_bf(val[0], val[1]); xf[j][1] = pack_bf(val[2], val[3]);
  }
#elif FL_WSMEM
  // Value v goes to column 16 (v / 16) + 8 ((v / 2) % 2) + 2 ((v / 4) % 4) + v % 2 of its row t,
  // so the eight columns 8 j .. 8 j + 7 are the values V(j, 0..7) of MMA n-block j and
  // ldmatrix.trans over rows t yields the B fragments directly.
  #pragma unroll
  for (int i = 0; i < SK_W * SK_P / 256; ++i) {
    const int ch = lane + 32 * i, t = ch / (SK_P / 8), v0 = 8 * (ch % (SK_P / 8));
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (t < wp) v = *reinterpret_cast<const uint4*>(xrp + t * st.xrs2 + v0);
    else if (t == wp) v = *reinterpret_cast<const uint4*>(xp + v0);
    const unsigned pair[4] = {v.x, v.y, v.z, v.w};
    bf16* xrow = sh.xw + t * XW_PITCH + 16 * (v0 >> 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int vv = (v0 & 15) + 2 * q;
      *reinterpret_cast<unsigned*>(xrow + 8 * ((vv >> 1) & 1) + 2 * (vv >> 2)) = pair[q];
    }
  }
#else
  const bf16* xt[4];
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int t = 2 * c + (q & 1) + 8 * (q >> 1);
    xt[q] = t < wp ? xrp + t * st.xrs2 : (t == wp ? xp : nullptr);
  }
  uint32_t xf[2 * SK_NP][2];
  #pragma unroll
  for (int j = 0; j < 2 * SK_NP; ++j) {
    const int v = vmap(j, g);
    bf16 val[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) val[q] = xt[q] ? xt[q][v] : __float2bfloat16(0.f);
    xf[j][0] = pack_bf(val[0], val[1]); xf[j][1] = pack_bf(val[2], val[3]);
  }
#endif
  __syncwarp();

  uw_t* up = Sketch + (meta_row * SKROWS + skoff) * SK_P;
  const bf16* cp = C + row * st.cs0 + group * st.cs1;
#if FL_OUTSMEM
  if (lane < SK_P / 4) *reinterpret_cast<float4*>(sh.o + 4 * lane) = make_float4(0.f, 0.f, 0.f, 0.f);
  float out[4 * SK_NP];   // only the final readout after the loop
#else
  float out[4 * SK_NP];
  #pragma unroll
  for (int i = 0; i < 4 * SK_NP; ++i) out[i] = 0.f;
#endif
#if FL_Q0SMEM
  float q0[1];     // unused: direction 0 is read from shared memory
#else
  float q0[4 * SK_NP];
#endif
#if QREG
  float q1[4 * SK_NP], q2[4 * SK_NP], q3[4 * SK_NP];
#endif
#if FL_PF_ROWS > 0
  // Prologue inputs of row + FL_PF_ROWS for this head: one 128-byte line per lane, issued
  // once block 0 is under way so the bookkeeping loads above are off the critical path.
  auto pf_next = [&]() {
    if (sp2 == null_slot) return;
    fl_pf_if(XR + (long)sp2 * st.xrs0 + h * st.xrs1 + lane * st.xrs2, lane < wp2);
    fl_pf_if(reinterpret_cast<const char*>(S + (long)sp2 * st.ss0 + h * st.ss1) + lane * 128, true);   // state block 0
    fl_pf_if(X + (long)rp * st.xs0 + h * st.xs1, lane == 0);
    fl_pf_if(DR + (long)sp2 * st.drs0 + h * st.drs1, lane == 1);
    fl_pf_if(C + (long)rp * st.cs0 + group * st.cs1 + 64 * (lane & 1), lane < 4 && lane >= 2);
    if ((h % SK_HPG) == 0) {   // window keys are per group
      fl_pf_if(BR + (long)sp2 * st.brs0 + group * st.brs1 + (lane >> 1) * st.brs2 + 64 * (lane & 1), (lane >> 1) < wp2);
      fl_pf_if(B + (long)rp * st.bs0 + group * st.bs1 + 64 * (lane & 1), lane < 2);
    }
  };
#endif
#if FL_STAGES
#if !FL_EARLY
  fl_issue_prologue();
#endif
#else
  float4 ob[8];
  #pragma unroll
  for (int rr = 0; rr < 2; ++rr)
    #pragma unroll
    for (int p = 0; p < 4; ++p)
      ob[rr * 4 + p] = *reinterpret_cast<const float4*>(sp + (g + 8 * rr) * st.ss3 + 16 * p + 4 * c);
#endif

  constexpr int kUnroll = UNROLL;
  #pragma unroll kUnroll
  for (int kb = 0; kb < SK_NKB; ++kb) {
    const int k0 = kb * 16;
#if FL_PF_ROWS > 0
    if (kb == 1) pf_next();
#endif
#if FL_STAGES && FL_PF_BLOCKS > 0
    if (kb > 0) pf_block(kb - 1 + PF_FIRST + FL_PF_BLOCKS);   // keeps FL_PF_BLOCKS blocks ahead of the stages
#endif
#if FL_CSMEM
    const float cq0 = __bfloat162float(sh.crow[k0 + g]);
    const float cq1 = __bfloat162float(sh.crow[k0 + g + 8]);
#else
    const float cq0 = __bfloat162float(cp[(k0 + g) * st.cs2]);
    const float cq1 = __bfloat162float(cp[(k0 + g + 8) * st.cs2]);
#endif
#if FL_STAGES && !FL_HALF
    wait_unit(kb);
    const float4* tile = sh.stile[kb % FL_STAGES];
#endif
    float4 nb[8];
#if PREFETCH && !FL_STAGES
    if (kb + 1 < SK_NKB) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int p = 0; p < 4; ++p)
          nb[rr * 4 + p] = *reinterpret_cast<const float4*>(sp + (k0 + 16 + g + 8 * rr) * st.ss3 + 16 * p + 4 * c);
    }
#endif
    // Window contribution of this 16-key block, accumulated over the 16-step window tiles:
    // acc = sum_t (s_t B_t)(keys) x_t(values), with the scaled keys split into three BF16 parts.
    float acc[2 * SK_NP][4];
    #pragma unroll
    for (int j = 0; j < 2 * SK_NP; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    #pragma unroll
    for (int wt = 0; wt < SK_WT; ++wt) {
      // A fragments for this 16-key block and window tile through transposed ldmatrix.
      uint32_t a[4];
      {
        const int qd = lane >> 3, r = lane & 7;
        ldmatrix_x4_trans(a, sh.ring + (16 * wt + (qd >> 1) * 8 + r) * RING_PITCH + k0 + (qd & 1) * 8);
      }
#if FL_WSMEM
      const float2 sa = *reinterpret_cast<const float2*>(sh.sc + 16 * wt + 2 * c);
      const float2 sb = *reinterpret_cast<const float2*>(sh.sc + 16 * wt + 2 * c + 8);
      const float s0 = sa.x, s1 = sa.y, s2 = sb.x, s3 = sb.y;
      uint32_t xf[2 * SK_NP][2];
      #pragma unroll
      for (int jp = 0; jp < SK_NP; ++jp) {
        uint32_t r4[4];
        ldmatrix_x4_trans(r4, sh.xw + (16 * wt + 8 * ((lane >> 3) & 1) + (lane & 7)) * XW_PITCH + 8 * (2 * jp + (lane >> 4)));
        xf[2 * jp][0] = r4[0]; xf[2 * jp][1] = r4[1]; xf[2 * jp + 1][0] = r4[2]; xf[2 * jp + 1][1] = r4[3];
      }
#endif
      uint32_t p0[4], p1[4], p2[4];
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        float2 f = unpack2(a[r]);
        f.x *= (r & 2) ? s2 : s0; f.y *= (r & 2) ? s3 : s1;
        p0[r] = pack2(f.x, f.y);
        float2 y = unpack2(p0[r]); f.x -= y.x; f.y -= y.y;
        p1[r] = pack2(f.x, f.y);
        y = unpack2(p1[r]); f.x -= y.x; f.y -= y.y;
        p2[r] = pack2(f.x, f.y);
      }
      #pragma unroll
      for (int j = 0; j < 2 * SK_NP; ++j) {
        mma16816(acc[j], p2, xf[j]);
        mma16816(acc[j], p1, xf[j]);
        mma16816(acc[j], p0, xf[j]);
      }
    }
#if !PREFETCH && !FL_STAGES
    if (kb > 0) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int p = 0; p < 4; ++p)
          ob[rr * 4 + p] = *reinterpret_cast<const float4*>(sp + (k0 + g + 8 * rr) * st.ss3 + 16 * p + 4 * c);
    }
#endif
    // State update and state store; sketch rows only while the block has kept rows.
    const bool block_kept = k0 < keep;
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int k = k0 + g + 8 * rr;
#if FL_STAGES && FL_DENSE_T
      const int hu = 2 * kb + rr;
      wait_unit(hu);
      const float* dtile = sh.dstile[hu % FL_STAGES] + g;              // key g of the half tile
#elif FL_STAGES && FL_HALF
      const int hu = 2 * kb + rr;
      wait_unit(hu);
      const float4* tile = sh.stile[hu % FL_STAGES] - 8 * (SK_P / 4) * rr;   // rows g + 8 rr map to the half tile
#endif
      #pragma unroll
      for (int p = 0; p < SK_NP; ++p) {
#if FL_STAGES && FL_DENSE_T
        const int vb = 16 * p + 4 * c;
        const float4 o = make_float4(dtile[vb * DENSE_PITCH], dtile[(vb + 1) * DENSE_PITCH],
                                     dtile[(vb + 2) * DENSE_PITCH], dtile[(vb + 3) * DENSE_PITCH]);
#elif FL_STAGES
        const float4 o = tile[(g + 8 * rr) * (SK_P / 4) + 4 * p + c];
#else
        float4 o = ob[rr * 4 + p];
#endif
        float4 u;
        u.x = fmaf(o.x, decay, acc[2 * p][2 * rr]);
        u.y = fmaf(o.y, decay, acc[2 * p][2 * rr + 1]);
        u.z = fmaf(o.z, decay, acc[2 * p + 1][2 * rr]);
        u.w = fmaf(o.w, decay, acc[2 * p + 1][2 * rr + 1]);
        acc[2 * p][2 * rr] = u.x; acc[2 * p][2 * rr + 1] = u.y;
        acc[2 * p + 1][2 * rr] = u.z; acc[2 * p + 1][2 * rr + 1] = u.w;
#if FL_DENSE_T
        {
          float* dp = sp + (long)(16 * p + 4 * c) * st.ss2 + k;      // [value][key], keys contiguous
          dp[0] = u.x; dp[st.ss2] = u.y; dp[2 * st.ss2] = u.z; dp[3 * st.ss2] = u.w;
        }
#else
        *reinterpret_cast<float4*>(sp + k * st.ss3 + 16 * p + 4 * c) = u;
#if !FL_DENSE
        if (block_kept) st_pred_uw(up + k * SK_P + 16 * p + 4 * c, u, k < keep);
#endif
#endif
      }
#if FL_STAGES && FL_HALF
      __syncwarp();                                 // this half tile is consumed: reuse its buffer
      if (hu + FL_STAGES < UNITS) issue_tile(hu + FL_STAGES);
#endif
    }
#if FL_STAGES && !FL_HALF
    __syncwarp();                                   // every lane is done with this stage buffer
    if (kb + FL_STAGES < SK_NKB) issue_tile(kb + FL_STAGES);
#elif FL_STAGES
#elif PREFETCH
    #pragma unroll
    for (int i = 0; i < 8; ++i) ob[i] = nb[i];
#endif
    if (m > 0) {
      if (kb == 0) {
        // Rows 0..3 of the updated state seed the P4 directions.
        if (g < 4) {
          #pragma unroll
          for (int p = 0; p < SK_NP; ++p)
            *reinterpret_cast<float4*>(sh.q + g * SK_P + 16 * p + 4 * c) =
                make_float4(acc[2 * p][0], acc[2 * p][1], acc[2 * p + 1][0], acc[2 * p + 1][1]);
        }
        __syncwarp();
        build_directions(lane, m, sh.q);
        #pragma unroll
        for (int p = 0; p < SK_NP; ++p) {
#if !FL_Q0SMEM
          const float4 t0 = *reinterpret_cast<const float4*>(sh.q + 16 * p + 4 * c);
          q0[4 * p] = t0.x; q0[4 * p + 1] = t0.y; q0[4 * p + 2] = t0.z; q0[4 * p + 3] = t0.w;
#endif
#if QREG
          const float4 t1 = *reinterpret_cast<const float4*>(sh.q + SK_P + 16 * p + 4 * c);
          q1[4 * p] = t1.x; q1[4 * p + 1] = t1.y; q1[4 * p + 2] = t1.z; q1[4 * p + 3] = t1.w;
          const float4 t2 = *reinterpret_cast<const float4*>(sh.q + 2 * SK_P + 16 * p + 4 * c);
          q2[4 * p] = t2.x; q2[4 * p + 1] = t2.y; q2[4 * p + 2] = t2.z; q2[4 * p + 3] = t2.w;
          const float4 t3 = *reinterpret_cast<const float4*>(sh.q + 3 * SK_P + 16 * p + 4 * c);
          q3[4 * p] = t3.x; q3[4 * p + 1] = t3.y; q3[4 * p + 2] = t3.z; q3[4 * p + 3] = t3.w;
#endif
        }
      }
#if QREG
#define QARGS q1, q2, q3
#else
#define QARGS sh.q
#endif
      if (m > 3) block_stats<4>(acc, q0, QARGS, k0, g, c, sh.e, sh.z);
      else if (m > 2) block_stats<3>(acc, q0, QARGS, k0, g, c, sh.e, sh.z);
      else if (m > 1) block_stats<2>(acc, q0, QARGS, k0, g, c, sh.e, sh.z);
      else block_stats<1>(acc, q0, QARGS, k0, g, c, sh.e, sh.z);
    }
    // Output contraction with the current query along keys.
#if FL_OUTSMEM
    {
      float ob[4 * SK_NP];
      #pragma unroll
      for (int p = 0; p < SK_NP; ++p) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = 4 * p + e;
          ob[i] = fmaf(cq0, acc[2 * p + (e >> 1)][e & 1], cq1 * acc[2 * p + (e >> 1)][2 + (e & 1)]);
        }
      }
      #pragma unroll
      for (int i = 0; i < 4 * SK_NP; ++i) {
        ob[i] += __shfl_xor_sync(FULL, ob[i], 4);
        ob[i] += __shfl_xor_sync(FULL, ob[i], 8);
        ob[i] += __shfl_xor_sync(FULL, ob[i], 16);
      }
      if (g == 0) {
        #pragma unroll
        for (int p = 0; p < SK_NP; ++p) {
          float4* op = reinterpret_cast<float4*>(sh.o + 16 * p + 4 * c);
          const float4 cur = *op;
          *op = make_float4(cur.x + ob[4 * p], cur.y + ob[4 * p + 1], cur.z + ob[4 * p + 2], cur.w + ob[4 * p + 3]);
        }
      }
    }
#else
    #pragma unroll
    for (int p = 0; p < SK_NP; ++p) {
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 4 * p + e;
        out[i] = fmaf(cq0, acc[2 * p + (e >> 1)][e & 1], out[i]);
        out[i] = fmaf(cq1, acc[2 * p + (e >> 1)][2 + (e & 1)], out[i]);
      }
    }
#endif
  }
#if FL_OUTSMEM
  __syncwarp();
  if (g == 0) {
    #pragma unroll
    for (int p = 0; p < SK_NP; ++p) {
      const float4 v = *reinterpret_cast<const float4*>(sh.o + 16 * p + 4 * c);
      out[4 * p] = v.x; out[4 * p + 1] = v.y; out[4 * p + 2] = v.z; out[4 * p + 3] = v.w;
    }
  }
#else
  // Reduce output partials over the eight row groups.
  #pragma unroll
  for (int i = 0; i < 4 * SK_NP; ++i) {
    out[i] += __shfl_xor_sync(FULL, out[i], 4);
    out[i] += __shfl_xor_sync(FULL, out[i], 8);
    out[i] += __shfl_xor_sync(FULL, out[i], 16);
  }
#endif
  if (g == 0) {
    #pragma unroll
    for (int p = 0; p < SK_NP; ++p) {
      const int v = 16 * p + 4 * c;
      const uint2 xr = *reinterpret_cast<const uint2*>(xp + v);
      const float2 xa = unpack2(xr.x), xb = unpack2(xr.y);
      const float x4[4] = {xa.x, xa.y, xb.x, xb.y};
      float r[4];
      #pragma unroll
      for (int e = 0; e < 4; ++e)
        r[e] = fmaf(x4[e], __bfloat162float(D[h * st.ds0 + (v + e) * st.ds1]), out[4 * p + e]);
      if (out_f32) {
        *reinterpret_cast<float4*>(reinterpret_cast<float*>(O) + row * st.os0 + h * st.os1 + v) = make_float4(r[0], r[1], r[2], r[3]);
      } else {
        *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(O) + row * st.os0 + h * st.os1 + v) = make_uint2(pack2(r[0], r[1]), pack2(r[2], r[3]));
      }
#if !FL_DENSE
      *reinterpret_cast<uint2*>(XR + slot * st.xrs0 + h * st.xrs1 + wp * st.xrs2 + v) = xr;
#endif
    }
  }
#if !FL_DENSE
  if (lane == 0) DR[slot * st.drs0 + h * st.drs1 + wp * st.drs2] = dt_cur;
  if ((h % SK_HPG) == 0) {
    #pragma unroll
    for (int i = 0; i < (SK_N + 255) / 256; ++i) {
      const int ch = lane + 32 * i;
      if (ch < SK_N / 8)
        *reinterpret_cast<uint4*>(BR + slot * st.brs0 + group * st.brs1 + wp * st.brs2 + ch * 8) = *reinterpret_cast<const uint4*>(bp + ch * 8);
    }
  }
  __syncwarp();
  if (m > 0) finish(lane, m, meta_row, WOff[h], Offsets[h], sh, Tail, AG, SM, WROWS);
#endif
#if FL_ROW_LIST
  __syncwarp();
  }
#endif
}
