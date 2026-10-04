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
#define SK_P 64         // head dim
#endif
#ifndef SK_N
#define SK_N 128        // state size
#endif
// Lane sl of a head owns keys SK_KPL * sl .. + SK_KPL - 1 of the query and of every map.
#define SK_KPL (SK_N / 16)
static_assert(SK_N == 64 || SK_N == 128 || SK_N == 256, "state size must be 64, 128 or 256");
#ifndef SK_W
#define SK_W 16         // window length
#endif
static_assert(SK_W % 16 == 0 && SK_W >= 16, "window must be a multiple of 16");
// The window is replayed in 16-step tiles; lane sl of a head holds steps sl + 16 j of tile j.
#define SK_WT (SK_W / 16)
// Ring tiles staged in shared memory per head: the window itself when it is one tile,
// otherwise a double buffer that the replay streams the tiles through.
#define SK_RT (SK_WT > 1 ? 2 : 1)
// Lane sl of a head owns values SK_VPL * sl .. + SK_VPL - 1; lanes sl >= SK_VL own none.
#define SK_VPL (SK_P <= 64 ? 4 : 8)
#define SK_VL (SK_P / SK_VPL)
static_assert(SK_P % SK_VPL == 0 && SK_P <= 128, "head dim must be a multiple of 4 (<= 64) or 8 (<= 128)");
typedef __nv_bfloat16 bf16;
// SUPER_NF_BF16_UW=1: packed sketch rows (U) and coefficient maps (W) are stored as bf16.
// Both are regenerated from the fp32 state at every flush, so the rounding does not
// accumulate across steps; the recurrent state itself stays fp32.
#ifndef UW_BF16
#define UW_BF16 0
#endif
#if UW_BF16
typedef bf16 uw_t;
#define UW_BYTES 2
#else
typedef float uw_t;
#define UW_BYTES 4
#endif

#ifndef NF_MINB
#define NF_MINB 6
#endif
#ifndef NF_UROWS
#define NF_UROWS 8
#endif
#ifndef NF_UBATCH
#define NF_UBATCH 4
#endif
#ifndef NF_HEADS
#define NF_HEADS 8
#endif
static_assert(SK_HPG % NF_HEADS == 0 && SK_NHEADS % SK_HPG == 0, "CTA heads share one group");
#ifndef NF_SMAPS
#define NF_SMAPS 4
#endif
#ifndef NF_ABLATE
#define NF_ABLATE 0
#endif
// NF_LEAN=1: issue-lean readout. Ring weights are staged in shared instead of 15 lane
// broadcasts, the four pivot dots are reduced with a divergence-free transpose reduction
// (no per-pivot branches, so no collective fallbacks), and heads with nf_h == 0 and m <= 4
// use the dots directly as coefficients without the shared coefficient loop.
#ifndef NF_LEAN
#define NF_LEAN 0
#endif
// NF_FFMA2=1 (with NF_LEAN): packed fp32x2 FMA (Blackwell fma.rn.f32x2) for the ring replay,
// the readout accumulation and the pivot dots; halves the FMA instruction count.
#ifndef NF_FFMA2
#define NF_FFMA2 0
#endif
// NF_FASTSP=1: softplus through the fast intrinsics __logf(1 + __expf(x)) instead of log1pf.
#ifndef NF_FASTSP
#define NF_FASTSP 0
#endif
// NF_CONST_STRIDES=1: tensor strides are compiled in (NF_STRIDE_INIT, one build per stride
// set chosen by the Python dispatcher), so the address arithmetic folds to constants.
#ifndef NF_CONST_STRIDES
#define NF_CONST_STRIDES 0
#endif
// NF_MMA=1 (with NF_LEAN): ring replay on tensor cores. X^T (v x t) is the A operand through
// ldmatrix.trans over the swizzled ring rows, the window weights (rounded to bf16, zero at
// t >= wp) fill every column of the B operand, four m16n8k16 MMAs per head accumulate in fp32,
// and the fragments go through shared memory back to the lane layout.
#ifndef NF_MMA
#define NF_MMA 0
#endif
// NF_LANES8=1: quarter-warp heads. Eight lanes own eight values each, so one warp serves four
// heads and every per-head scalar/control instruction is amortized over twice as many heads;
// ring rows load as 16-byte chunks. Same fp32 arithmetic as the lean path (the 16-step decay
// prefix sum is evaluated pairwise per lane, so weights can differ from the 16-lane scan in
// the last bit); requires UW_BF16 or fp32 U/W, honours NF_FFMA2 and NF_CONST_STRIDES.
#ifndef NF_LANES8
#define NF_LANES8 0
#endif
// NF_ROWS=n (> 1): each CTA walks n consecutive rows for its heads and issues the next row's
// slot-dependent ring/dt loads right after the current row's replay, so the second DRAM hop
// overlaps the current row's dots, readout and stores. 16-lane lean path (bf16/fp32 U/W,
// NF_FFMA2, NF_CONST_STRIDES); same arithmetic as the lean kernel.
#ifndef NF_ROWS
#define NF_ROWS 1
#endif
// NF_PF_ROWS=d (> 0): each CTA issues L2 line prefetches for row b+d of its own heads (ring rows,
// dt ring, U rows, maps, query, B.C weights, x, dt) so that, d rows later, the CTA that owns that
// row finds its two dependent DRAM hops already in L2. Bytes moved are unchanged; only the
// latency seen by the loads changes. Numerics unchanged.
#ifndef NF_PF_ROWS
#define NF_PF_ROWS 0
#endif
// NF_PF_FULL=1 also prefetches the activation rows (x, dt, B.C weights, query); by default only
// the slot/flush-time data (ring rows, dt ring, sketch rows, maps) is prefetched, since the
// activations were just written by the preceding kernels and are usually L2 resident.
#ifndef NF_PF_FULL
#define NF_PF_FULL 2
#endif
// NF_PF_BULK=1: contiguous ranges (ring rows, sketch rows, maps, query) are prefetched with one
// cp.async.bulk.prefetch.L2 per range from a single lane instead of one line per lane.
#ifndef NF_PF_BULK
#define NF_PF_BULK 0
#endif
#if (SK_P != 64 || SK_N != 128 || SK_W != 16) && (!NF_LEAN || NF_MMA || NF_LANES8 || NF_ROWS > 1 || NF_ABLATE)
#error "shapes other than head dim 64, state size 128, window 16 support the lean 16-lane kernel only"
#endif

struct NfStrides {
  long x_b, x_h;            // x: row, head (value stride 1)
  long dt_b, dt_h;          // dt: row, head
  long bias_h, A_h, D_h;    // per-head scalars
  long B_b, B_g;            // B: row, group (key stride 1)
  long C_b, C_g;            // C (the FP32 query): row, group (key stride 1)
  long o_b, o_h;            // out: row, head (value stride 1)
  long xc_b, xc_h, xc_k;    // x ring: slot, head, t (value stride 1)
  long dc_b, dc_h, dc_k;    // dt ring: slot, head, t
  long Bc_b, Bc_g, Bc_k;    // B ring: slot, group, t (key stride 1)
  long bc_b, bc_g, bc_k;    // bc_pre: row, group, t
  long q_b, q_g;            // FP32 query (rotated C): row, group (key stride 1)
  long w_rows, sketch_rows, sm;
};

__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
// Predicated prefetch: no divergent block, the predicate goes into the instruction itself.
__device__ __forceinline__ void pf_l2_if(const void* p, bool pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %1, 0;\n @q prefetch.global.L2 [%0];\n}\n" :: "l"(p), "r"((int)pred));
}
__device__ __forceinline__ void pf_bulk(const void* p, unsigned bytes) {
  if (bytes) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait_1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p) {
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v)); }
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
#if NF_FFMA2
  unsigned long long x = *reinterpret_cast<unsigned long long*>(&a), y = *reinterpret_cast<unsigned long long*>(&b), z = *reinterpret_cast<unsigned long long*>(&c);
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(z) : "l"(x), "l"(y), "l"(z));
  return *reinterpret_cast<float2*>(&z);
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}
// Ring rows are staged with the 16-byte chunk index XOR-swizzled by the row so
// that ldmatrix over eight consecutive rows is bank-conflict free.
__device__ __forceinline__ int ring_off(int t, int c) { return t * SK_P + 8 * (c ^ (t & 7)); }
// NF_LEAN reads the ring with 8-byte lane loads over contiguous chunks (conflict-free without
// the swizzle), so the per-row XOR and address math fold into immediate offsets.
__device__ __forceinline__ int ring_at(int t, int c) {
#if NF_LEAN && !NF_MMA
  return t * SK_P + 8 * c;
#else
  return ring_off(t, c);
#endif
}
__device__ __forceinline__ float bf_lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
// U/W element loads in the storage type: four consecutive values (16 B fp32 / 8 B bf16).
__device__ __forceinline__ float4 ld_uw4(const uw_t* p) {
#if UW_BF16
  const uint2 r = *reinterpret_cast<const uint2*>(p);
  return make_float4(bf_lo(r.x), bf_hi(r.x), bf_lo(r.y), bf_hi(r.y));
#else
  return *reinterpret_cast<const float4*>(p);
#endif
}
__device__ __forceinline__ float to_f(uw_t v) {
#if UW_BF16
  return __bfloat162float(v);
#else
  return v;
#endif
}
// A lane's SK_VPL values: raw bf16 words and fp32 loads of U rows.
struct XRaw { unsigned w[SK_VPL / 2]; };
__device__ __forceinline__ XRaw ld_xraw(const bf16* p) {
  XRaw r;
#if SK_VPL == 4
  const uint2 t = *reinterpret_cast<const uint2*>(p); r.w[0] = t.x; r.w[1] = t.y;
#else
  const uint4 t = *reinterpret_cast<const uint4*>(p); r.w[0] = t.x; r.w[1] = t.y; r.w[2] = t.z; r.w[3] = t.w;
#endif
  return r;
}
__device__ __forceinline__ void st_xraw(bf16* p, const XRaw& r) {
#if SK_VPL == 4
  *reinterpret_cast<uint2*>(p) = make_uint2(r.w[0], r.w[1]);
#else
  *reinterpret_cast<uint4*>(p) = make_uint4(r.w[0], r.w[1], r.w[2], r.w[3]);
#endif
}
__device__ __forceinline__ void ld_uwv(const uw_t* p, float (&v)[SK_VPL]) {
  #pragma unroll
  for (int i = 0; i < SK_VPL; i += 4) {
    const float4 t = ld_uw4(p + i);
    v[i] = t.x; v[i + 1] = t.y; v[i + 2] = t.z; v[i + 3] = t.w;
  }
}
__device__ __forceinline__ unsigned pack_rt(unsigned u) {
  // Same BF16 -> FP32 -> BF16 round trip as the accepted ring append.
  const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(bf_lo(u)));
  const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(bf_hi(u)));
  return lo | (hi << 16);
}
// Stage a lane's SK_KPL map entries (16-byte chunks, or one 8-byte chunk).
__device__ __forceinline__ void cp_keys(uw_t* dst, const uw_t* src) {
#if SK_KPL * UW_BYTES < 16
  cp_async8(dst, src);
#else
  #pragma unroll
  for (int i = 0; i < SK_KPL * UW_BYTES / 16; ++i) cp_async16(dst + i * (16 / (int)sizeof(uw_t)), src + i * (16 / (int)sizeof(uw_t)));
#endif
}
// Ring append of B keys: SK_KPL / KV vectors of KV keys per lane, with the same
// BF16 -> FP32 -> BF16 round trip as the accepted append.
#if SK_KPL == 4
typedef uint2 keyvec;
__device__ __forceinline__ uint2 pack_rt_vec(uint2 r) { return make_uint2(pack_rt(r.x), pack_rt(r.y)); }
#else
typedef uint4 keyvec;
__device__ __forceinline__ uint4 pack_rt_vec(uint4 r) { return make_uint4(pack_rt(r.x), pack_rt(r.y), pack_rt(r.z), pack_rt(r.w)); }
#endif
#define KV ((int)sizeof(keyvec) / 2)

struct __align__(16) NfShared {
  bf16 ring[NF_HEADS][SK_RT * 16 * SK_P];   // x-ring rows t < wp per head (SK_RT 16-row tiles)
  uw_t maps[NF_HEADS][NF_SMAPS * SK_N];    // packed P4 maps staged in shared (the rest are read directly)
  float coef[NF_HEADS][SK_N];       // readout coefficients per head
  float wk[NF_HEADS][SK_W];         // NF_LEAN: ring weights per window position
};
static_assert(sizeof(NfShared) <= 48 * 1024,
              "non-flush kernel: shared memory (ring tiles, maps, coefficients and weights of NF_HEADS heads) "
              "exceeds the 48 KB static shared-memory limit per block; lower NF_HEADS or NF_SMAPS");

// One CTA: row b, eight consecutive heads of one group; half-warp per head.
// Meta row == b: the control kernel `prepare_rows` stores smap[slot] = row for
// every active row, and the accepted kernels index U/maps/AG/query by
// sk_map[slot]; using b directly removes one dependent global round trip.
extern "C" __global__ void __launch_bounds__(16 * NF_HEADS, NF_MINB)
nf_kernel(const bf16* __restrict__ X, const bf16* __restrict__ DT, const bf16* __restrict__ Bias,
          const float* __restrict__ A, const bf16* __restrict__ B, const float* __restrict__ C,
          const bf16* __restrict__ D, void* __restrict__ O, bf16* __restrict__ XR, float* __restrict__ DR,
          bf16* __restrict__ BR, const float* __restrict__ BC, const int* __restrict__ WP,
          const signed char* __restrict__ FL, const int* __restrict__ Slots, int null_slot,
          const int* __restrict__ NFH, const int* __restrict__ MH, const int* __restrict__ OFF,
          const int* __restrict__ Map, const uw_t* __restrict__ W, const float* __restrict__ AG,
          const float* __restrict__ Q, const uw_t* __restrict__ U, const int* __restrict__ SkOff,
          const int* __restrict__ WOff, NfStrides st_rt, int out_f32, int batch) {
#if NF_CONST_STRIDES
  NfStrides st = NF_STRIDE_INIT;
  st.w_rows = st_rt.w_rows; st.sketch_rows = st_rt.sketch_rows; st.sm = st_rt.sm;
#else
  const NfStrides st = st_rt;
#endif
  __shared__ NfShared sh;
  const int b = blockIdx.y;
  const signed char fl = FL[b];
  const long slot = Slots[b];                      // consumed only at hop 2
  const int wp = WP[b];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int hs = 2 * warp + (lane >> 4);           // head slot within the CTA
  const int sl = lane & 15;                        // lane within the head
  const int h = NF_HEADS * blockIdx.x + hs;
  const int g = h / SK_HPG;
  const unsigned hmask = 0xffffu << (lane & 16);
  const long di = Map[b];
  if (fl != 0) return;

  // ---- hop 1: everything independent of the slot ------------------------------
  const int m_h = MH[h];
  const int nf_h = NFH ? min(NFH[h], SK_N) : 0;
  const int off_h = OFF[h];
  const int skoff = SkOff[h];
  const int woff = WOff[h];
  const int nrow = max(nf_h, m_h);
#if NF_PF_ROWS > 0
  {
    // Look-ahead L2 prefetch for row b + NF_PF_ROWS of this head (one 128-byte line per lane).
    const int bp = b + NF_PF_ROWS;
    if (bp < batch && FL[bp] == 0) {                   // flush rows are not read here
      const long ps = Slots[bp]; const int pw = WP[bp];
      if (ps != null_slot) {
        constexpr int UL = 128 / (int)sizeof(uw_t);      // elements per 128-byte line
#if NF_PF_BULK
        if (sl == 0) pf_bulk(XR + ps * st.xc_b + h * st.xc_h, (unsigned)pw * 128u);
        if (sl == 1) pf_bulk(U + ((long)Map[bp] * st.sketch_rows + skoff) * SK_P, (unsigned)min(nrow, 16) * (unsigned)SK_P * (unsigned)sizeof(uw_t));
        if (sl == 2) pf_bulk(W + ((long)Map[bp] * st.w_rows + woff) * SK_N, (unsigned)min(m_h, 4) * (unsigned)SK_N * (unsigned)sizeof(uw_t));
        if (sl == 3) pf_bulk(DR + ps * st.dc_b + h * st.dc_h, 4u * SK_W);
#if NF_PF_FULL
        if (sl == 4) pf_bulk(Q + (long)bp * st.q_b + g * st.q_g, 4u * SK_N);
        if (sl == 5) pf_bulk(X + (long)bp * st.x_b + h * st.x_h, 2u * SK_P);
        if (sl == 6) pf_bulk(BC + (long)bp * st.bc_b + g * st.bc_g, 4u * SK_W);
        if (sl == 7) pf_l2(DT + (long)bp * st.dt_b + h * st.dt_h);
#endif
#else
        #pragma unroll
        for (int j = 0; j < SK_WT; ++j) pf_l2_if(XR + ps * st.xc_b + h * st.xc_h + (sl + 16 * j) * SK_P, sl + 16 * j < pw);
        pf_l2_if(U + ((long)Map[bp] * st.sketch_rows + skoff) * SK_P + sl * UL, sl < nrow * SK_P / UL && sl < 16);
        pf_l2_if(W + ((long)Map[bp] * st.w_rows + woff) * SK_N + sl * UL, sl < min(m_h, 4) * (SK_N / UL));
        #pragma unroll
        for (int i = 0; i < (4 * SK_W + 127) / 128; ++i) pf_l2_if(DR + ps * st.dc_b + h * st.dc_h + 32 * i, sl == 0);
#if NF_PF_FULL >= 1
        pf_l2_if(Q + (long)bp * st.q_b + g * st.q_g + 32 * (sl & (SK_N / 32 - 1)), sl < SK_N / 32);
#endif
#if NF_PF_FULL >= 2
        pf_l2_if(X + (long)bp * st.x_b + h * st.x_h, sl == 3);
        pf_l2_if(DT + (long)bp * st.dt_b + h * st.dt_h, sl == 4);
        #pragma unroll
        for (int i = 0; i < (4 * SK_W + 127) / 128; ++i) pf_l2_if(BC + (long)bp * st.bc_b + g * st.bc_g + 32 * i, sl == 5);
        // pivot coefficient rows (a, g0..g3) of heads with m > 4: lanes 8..12, one line each
        pf_l2_if(AG + (long)bp * (5L * st.sm) + (sl - 8) * (long)st.sm + off_h, m_h > 4 && sl >= 8 && sl < 13);
#endif
#endif
      }
    }
  }
#endif
  float dt_cur = __bfloat162float(DT[b * st.dt_b + h * st.dt_h]) + __bfloat162float(Bias[h * st.bias_h]);
  const float Ah = A[h * st.A_h];
  const float Dh = __bfloat162float(D[h * st.D_h]);
  const bf16* xp = X + b * st.x_b + h * st.x_h;
  const bool vlane = sl < SK_VL;                  // lane owns values
  XRaw xraw;
  #pragma unroll
  for (int i = 0; i < SK_VPL / 2; ++i) xraw.w[i] = 0u;
  if (vlane) xraw = ld_xraw(xp + SK_VPL * sl);
  float bcw[SK_WT];
  {
    const float* bcp = BC + b * st.bc_b + g * st.bc_g + sl;
    #pragma unroll
    for (int j = 0; j < SK_WT; ++j) bcw[j] = bcp[16 * j];
  }
  const float* qp = Q + b * st.q_b + g * st.q_g;
  // Lane sl owns keys j = SK_KPL*sl .. SK_KPL*sl+SK_KPL-1 of the query and of every map.
  float qv[SK_KPL];
  #pragma unroll
  for (int k = 0; k < SK_KPL; k += 4) {
    const float4 qr = *reinterpret_cast<const float4*>(qp + SK_KPL * sl + k);
    qv[k] = qr.x; qv[k + 1] = qr.y; qv[k + 2] = qr.z; qv[k + 3] = qr.w;
  }
  const uw_t* w = W + (di * st.w_rows + woff) * SK_N;
  #pragma unroll
  for (int ip = 0; ip < NF_SMAPS; ++ip) {
    if (m_h > ip) cp_keys(sh.maps[hs] + ip * SK_N + SK_KPL * sl, w + ip * SK_N + SK_KPL * sl);
  }
  const uw_t* up = U + (di * st.sketch_rows + skoff) * SK_P;
  float ur[NF_UROWS][SK_VPL];
  #pragma unroll
  for (int n = 0; n < NF_UROWS; ++n) {
    if (n < nrow && vlane) ld_uwv(up + n * SK_P + SK_VPL * sl, ur[n]);
    else {
      #pragma unroll
      for (int i = 0; i < SK_VPL; ++i) ur[n][i] = 0.f;
    }
  }
  // ---- hop 2: slot-dependent ------------------------------------------------
  if (slot == null_slot) return;                   // uniform over the CTA
  float dtk[SK_WT];
  #pragma unroll
  for (int j = 0; j < SK_WT; ++j) {
    dtk[j] = 0.f;
    if (sl + 16 * j < wp) dtk[j] = DR[slot * st.dc_b + h * st.dc_h + sl + 16 * j];
  }
  // Window tile j (rows 16 j .. 16 j + 15) goes to ring buffer j % SK_RT; lane sl stages
  // 16-byte chunks sl + 16 k of its rows t < wp (SK_P / 8 chunks per row).
  auto stage_tile = [&](int j) {
    const bf16* xr = XR + slot * st.xc_b + h * st.xc_h + 16 * j * SK_P;
    constexpr int CPR = SK_P / 8;
    #pragma unroll
    for (int k = 0; k < CPR; ++k) {
      const int ch = sl + 16 * k, t = ch / CPR, c = ch % CPR;
      if (t < wp - 16 * j) cp_async16(sh.ring[hs] + (j % SK_RT) * (16 * SK_P) + ring_at(t, c), xr + t * SK_P + 8 * c);
    }
    cp_async_commit();
  };
  stage_tile(0);
  if (SK_WT > 1) stage_tile(1);

  // ---- window decay and ring weights (lanes 0..15 of the head hold steps) ----
#if NF_FASTSP
  if (dt_cur <= 20.f) dt_cur = __logf(1.f + __expf(dt_cur));
#else
  if (dt_cur <= 20.f) dt_cur = log1pf(__expf(dt_cur));
#endif
  // Prefix sums over the window: a 16-lane scan per tile, carried across tiles.
  float cum[SK_WT], tot = 0.f;
  #pragma unroll
  for (int j = 0; j < SK_WT; ++j) {
    if (sl + 16 * j == wp) dtk[j] = dt_cur;
    cum[j] = dtk[j];
    #pragma unroll
    for (int o = 1; o < 16; o <<= 1) { const float t = __shfl_up_sync(FULL, cum[j], o, 16); if (sl >= o) cum[j] += t; }
    if (j > 0) cum[j] += tot;
    tot = __shfl_sync(FULL, cum[j], 15, 16);
  }
  const float dAt = Ah * tot;
  const float decay = __expf(dAt);
  float wkt[SK_WT];
  #pragma unroll
  for (int j = 0; j < SK_WT; ++j) wkt[j] = (sl + 16 * j <= wp) ? dtk[j] * __expf(dAt - Ah * cum[j]) * bcw[j] : 0.f;
#if !NF_LEAN || NF_MMA || NF_ABLATE
  const float wk = wkt[0];   // these variants take window 16 only: one tile
#endif

  // ---- ring replay from shared ----------------------------------------------
  if (SK_WT > 1) cp_async_wait_1();                // tile 0 has landed; tile 1 may be in flight
  else cp_async_wait_all();
  __syncwarp();
  float y[SK_VPL];
  #pragma unroll
  for (int i = 0; i < SK_VPL; ++i) y[i] = 0.f;
#if NF_ABLATE
  {
    float t = wk + qv[0] + to_f(sh.maps[hs][sl]) + bf_lo(*reinterpret_cast<const unsigned*>(sh.ring[hs] + 2 * sl));
    #pragma unroll
    for (int n = 0; n < NF_UROWS; ++n) t += ur[n][0];
    y[0] = t * decay;
    if (m_h > 4) y[1] = AG[di * (5L * st.sm) + off_h + sl];
    if (nrow > NF_UROWS) y[2] = ld_uw4(up + NF_UROWS * SK_P + 4 * sl).x;
  }
#else
  {
    const int c0 = (4 * sl) >> 3, e0 = (4 * sl) & 7;
#if NF_LEAN && NF_MMA
    float2 y01, y23;
    // Rows t >= wp were not staged; zero them (whole 128-byte rows, so the swizzle is moot)
    // so that stale shared memory cannot poison the MMA through 0 x NaN.
    for (int t = wp + (sl >> 3); t < 16; t += 2)
      *reinterpret_cast<uint4*>(sh.ring[hs] + t * 64 + 8 * (sl & 7)) = make_uint4(0u, 0u, 0u, 0u);
    sh.wk[hs][sl] = (sl < wp) ? wk : 0.f;
    __syncwarp();
    {
      const int q = lane & 3, j = lane >> 3, r = lane & 7;
      #pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int H = (hs & ~1) | hh;                 // both half-warps serve head H
        const float* wq = sh.wk[H];
        const uint32_t bfrag[2] = {pack2(wq[2 * q], wq[2 * q + 1]), pack2(wq[2 * q + 8], wq[2 * q + 9])};
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          uint32_t a[4];
          float d[4] = {0.f, 0.f, 0.f, 0.f};
          ldmatrix_x4_trans(a, sh.ring[H] + ring_off(8 * (j >> 1) + r, 2 * mt + (j & 1)));
          mma16816(d, a, bfrag);
          if (q == 0) { sh.coef[H][16 * mt + (lane >> 2)] = d[0]; sh.coef[H][16 * mt + 8 + (lane >> 2)] = d[2]; }
        }
      }
    }
    __syncwarp();
    {
      const float4 yy = *reinterpret_cast<const float4*>(sh.coef[hs] + 4 * sl);
      y01 = make_float2(yy.x, yy.y); y23 = make_float2(yy.z, yy.w);
    }
    __syncwarp();
    y[0] = y01.x; y[1] = y01.y; y[2] = y23.x; y[3] = y23.y;
#elif NF_LEAN
    // Pairs of values (2 i, 2 i + 1) of this lane.
    float2 yp[SK_VPL / 2];
    #pragma unroll
    for (int i = 0; i < SK_VPL / 2; ++i) yp[i] = make_float2(0.f, 0.f);
    #pragma unroll
    for (int j = 0; j < SK_WT; ++j) sh.wk[hs][sl + 16 * j] = wkt[j];
    __syncwarp();
    const float4* wq = reinterpret_cast<const float4*>(sh.wk[hs]);
    // Tile j of the window, rows t = 16 j + tl; each ring row is read once.
    #pragma unroll
    for (int j = 0; j < SK_WT; ++j) {
      if (j > 0) {
        if (16 * j >= wp) break;                     // no ring rows left (uniform over the head)
        if (j + 1 < SK_WT) cp_async_wait_1();
        else cp_async_wait_all();
        __syncwarp();
      }
      const bf16* rb = sh.ring[hs] + (j % SK_RT) * 16 * SK_P + SK_VPL * sl;
      // Two rows per uniform predicate: rows t and t+1 are both below wp or handled by the tail.
      #pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        const float4 w4 = wq[4 * j + t4];
        const float ws[4] = {w4.x, w4.y, w4.z, w4.w};
        #pragma unroll
        for (int i = 0; i < 4; i += 2) {
          const int tl = 4 * t4 + i, t = 16 * j + tl;
          if (t + 1 < wp && vlane) {
            const XRaw r0 = ld_xraw(rb + tl * SK_P);
            const XRaw r1 = ld_xraw(rb + (tl + 1) * SK_P);
            const float2 w0 = make_float2(ws[i], ws[i]), w1 = make_float2(ws[i + 1], ws[i + 1]);
            #pragma unroll
            for (int q = 0; q < SK_VPL / 2; ++q) yp[q] = ffma2(make_float2(bf_lo(r0.w[q]), bf_hi(r0.w[q])), w0, yp[q]);
            #pragma unroll
            for (int q = 0; q < SK_VPL / 2; ++q) yp[q] = ffma2(make_float2(bf_lo(r1.w[q]), bf_hi(r1.w[q])), w1, yp[q]);
          } else if (t < wp && vlane) {
            const XRaw r0 = ld_xraw(rb + tl * SK_P);
            const float2 w0 = make_float2(ws[i], ws[i]);
            #pragma unroll
            for (int q = 0; q < SK_VPL / 2; ++q) yp[q] = ffma2(make_float2(bf_lo(r0.w[q]), bf_hi(r0.w[q])), w0, yp[q]);
          }
        }
      }
      if (j + 2 < SK_WT) {
        __syncwarp();                                // every lane is done with this ring buffer
        stage_tile(j + 2);
      }
    }
    #pragma unroll
    for (int q = 0; q < SK_VPL / 2; ++q) { y[2 * q] = yp[q].x; y[2 * q + 1] = yp[q].y; }
#else
    #pragma unroll
    for (int t = 0; t < 15; ++t) {
      if (t < wp) {
        const float w = __shfl_sync(FULL, wk, t, 16);
        const uint2 r = *reinterpret_cast<const uint2*>(sh.ring[hs] + ring_off(t, c0) + e0);
        y[0] = fmaf(bf_lo(r.x), w, y[0]); y[1] = fmaf(bf_hi(r.x), w, y[1]);
        y[2] = fmaf(bf_lo(r.y), w, y[2]); y[3] = fmaf(bf_hi(r.y), w, y[3]);
      }
    }
#endif
  }
  float xc[SK_VPL];
  #pragma unroll
  for (int q = 0; q < SK_VPL / 2; ++q) { xc[2 * q] = bf_lo(xraw.w[q]); xc[2 * q + 1] = bf_hi(xraw.w[q]); }
  {
    // Weight of the current token (step wp); longer windows read it back from shared memory.
    float w;
    if (SK_WT > 1) w = sh.wk[hs][wp];
    else w = __shfl_sync(FULL, wkt[0], wp, 16);
    #pragma unroll
    for (int i = 0; i < SK_VPL; ++i) y[i] = fmaf(xc[i], w, y[i]);
  }
  // ---- query dots with the packed maps (same j order and tree as accepted) --
  float dots[4] = {0.f, 0.f, 0.f, 0.f};
#if NF_LEAN
  {
    // Partial dots for all four pivots without branching: rows beyond m_h read row 0
    // (always staged when m_h > 0) and are zeroed afterwards, so no lane diverges and
    // the shuffles below run with the full mask.
    float d[4];
    #pragma unroll
    for (int ip = 0; ip < 4; ++ip) {
      const int ipc = (ip < m_h) ? ip : 0;
      const uw_t* mp = (ipc < NF_SMAPS) ? (sh.maps[hs] + ipc * SK_N + SK_KPL * sl) : (w + ipc * SK_N + SK_KPL * sl);
      // Keys in groups of eight (two four-key loads), or one group of four.
#if NF_FFMA2
      float2 dd = make_float2(0.f, 0.f);
#else
      float dot = 0.f;
#endif
      #pragma unroll
      for (int k = 0; k < SK_KPL; k += 8) {
        const float4 a = ld_uw4(mp + k), bq = SK_KPL > 4 ? ld_uw4(mp + k + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
#if NF_FFMA2
        dd = ffma2(make_float2(a.x, a.y), make_float2(qv[k], qv[k + 1]), dd);
        dd = ffma2(make_float2(a.z, a.w), make_float2(qv[k + 2], qv[k + 3]), dd);
        if (SK_KPL > 4) {
          dd = ffma2(make_float2(bq.x, bq.y), make_float2(qv[k + 4], qv[k + 5]), dd);
          dd = ffma2(make_float2(bq.z, bq.w), make_float2(qv[k + 6], qv[k + 7]), dd);
        }
#else
        dot = fmaf(a.x, qv[k], dot); dot = fmaf(a.y, qv[k + 1], dot); dot = fmaf(a.z, qv[k + 2], dot); dot = fmaf(a.w, qv[k + 3], dot);
        if (SK_KPL > 4) {
          dot = fmaf(bq.x, qv[k + 4], dot); dot = fmaf(bq.y, qv[k + 5], dot); dot = fmaf(bq.z, qv[k + 6], dot); dot = fmaf(bq.w, qv[k + 7], dot);
        }
#endif
      }
#if NF_FFMA2
      const float dot = dd.x + dd.y;
#endif
      d[ip] = (ip < m_h) ? dot : 0.f;
    }
    // Transpose reduction over the 16 lanes of the head: 2+1+1+1 shuffles leave the
    // total of pivot k in lanes with (sl & 12) == 4k; four broadcasts fan them out.
    const bool hi8 = (sl & 8) != 0, hi4 = (sl & 4) != 0;
    const float r0 = __shfl_xor_sync(FULL, hi8 ? d[0] : d[2], 8, 16);
    const float r1 = __shfl_xor_sync(FULL, hi8 ? d[1] : d[3], 8, 16);
    const float e0 = (hi8 ? d[2] : d[0]) + r0, e1 = (hi8 ? d[3] : d[1]) + r1;
    const float r2 = __shfl_xor_sync(FULL, hi4 ? e0 : e1, 4, 16);
    float f = (hi4 ? e1 : e0) + r2;
    f += __shfl_xor_sync(FULL, f, 2, 16);
    f += __shfl_xor_sync(FULL, f, 1, 16);
    #pragma unroll
    for (int k = 0; k < 4; ++k) dots[k] = __shfl_sync(FULL, f, 4 * k, 16);
  }
#else
  #pragma unroll
  for (int ip = 0; ip < 4; ++ip) {
    if (m_h > ip) {
      const uw_t* mp = (ip < NF_SMAPS) ? (sh.maps[hs] + ip * SK_N + 8 * sl) : (w + ip * SK_N + 8 * sl);
      const float4 a = ld_uw4(mp), bq = ld_uw4(mp + 4);
      float dot = 0.f;
      dot = fmaf(a.x, qv[0], dot); dot = fmaf(a.y, qv[1], dot); dot = fmaf(a.z, qv[2], dot); dot = fmaf(a.w, qv[3], dot);
      dot = fmaf(bq.x, qv[4], dot); dot = fmaf(bq.y, qv[5], dot); dot = fmaf(bq.z, qv[6], dot); dot = fmaf(bq.w, qv[7], dot);
      #pragma unroll
      for (int o = 8; o > 0; o >>= 1) dot += __shfl_down_sync(hmask, dot, o, 16);
      dots[ip] = __shfl_sync(hmask, dot, 0, 16);
    }
  }
#endif
  float acc[SK_VPL];
  #pragma unroll
  for (int i = 0; i < SK_VPL; ++i) acc[i] = 0.f;
#if NF_LEAN
  if (nf_h == 0 && m_h <= 4) {
    // The dots are the coefficients of the (at most four) sketch rows.
    float2 ap[SK_VPL / 2];
    #pragma unroll
    for (int q = 0; q < SK_VPL / 2; ++q) ap[q] = make_float2(0.f, 0.f);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      if (n < m_h) {
        float u[SK_VPL];
        if (n < NF_UROWS) {
          #pragma unroll
          for (int i = 0; i < SK_VPL; ++i) u[i] = ur[n < NF_UROWS ? n : 0][i];
        } else if (vlane) {
          ld_uwv(up + n * SK_P + SK_VPL * sl, u);
        } else {
          #pragma unroll
          for (int i = 0; i < SK_VPL; ++i) u[i] = 0.f;
        }
        const float2 cc = make_float2(dots[n], dots[n]);
        #pragma unroll
        for (int q = 0; q < SK_VPL / 2; ++q) ap[q] = ffma2(make_float2(u[2 * q], u[2 * q + 1]), cc, ap[q]);
      }
    }
    #pragma unroll
    for (int q = 0; q < SK_VPL / 2; ++q) { acc[2 * q] = ap[q].x; acc[2 * q + 1] = ap[q].y; }
  } else {
  // ---- coefficients (C only for dense heads; AG only for m > 4) -------------
  const float* cp = C + b * st.C_b + g * st.C_g;
  for (int n = sl; n < nrow; n += 16) {
    float c = (n < nf_h) ? cp[n] : 0.f;
    if (n < m_h) {
      if (m_h <= 4) c += (n == 0 ? dots[0] : n == 1 ? dots[1] : n == 2 ? dots[2] : dots[3]);
      else {
        const long index = di * (5L * st.sm) + off_h + n;
        c += fmaf(AG[index + st.sm], dots[0], AG[index] * qp[n]);
        #pragma unroll
        for (int ip = 1; ip < 4; ++ip) c = fmaf(AG[index + (ip + 1) * st.sm], dots[ip], c);
      }
    }
    sh.coef[hs][n] = c;
  }
  __syncwarp();
  // ---- compact-U readout ---------------------------------------------------
  #pragma unroll
  for (int n = 0; n < NF_UROWS; ++n) {
    if (n < nrow) {
      const float c = sh.coef[hs][n];
      #pragma unroll
      for (int i = 0; i < SK_VPL; ++i) acc[i] = fmaf(ur[n][i], c, acc[i]);
    }
  }
  if (vlane) {
  for (int n0 = NF_UROWS; n0 < nrow; n0 += NF_UBATCH) {
    float v[NF_UBATCH][SK_VPL];
    #pragma unroll
    for (int u = 0; u < NF_UBATCH; ++u)
      if (n0 + u < nrow) ld_uwv(up + (n0 + u) * SK_P + SK_VPL * sl, v[u]);
    #pragma unroll
    for (int u = 0; u < NF_UBATCH; ++u) {
      if (n0 + u < nrow) {
        const float c = sh.coef[hs][n0 + u];
        #pragma unroll
        for (int i = 0; i < SK_VPL; ++i) acc[i] = fmaf(v[u][i], c, acc[i]);
      }
    }
  }
  }
  }
#else
  // ---- coefficients (C only for dense heads; AG only for m > 4) -------------
  const float* cp = C + b * st.C_b + g * st.C_g;
  for (int n = sl; n < nrow; n += 16) {
    float c = (n < nf_h) ? cp[n] : 0.f;
    if (n < m_h) {
      if (m_h <= 4) c += (n == 0 ? dots[0] : n == 1 ? dots[1] : n == 2 ? dots[2] : dots[3]);
      else {
        const long index = di * (5L * st.sm) + off_h + n;
        c += fmaf(AG[index + st.sm], dots[0], AG[index] * qp[n]);
        #pragma unroll
        for (int ip = 1; ip < 4; ++ip) c = fmaf(AG[index + (ip + 1) * st.sm], dots[ip], c);
      }
    }
    sh.coef[hs][n] = c;
  }
  __syncwarp();
  // ---- compact-U readout ---------------------------------------------------
  #pragma unroll
  for (int n = 0; n < NF_UROWS; ++n) {
    if (n < nrow) {
      const float c = sh.coef[hs][n];
      #pragma unroll
      for (int i = 0; i < SK_VPL; ++i) acc[i] = fmaf(ur[n][i], c, acc[i]);
    }
  }
  if (vlane) {
  for (int n0 = NF_UROWS; n0 < nrow; n0 += NF_UBATCH) {
    float v[NF_UBATCH][SK_VPL];
    #pragma unroll
    for (int u = 0; u < NF_UBATCH; ++u)
      if (n0 + u < nrow) ld_uwv(up + (n0 + u) * SK_P + SK_VPL * sl, v[u]);
    #pragma unroll
    for (int u = 0; u < NF_UBATCH; ++u) {
      if (n0 + u < nrow) {
        const float c = sh.coef[hs][n0 + u];
        #pragma unroll
        for (int i = 0; i < SK_VPL; ++i) acc[i] = fmaf(v[u][i], c, acc[i]);
      }
    }
  }
  }
#endif
  #pragma unroll
  for (int i = 0; i < SK_VPL; ++i) { y[i] = fmaf(acc[i], decay, y[i]); y[i] = fmaf(xc[i], Dh, y[i]); }
#endif
  // ---- output and ring append ----------------------------------------------
  if (vlane) {
    if (out_f32) {
      float* op = reinterpret_cast<float*>(O) + b * st.o_b + h * st.o_h + SK_VPL * sl;
      #pragma unroll
      for (int i = 0; i < SK_VPL; i += 4) *reinterpret_cast<float4*>(op + i) = make_float4(y[i], y[i + 1], y[i + 2], y[i + 3]);
    } else {
      XRaw o;
      #pragma unroll
      for (int q = 0; q < SK_VPL / 2; ++q) {
        __nv_bfloat162 p = __floats2bfloat162_rn(y[2 * q], y[2 * q + 1]);
        o.w[q] = *reinterpret_cast<unsigned*>(&p);
      }
      st_xraw(reinterpret_cast<bf16*>(O) + b * st.o_b + h * st.o_h + SK_VPL * sl, o);
    }
  }
  if (wp < SK_W) {
    if (vlane) st_xraw(XR + slot * st.xc_b + h * st.xc_h + wp * SK_P + SK_VPL * sl, xraw);
    if (sl == 0) DR[slot * st.dc_b + h * st.dc_h + wp] = dt_cur;
    if ((h % SK_HPG) == 0) {
      #pragma unroll
      for (int k = 0; k < SK_KPL; k += KV) {
        const keyvec raw = *reinterpret_cast<const keyvec*>(B + b * st.B_b + g * st.B_g + SK_KPL * sl + k);
        *reinterpret_cast<keyvec*>(BR + slot * st.Bc_b + g * st.Bc_g + wp * SK_N + SK_KPL * sl + k) = pack_rt_vec(raw);
      }
    }
  }
}

#if NF_LANES8
struct __align__(16) NfShared8 {
  bf16 ring[NF_HEADS][16 * 64];          // x-ring rows t < wp per head (unswizzled); coefficients alias it afterwards
  uw_t maps[NF_HEADS][NF_SMAPS * 128];   // staged pivot maps
  float wk[NF_HEADS][16];                // ring weights per window position
};

__device__ __forceinline__ void ld_uw8(const uw_t* p, float* v) {
#if UW_BF16
  const uint4 r = *reinterpret_cast<const uint4*>(p);
  v[0] = bf_lo(r.x); v[1] = bf_hi(r.x); v[2] = bf_lo(r.y); v[3] = bf_hi(r.y);
  v[4] = bf_lo(r.z); v[5] = bf_hi(r.z); v[6] = bf_lo(r.w); v[7] = bf_hi(r.w);
#else
  const float4 a = *reinterpret_cast<const float4*>(p), c = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#endif
}

// One CTA: row b, NF_HEADS consecutive heads; a quarter-warp (8 lanes) per head, lane s owns
// values 8s..8s+7, keys 16s..16s+15 of the query/maps, window positions 2s, 2s+1.
extern "C" __global__ void __launch_bounds__(8 * NF_HEADS, NF_MINB)
nf_kernel8(const bf16* __restrict__ X, const bf16* __restrict__ DT, const bf16* __restrict__ Bias,
           const float* __restrict__ A, const bf16* __restrict__ B, const float* __restrict__ C,
           const bf16* __restrict__ D, void* __restrict__ O, bf16* __restrict__ XR, float* __restrict__ DR,
           bf16* __restrict__ BR, const float* __restrict__ BC, const int* __restrict__ WP,
           const signed char* __restrict__ FL, const int* __restrict__ Slots, int null_slot,
           const int* __restrict__ NFH, const int* __restrict__ MH, const int* __restrict__ OFF,
           const int* __restrict__ Map, const uw_t* __restrict__ W, const float* __restrict__ AG,
           const float* __restrict__ Q, const uw_t* __restrict__ U, const int* __restrict__ SkOff,
           const int* __restrict__ WOff, NfStrides st_rt, int out_f32, int batch) {
#if NF_CONST_STRIDES
  NfStrides st = NF_STRIDE_INIT;
  st.w_rows = st_rt.w_rows; st.sketch_rows = st_rt.sketch_rows; st.sm = st_rt.sm;
#else
  const NfStrides st = st_rt;
#endif
  __shared__ NfShared8 sh;
  const int b = blockIdx.y;
  if (FL[b] != 0) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int hs = 4 * warp + (lane >> 3);           // head slot within the CTA
  const int s = lane & 7;                          // lane within the head
  const int h = NF_HEADS * blockIdx.x + hs;
  const int g = h / SK_HPG;
  const long slot = Slots[b];
  const int wp = WP[b];
  const long di = Map[b];

  // ---- hop 1: everything independent of the slot ------------------------------
  const int m_h = MH[h];
  const int nf_h = NFH ? min(NFH[h], 128) : 0;
  const int off_h = OFF[h];
  const int skoff = SkOff[h];
  const int woff = WOff[h];
  const int nrow = max(nf_h, m_h);
#if NF_PF_ROWS > 0
  {
    // Look-ahead L2 prefetch for row b + NF_PF_ROWS of this head (eight lanes, two lines each).
    const int bp = b + NF_PF_ROWS;
    if (bp < batch && FL[bp] == 0) {
      const long ps = Slots[bp]; const int pw = WP[bp];
      if (ps != null_slot) {
        constexpr int UL = 128 / (int)sizeof(uw_t);
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int l = s + 8 * k;
          if (l < pw) pf_l2(XR + ps * st.xc_b + h * st.xc_h + l * 64);
          if (l < nrow * (64 / UL) && l < 16) pf_l2(U + ((long)Map[bp] * st.sketch_rows + skoff) * 64 + l * UL);
          if (l < min(m_h, 4) * (128 / UL)) pf_l2(W + ((long)Map[bp] * st.w_rows + woff) * 128 + l * UL);
        }
        if (s == 0) pf_l2(DR + ps * st.dc_b + h * st.dc_h);
#if NF_PF_FULL
        if (s == 3) { pf_l2(X + (long)bp * st.x_b + h * st.x_h); pf_l2(DT + (long)bp * st.dt_b + h * st.dt_h); pf_l2(BC + (long)bp * st.bc_b + g * st.bc_g); }
        if (s == 1) pf_l2(Q + (long)bp * st.q_b + g * st.q_g);
        if (s == 2) pf_l2(Q + (long)bp * st.q_b + g * st.q_g + 32);
        if (s == 4) pf_l2(Q + (long)bp * st.q_b + g * st.q_g + 64);
        if (s == 5) pf_l2(Q + (long)bp * st.q_b + g * st.q_g + 96);
#endif
      }
    }
  }
#endif
  float dt_cur = __bfloat162float(DT[b * st.dt_b + h * st.dt_h]) + __bfloat162float(Bias[h * st.bias_h]);
  const float Ah = A[h * st.A_h];
  const float Dh = __bfloat162float(D[h * st.D_h]);
  const bf16* xp = X + b * st.x_b + h * st.x_h;
  const uint4 xraw = *reinterpret_cast<const uint4*>(xp + 8 * s);
  const float2 bc2 = *reinterpret_cast<const float2*>(BC + b * st.bc_b + g * st.bc_g + 2 * s);
  const float* qp = Q + b * st.q_b + g * st.q_g;
  float qv[16];
  #pragma unroll
  for (int k = 0; k < 16; k += 4) {
    const float4 qr = *reinterpret_cast<const float4*>(qp + 16 * s + k);
    qv[k] = qr.x; qv[k + 1] = qr.y; qv[k + 2] = qr.z; qv[k + 3] = qr.w;
  }
  const uw_t* w = W + (di * st.w_rows + woff) * 128;
  #pragma unroll
  for (int ip = 0; ip < NF_SMAPS; ++ip) {
    if (m_h > ip) {
      cp_async16(sh.maps[hs] + ip * 128 + 16 * s, w + ip * 128 + 16 * s);
      cp_async16(sh.maps[hs] + ip * 128 + 16 * s + 8, w + ip * 128 + 16 * s + 8);
#if !UW_BF16
      cp_async16(sh.maps[hs] + ip * 128 + 16 * s + 4, w + ip * 128 + 16 * s + 4);
      cp_async16(sh.maps[hs] + ip * 128 + 16 * s + 12, w + ip * 128 + 16 * s + 12);
#endif
    }
  }
  const uw_t* up = U + (di * st.sketch_rows + skoff) * 64;
  float ur[NF_UROWS][8];
  #pragma unroll
  for (int n = 0; n < NF_UROWS; ++n) {
    if (n < nrow) ld_uw8(up + n * 64 + 8 * s, ur[n]);
    else {
      #pragma unroll
      for (int i = 0; i < 8; ++i) ur[n][i] = 0.f;
    }
  }
  // ---- hop 2: slot-dependent ------------------------------------------------
  if (slot == null_slot) return;                   // uniform over the CTA
  float dk0 = 0.f, dk1 = 0.f;
  if (2 * s < wp) {
    const float2 d2 = *reinterpret_cast<const float2*>(DR + slot * st.dc_b + h * st.dc_h + 2 * s);
    dk0 = d2.x; dk1 = (2 * s + 1 < wp) ? d2.y : 0.f;
  }
  {
    const bf16* xr = XR + slot * st.xc_b + h * st.xc_h;
    #pragma unroll
    for (int t = 0; t < 15; ++t)
      if (t < wp) cp_async16(sh.ring[hs] + t * 64 + 8 * s, xr + t * 64 + 8 * s);
  }
  cp_async_commit();

  // ---- window decay and ring weights (lane s holds steps 2s and 2s+1) --------
  if (dt_cur <= 20.f) dt_cur = log1pf(__expf(dt_cur));
  if (2 * s == wp) dk0 = dt_cur;
  if (2 * s + 1 == wp) dk1 = dt_cur;
  const float pair = dk0 + dk1;
  float scan = pair;
  #pragma unroll
  for (int o = 1; o < 8; o <<= 1) { const float t = __shfl_up_sync(FULL, scan, o, 8); if (s >= o) scan += t; }
  const float cum1 = scan, cum0 = scan - dk1;
  const float tot = __shfl_sync(FULL, scan, 7, 8);
  const float dAt = Ah * tot;
  const float decay = __expf(dAt);
  const float w0 = (2 * s <= wp) ? dk0 * __expf(dAt - Ah * cum0) * bc2.x : 0.f;
  const float w1 = (2 * s + 1 <= wp) ? dk1 * __expf(dAt - Ah * cum1) * bc2.y : 0.f;
  *reinterpret_cast<float2*>(sh.wk[hs] + 2 * s) = make_float2(w0, w1);

  // ---- ring replay from shared ----------------------------------------------
  cp_async_wait_all();
  __syncwarp();
  float2 ya = make_float2(0.f, 0.f), yb = make_float2(0.f, 0.f), yc = make_float2(0.f, 0.f), yd = make_float2(0.f, 0.f);
  {
    const float4* wq = reinterpret_cast<const float4*>(sh.wk[hs]);
    #pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
      const float4 w4 = wq[t4];
      const float ws[4] = {w4.x, w4.y, w4.z, w4.w};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = 4 * t4 + i;
        if (t < 15 && t < wp) {
          const uint4 r = *reinterpret_cast<const uint4*>(sh.ring[hs] + t * 64 + 8 * s);
          const float2 ww = make_float2(ws[i], ws[i]);
          ya = ffma2(make_float2(bf_lo(r.x), bf_hi(r.x)), ww, ya);
          yb = ffma2(make_float2(bf_lo(r.y), bf_hi(r.y)), ww, yb);
          yc = ffma2(make_float2(bf_lo(r.z), bf_hi(r.z)), ww, yc);
          yd = ffma2(make_float2(bf_lo(r.w), bf_hi(r.w)), ww, yd);
        }
      }
    }
  }
  float y[8] = {ya.x, ya.y, yb.x, yb.y, yc.x, yc.y, yd.x, yd.y};
  const float xc[8] = {bf_lo(xraw.x), bf_hi(xraw.x), bf_lo(xraw.y), bf_hi(xraw.y),
                       bf_lo(xraw.z), bf_hi(xraw.z), bf_lo(xraw.w), bf_hi(xraw.w)};
  {
    const float wcur = sh.wk[hs][wp];
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = fmaf(xc[i], wcur, y[i]);
  }
  // ---- pivot dots (all four, no divergence) and the 8-lane transpose reduction --
  float dots[4];
  {
    float d[4];
    #pragma unroll
    for (int ip = 0; ip < 4; ++ip) {
      const int ipc = (ip < m_h) ? ip : 0;
      const uw_t* mp = (ipc < NF_SMAPS) ? (sh.maps[hs] + ipc * 128 + 16 * s) : (w + ipc * 128 + 16 * s);
      float mv[16];
      ld_uw8(mp, mv); ld_uw8(mp + 8, mv + 8);
      float2 dd = make_float2(0.f, 0.f);
      #pragma unroll
      for (int i = 0; i < 8; ++i) dd = ffma2(make_float2(mv[2 * i], mv[2 * i + 1]), make_float2(qv[2 * i], qv[2 * i + 1]), dd);
      d[ip] = (ip < m_h) ? dd.x + dd.y : 0.f;
    }
    const bool hi4 = (s & 4) != 0, hi2 = (s & 2) != 0;
    const float r0 = __shfl_xor_sync(FULL, hi4 ? d[0] : d[2], 4, 8);
    const float r1 = __shfl_xor_sync(FULL, hi4 ? d[1] : d[3], 4, 8);
    const float e0 = (hi4 ? d[2] : d[0]) + r0, e1 = (hi4 ? d[3] : d[1]) + r1;
    const float r2 = __shfl_xor_sync(FULL, hi2 ? e0 : e1, 2, 8);
    float f = (hi2 ? e1 : e0) + r2;
    f += __shfl_xor_sync(FULL, f, 1, 8);
    // total of pivot k sits in lanes with (s & 6) == 2k
    #pragma unroll
    for (int k = 0; k < 4; ++k) dots[k] = __shfl_sync(FULL, f, 2 * k, 8);
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (nf_h == 0 && m_h <= 4) {
    float2 a0 = make_float2(0.f, 0.f), a1 = make_float2(0.f, 0.f), a2 = make_float2(0.f, 0.f), a3 = make_float2(0.f, 0.f);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      if (n < m_h) {
        float u[8];
        if (n < NF_UROWS) {
          #pragma unroll
          for (int i = 0; i < 8; ++i) u[i] = ur[n < NF_UROWS ? n : 0][i];
        } else ld_uw8(up + n * 64 + 8 * s, u);
        const float2 cc = make_float2(dots[n], dots[n]);
        a0 = ffma2(make_float2(u[0], u[1]), cc, a0); a1 = ffma2(make_float2(u[2], u[3]), cc, a1);
        a2 = ffma2(make_float2(u[4], u[5]), cc, a2); a3 = ffma2(make_float2(u[6], u[7]), cc, a3);
      }
    }
    acc[0] = a0.x; acc[1] = a0.y; acc[2] = a1.x; acc[3] = a1.y; acc[4] = a2.x; acc[5] = a2.y; acc[6] = a3.x; acc[7] = a3.y;
  } else {
    // ---- coefficients (C only for dense heads; AG only for m > 4); the ring is free now ----
    float* coef = reinterpret_cast<float*>(sh.ring[hs]);
    const float* cp = C + b * st.C_b + g * st.C_g;
    for (int n = s; n < nrow; n += 8) {
      float c = (n < nf_h) ? cp[n] : 0.f;
      if (n < m_h) {
        if (m_h <= 4) c += (n == 0 ? dots[0] : n == 1 ? dots[1] : n == 2 ? dots[2] : dots[3]);
        else {
          const long index = di * (5L * st.sm) + off_h + n;
          c += fmaf(AG[index + st.sm], dots[0], AG[index] * qp[n]);
          #pragma unroll
          for (int ip = 1; ip < 4; ++ip) c = fmaf(AG[index + (ip + 1) * st.sm], dots[ip], c);
        }
      }
      coef[n] = c;
    }
    __syncwarp();
    #pragma unroll
    for (int n = 0; n < NF_UROWS; ++n) {
      if (n < nrow) {
        const float c = coef[n];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(ur[n][i], c, acc[i]);
      }
    }
    for (int n0 = NF_UROWS; n0 < nrow; n0 += NF_UBATCH) {
      float v[NF_UBATCH][8];
      #pragma unroll
      for (int u = 0; u < NF_UBATCH; ++u)
        if (n0 + u < nrow) ld_uw8(up + (n0 + u) * 64 + 8 * s, v[u]);
      #pragma unroll
      for (int u = 0; u < NF_UBATCH; ++u) {
        if (n0 + u < nrow) {
          const float c = coef[n0 + u];
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] = fmaf(v[u][i], c, acc[i]);
        }
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) { y[i] = fmaf(acc[i], decay, y[i]); y[i] = fmaf(xc[i], Dh, y[i]); }
  // ---- output and ring append ----------------------------------------------
  if (out_f32) {
    float* op = reinterpret_cast<float*>(O) + b * st.o_b + h * st.o_h + 8 * s;
    *reinterpret_cast<float4*>(op) = make_float4(y[0], y[1], y[2], y[3]);
    *reinterpret_cast<float4*>(op + 4) = make_float4(y[4], y[5], y[6], y[7]);
  } else {
    *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(O) + b * st.o_b + h * st.o_h + 8 * s) =
        make_uint4(pack2(y[0], y[1]), pack2(y[2], y[3]), pack2(y[4], y[5]), pack2(y[6], y[7]));
  }
  if (wp < 16) {
    *reinterpret_cast<uint4*>(XR + slot * st.xc_b + h * st.xc_h + wp * 64 + 8 * s) = xraw;
    if (s == 0) DR[slot * st.dc_b + h * st.dc_h + wp] = dt_cur;
    if ((h % SK_HPG) == 0) {
      const bf16* bp = B + b * st.B_b + g * st.B_g + 16 * s;
      bf16* brp = BR + slot * st.Bc_b + g * st.Bc_g + wp * 128 + 16 * s;
      const uint4 r0 = *reinterpret_cast<const uint4*>(bp), r1 = *reinterpret_cast<const uint4*>(bp + 8);
      *reinterpret_cast<uint4*>(brp) = make_uint4(pack_rt(r0.x), pack_rt(r0.y), pack_rt(r0.z), pack_rt(r0.w));
      *reinterpret_cast<uint4*>(brp + 8) = make_uint4(pack_rt(r1.x), pack_rt(r1.y), pack_rt(r1.z), pack_rt(r1.w));
    }
  }
}
#endif

#if NF_ROWS > 1
extern "C" __global__ void __launch_bounds__(16 * NF_HEADS, NF_MINB)
nf_kernel_rows(const bf16* __restrict__ X, const bf16* __restrict__ DT, const bf16* __restrict__ Bias,
               const float* __restrict__ A, const bf16* __restrict__ B, const float* __restrict__ C,
               const bf16* __restrict__ D, void* __restrict__ O, bf16* __restrict__ XR, float* __restrict__ DR,
               bf16* __restrict__ BR, const float* __restrict__ BC, const int* __restrict__ WP,
               const signed char* __restrict__ FL, const int* __restrict__ Slots, int null_slot,
               const int* __restrict__ NFH, const int* __restrict__ MH, const int* __restrict__ OFF,
               const int* __restrict__ Map, const uw_t* __restrict__ W, const float* __restrict__ AG,
               const float* __restrict__ Q, const uw_t* __restrict__ U, const int* __restrict__ SkOff,
               const int* __restrict__ WOff, NfStrides st_rt, int out_f32, int batch) {
#if NF_CONST_STRIDES
  NfStrides st = NF_STRIDE_INIT;
  st.w_rows = st_rt.w_rows; st.sketch_rows = st_rt.sketch_rows; st.sm = st_rt.sm;
#else
  const NfStrides st = st_rt;
#endif
  __shared__ NfShared sh;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int hs = 2 * warp + (lane >> 4);
  const int sl = lane & 15;
  const int h = NF_HEADS * blockIdx.x + hs;
  const int g = h / SK_HPG;
  const unsigned hmask = 0xffffu << (lane & 16);   // this head's half of the warp
  // per-head constants, loaded once for all rows of this CTA
  const int m_h = MH[h];
  const int nf_h = NFH ? min(NFH[h], 128) : 0;
  const int off_h = OFF[h];
  const int skoff = SkOff[h];
  const int woff = WOff[h];
  const int nrow = max(nf_h, m_h);
  const float Ah = A[h * st.A_h];
  const float Dh = __bfloat162float(D[h * st.D_h]);
  const float bias = __bfloat162float(Bias[h * st.bias_h]);
  const int b0 = blockIdx.y * NF_ROWS;
  // row-r state that the previous iteration prefetches: slot, wp, dt-ring value, ring rows in shared
  long slot = 0; int wp = 0; bool active = false; float dtk = 0.f;
  auto stage_row = [&](int b) {
    active = (b < batch) && FL[b] == 0;
    if (!active) return;
    slot = Slots[b]; wp = WP[b];
    if (slot == null_slot) { active = false; return; }
    dtk = (sl < wp) ? DR[slot * st.dc_b + h * st.dc_h + sl] : 0.f;
    const bf16* xr = XR + slot * st.xc_b + h * st.xc_h;
    const int c = sl & 7, t0 = sl >> 3;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int t = t0 + 2 * k;
      if (t < wp) cp_async16(sh.ring[hs] + ring_at(t, c), xr + t * 64 + 8 * c);
    }
    cp_async_commit();
  };
  stage_row(b0);
  #pragma unroll 1
  for (int rr = 0; rr < NF_ROWS; ++rr) {
    const int b = b0 + rr;
    if (b >= batch) break;
    const bool row_active = active;
    const long row_slot = slot; const int row_wp = wp; const float row_dtk = dtk;
    const long di = Map[b];
    // ---- hop 1 for this row -------------------------------------------------
    float dt_cur = 0.f; uint2 xraw = make_uint2(0u, 0u); float bcw = 0.f; float qv[8]; float4 ur[NF_UROWS];
    const uw_t* w = W + (di * st.w_rows + woff) * 128;
    const uw_t* up = U + (di * st.sketch_rows + skoff) * 64;
    const float* qp = Q + b * st.q_b + g * st.q_g;
    if (row_active) {
      dt_cur = __bfloat162float(DT[b * st.dt_b + h * st.dt_h]) + bias;
      xraw = *reinterpret_cast<const uint2*>(X + b * st.x_b + h * st.x_h + 4 * sl);
      bcw = BC[b * st.bc_b + g * st.bc_g + sl];
      const float4 q0 = *reinterpret_cast<const float4*>(qp + 8 * sl), q1 = *reinterpret_cast<const float4*>(qp + 8 * sl + 4);
      qv[0] = q0.x; qv[1] = q0.y; qv[2] = q0.z; qv[3] = q0.w; qv[4] = q1.x; qv[5] = q1.y; qv[6] = q1.z; qv[7] = q1.w;
      #pragma unroll
      for (int ip = 0; ip < NF_SMAPS; ++ip) {
        if (m_h > ip) {
          cp_async16(sh.maps[hs] + ip * 128 + 8 * sl, w + ip * 128 + 8 * sl);
#if !UW_BF16
          cp_async16(sh.maps[hs] + ip * 128 + 8 * sl + 4, w + ip * 128 + 8 * sl + 4);
#endif
        }
      }
      cp_async_commit();
      #pragma unroll
      for (int n = 0; n < NF_UROWS; ++n)
        if (n < nrow) ur[n] = ld_uw4(up + n * 64 + 4 * sl);
        else ur[n] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // ---- window decay and ring weights -----------------------------------------
    float y[4] = {0.f, 0.f, 0.f, 0.f};
    float decay = 1.f;
    if (row_active) {
      if (dt_cur <= 20.f) dt_cur = log1pf(__expf(dt_cur));
      float dk = row_dtk;
      if (sl == row_wp) dk = dt_cur;
      float cum = dk;
      #pragma unroll
      for (int o = 1; o < 16; o <<= 1) { const float t = __shfl_up_sync(FULL, cum, o, 16); if (sl >= o) cum += t; }
      const float tot = __shfl_sync(FULL, cum, 15, 16);
      const float dAt = Ah * tot;
      decay = __expf(dAt);
      float wk = 0.f;
      if (sl <= row_wp) wk = dk * __expf(dAt - Ah * cum) * bcw;
      sh.wk[hs][sl] = wk;
    }
    cp_async_wait_all();
    __syncwarp();
    if (row_active) {
      float2 y01 = make_float2(0.f, 0.f), y23 = make_float2(0.f, 0.f);
      const int c0 = (4 * sl) >> 3, e0 = (4 * sl) & 7;
      const float4* wq = reinterpret_cast<const float4*>(sh.wk[hs]);
      #pragma unroll
      for (int t4 = 0; t4 < 4; ++t4) {
        const float4 w4 = wq[t4];
        const float ws[4] = {w4.x, w4.y, w4.z, w4.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int t = 4 * t4 + i;
          if (t < 15 && t < row_wp) {
            const uint2 r = *reinterpret_cast<const uint2*>(sh.ring[hs] + ring_at(t, c0) + e0);
            const float2 ww = make_float2(ws[i], ws[i]);
            y01 = ffma2(make_float2(bf_lo(r.x), bf_hi(r.x)), ww, y01);
            y23 = ffma2(make_float2(bf_lo(r.y), bf_hi(r.y)), ww, y23);
          }
        }
      }
      y[0] = y01.x; y[1] = y01.y; y[2] = y23.x; y[3] = y23.y;
    }
    // ---- the ring is consumed: prefetch the next row's ring and dt-ring value ------
    __syncwarp();
    if (rr + 1 < NF_ROWS) stage_row(b + 1);
    if (!row_active) continue;
    const float xc[4] = {bf_lo(xraw.x), bf_hi(xraw.x), bf_lo(xraw.y), bf_hi(xraw.y)};
    {
      const float wcur = __shfl_sync(FULL, sh.wk[hs][sl], row_wp, 16);
      #pragma unroll
      for (int i = 0; i < 4; ++i) y[i] = fmaf(xc[i], wcur, y[i]);
    }
    // ---- pivot dots ------------------------------------------------------------
    float dots[4] = {0.f, 0.f, 0.f, 0.f};
    {
      float d[4];
      #pragma unroll
      for (int ip = 0; ip < 4; ++ip) {
        const int ipc = (ip < m_h) ? ip : 0;
        const uw_t* mp = (ipc < NF_SMAPS) ? (sh.maps[hs] + ipc * 128 + 8 * sl) : (w + ipc * 128 + 8 * sl);
        const float4 a = ld_uw4(mp), bq = ld_uw4(mp + 4);
        float2 dd = make_float2(0.f, 0.f);
        dd = ffma2(make_float2(a.x, a.y), make_float2(qv[0], qv[1]), dd);
        dd = ffma2(make_float2(a.z, a.w), make_float2(qv[2], qv[3]), dd);
        dd = ffma2(make_float2(bq.x, bq.y), make_float2(qv[4], qv[5]), dd);
        dd = ffma2(make_float2(bq.z, bq.w), make_float2(qv[6], qv[7]), dd);
        d[ip] = (ip < m_h) ? dd.x + dd.y : 0.f;
      }
      const bool hi8 = (sl & 8) != 0, hi4 = (sl & 4) != 0;
      const float r0 = __shfl_xor_sync(FULL, hi8 ? d[0] : d[2], 8, 16);
      const float r1 = __shfl_xor_sync(FULL, hi8 ? d[1] : d[3], 8, 16);
      const float e0 = (hi8 ? d[2] : d[0]) + r0, e1 = (hi8 ? d[3] : d[1]) + r1;
      const float r2 = __shfl_xor_sync(FULL, hi4 ? e0 : e1, 4, 16);
      float f = (hi4 ? e1 : e0) + r2;
      f += __shfl_xor_sync(FULL, f, 2, 16);
      f += __shfl_xor_sync(FULL, f, 1, 16);
      #pragma unroll
      for (int k = 0; k < 4; ++k) dots[k] = __shfl_sync(FULL, f, 4 * k, 16);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (nf_h == 0 && m_h <= 4) {
      float2 a01 = make_float2(0.f, 0.f), a23 = make_float2(0.f, 0.f);
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        if (n < m_h) {
          const float4 u = (n < NF_UROWS) ? ur[n < NF_UROWS ? n : 0] : ld_uw4(up + n * 64 + 4 * sl);
          const float2 cc = make_float2(dots[n], dots[n]);
          a01 = ffma2(make_float2(u.x, u.y), cc, a01); a23 = ffma2(make_float2(u.z, u.w), cc, a23);
        }
      }
      acc[0] = a01.x; acc[1] = a01.y; acc[2] = a23.x; acc[3] = a23.y;
    } else {
      const float* cp = C + b * st.C_b + g * st.C_g;
      for (int n = sl; n < nrow; n += 16) {
        float c = (n < nf_h) ? cp[n] : 0.f;
        if (n < m_h) {
          if (m_h <= 4) c += (n == 0 ? dots[0] : n == 1 ? dots[1] : n == 2 ? dots[2] : dots[3]);
          else {
            const long index = di * (5L * st.sm) + off_h + n;
            c += fmaf(AG[index + st.sm], dots[0], AG[index] * qp[n]);
            #pragma unroll
            for (int ip = 1; ip < 4; ++ip) c = fmaf(AG[index + (ip + 1) * st.sm], dots[ip], c);
          }
        }
        sh.coef[hs][n] = c;
      }
      __syncwarp(hmask);
      #pragma unroll
      for (int n = 0; n < NF_UROWS; ++n) {
        if (n < nrow) {
          const float c = sh.coef[hs][n];
          acc[0] = fmaf(ur[n].x, c, acc[0]); acc[1] = fmaf(ur[n].y, c, acc[1]);
          acc[2] = fmaf(ur[n].z, c, acc[2]); acc[3] = fmaf(ur[n].w, c, acc[3]);
        }
      }
      for (int n0 = NF_UROWS; n0 < nrow; n0 += NF_UBATCH) {
        float4 v[NF_UBATCH];
        #pragma unroll
        for (int u = 0; u < NF_UBATCH; ++u)
          if (n0 + u < nrow) v[u] = ld_uw4(up + (n0 + u) * 64 + 4 * sl);
        #pragma unroll
        for (int u = 0; u < NF_UBATCH; ++u) {
          if (n0 + u < nrow) {
            const float c = sh.coef[hs][n0 + u];
            acc[0] = fmaf(v[u].x, c, acc[0]); acc[1] = fmaf(v[u].y, c, acc[1]);
            acc[2] = fmaf(v[u].z, c, acc[2]); acc[3] = fmaf(v[u].w, c, acc[3]);
          }
        }
      }
      __syncwarp(hmask);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) { y[i] = fmaf(acc[i], decay, y[i]); y[i] = fmaf(xc[i], Dh, y[i]); }
    if (out_f32) {
      *reinterpret_cast<float4*>(reinterpret_cast<float*>(O) + b * st.o_b + h * st.o_h + 4 * sl) = make_float4(y[0], y[1], y[2], y[3]);
    } else {
      *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(O) + b * st.o_b + h * st.o_h + 4 * sl) = make_uint2(pack2(y[0], y[1]), pack2(y[2], y[3]));
    }
    if (row_wp < 16) {
      *reinterpret_cast<uint2*>(XR + row_slot * st.xc_b + h * st.xc_h + row_wp * 64 + 4 * sl) = xraw;
      if (sl == 0) DR[row_slot * st.dc_b + h * st.dc_h + row_wp] = dt_cur;
      if ((h % SK_HPG) == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(B + b * st.B_b + g * st.B_g + 8 * sl);
        *reinterpret_cast<uint4*>(BR + row_slot * st.Bc_b + g * st.Bc_g + row_wp * 128 + 8 * sl) =
            make_uint4(pack_rt(raw.x), pack_rt(raw.y), pack_rt(raw.z), pack_rt(raw.w));
      }
    }
  }
}
#endif
