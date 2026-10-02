// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
// SketchSSM decode kernels (https://arxiv.org/abs/2609.33051).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#define FULL 0xffffffffu
#ifndef NW
#define NW 4
#endif
#ifndef S8_PREFETCH
#define S8_PREFETCH 0
#endif
#ifndef S7_MINB
#define S7_MINB 5
#endif
#ifndef KDA_W
#define KDA_W 16                // window rows W (a multiple of 16): ring rows per head, f history columns
#endif
static_assert(KDA_W % 16 == 0 && KDA_W >= 16, "kda step: the window must be a positive multiple of 16");
#define NC8 (KDA_W / 8)         // 8-row chunks of the window
#define ROWB 256                // staged rows: 256 B, 16-byte chunks XOR-swizzled by (row & 7)
#define CB (8 * ROWB)           // 8-row buffers: Y0, Y1 (d / u row chunks; Y1 also the second Phi / U chunk), X (Phi / U rows)
#define XBYTES CB
#define YBYTES (2 * CB)
#define BBYTES 1024             // [j 8][slot 16] x 8 B: B fragments of [cd k^ hi, lo ; cd q^ hi, lo], slot = (n 4 + c) ^ SW(j)
#define SW(j) ((((j) & 1) | (((j) >> 1) << 2)) * 8)   // bank-conflict-free for both the writer (lanes 4 j + a) and the reader (lanes (n, c))
#define WBYTES (XBYTES + YBYTES + BBYTES + 8 * KDA_W + 1024 + (KDA_W > 16 ? BBYTES : 0))   // + s_kk[2 W] + s_pk / s_c [256] (+ W > 16: B2)
static_assert(NW * WBYTES + 1024 <= 48 * 1024, "kda step: NW warps x the window's per-warp shared memory exceed the 48 KB static shared memory limit; lower NW");

__device__ __forceinline__ float fexp(float x) {              // ex2.approx.ftz(x log2 e): no denormal fix-up path (both step kernels)
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f)); return y;
}
__device__ __forceinline__ float fsig(float x) { return __fdividef(1.f, 1.f + fexp(-x)); }   // sigmoid: rcp.approx
__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
    return x;
}
__device__ __forceinline__ float4 unpack4(uint2 u) {
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
__device__ __forceinline__ float4 ld4_bf16(const __nv_bfloat16* p) { return unpack4(*(const uint2*)p); }
__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
    const __nv_bfloat162 h2 = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&h2);
}
__device__ __forceinline__ void st4_bf16(__nv_bfloat16* p, float4 v) { *(uint2*)p = make_uint2(pack_bf16(v.x, v.y), pack_bf16(v.z, v.w)); }
__device__ __forceinline__ float dot4(float4 a, float4 b) { return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w))); }
__device__ __forceinline__ void split2(float x0, float x1, unsigned& h, unsigned& l) {
    h = pack_bf16(x0, x1);
    l = pack_bf16(x0 - __uint_as_float(h << 16), x1 - __uint_as_float(h & 0xffff0000u));
}
__device__ __forceinline__ void mma16816_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(unsigned* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(unsigned* r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ uint2 lds64(unsigned a) {
    uint2 v; asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ void cp16z(unsigned dst, const void* src, unsigned size) {   // size 16 or 0 (zero fill)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(size) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
// the rings of a state page: every ring shares the page stride RS (bytes), rp = page * RS
template <typename T> __device__ __forceinline__ T* page_ring(T* base, long rp) { return (T*)((char*)base + rp); }

// staging of one 8-row chunk (rows r0 + 2 i, i < 4; chunk ch): row r of the chunk from src + (first + r) rows when
// first + r < nrows, else zeros; the lane's four swizzled destination bases and its source offset are precomputed
struct Stage { unsigned d[4]; unsigned so; int r0; };
__device__ __forceinline__ Stage stage_init(unsigned y_u, int lane) {
    Stage st; const int r0 = lane >> 4, ch = lane & 15;
    #pragma unroll
    for (int k = 0; k < 4; ++k) st.d[k] = y_u + r0 * ROWB + ((ch ^ r0 ^ (2 * k)) << 4);
    st.so = r0 * ROWB + ch * 16; st.r0 = r0;
    return st;
}
__device__ __forceinline__ void stage8(const Stage& st, unsigned buf_off, const __nv_bfloat16* src, int first, int nrows, int skip = -1) {
    const char* s = (const char*)src + (size_t)first * ROWB + st.so;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = first + st.r0 + 2 * i;
        if (r != skip) cp16z(st.d[i] + buf_off + i * 2 * ROWB, s + i * 2 * ROWB, (r < nrows) ? 16u : 0u);
    }
}

// W > 16: B fragments [cd k^ hi, lo ; cd q^ hi, lo] of the d rows of an 8-row chunk c >= 1, cd = exp(pre - ref_c), in
// the fragment order of the Bd block (read by the lanes g >= 4: MMA columns 4..7)
__device__ __forceinline__ void write_bd(unsigned char* blk, int lane, float4 cd, float4 kh, float4 qh) {
    const int j = lane >> 2, a = lane & 3;
    const int c0 = (a < 2) ? 2 * a : 2 * a - 4, word = (a < 2) ? 0 : 1;
    const unsigned sw = SW(j) >> 3;
    unsigned* e = (unsigned*)blk + j * 32 + word;
    unsigned h0, l0, h1, l1;
    split2(cd.x * kh.x, cd.y * kh.y, h0, l0); split2(cd.z * kh.z, cd.w * kh.w, h1, l1);
    e[((0 + c0) ^ sw) * 2] = h0; e[((0 + c0 + 1) ^ sw) * 2] = h1; e[((4 + c0) ^ sw) * 2] = l0; e[((4 + c0 + 1) ^ sw) * 2] = l1;
    split2(cd.x * qh.x, cd.y * qh.y, h0, l0); split2(cd.z * qh.z, cd.w * qh.w, h1, l1);
    e[((8 + c0) ^ sw) * 2] = h0; e[((8 + c0 + 1) ^ sw) * 2] = h1; e[((12 + c0) ^ sw) * 2] = l0; e[((12 + c0 + 1) ^ sw) * 2] = l1;
}

// token row strides (elements) of the step inputs: q / k / v / gate rows are [H][128] with dense heads, beta rows [H]
struct InStrides { unsigned q, k, v, g, beta; };

__global__ void __launch_bounds__(32 * NW, S7_MINB)
kda_step_kernel(const int* __restrict__ Heads, int NH,
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ Kin, const __nv_bfloat16* __restrict__ Vin,
    const __nv_bfloat16* __restrict__ Gate, const __nv_bfloat16* __restrict__ Beta, const InStrides IS,
    const float* __restrict__ A, const float* __restrict__ Bias,
    const int* __restrict__ Slots, const int* __restrict__ Meta, const int* __restrict__ Pos,
    float* __restrict__ State, long S0, long S1,
    const __nv_bfloat16* __restrict__ U, const __nv_bfloat16* __restrict__ Phi, const int* __restrict__ Ranks,
    __nv_bfloat16* __restrict__ KR, __nv_bfloat16* __restrict__ VR, float* __restrict__ BR,
    float* __restrict__ PrefixR, __nv_bfloat16* __restrict__ FR, __nv_bfloat16* __restrict__ UR, __nv_bfloat16* __restrict__ DR, long RS,
    __nv_bfloat16* __restrict__ Out, int H, int G, float scale)
{
    __shared__ __align__(128) unsigned char sm[NW * WBYTES];
    __shared__ __align__(16) unsigned sZ[256];                    // 1 KB of zeros: B fragments of lanes g >= 4
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, c = lane & 3;
    for (int i = threadIdx.x; i < 256; i += 32 * NW) sZ[i] = 0u;
    __syncthreads();
    const int row = blockIdx.x, hidx = blockIdx.y * NW + warp;
    if (hidx >= NH) return;
    const int head = Heads[hidx];
    const unsigned io = (unsigned)(row * H + head) * 128u + 4u * lane, hl = (unsigned)head * 128u + 4u * lane;   // io: output
    const uint2 kraw = *(const uint2*)(Kin + ((unsigned)row * IS.k + hl)), vraw = *(const uint2*)(Vin + ((unsigned)row * IS.v + hl));   // row-only operands: issued before the slot resolves
    const float4 q0 = ld4_bf16(Q + ((unsigned)row * IS.q + hl)), g0 = ld4_bf16(Gate + ((unsigned)row * IS.g + hl));
    const float4 b0 = *(const float4*)(Bias + head * 128 + 4 * lane);
    const float amp = fexp(A[head]);
    const float beta = fsig(__bfloat162float(Beta[(unsigned)row * IS.beta + (unsigned)head]));
    const int page = Slots[row];
    if (page <= 0) { *(uint2*)(Out + io) = make_uint2(0u, 0u); return; }   // null row: zero output
    const int slot = Meta[row], pos = Pos[row];                  // slot: the request's sketch row
    const int m = Ranks[head];
    const unsigned base = (unsigned)head * (unsigned)KDA_W;      // ring row 0 of head in the page's rings
    const long rp = (long)page * RS;                             // the rings share the page stride RS (bytes)
    const unsigned sh = (unsigned)(slot * H + head) * (unsigned)G;   // sketch row 0 of (slot, head)
    const bool sketch = (m > 0) && (pos < KDA_W - 1);
    unsigned char* wsm = sm + warp * WBYTES;
    const unsigned y_u = smem_u32(wsm), b_u = y_u + YBYTES + XBYTES;
    float* s_kk = (float*)(wsm + XBYTES + YBYTES + BBYTES);      // kk_t (0..W-1) | kq_t (W..2W-1)
    float* s_c = s_kk + 2 * KDA_W;                               // pk_g (0..127) | pq_g (128..255), then c_g in place
    const __nv_bfloat16* phi = Phi + (size_t)sh * 128;
    const __nv_bfloat16* ul = U + (size_t)sh * 128;
    __nv_bfloat16* fr = FR + (size_t)sh * KDA_W + g * KDA_W + 2 * c;   // [g][t]: this lane's fragment words
    const Stage st = stage_init(y_u, lane);
    const __nv_bfloat16* dr = page_ring(DR, rp) + (size_t)base * 128;
    const __nv_bfloat16* ur = page_ring(UR, rp) + (size_t)base * 128;
    unsigned fa[4];                                              // f^T fragment of ranks 0..15
    const int nd = (pos + 7) >> 3, np = (m + 7) >> 3;            // 8-row chunks of the d rows / Phi rows
    if (sketch) {
        stage8(st, 0, dr, 0, pos);                               // d rows t < 8 -> Y0
        stage8(st, YBYTES, phi, 0, m);                           // Phi rows g < 8 -> X
        if (nd > 1) stage8(st, CB, dr, 8, pos);                  // second d chunk -> Y1, else a second Phi chunk -> Y1
        else if (np > 1) stage8(st, CB, phi, 8, m);
        cp_commit();
        fa[0] = *(const unsigned*)(fr); fa[1] = *(const unsigned*)(fr + 8 * KDA_W);
        fa[2] = *(const unsigned*)(fr + 8); fa[3] = *(const unsigned*)(fr + 8 * KDA_W + 8);
#if S8_PREFETCH
        #pragma unroll
        for (int i = 0; i < 4; ++i) {                            // the readout rows (u rows t < pos, U rows g < m) into L2 now
            const int r = st.r0 + 2 * i;
            if (r < pos) prefetch_l2((const char*)ur + st.so + i * 2 * ROWB);
            if (r < m) prefetch_l2((const char*)ul + st.so + i * 2 * ROWB);
        }
#endif
    }
    const float4 k0 = unpack4(kraw);
    float4 prev = make_float4(0.f, 0.f, 0.f, 0.f);
    if (pos > 0) prev = *(const float4*)(page_ring(PrefixR, rp) + (size_t)(base + pos - 1) * 128 + 4 * lane);
    // W > 16: the d rows of 8-row chunk c >= 1 are rebased to ref_c = prefix row 8 c - 1 (exponents <= 40 within a chunk)
    float4 refc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (KDA_W > 16 && pos >= 8) refc = *(const float4*)(page_ring(PrefixR, rp) + (size_t)(base + (pos & ~7) - 1) * 128 + 4 * lane);
    // ── current step ──
    float qn = dot4(q0, q0), kn = dot4(k0, k0);
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) { qn += __shfl_xor_sync(FULL, qn, o); kn += __shfl_xor_sync(FULL, kn, o); }
    const float qf = rsqrtf(qn + 1.e-6f) * scale, kf = rsqrtf(kn + 1.e-6f);
    const float4 qh = make_float4(q0.x * qf, q0.y * qf, q0.z * qf, q0.w * qf);
    const float4 kh = make_float4(k0.x * kf, k0.y * kf, k0.z * kf, k0.w * kf);
    float4 la;
    la.x = -5.f * fsig(amp * (g0.x + b0.x)); la.y = -5.f * fsig(amp * (g0.y + b0.y));   // same arithmetic as the fp32-storage step (bitwise state parity)
    la.z = -5.f * fsig(amp * (g0.z + b0.z)); la.w = -5.f * fsig(amp * (g0.w + b0.w));
    const float4 pre = make_float4(prev.x + la.x, prev.y + la.y, prev.z + la.z, prev.w + la.w);
    *(uint2*)(page_ring(KR, rp) + (size_t)(base + pos) * 128 + 4 * lane) = kraw;
    *(uint2*)(page_ring(VR, rp) + (size_t)(base + pos) * 128 + 4 * lane) = vraw;
    *(float4*)(page_ring(PrefixR, rp) + (size_t)(base + pos) * 128 + 4 * lane) = pre;
    if (lane == 0) page_ring(BR, rp)[base + pos] = beta;
    __nv_bfloat16* po = Out + io;
    if (m <= 0) {
        float* sp = State + (long)page * S0 + (long)head * S1 + 4 * lane;
        const float4 ea = make_float4(fexp(la.x), fexp(la.y), fexp(la.z), fexp(la.w)), v0 = unpack4(vraw);
        float4 o4 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int r = 0; r < 128; ++r) {
            float4 s = *(const float4*)(sp + (long)r * 128);
            s.x *= ea.x; s.y *= ea.y; s.z *= ea.z; s.w *= ea.w;
            const int e = r & 3;
            const float vsel = (e == 0) ? v0.x : (e == 1) ? v0.y : (e == 2) ? v0.z : v0.w;
            const float d = beta * (__shfl_sync(FULL, vsel, r >> 2) - warp_sum(dot4(s, kh)));
            s.x = fmaf(d, kh.x, s.x); s.y = fmaf(d, kh.y, s.y); s.z = fmaf(d, kh.z, s.z); s.w = fmaf(d, kh.w, s.w);
            *(float4*)(sp + (long)r * 128) = s;
            const float ov = warp_sum(dot4(s, qh));
            if ((r >> 2) == lane) { if (e == 0) o4.x = ov; else if (e == 1) o4.y = ov; else if (e == 2) o4.z = ov; else o4.w = ov; }
        }
        st4_bf16(po, o4);
        return;
    }
    if (!sketch) return;
    const float4 cd = make_float4(fexp(pre.x), fexp(pre.y), fexp(pre.z), fexp(pre.w));
    if constexpr (KDA_W > 16)
        st4_bf16(page_ring(DR, rp) + (size_t)(base + pos) * 128 + 4 * lane, make_float4(kh.x * fexp(refc.x - pre.x), kh.y * fexp(refc.y - pre.y), kh.z * fexp(refc.z - pre.z), kh.w * fexp(refc.w - pre.w)));
    else
    st4_bf16(page_ring(DR, rp) + (size_t)(base + pos) * 128 + 4 * lane, make_float4(kh.x * fexp(-pre.x), kh.y * fexp(-pre.y), kh.z * fexp(-pre.z), kh.w * fexp(-pre.w)));
    // ── B operands: four columns n = (first hi, first lo, second hi, second lo), one MMA per k-step; lane (g, c)
    //    holds column g.  Bd = [cd k^ ; cd q^] goes to shared memory in fragment order: entry (n, j, c) = {b0, b1}. ──
    {
        const int j = lane >> 2, a = lane & 3;
        const int c0 = (a < 2) ? 2 * a : 2 * a - 4, word = (a < 2) ? 0 : 1;
        const unsigned sw = SW(j) >> 3;                          // slot swizzle of this j
        unsigned* e = (unsigned*)(wsm + YBYTES + XBYTES) + j * 32 + word;   // block j; entry (n, c) at slot ((n 4 + c) ^ sw) * 2 words
        unsigned h0, l0, h1, l1;
        split2(cd.x * kh.x, cd.y * kh.y, h0, l0); split2(cd.z * kh.z, cd.w * kh.w, h1, l1);
        e[((0 + c0) ^ sw) * 2] = h0; e[((0 + c0 + 1) ^ sw) * 2] = h1; e[((4 + c0) ^ sw) * 2] = l0; e[((4 + c0 + 1) ^ sw) * 2] = l1;
        split2(cd.x * qh.x, cd.y * qh.y, h0, l0); split2(cd.z * qh.z, cd.w * qh.w, h1, l1);
        e[((8 + c0) ^ sw) * 2] = h0; e[((8 + c0 + 1) ^ sw) * 2] = h1; e[((12 + c0) ^ sw) * 2] = l0; e[((12 + c0 + 1) ^ sw) * 2] = l1;
    }
    const float kq_cur = warp_sum(dot4(kh, qh));
    const unsigned bbase = (g < 4) ? b_u : smem_u32(sZ), bslot = (unsigned)(((g & 3) * 4 + c) * 8);
#define BFRAG(j) (bbase + (bslot ^ SW(j)) + (j) * 128)
    // ldmatrix lane addresses (swizzled): non-trans A rows 0..7 from Y (lanes lt = 0 / 2), rows 8..15 from X (lt = 1 / 3);
    // trans A^T k 0..7 from Y (lt < 2), k 8..15 from X (lt >= 2).  Rows lr of each buffer, chunk (2 j + parity) ^ lr.
    const int lr = lane & 7, lt = lane >> 3;
    unsigned ax[4], tx[4];
    {
        const unsigned a_x = (unsigned)(((lt >> 1) ^ lr) << 4), t_x = (unsigned)(((lt & 1) ^ lr) << 4);
        #pragma unroll
        for (int k = 0; k < 4; ++k) { ax[k] = (32 * k) ^ a_x; tx[k] = (32 * k) ^ t_x; }
    }
    const unsigned rowY0 = y_u + lr * ROWB, rowY1 = rowY0 + CB, rowX = rowY0 + YBYTES, zrow = smem_u32(sZ);   // zero rows: 8 chunks of sZ
    const int npass = (nd > np) ? nd : np;
    // W > 16: the rebased B fragments of d chunks >= 1 (B2, columns 4..7)
    unsigned char* b2 = wsm + XBYTES + YBYTES + BBYTES + 8 * KDA_W + 1024;
    // ── dots: pass i = d chunk i (rows 0..7) + Phi chunk i (rows 8..15) against Bd ──
    //    lane c = 0: kk_t = C[t][0] + C[t][1] (rows t = g: the d row), pk_g = C[g + 8][0] + C[g + 8][1]; c = 1: kq_t, pq_g
    //    W > 16: d chunks alternate between Y0 / Y1 (chunk i + 1 prefetched during pass i); chunks i >= 1 are rebased,
    //    their kk_t / kq_t are columns 4 + 5 / 6 + 7 (lanes c = 2 / 3) against B2
    for (int i = 0; i < npass; ++i) {
        unsigned lo = zrow, hi = rowX;                           // pass sources: rows 0..7 (d chunk i), rows 8..15 (Phi chunk i)
        if (i < nd) lo = (KDA_W > 16 ? (i & 1) : i) ? rowY1 : rowY0;
        bool next = false;                                       // W > 16: d chunk i + 1 in flight (one group behind)
        if (i == 1 && nd <= 1) hi = rowY1;                       // Phi chunk 1 was staged with chunk 0
        else if (i > 0) {                                        // Phi chunk i into X once chunk i - 1 is consumed (exposed; rare)
            __syncwarp();
            if constexpr (KDA_W > 16) {
                float4 rf = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i < nd) rf = *(const float4*)(page_ring(PrefixR, rp) + (size_t)(base + 8 * i - 1) * 128 + 4 * lane);
                stage8(st, YBYTES, phi, 8 * i, m);
                cp_commit();
                if (i + 1 < nd) { stage8(st, ((i + 1) & 1) * CB, dr, 8 * (i + 1), pos); cp_commit(); next = true; }
                if (i < nd) write_bd(b2, lane, make_float4(fexp(pre.x - rf.x), fexp(pre.y - rf.y), fexp(pre.z - rf.z), fexp(pre.w - rf.w)), kh, qh);
            } else {
            stage8(st, YBYTES, phi, 8 * i, m);
            cp_commit();
            }
        }
        const unsigned ra = (lt & 1) ? hi : lo;
        if constexpr (KDA_W > 16) { if (next) cp_wait<1>(); else cp_wait<0>(); }
        else cp_wait<0>();
        __syncwarp();
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if constexpr (KDA_W > 16) {
            const unsigned bb = (g >= 4 && i >= 1 && i < nd) ? smem_u32(b2) : bbase;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned a[4]; ldsm4(a, ra + ax[j & 3] + (j >> 2) * 128);
                const uint2 b = lds64(bb + (bslot ^ SW(j)) + j * 128); mma16816_bf16(acc, a, b.x, b.y);
            }
            const int cc = (i >= 1) ? c - 2 : c;                 // the lanes holding kk / kq of this chunk
            if (cc >= 0 && cc < 2 && i < NC8) s_kk[KDA_W * cc + 8 * i + g] = (8 * i + g < pos) ? acc[0] + acc[1] : 0.f;
        } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned a[4]; ldsm4(a, ra + ax[j & 3] + (j >> 2) * 128);
            const uint2 b = lds64(BFRAG(j)); mma16816_bf16(acc, a, b.x, b.y);
        }
        if (c < 2 && i < 2) s_kk[16 * c + 8 * i + g] = (8 * i + g < pos) ? acc[0] + acc[1] : 0.f;   // passes i >= 2: Phi chunks only
        }
        if (c < 2) s_c[128 * c + 8 * i + g] = acc[2] + acc[3];
    }
    __syncwarp();
    if constexpr (KDA_W > 16) {                                  // chunks without a pass: kk / kq are zero
        if (c < 2) for (int i = npass; i < NC8; ++i) s_kk[KDA_W * c + 8 * i + g] = 0.f;
    } else
    if (c < 2 && pos <= 8) s_kk[16 * c + 8 + g] = 0.f;          // no second d chunk: kk / kq of t >= 8 are zero
    __syncwarp();
    // ── coefficients per 16 ranks: sk / sq = f^T [kk ; kq] (one MMA; f^T straight from global), f_p, c ──
    unsigned kb[2];
    {
        const float2 w0 = *(const float2*)(s_kk + KDA_W * (g >> 1) + 2 * c), w1 = *(const float2*)(s_kk + KDA_W * (g >> 1) + 2 * c + 8);
        unsigned h0, l0, h1, l1;
        split2(w0.x, w0.y, h0, l0); split2(w1.x, w1.y, h1, l1);
        kb[0] = (g >= 4) ? 0u : (g & 1) ? l0 : h0; kb[1] = (g >= 4) ? 0u : (g & 1) ? l1 : h1;
    }
    for (int gc = 0; gc < m; gc += 16) {
        if (gc > 0) {
            fa[0] = *(const unsigned*)(fr + gc * KDA_W); fa[1] = *(const unsigned*)(fr + gc * KDA_W + 8 * KDA_W);
            fa[2] = *(const unsigned*)(fr + gc * KDA_W + 8); fa[3] = *(const unsigned*)(fr + gc * KDA_W + 8 * KDA_W + 8);
        }
        float sk[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816_bf16(sk, fa, kb[0], kb[1]);
        if constexpr (KDA_W > 16) {                              // k-steps t = 16 kt .. 16 kt + 15 (kk / kq are zero for t >= pos)
            for (int kt = 1; kt < KDA_W / 16 && 16 * kt < pos; ++kt) {
                unsigned fb[4], kbt[2];
                const __nv_bfloat16* fk = fr + gc * KDA_W + 16 * kt;
                fb[0] = *(const unsigned*)(fk); fb[1] = *(const unsigned*)(fk + 8 * KDA_W);
                fb[2] = *(const unsigned*)(fk + 8); fb[3] = *(const unsigned*)(fk + 8 * KDA_W + 8);
                const float* sw = s_kk + KDA_W * (g >> 1) + 16 * kt + 2 * c;
                const float2 w0 = *(const float2*)(sw), w1 = *(const float2*)(sw + 8);
                unsigned h0, l0, h1, l1;
                split2(w0.x, w0.y, h0, l0); split2(w1.x, w1.y, h1, l1);
                kbt[0] = (g >= 4) ? 0u : (g & 1) ? l0 : h0; kbt[1] = (g >= 4) ? 0u : (g & 1) ? l1 : h1;
                mma16816_bf16(sk, fb, kbt[0], kbt[1]);
            }
        }
        const float p0 = s_c[128 * (c & 1) + gc + g], p1 = s_c[128 * (c & 1) + gc + g + 8];   // c = 0: pk; c = 1: pq (lanes c >= 2: unused)
        const float d0 = p0 - sk[0] - sk[1], d1 = p1 - sk[2] - sk[3];
        const float f0 = beta * __shfl_sync(FULL, d0, lane & ~3), f1 = beta * __shfl_sync(FULL, d1, lane & ~3);   // f_p of rows g, g + 8
        const bool r0ok = gc + g < m, r1ok = gc + g + 8 < m;
        if (c == 0 && r0ok) fr[gc * KDA_W + pos] = __float2bfloat16_rn(f0);
        if (c == 0 && r1ok) fr[gc * KDA_W + 8 * KDA_W + pos] = __float2bfloat16_rn(f1);
        __syncwarp();
        if (c == 1) s_c[gc + g] = r0ok ? d0 - f0 * kq_cur : 0.f;   // c_g in place of pk_g
        if (c == 1) s_c[gc + g + 8] = r1ok ? d1 - f1 * kq_cur : 0.f;
    }
    __syncwarp();
    // ── readout: pass i = u rows chunk i (k 0..7, v at t = vr) + U rows chunk i (k 8..15) ──
    //    B columns (w_u hi, w_u lo, w_o hi, w_o lo): b0 = w pairs t = 8 i + 2 c, + 1 (w_u[t] = -beta kk_t + beta [t = vr],
    //    w_o[t] = kq_t + kq w_u[t]); b1 = c pairs g = 8 i + 2 c, + 1 in the out columns
    const int vr = (KDA_W > 16) ? (pos | 7) : (pos <= 7) ? 7 : 15;   // the last row of pos's chunk (not a u row)
    const int nu = (pos + 8) >> 3, nU = np;                      // u chunks (incl. v), U chunks
    const int nout = (nu > nU) ? nu : nU;
    float o[8][4];
    #pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
    __syncwarp();
    stage8(st, 0, ur, 0, pos, vr);                               // u rows t < 8 (row vr left for v) -> Y0
    stage8(st, YBYTES, ul, 0, m);                                // U rows g < 8 -> X
    if (nu > 1) stage8(st, CB, ur, 8, pos, vr);                  // second u chunk -> Y1, else a second U chunk -> Y1
    else if (nU > 1) stage8(st, CB, ul, 8, m);
    cp_commit();
    if (KDA_W == 16 || vr < 16)
    *(uint2*)(wsm + (vr >> 3) * CB + 7 * ROWB + (((lane >> 1) ^ 7) << 4) + (lane & 1) * 8) = vraw;   // v at row vr
    for (int i = 0; i < nout; ++i) {
        unsigned lo = zrow, hi = rowX;                           // pass sources: k 0..7 (u chunk i), k 8..15 (U chunk i)
        if (i < nu) lo = (KDA_W > 16 ? (i & 1) : i) ? rowY1 : rowY0;
        bool next = false;                                       // W > 16: u chunk i + 1 in flight (one group behind)
        if (i == 1 && nu <= 1) hi = rowY1;                       // U chunk 1 was staged with chunk 0
        else if (i > 0) {                                        // U chunk i into X once chunk i - 1 is consumed (exposed; rare)
            __syncwarp();
            stage8(st, YBYTES, ul, 8 * i, m);
            cp_commit();
            if constexpr (KDA_W > 16) {                          // u chunk i + 1 -> Y((i + 1) & 1), v at its row vr
                if (i + 1 < nu) {
                    const int bo = ((i + 1) & 1) * CB;
                    stage8(st, bo, ur, 8 * (i + 1), pos, vr);
                    cp_commit(); next = true;
                    if (vr >> 3 == i + 1) *(uint2*)(wsm + bo + 7 * ROWB + (((lane >> 1) ^ 7) << 4) + (lane & 1) * 8) = vraw;
                }
            }
        }
        const unsigned rt = (lt >> 1) ? hi : lo;
        unsigned bw[2];
        {
            const int t0 = 8 * i + 2 * c;
            const float2 kk0 = *(const float2*)(s_kk + t0), kq0 = *(const float2*)(s_kk + KDA_W + t0);
            float2 wu = make_float2(-beta * kk0.x, -beta * kk0.y);
            if (t0 + 1 == vr) wu.y = beta;                       // kk at t = vr is zero
            const float2 wo = make_float2(fmaf(kq_cur, wu.x, kq0.x), fmaf(kq_cur, wu.y, kq0.y));
            const float2 w = (i >= NC8) ? make_float2(0.f, 0.f) : (g < 2) ? wu : wo;   // passes i >= W / 8: U chunks only
            const float2 cc = *(const float2*)(s_c + t0);        // c_g pairs (columns 2, 3 only)
            unsigned h0, l0, h1, l1;
            split2(w.x, w.y, h0, l0); split2(cc.x, cc.y, h1, l1);
            bw[0] = (g >= 4) ? 0u : (g & 1) ? l0 : h0;
            bw[1] = (g == 2) ? h1 : (g == 3) ? l1 : 0u;
        }
        if constexpr (KDA_W > 16) { if (next) cp_wait<1>(); else cp_wait<0>(); }
        else cp_wait<0>();
        __syncwarp();
        #pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
            unsigned a[4]; ldsm4t(a, rt + tx[jn & 3] + (jn >> 2) * 128); mma16816_bf16(o[jn], a, bw[0], bw[1]);
        }
    }
    // epilogue: lane c = 0 holds u (columns 0 + 1), lane c = 1 holds out (2 + 3) of columns n = 16 jn + g, + 8:
    // transposed through the Y / X area, then bf16 row stores
    __syncwarp();
    float* sT = (float*)wsm + c * 136;                           // c = 0: u, c = 1: out (lanes c >= 2 write dead copies); pitch 136: no bank conflicts
    #pragma unroll
    for (int jn = 0; jn < 8; ++jn) { sT[16 * jn + g] = o[jn][0] + o[jn][1]; sT[16 * jn + g + 8] = o[jn][2] + o[jn][3]; }
    __syncwarp();
    const float4 un = *(const float4*)((float*)wsm + 4 * lane), on = *(const float4*)((float*)wsm + 136 + 4 * lane);
    st4_bf16(page_ring(UR, rp) + (size_t)(base + pos) * 128 + 4 * lane, un);
    st4_bf16(po, on);
}

