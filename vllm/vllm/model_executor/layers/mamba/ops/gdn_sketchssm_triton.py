# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Gated DeltaNet decode kernels (Triton).

q and k arrive unrotated. The key ring holds the BF16 keys as they are; ring
dots are rotation invariant, and the sketch and state reads take the FP32
rotated q and k. For the flush, the steps write the residual of the BF16 d ring
and the unit keys in the rotated frame; it folds the window exactly from them,
as the reference recurrence does in FP32 (see the CUDA flush).
"""

import torch

from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_common import (
    GDN_SKETCH_HEAD_DIM,
    GDN_SKETCH_PIVOTS,
    GDNSketchArgs,
    gdn_sketch_build,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# Flush rows per program of the flush and rebuild launches.
TRITON_ROWS_PER_PROGRAM = 4
# Unit keys go to the flush as fp16 hi + lo of KSC k (|k_i| <= 1).
KSC = 1024.0


@triton.jit
def _gdn_sketch_step_kernel(
    qkv, qkr, a_act, b_act, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
    slots, write_pos, meta, u, phi, fs, beta_ring, ranks, layout, scale, s_qkv,
    s_qkr, s_a, s_b, s_st_slot, s_st_head, s_st_v, s_d_slot, s_d_head, s_d_pos,
    s_k_slot, s_k_head, s_k_pos, s_g_slot, s_g_head, s_u, s_phi, s_fs, s_beta,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, W: tl.constexpr, WP: tl.constexpr, P: tl.constexpr,
    BG: tl.constexpr, BK: tl.constexpr, KSCALE: tl.constexpr,
):  # fmt: skip
    # One (row, value head). Window rows are padded to WP (a power of two).
    # qkr: the q and k the state and sketch meet (FP32 rotated, or qkv itself
    # without a frame); k at offset H * K in both layouts.
    n = tl.program_id(0)
    hv = tl.program_id(1)
    i_h = hv // (HV // H)
    kk = tl.arange(0, K)
    vv = tl.arange(0, V)
    ww = tl.arange(0, WP)
    pp = tl.arange(0, P)
    slot = tl.load(slots + n).to(tl.int64)
    p_o = out + (n * HV + hv) * V + vv
    if slot <= 0:
        tl.store(p_o, tl.zeros([V], tl.float32).to(p_o.dtype.element_ty))
    else:
        wp = tl.load(write_pos + n)
        cidx = tl.load(meta + n).to(tl.int64)
        m = tl.load(ranks + hv)
        p_q = qkv + n * s_qkv + i_h * K
        p_k = p_q + H * K
        p_qr = qkr + n * s_qkr + i_h * K
        p_kr = p_qr + H * K
        q = tl.load(p_q + kk).to(tl.float32)
        k = tl.load(p_k + kk).to(tl.float32)
        v = tl.load(qkv + n * s_qkv + 2 * H * K + hv * V + vv).to(tl.float32)
        a_val = tl.load(a_act + n * s_a + hv).to(tl.float32)
        b_val = tl.load(b_act + n * s_b + hv).to(tl.float32)
        xg = a_val + tl.load(dt_bias + hv).to(tl.float32)
        sp = tl.where(xg <= 20.0, tl.log(1.0 + tl.exp(xg)), xg)
        g_val = -tl.exp(tl.load(A_log + hv).to(tl.float32)) * sp
        alpha = tl.exp(g_val)
        beta = 1.0 / (1.0 + tl.exp(-b_val))
        tl.store(beta_ring + cidx * s_beta + hv * W + wp, beta)
        q_sc = 1.0 / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        k_rn = 1.0 / tl.sqrt(tl.sum(k * k) + 1e-6)
        qn = q * q_sc
        kn = k * k_rn
        cur_kq = tl.sum(qn * kn)

        # Ring replay weights over the window rows s < wp (unrotated unit keys).
        valid = ww < wp
        k_ring = k_cache + slot * s_k_slot + i_h * s_k_head
        keys = tl.load(
            k_ring + ww[:, None] * s_k_pos + kk[None, :], mask=valid[:, None], other=0.0
        ).to(tl.float32)
        keys = keys / tl.sqrt(tl.sum(keys * keys, axis=1) + 1e-6)[:, None]
        kq_s = tl.sum(keys * qn[None, :], axis=1)
        kk_s = tl.sum(keys * kn[None, :], axis=1)
        g_ring = g_cache + slot * s_g_slot + hv * s_g_head
        gs = tl.load(g_ring + ww, mask=valid, other=0.0)
        pre = tl.cumsum(gs, axis=0)
        gtot = tl.sum(gs)
        rep = tl.where(valid, tl.exp(gtot - pre), 0.0)
        tot = tl.exp(gtot)
        d_ring = d_cache + slot * s_d_slot + hv * s_d_head
        ds = tl.load(
            d_ring + ww[:, None] * s_d_pos + vv[None, :], mask=valid[:, None], other=0.0
        ).to(tl.float32)
        s_q = tl.sum(ds * (kq_s * rep)[:, None], axis=0)
        s_k = tl.sum(ds * (kk_s * rep)[:, None], axis=0)

        hq = tl.zeros([V], tl.float32)
        hk = tl.zeros([V], tl.float32)
        u_off = tl.load(layout + hv * 4)
        if (m == 0) & (u_off >= 0):
            # Dense head: its BF16 rows of the window-start state.
            p_u = u + cidx * s_u + u_off
            for k0 in range(0, K, BK):
                kc = k0 + tl.arange(0, BK)
                s = tl.load(p_u + kc[:, None] * V + vv[None, :]).to(tl.float32)
                qc = tl.load(p_qr + kc).to(tl.float32) * q_sc
                kcv = tl.load(p_kr + kc).to(tl.float32) * k_rn
                hq += tl.sum(s * qc[:, None], axis=0)
                hk += tl.sum(s * kcv[:, None], axis=0)
        elif m == 0:
            p_s = state + slot * s_st_slot + hv * s_st_head + vv[:, None] * s_st_v
            for k0 in range(0, K, BK):
                kc = k0 + tl.arange(0, BK)
                s = tl.load(p_s + kc[None, :])
                qc = tl.load(p_qr + kc).to(tl.float32) * q_sc
                kcv = tl.load(p_kr + kc).to(tl.float32) * k_rn
                hq += tl.sum(s * qc[None, :], axis=1)
                hk += tl.sum(s * kcv[None, :], axis=1)
        else:
            phi_off = tl.load(layout + hv * 4 + 1)
            fs_off = tl.load(layout + hv * 4 + 2)
            fg = tl.load(layout + hv * 4 + 3)
            p_phi = phi + cidx * s_phi + phi_off
            p_fs = fs + cidx * s_fs + fs_off
            p_u = u + cidx * s_u + u_off
            qrn = tl.load(p_qr + kk).to(tl.float32) * q_sc
            krn = tl.load(p_kr + kk).to(tl.float32) * k_rn
            # Pivot rows (or the merged maps for m <= 4) dotted with q and k.
            piv = tl.load(
                p_phi + pp[:, None] * K + kk[None, :],
                mask=(pp[:, None] < tl.minimum(m, P)) & (m < K),
                other=0.0,
            ).to(tl.float32)
            tq = tl.sum(piv * qrn[None, :], axis=1)
            tk = tl.sum(piv * krn[None, :], axis=1)
            mid = (m > P) & (m < K)
            for g0 in range(0, m, BG):
                g = g0 + tl.arange(0, BG)
                gm = g < m
                qg = tl.load(p_qr + g, mask=gm, other=0.0).to(tl.float32) * q_sc
                kg = tl.load(p_kr + g, mask=gm, other=0.0).to(tl.float32) * k_rn
                sel = g[:, None] == pp[None, :]
                xq = tl.sum(tl.where(sel, tq[None, :], 0.0), axis=1)
                xk = tl.sum(tl.where(sel, tk[None, :], 0.0), axis=1)
                ad = tl.load(p_phi + P * K + g, mask=gm & mid, other=0.0)
                gains = tl.load(
                    p_phi + P * K + (pp[:, None] + 1) * fg + g[None, :],
                    mask=gm[None, :] & mid,
                    other=0.0,
                ).to(tl.float32)
                xq_mid = ad.to(tl.float32) * qg + tl.sum(gains * tq[:, None], axis=0)
                xk_mid = ad.to(tl.float32) * kg + tl.sum(gains * tk[:, None], axis=0)
                xq = tl.where(m <= P, xq, tl.where(m < K, xq_mid, qg))
                xk = tl.where(m <= P, xk, tl.where(m < K, xk_mid, kg))
                f = tl.load(
                    p_fs + ww[:, None] * fg + g[None, :],
                    mask=valid[:, None] & gm[None, :],
                    other=0.0,
                ).to(tl.float32)
                eq = tl.sum(f * kq_s[:, None], axis=0)
                ek = tl.sum(f * kk_s[:, None], axis=0)
                fcur = beta * (xk - ek)
                x = xq - eq - fcur * cur_kq
                tl.store(p_fs + wp * fg + g, fcur.to(fs.dtype.element_ty), mask=gm)
                us = tl.load(
                    p_u + g[:, None] * V + vv[None, :], mask=gm[:, None], other=0.0
                ).to(tl.float32)
                hq += tl.sum(us * x[:, None], axis=0)

        stq = alpha * (hq * tot + s_q)
        stk = alpha * (hk * tot + s_k)
        dc = tl.where(m > 0, beta * (v - alpha * s_k), beta * (v - stk))
        tl.store(p_o, (stq + dc * cur_kq).to(p_o.dtype.element_ty))
        tl.store(g_ring + wp, g_val)
        # The d ring row (BF16) and, for the flush, its residual in units of the
        # bf16 ulp (fp16) and the unit key in the rotated frame (fp16 hi + lo).
        hi = dc.to(d_cache.dtype.element_ty)
        tl.store(d_ring + wp * s_d_pos + vv, hi)
        hf = hi.to(tl.float32)
        ex = (hf.to(tl.int32, bitcast=True) >> 23) & 255
        inv_ulp = ((261 - ex) << 23).to(tl.float32, bitcast=True)
        lo = tl.where((ex >= 8) & (ex <= 253), (dc - hf) * inv_ulp, 0.0)
        # fp16 rows are kept bit for bit in the BF16 ring storage
        f16 = d_cache.dtype.element_ty
        tl.store(d_ring + (W + wp) * s_d_pos + vv, lo.to(tl.float16).to(f16, bitcast=True))
        if hv % (HV // H) == 0:
            if wp < W - 1:
                tl.store(k_ring + wp * s_k_pos + kk, k.to(k_cache.dtype.element_ty))
            ku = tl.load(p_kr + kk).to(tl.float32) * (k_rn * KSCALE)
            ku_hi = ku.to(tl.float16)
            ku_lo = (ku - ku_hi.to(tl.float32)).to(tl.float16)
            tl.store(k_ring + (W + wp) * s_k_pos + kk, ku_hi.to(f16, bitcast=True))
            tl.store(k_ring + (2 * W + wp) * s_k_pos + kk, ku_lo.to(f16, bitcast=True))


@triton.jit
def _gdn_sketch_flush_kernel(
    qkr, out, state, d_cache, k_cache, g_cache, slots, meta, flush_rows, batch,
    beta_ring, ranks, layout, scale, s_qkr, s_st_slot, s_st_head, s_st_v,
    s_d_slot, s_d_head, s_d_pos, s_k_slot, s_k_head, s_k_pos, s_g_slot, s_g_head,
    s_beta,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    W: tl.constexpr, WP: tl.constexpr, BV: tl.constexpr, KSCALE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):  # fmt: skip
    # Exact flush of one value head of the rows in flush_rows (-1 padded). The
    # steps stored d^ = H + R (BF16 ring + residual) computed with the exact
    # unit keys K, the BF16 ring rows and the window-start state as far as they
    # read it exactly; with M = (I + Gamma)^-1 and P_c = (S - S_read) K^T
    # (S for sketch heads, S - S~ for dense heads with BF16 rows, none without),
    # D = H + (R - P_c diag(beta exp(pre))) M and S' = tot S + D diag(rep) K.
    # Window rows are padded to WP with zero keys, updates and gates.
    hv = tl.program_id(1)
    i_h = hv // (HV // H)
    kk = tl.arange(0, K)
    ww = tl.arange(0, WP)
    inw = ww < W
    it = tl.program_id(0)
    row = tl.load(flush_rows + it, mask=it < batch, other=-1)
    while row >= 0:
        slot = tl.load(slots + row).to(tl.int64)
        if slot > 0:
            cidx = tl.load(meta + row).to(tl.int64)
            m = tl.load(ranks + hv)
            rows_ = (m == 0) & (tl.load(layout + hv * 4) >= 0)
            p_ku = (
                k_cache + slot * s_k_slot + i_h * s_k_head
                + (W + ww[:, None]) * s_k_pos + kk[None, :]
            )  # fmt: skip
            k_hi = tl.load(p_ku, mask=inw[:, None], other=0.0)
            k_lo = tl.load(p_ku + W * s_k_pos, mask=inw[:, None], other=0.0)
            k_hi = k_hi.to(tl.float16, bitcast=True).to(tl.float32)
            k_lo = k_lo.to(tl.float16, bitcast=True).to(tl.float32)
            keys = (k_hi + k_lo) * (1.0 / KSCALE)
            gs = tl.load(
                g_cache + slot * s_g_slot + hv * s_g_head + ww, mask=inw, other=0.0
            )
            pre = tl.cumsum(gs, axis=0)
            gt = tl.sum(gs)
            rep = tl.where(inw, tl.exp(gt - pre), 0.0)
            tot = tl.exp(gt)
            q = tl.load(qkr + row * s_qkr + i_h * K + kk).to(tl.float32)
            qn = q * (1.0 / tl.sqrt(tl.sum(q * q) + 1e-6) * scale)
            bs = tl.load(beta_ring + cidx * s_beta + hv * W + ww, mask=inw, other=0.0)
            gram = tl.dot(keys, tl.trans(keys), input_precision=DOT_PRECISION)
            upper = ww[:, None] < ww[None, :]
            gamma = gram * bs[None, :] * tl.exp(pre[None, :] - pre[:, None])
            gamma = tl.where(upper, gamma, 0.0)
            inv = tl.zeros([WP, WP], tl.float32)
            if W == 16:
                for i in tl.static_range(W):
                    r = W - 1 - i
                    grow = tl.sum(tl.where(ww[:, None] == r, gamma, 0.0), axis=0)
                    new = tl.where(ww == r, 1.0, 0.0)
                    new -= tl.sum(grow[:, None] * inv, axis=0)
                    inv = tl.where(ww[:, None] == r, new[None, :], inv)
            else:
                for i in range(W):
                    r = W - 1 - i
                    grow = tl.sum(tl.where(ww[:, None] == r, gamma, 0.0), axis=0)
                    new = tl.where(ww == r, 1.0, 0.0)
                    new -= tl.sum(grow[:, None] * inv, axis=0)
                    inv = tl.where(ww[:, None] == r, new[None, :], inv)
            mr = tl.where(ww[:, None] <= ww[None, :], inv * rep[None, :], 0.0)
            zp = bs * tl.exp(pre)
            p_s = state + slot * s_st_slot + hv * s_st_head
            d_ring = d_cache + slot * s_d_slot + hv * s_d_head
            for v0 in range(0, V, BV):
                vb = v0 + tl.arange(0, BV)
                ptr = p_s + vb[:, None] * s_st_v + kk[None, :]
                s = tl.load(ptr)
                hi = tl.load(
                    d_ring + ww[None, :] * s_d_pos + vb[:, None], mask=inw[None, :],
                    other=0.0,
                ).to(tl.float32)  # fmt: skip
                lo = tl.load(
                    d_ring + (W + ww[None, :]) * s_d_pos + vb[:, None],
                    mask=inw[None, :], other=0.0,
                )  # fmt: skip
                lo = lo.to(tl.float16, bitcast=True).to(tl.float32)
                ex = (hi.to(tl.int32, bitcast=True) >> 23) & 255
                ulp = ((ex - 7) << 23).to(tl.float32, bitcast=True)
                z = tl.where(ex >= 8, lo * ulp, 0.0)
                if (m > 0) | rows_:
                    sc = tl.where(m > 0, s, s - s.to(tl.bfloat16).to(tl.float32))
                    proj = tl.dot(sc, tl.trans(keys), input_precision=DOT_PRECISION)
                    z -= proj * zp[None, :]
                x = hi * rep[None, :] + tl.dot(z, mr, input_precision=DOT_PRECISION)
                s = s * tot + tl.dot(x, keys, input_precision=DOT_PRECISION)
                tl.store(ptr, s)
                o = tl.sum(s * qn[None, :], axis=1)
                tl.store(out + (row * HV + hv) * V + vb, o.to(out.dtype.element_ty))
        it += tl.num_programs(0)
        row = tl.load(flush_rows + it, mask=it < batch, other=-1)


def gdn_sketch_triton_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    write_pos: torch.Tensor,
    meta: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: GDNSketchArgs,
    scale: float,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
    *,
    qk: torch.Tensor | None = None,
) -> None:
    """One SketchSSM decode step of a GDN layer: q/k of ``mixed_qkv``
    unrotated, ``qk`` their FP32 rotation ``(batch, 2, H, K)`` (None without
    a frame)."""
    batch = mixed_qkv.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    h, hv = k_cache.shape[1], state.shape[1]
    k = v = GDN_SKETCH_HEAD_DIM
    w = t.window
    wp = triton.next_power_of_2(w)
    assert state.shape[2:] == (v, k) and state.stride(3) == 1
    # d ring: W BF16 rows, then W fp16 residual rows; k ring: W raw BF16 keys,
    # then W fp16 rows each of the unit keys' hi and lo (flush-only rows).
    assert d_cache.shape[2:] == (2 * w, v) and d_cache.stride(3) == 1
    assert k_cache.shape[2:] == (3 * w, k) and k_cache.stride(3) == 1
    assert g_cache.shape[2] == w and g_cache.stride(2) == 1
    assert hv % h == 0 and mixed_qkv.stride(1) == 1
    if slots.dim() == 2:
        slots = slots[:, 0]
    # Slots <= 0 are padding.
    assert null_block_id == 0
    assert out.is_contiguous() and out.dtype == mixed_qkv.dtype
    s = sketch
    assert s.beta[0].is_contiguous()
    qkr = mixed_qkv if qk is None else qk.view(batch, 2 * h * k)
    strides = (
        *state.stride()[:3], *d_cache.stride()[:3], *k_cache.stride()[:3],
        *g_cache.stride()[:2], s.u.stride(0),
    )  # fmt: skip
    _gdn_sketch_step_kernel[(batch, hv)](
        mixed_qkv, qkr, a, b, A_log, dt_bias, out, state, d_cache, k_cache,
        g_cache, slots, write_pos, meta, s.u, s.phi, s.fs, s.beta, t.ranks,
        t.layout, scale, mixed_qkv.stride(0), qkr.stride(0), a.stride(0),
        b.stride(0), *strides, s.phi.stride(0), s.fs.stride(0), s.beta.stride(0),
        H=h, HV=hv, K=k,
        V=v, W=w, WP=wp, P=GDN_SKETCH_PIVOTS, BG=4, BK=8, KSCALE=KSC,
        num_warps=min(8, wp // 16),
    )  # fmt: skip
    if not has_flush_rows:
        return
    programs = max(1, triton.cdiv(batch, TRITON_ROWS_PER_PROGRAM))
    rocm = current_platform.is_rocm()
    # Above 32 rows, FMA dots on 16-row value blocks need less shared memory.
    fma = wp > 32
    _gdn_sketch_flush_kernel[(programs, hv)](
        qkr, out, state, d_cache, k_cache, g_cache, slots, meta, flush_rows, batch,
        s.beta, t.ranks, t.layout, scale, qkr.stride(0), *state.stride()[:3],
        *d_cache.stride()[:3], *k_cache.stride()[:3], *g_cache.stride()[:2],
        s.beta.stride(0), H=h, HV=hv, K=k,
        V=v, W=w, WP=wp, BV=16 if fma else 32, KSCALE=KSC,
        DOT_PRECISION=None if rocm else "ieee" if fma else "tf32x3",
        num_warps=8 if fma else 4,
    )  # fmt: skip
    # Rebuild the flushed rows' sketches from the new state.
    gdn_sketch_build(
        state, flush_rows, slots, meta, sketch, null_block_id, rows=flush_rows,
        rows_per_program=TRITON_ROWS_PER_PROGRAM,
    )  # fmt: skip
