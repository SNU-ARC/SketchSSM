# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Gated DeltaNet decode kernels (Triton)."""

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


@triton.jit
def _gdn_sketch_step_kernel(
    qkv, a_act, b_act, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
    slots, write_pos, meta, u, phi, fs, beta_ring, current_d, current_k, ranks,
    layout, scale, s_qkv, s_a, s_b, s_st_slot, s_st_head, s_st_v, s_d_slot,
    s_d_head, s_d_pos, s_k_slot, s_k_head, s_k_pos, s_g_slot, s_g_head, s_u,
    s_phi, s_fs, s_beta, s_cd, s_ck, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
    P: tl.constexpr, BG: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    # One (row, value head). Window rows are padded to WP (a power of two).
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

        # Ring replay weights over the window rows s < wp.
        valid = ww < wp
        k_ring = k_cache + slot * s_k_slot + i_h * s_k_head
        keys = tl.load(
            k_ring + ww[:, None] * s_k_pos + kk[None, :], mask=valid[:, None], other=0.0
        ).to(tl.float32)
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
                qc = tl.load(p_q + kc).to(tl.float32) * q_sc
                kcv = tl.load(p_k + kc).to(tl.float32) * k_rn
                hq += tl.sum(s * qc[:, None], axis=0)
                hk += tl.sum(s * kcv[:, None], axis=0)
        elif m == 0:
            p_s = state + slot * s_st_slot + hv * s_st_head + vv[:, None] * s_st_v
            for k0 in range(0, K, BK):
                kc = k0 + tl.arange(0, BK)
                s = tl.load(p_s + kc[None, :])
                qc = tl.load(p_q + kc).to(tl.float32) * q_sc
                kcv = tl.load(p_k + kc).to(tl.float32) * k_rn
                hq += tl.sum(s * qc[None, :], axis=1)
                hk += tl.sum(s * kcv[None, :], axis=1)
        else:
            phi_off = tl.load(layout + hv * 4 + 1)
            fs_off = tl.load(layout + hv * 4 + 2)
            fg = tl.load(layout + hv * 4 + 3)
            p_phi = phi + cidx * s_phi + phi_off
            p_fs = fs + cidx * s_fs + fs_off
            p_u = u + cidx * s_u + u_off
            # Pivot rows (or the merged maps for m <= 4) dotted with q and k.
            piv = tl.load(
                p_phi + pp[:, None] * K + kk[None, :],
                mask=(pp[:, None] < tl.minimum(m, P)) & (m < K),
                other=0.0,
            ).to(tl.float32)
            tq = tl.sum(piv * qn[None, :], axis=1)
            tk = tl.sum(piv * kn[None, :], axis=1)
            mid = (m > P) & (m < K)
            for g0 in range(0, m, BG):
                g = g0 + tl.arange(0, BG)
                gm = g < m
                qg = tl.load(p_q + g, mask=gm, other=0.0).to(tl.float32) * q_sc
                kg = tl.load(p_k + g, mask=gm, other=0.0).to(tl.float32) * k_rn
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
        if wp == W - 1:
            tl.store(current_d + cidx * s_cd + hv * V + vv, dc)
        else:
            tl.store(d_ring + wp * s_d_pos + vv, dc.to(d_cache.dtype.element_ty))
        if hv % (HV // H) == 0:
            if wp == W - 1:
                tl.store(current_k + cidx * s_ck + i_h * K + kk, kn)
            else:
                tl.store(k_ring + wp * s_k_pos + kk, kn.to(k_cache.dtype.element_ty))


@triton.jit
def _gdn_sketch_flush_kernel(
    qkv, out, state, d_cache, k_cache, g_cache, slots, meta, flush_rows, batch,
    beta_ring, current_d, current_k, ranks, layout, scale, s_qkv, s_st_slot,
    s_st_head, s_st_v, s_d_slot, s_d_head, s_d_pos, s_k_slot, s_k_head,
    s_k_pos, s_g_slot, s_g_head, s_beta, s_cd, s_ck, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
    WP: tl.constexpr, BV: tl.constexpr, PROJ_PRECISION: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):  # fmt: skip
    # Flush of one value head of the rows in flush_rows (-1 padded). The
    # stored updates lack the window-start state's erase, which the WY form
    # restores: S' = tot S + X K with X = D rep - (S K^T) T. Dense heads
    # with BF16 rows stored full updates computed from those rows (S~ = S
    # rounded to BF16 by the build), so only the rounding residual is
    # erased: X = D rep - ((S - S~) K^T) T.
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
            u_off = tl.load(layout + hv * 4)
            rows = (m == 0) & (u_off >= 0)
            last = ww == W - 1
            ring = inw & ~last
            keys = tl.load(
                k_cache + slot * s_k_slot + i_h * s_k_head + ww[:, None] * s_k_pos
                + kk[None, :],
                mask=ring[:, None],
                other=0.0,
            ).to(tl.float32)  # fmt: skip
            k_cur = tl.load(current_k + cidx * s_ck + i_h * K + kk)
            k_cur = k_cur.to(tl.bfloat16).to(tl.float32)
            keys = tl.where(last[:, None], k_cur[None, :], keys)
            gs = tl.load(
                g_cache + slot * s_g_slot + hv * s_g_head + ww, mask=inw, other=0.0
            )
            pre = tl.cumsum(gs, axis=0)
            gt = tl.sum(gs)
            rep = tl.exp(gt - pre)
            tot = tl.exp(gt)
            q = tl.load(qkv + row * s_qkv + i_h * K + kk).to(tl.float32)
            qn = q * (1.0 / tl.sqrt(tl.sum(q * q) + 1e-6) * scale)
            tm = tl.zeros([WP, WP], tl.float32)
            if (m > 0) | rows:
                bs = tl.load(
                    beta_ring + cidx * s_beta + hv * W + ww, mask=inw, other=0.0
                )
                gram = tl.dot(keys, tl.trans(keys), input_precision=PROJ_PRECISION)
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
                tm = inv * bs[:, None] * tl.exp(pre[:, None] + gt - pre[None, :])
                tm = tl.where(ww[:, None] <= ww[None, :], tm, 0.0)
            p_s = state + slot * s_st_slot + hv * s_st_head
            d_ring = d_cache + slot * s_d_slot + hv * s_d_head
            for v0 in range(0, V, BV):
                vb = v0 + tl.arange(0, BV)
                ptr = p_s + vb[:, None] * s_st_v + kk[None, :]
                s = tl.load(ptr)
                d = tl.load(
                    d_ring + ww[None, :] * s_d_pos + vb[:, None],
                    mask=ring[None, :],
                    other=0.0,
                ).to(tl.float32)
                d_cur = tl.load(current_d + cidx * s_cd + hv * V + vb)
                x = tl.where(last[None, :], d_cur[:, None], d) * rep[None, :]
                if (m > 0) | rows:
                    # The rows are S rounded to BF16: S - S~ is exact.
                    res = s - s.to(tl.bfloat16).to(tl.float32)
                    res = tl.where(rows, res, s)
                    proj = tl.dot(res, tl.trans(keys), input_precision=PROJ_PRECISION)
                    x -= tl.dot(proj, tm, input_precision=DOT_PRECISION)
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
) -> None:
    """One SketchSSM decode step of a GDN layer."""
    batch = mixed_qkv.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    h, hv = k_cache.shape[1], state.shape[1]
    k = v = GDN_SKETCH_HEAD_DIM
    w = t.window
    wp = triton.next_power_of_2(w)
    assert state.shape[2:] == (v, k) and state.stride(3) == 1
    assert d_cache.shape[2:] == (w, v) and d_cache.stride(3) == 1
    assert k_cache.shape[2:] == (w, k) and k_cache.stride(3) == 1
    assert g_cache.shape[2] == w and g_cache.stride(2) == 1
    assert hv % h == 0 and mixed_qkv.stride(1) == 1
    if slots.dim() == 2:
        slots = slots[:, 0]
    # Slots <= 0 are padding.
    assert null_block_id == 0
    assert out.is_contiguous() and out.dtype == mixed_qkv.dtype
    s = sketch
    for x in (s.beta, s.current_d, s.current_k):
        assert x[0].is_contiguous()
    strides = (
        *state.stride()[:3], *d_cache.stride()[:3], *k_cache.stride()[:3],
        *g_cache.stride()[:2], s.u.stride(0),
    )  # fmt: skip
    _gdn_sketch_step_kernel[(batch, hv)](
        mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
        slots, write_pos, meta, s.u, s.phi, s.fs, s.beta, s.current_d,
        s.current_k, t.ranks, t.layout, scale, mixed_qkv.stride(0), a.stride(0),
        b.stride(0), *strides, s.phi.stride(0), s.fs.stride(0), s.beta.stride(0),
        s.current_d.stride(0), s.current_k.stride(0), H=h, HV=hv, K=k, V=v, W=w,
        WP=wp, P=GDN_SKETCH_PIVOTS, BG=4, BK=8, num_warps=min(8, wp // 16),
    )  # fmt: skip
    if not has_flush_rows:
        return
    programs = max(1, triton.cdiv(batch, TRITON_ROWS_PER_PROGRAM))
    rocm = current_platform.is_rocm()
    # Above 32 rows, FMA dots on 16-row value blocks need less shared memory.
    fma = wp > 32
    _gdn_sketch_flush_kernel[(programs, hv)](
        mixed_qkv, out, state, d_cache, k_cache, g_cache, slots, meta, flush_rows,
        batch, s.beta, s.current_d, s.current_k, t.ranks, t.layout, scale,
        mixed_qkv.stride(0), *strides[:-1], s.beta.stride(0), s.current_d.stride(0),
        s.current_k.stride(0), H=h, HV=hv, K=k, V=v, W=w, WP=wp, BV=16 if fma else 32,
        PROJ_PRECISION=None if rocm else "ieee" if fma else "tf32",
        DOT_PRECISION=None if rocm else "ieee" if fma else "tf32x3",
        num_warps=8 if fma else 4,
    )  # fmt: skip
    # Rebuild the flushed rows' sketches from the new state.
    gdn_sketch_build(
        state, flush_rows, slots, meta, sketch, null_block_id, rows=flush_rows,
        rows_per_program=TRITON_ROWS_PER_PROGRAM,
    )  # fmt: skip
