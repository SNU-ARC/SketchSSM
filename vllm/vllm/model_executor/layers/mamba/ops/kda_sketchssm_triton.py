# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Kimi Delta Attention (KDA) decode kernels (Triton).

The step reads each sketch head's sketch, and each dense head's BF16 state
rows; the flush folds the window into the state (sketch heads: the WY form;
dense heads: the recurrence), then rebuilds the sketch in a stats and a finish
launch and the dense heads' BF16 state rows.
"""

import functools

import torch

from vllm.model_executor.layers.mamba.ops.kda_sketchssm_common import (
    _FRAME_F,
    KDA_SKETCH_HEAD_DIM,
    KDA_SKETCH_SCRATCH_F,
    KDA_SKETCH_WINDOW,
    KDASketchArgs,
    KDASketchRings,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# Launch shapes of the step and of the flush / cold-build main and finish.
STEP_WARPS = 1
STEP_BG = 8
FLUSH_BK = 32
FLUSH_BV = 128
FLUSH_WARPS = 2
FLUSH_STAGES = 1
FLUSH_PROGRAMS_PER_SM = 32
FINISH_PROGRAMS_PER_SM = 32
# The blocked window fold (W > 16): the whole state per program.
FOLD_WARPS = 8
FOLD_PROGRAMS_PER_SM = 2


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@triton.jit
def _kda_sketch_step_kernel(
    q, k, v, gate, beta, s_q, s_k, s_v, s_g, s_b, A_log, bias, slots, meta,
    pos_ptr, state, s_st0, s_st1, u, phi, f, ranks, kr, vr, br, pr, ur, dr,
    s_kr, s_vr, s_br, s_pr, s_ur, s_dr, out, scale, G, H, dense, s_dn,
    dense_rows, D: tl.constexpr, W: tl.constexpr,
    BV: tl.constexpr, BG: tl.constexpr,
):  # fmt: skip
    LOG2E: tl.constexpr = 1.4426950408889634
    row = tl.program_id(0)
    head = tl.program_id(1)
    kk = tl.arange(0, D)
    ww = tl.arange(0, W)
    io = (row * H + head).to(tl.int64) * D
    # q / k / v / gate / beta may be row-strided views (s_*: token row strides).
    r64 = row.to(tl.int64)
    hd = head * D
    page = tl.load(slots + row).to(tl.int64)
    if page <= 0:
        tl.store(out + io + kk, tl.zeros([D], tl.float32).to(out.dtype.element_ty))
    else:
        slot = tl.load(meta + row).to(tl.int64)
        p = tl.load(pos_ptr + row)
        m = tl.load(ranks + head)
        k_raw = tl.load(k + r64 * s_k + hd + kk)
        v_raw = tl.load(v + r64 * s_v + hd + kk)
        q0 = tl.load(q + r64 * s_q + hd + kk).to(tl.float32)
        k0 = k_raw.to(tl.float32)
        g0 = tl.load(gate + r64 * s_g + hd + kk).to(tl.float32)
        b0 = tl.load(bias + head * D + kk)
        amp = tl.exp2(tl.load(A_log + head) * LOG2E)
        b_in = tl.load(beta + r64 * s_b + head).to(tl.float32)
        bet = 1.0 / (1.0 + tl.exp2(-b_in * LOG2E))
        qh = q0 * (tl.math.rsqrt(tl.sum(q0 * q0) + 1e-6) * scale)
        kh = k0 * tl.math.rsqrt(tl.sum(k0 * k0) + 1e-6)
        la = -5.0 / (1.0 + tl.exp2(-(amp * (g0 + b0)) * LOG2E))
        ring = head * W
        p_pre = pr + page * s_pr + ring * D
        prev = tl.load(p_pre + (p - 1) * D + kk, mask=(kk < D) & (p > 0), other=0.0)
        pre = prev + la
        tl.store(kr + page * s_kr + (ring + p) * D + kk, k_raw)
        tl.store(vr + page * s_vr + (ring + p) * D + kk, v_raw)
        tl.store(p_pre + p * D + kk, pre)
        tl.store(br + page * s_br + ring + p, bet)
        if m <= 0:
            if p < W - 1:
                _kda_dense_replay(
                    page, slot, head, p, pre, kh, qh, bet, v, r64 * s_v + hd,
                    kr + page * s_kr + ring * D, p_pre, ur, dr, page * s_ur,
                    page * s_dr, ring, dense, s_dn, dense_rows, out, io, D, W,
                    BV,
                )  # fmt: skip
        elif p < W - 1:
            # Sketch head.
            t2 = pre * LOG2E
            cd = tl.exp2(t2)
            p_d = dr + page * s_dr + ring * D
            p_u = ur + page * s_ur + ring * D
            bk = cd * kh
            bq = cd * qh
            kq_cur = tl.sum(kh * qh)
            tmask = ww < p
            if W == 16:
                dcur = kh * tl.exp2(-t2)
                tl.store(p_d + p * D + kk, dcur.to(dr.dtype.element_ty))
                dd = tl.load(
                    p_d + ww[:, None] * D + kk[None, :], mask=tmask[:, None], other=0.0
                ).to(tl.float32)
                kkv = tl.sum(dd * bk[None, :], axis=1)
                kqv = tl.sum(dd * bq[None, :], axis=1)
            else:
                # W > 16: d rows of 8-row chunk c >= 1 are rebased to
                # prefix[8 c - 1].
                ref = tl.load(
                    p_pre + ((p // 8) * 8 - 1) * D + kk, mask=(kk < D) & (p >= 8),
                    other=0.0,
                )  # fmt: skip
                dcur = kh * tl.exp2((ref - pre) * LOG2E)
                tl.store(p_d + p * D + kk, dcur.to(dr.dtype.element_ty))
                kkv = tl.zeros([W], tl.float32)
                kqv = tl.zeros([W], tl.float32)
                urep = tl.zeros([D], tl.float32)
                orep = tl.zeros([D], tl.float32)
                for t0 in range(0, p, 16):
                    rr = t0 + tl.arange(0, 16)
                    rm = rr < p
                    dd = tl.load(
                        p_d + rr[:, None] * D + kk[None, :], mask=rm[:, None], other=0.0
                    ).to(tl.float32)
                    rf = tl.load(
                        p_pre + ((rr // 8) * 8 - 1)[:, None] * D + kk[None, :],
                        mask=(rm & (rr >= 8))[:, None], other=0.0,
                    )  # fmt: skip
                    fac = tl.exp2((pre[None, :] - rf) * LOG2E)
                    kk_t = tl.sum(dd * fac * kh[None, :], axis=1)
                    kq_t = tl.sum(dd * fac * qh[None, :], axis=1)
                    hit = ww[None, :] == rr[:, None]
                    kkv += tl.sum(tl.where(hit, kk_t[:, None], 0.0), axis=0)
                    kqv += tl.sum(tl.where(hit, kq_t[:, None], 0.0), axis=0)
                    uu_t = tl.load(
                        p_u + rr[:, None] * D + kk[None, :], mask=rm[:, None], other=0.0
                    ).to(tl.float32)
                    wu_t = tl.where(rm, -bet * kk_t, 0.0)
                    urep += tl.sum(uu_t * wu_t[:, None], axis=0)
                    orep += tl.sum(uu_t * (kq_t + kq_cur * wu_t)[:, None], axis=0)
            sh = (slot * H + head) * G
            # c_g = (pq - f kq) - f_p kq_cur with f_p = beta (pk - f kk).
            acc = tl.zeros([D], tl.float32)
            for r0 in range(0, m, BG):
                gg = r0 + tl.arange(0, BG)
                gm = gg < m
                rows_g = (sh + gg)[:, None]
                ph = tl.load(
                    phi + rows_g * D + kk[None, :], mask=gm[:, None], other=0.0
                ).to(tl.float32)
                pk = tl.sum(ph * bk[None, :], axis=1)
                pq = tl.sum(ph * bq[None, :], axis=1)
                fr = tl.load(
                    f + rows_g * W + ww[None, :], mask=gm[:, None], other=0.0
                ).to(tl.float32)
                fcur = bet * (pk - tl.sum(fr * kkv[None, :], axis=1))
                tl.store(f + (sh + gg) * W + p, fcur.to(f.dtype.element_ty), mask=gm)
                cg = pq - tl.sum(fr * kqv[None, :], axis=1) - fcur * kq_cur
                cg = tl.where(gm, cg, 0.0)
                us = tl.load(
                    u + rows_g * D + kk[None, :], mask=gm[:, None], other=0.0
                ).to(tl.float32)
                acc += tl.sum(us * cg[:, None], axis=0)
            # Replay: u_p = sum_t w_u[t] u_t + beta v, w_u[t] = -beta kk_t;
            # out = sum_t kq_t u_t + kq_cur u_p + sum_g c_g U_g.
            vf = v_raw.to(tl.float32)
            if W == 16:
                uu = tl.load(
                    p_u + ww[:, None] * D + kk[None, :], mask=tmask[:, None], other=0.0
                ).to(tl.float32)
                wu = tl.where(tmask, -bet * kkv, 0.0)
                wo = kqv + kq_cur * wu
                urep = tl.sum(uu * wu[:, None], axis=0)
                orep = tl.sum(uu * wo[:, None], axis=0)
            unew = urep + bet * vf
            o = orep + (kq_cur * bet) * vf + acc
            tl.store(p_u + p * D + kk, unew.to(ur.dtype.element_ty))
            tl.store(out + io + kk, o.to(out.dtype.element_ty))


@triton.jit
def _kda_dense_replay(page, slot, head, p, pre, kh, qh, bet, v, v_off, p_k,
                      p_pre, ur, dr, ur_off, dr_off, ring, dense, s_dn,
                      dense_rows, out, io, D: tl.constexpr, W: tl.constexpr,
                      BV: tl.constexpr):  # fmt: skip
    # Dense head, non-flush step, from its BF16 state rows S^ and the window:
    # delta_p = beta (v - S^ (e k^) - sum_t delta_t kk_t),
    # out = S^ (e q^) + sum_t delta_t kq_t + delta_p kq_cur, e = exp(prefix_p),
    # kk_t / kq_t = <k^_t exp(prefix_p - prefix_t), k^ / q^>. FP32 delta_t
    # rows: u_ring bytes (t < W / 2), then d_ring bytes.
    LOG2E: tl.constexpr = 1.4426950408889634
    HALF: tl.constexpr = W // 2
    kk = tl.arange(0, D)
    ww = tl.arange(0, W)
    tmask = ww < p
    keys = tl.load(
        p_k + ww[:, None] * D + kk[None, :], mask=tmask[:, None], other=0.0
    ).to(tl.float32)
    keys *= tl.math.rsqrt(tl.sum(keys * keys, axis=1) + 1e-6)[:, None]
    pf = tl.load(p_pre + ww[:, None] * D + kk[None, :], mask=tmask[:, None], other=0.0)
    ell = tl.where(tmask[:, None], keys * tl.exp2((pre[None, :] - pf) * LOG2E), 0.0)
    kkv = tl.sum(ell * kh[None, :], axis=1)
    kqv = tl.sum(ell * qh[None, :], axis=1)
    kq_cur = tl.sum(kh * qh)
    ep = tl.exp2(pre * LOG2E)
    ek = ep * kh
    eq = ep * qh
    urf = (ur + ur_off + ring * D).to(tl.pointer_type(tl.float32))
    drf = (dr + dr_off + ring * D).to(tl.pointer_type(tl.float32))
    d_row = tl.where(ww < HALF, urf + ww * D, drf + (ww - HALF) * D)
    p_cur = tl.where(p < HALF, urf + p * D, drf + (p - HALF) * D)
    p_s = dense + slot * s_dn + tl.load(dense_rows + head).to(tl.int64) * D * D
    for v0 in tl.static_range(0, D, BV):
        vb = v0 + tl.arange(0, BV)
        s = tl.load(p_s + vb[:, None] * D + kk[None, :]).to(tl.float32)
        pk = tl.sum(s * ek[None, :], axis=1)
        pq = tl.sum(s * eq[None, :], axis=1)
        dl = tl.load(d_row[:, None] + vb[None, :], mask=tmask[:, None], other=0.0)
        rk = tl.sum(dl * kkv[:, None], axis=0)
        rq = tl.sum(dl * kqv[:, None], axis=0)
        vv = tl.load(v + v_off + vb).to(tl.float32)
        delta = bet * (vv - pk - rk)
        tl.store(p_cur + vb, delta)
        o = pq + rq + delta * kq_cur
        tl.store(out + io + vb, o.to(out.dtype.element_ty))


@triton.jit
def _dot(a, b, PREC: tl.constexpr):
    return tl.dot(a, b, input_precision=PREC)


@triton.jit
def _decayed_keys(p_k, p_pre, inv, k0, D: tl.constexpr, W: tl.constexpr,
                  BK: tl.constexpr):  # fmt: skip
    # Key chunk of a = k^ exp(prefix) and c = k^ exp(-prefix).
    LOG2E: tl.constexpr = 1.4426950408889634
    kb = k0 + tl.arange(0, BK)
    ww = tl.arange(0, W)
    kn = tl.load(p_k + ww[:, None] * D + kb[None, :]).to(tl.float32) * inv[:, None]
    t2 = tl.load(p_pre + ww[:, None] * D + kb[None, :]) * LOG2E
    return kn * tl.exp2(t2), kn * tl.exp2(-t2)


@triton.jit
def _kda_sketch_flush_kernel(
    rows, n_rows, slots, meta, heads_all, head_base, NH, NHall, state, s_st0,
    s_st1, frame, u, G, H, kr, vr, pr, br, s_kr, s_vr, s_pr, s_br, q, s_q, out,
    scale, scratch, D: tl.constexpr, W: tl.constexpr, MPAD: tl.constexpr,
    BV: tl.constexpr, GC: tl.constexpr, FLUSH: tl.constexpr,
    PREC: tl.constexpr, FRAME_F: tl.constexpr,
    SCRATCH_F: tl.constexpr, BK: tl.constexpr, STATS: tl.constexpr,
):  # fmt: skip
    # Items (row, head) of one rank bucket, walked until the first -1 row.
    # FLUSH: the window fold and output; STATS: the new sketch rows and the
    # finish's inputs in the scratch.
    LOG2E: tl.constexpr = 1.4426950408889634
    kk = tl.arange(0, D)
    ww = tl.arange(0, W)
    g16 = tl.arange(0, 16)
    total = n_rows * NH
    idx = tl.program_id(0)
    row = tl.load(rows + idx // NH, mask=idx < total, other=-1)
    while row >= 0:
        item = idx // NH
        hy = idx % NH
        info = tl.load(heads_all + head_base + hy)
        h = info & 65535
        m = info >> 16
        page = tl.load(slots + row).to(tl.int64)
        if page > 0:
            slot = tl.load(meta + row).to(tl.int64)
            sc = scratch + (item * NHall + head_base + hy).to(tl.int64) * SCRATCH_F
            p_s = state + page * s_st0 + h * s_st1
            sh = (slot * H + h) * G
            ring = h * W
            om = frame + h * FRAME_F  # Omega rows [g][k] (FP32)
            if FLUSH:
                # T'' = diag(beta) (I + L)^-T, L[i][j] = beta_i <a_i, c_j>.
                p_k = kr + page * s_kr + ring * D
                p_pre = pr + page * s_pr + ring * D
                keys = tl.load(p_k + ww[:, None] * D + kk[None, :]).to(tl.float32)
                inv = tl.math.rsqrt(tl.sum(keys * keys, axis=1) + 1e-6)
                lm = tl.zeros([W, W], tl.float32)
                for k0 in range(0, D, BK):
                    ka, kc = _decayed_keys(p_k, p_pre, inv, k0, D, W, BK)
                    lm += _dot(ka, tl.trans(kc), PREC)
                bet = tl.load(br + page * s_br + ring + ww)
                lm = tl.where(ww[None, :] < ww[:, None], bet[:, None] * lm, 0.0)
                tm = tl.zeros([W, W], tl.float32)
                for i in tl.static_range(W):
                    lrow = tl.sum(tl.where(ww[:, None] == i, lm, 0.0), axis=0)
                    new = tl.where(ww == i, 1.0, 0.0) - tl.sum(
                        lrow[:, None] * tm, axis=0
                    )
                    tm = tl.where(ww[:, None] == i, new[None, :], tm)
                tpp = tl.trans(tm) * bet[:, None]
                q0 = tl.load(q + row.to(tl.int64) * s_q + h * D + kk).to(tl.float32)
                qf = tl.math.rsqrt(tl.sum(q0 * q0) + 1e-6) * scale
                # X = (V - S A^T) T'', S' = (S + X C) diag(exp(prefix_15)).
                for v0 in range(0, D, BV):
                    vb = v0 + tl.arange(0, BV)
                    p_sv = p_s + vb[:, None] * D
                    pm = tl.zeros([BV, W], tl.float32)
                    for k0 in range(0, D, BK):
                        kb = k0 + tl.arange(0, BK)
                        ka, kc = _decayed_keys(p_k, p_pre, inv, k0, D, W, BK)
                        pm += _dot(tl.load(p_sv + kb[None, :]), tl.trans(ka), PREC)
                    vt = tl.load(
                        vr + page * s_vr + (ring + ww[None, :]) * D + vb[:, None]
                    ).to(tl.float32)
                    x = _dot(vt - pm, tpp, PREC)
                    o = tl.zeros([BV], tl.float32)
                    for k0 in range(0, D, BK):
                        kb = k0 + tl.arange(0, BK)
                        ka, kc = _decayed_keys(p_k, p_pre, inv, k0, D, W, BK)
                        pw = tl.exp2(tl.load(p_pre + (W - 1) * D + kb) * LOG2E)
                        sv = tl.load(p_sv + kb[None, :])
                        sv = (sv + _dot(x, kc, PREC)) * pw[None, :]
                        tl.store(p_sv + kb[None, :], sv)
                        qc = tl.load(q + row.to(tl.int64) * s_q + h * D + kb)
                        o += tl.sum(sv * (qc.to(tl.float32) * qf)[None, :], axis=1)
                    tl.store(
                        out + (row * H + h).to(tl.int64) * D + vb,
                        o.to(out.dtype.element_ty),
                    )
                tl.debug_barrier()
            if STATS:
                # U = S' Omega^T of ranks < 16 and (m <= 16) W = U4^T U.
                sumsq = tl.zeros([BV], tl.float32)
                wm = tl.zeros([16, 16], tl.float32)
                en16 = tl.zeros([16], tl.float32)
                for v0 in range(0, D, BV):
                    vb = v0 + tl.arange(0, BV)
                    u16 = tl.zeros([BV, 16], tl.float32)
                    for k0 in range(0, D, BK):
                        kb = k0 + tl.arange(0, BK)
                        sv = tl.load(p_s + vb[:, None] * D + kb[None, :])
                        sumsq += tl.sum(sv * sv, axis=1)
                        om16 = tl.load(
                            om + g16[None, :] * D + kb[:, None],
                            mask=g16[None, :] < m,
                            other=0.0,
                        )
                        u16 += _dot(sv, om16, PREC)
                    tl.store(
                        sc + g16[None, :] * D + vb[:, None], u16, mask=g16[None, :] < 4
                    )
                    if MPAD == 16:
                        u4 = tl.where(g16[None, :] < 4, u16, 0.0)
                        wm += _dot(tl.trans(u4), u16, PREC)
                    tl.store(
                        u + (sh + g16[None, :]) * D + vb[:, None],
                        u16.to(u.dtype.element_ty),
                        mask=g16[None, :] < m,
                    )
                    en16 += tl.sum(u16 * u16, axis=0)
                tl.store(sc + 1024 + g16, en16)
                tl.store(sc + 1152, tl.sum(sumsq))
                if MPAD == 16:
                    tl.store(
                        sc + 1280 + g16[:, None] * D + g16[None, :],
                        wm,
                        mask=g16[:, None] < 4,
                    )
                tl.debug_barrier()
                # Y4 = U4^T S' by key chunks.
                for k0 in range(0, D, BK):
                    kb = k0 + tl.arange(0, BK)
                    y4 = tl.zeros([16, BK], tl.float32)
                    for v0 in range(0, D, BV):
                        vb = v0 + tl.arange(0, BV)
                        u4t = tl.load(
                            sc + g16[:, None] * D + vb[None, :],
                            mask=g16[:, None] < 4,
                            other=0.0,
                        )
                        sv = tl.load(p_s + vb[:, None] * D + kb[None, :])
                        y4 += _dot(u4t, sv, PREC)
                    tl.store(
                        sc + 512 + g16[:, None] * D + kb[None, :],
                        y4,
                        mask=g16[:, None] < 4,
                    )
                if MPAD > 16:
                    # Ranks >= 16: U = S' Omega^T in rank chunks.
                    for g0 in range(16, m, GC):
                        gc = g0 + tl.arange(0, GC)
                        en = tl.zeros([GC], tl.float32)
                        for v0 in range(0, D, BV):
                            vb = v0 + tl.arange(0, BV)
                            uc = tl.zeros([BV, GC], tl.float32)
                            for k0 in range(0, D, BK):
                                kb = k0 + tl.arange(0, BK)
                                sv = tl.load(p_s + vb[:, None] * D + kb[None, :])
                                omc = tl.load(
                                    om + gc[None, :] * D + kb[:, None],
                                    mask=gc[None, :] < m,
                                    other=0.0,
                                )
                                uc += _dot(sv, omc, PREC)
                            tl.store(
                                u + (sh + gc[None, :]) * D + vb[:, None],
                                uc.to(u.dtype.element_ty),
                                mask=gc[None, :] < m,
                            )
                            en += tl.sum(uc * uc, axis=0)
                        tl.store(sc + 1024 + gc, en, mask=gc < D)
            tl.debug_barrier()
        idx += tl.num_programs(0)
        row = tl.load(rows + idx // NH, mask=idx < total, other=-1)


@triton.jit
def _kda_sketch_fold_blocked_kernel(
    rows, n_rows, slots, heads_all, head_base, NH, state, s_st0, s_st1, H, kr,
    vr, pr, br, s_kr, s_vr, s_pr, s_br, q, s_q, out, scale, D: tl.constexpr,
    W: tl.constexpr, PREC: tl.constexpr,
):  # fmt: skip
    # Blocked WY window fold for W > 16: the state stays in registers while
    # the rows are applied in 16-row blocks rebased to prefix[16 b - 1].
    LOG2E: tl.constexpr = 1.4426950408889634
    kk = tl.arange(0, D)
    i16 = tl.arange(0, 16)
    total = n_rows * NH
    idx = tl.program_id(0)
    row = tl.load(rows + idx // NH, mask=idx < total, other=-1)
    while row >= 0:
        info = tl.load(heads_all + head_base + idx % NH)
        h = info & 65535
        page = tl.load(slots + row).to(tl.int64)
        if page > 0:
            ring = h * W
            p_s = state + page * s_st0 + h * s_st1
            p_k = kr + page * s_kr + ring * D
            p_v = vr + page * s_vr + ring * D
            p_pre = pr + page * s_pr + ring * D
            p_b = br + page * s_br + ring
            sv = tl.load(p_s + kk[:, None] * D + kk[None, :])
            for b in range(0, W // 16):
                r = b * 16 + i16
                keys = tl.load(p_k + r[:, None] * D + kk[None, :]).to(tl.float32)
                kn = keys * tl.math.rsqrt(tl.sum(keys * keys, axis=1) + 1e-6)[:, None]
                ref = tl.load(
                    p_pre + (b * 16 - 1) * D + kk, mask=(kk < D) & (b > 0), other=0.0
                )
                t2 = (
                    tl.load(p_pre + r[:, None] * D + kk[None, :]) - ref[None, :]
                ) * LOG2E
                ka = kn * tl.exp2(t2)
                kc = kn * tl.exp2(-t2)
                bet = tl.load(p_b + r)
                lm = _dot(ka, tl.trans(kc), PREC)
                lm = tl.where(i16[None, :] < i16[:, None], bet[:, None] * lm, 0.0)
                tm = tl.zeros([16, 16], tl.float32)
                for i in tl.static_range(16):
                    lrow = tl.sum(tl.where(i16[:, None] == i, lm, 0.0), axis=0)
                    new = tl.where(i16 == i, 1.0, 0.0) - tl.sum(
                        lrow[:, None] * tm, axis=0
                    )
                    tm = tl.where(i16[:, None] == i, new[None, :], tm)
                tpp = tl.trans(tm) * bet[:, None]
                pm = _dot(sv, tl.trans(ka), PREC)
                vt = tl.load(p_v + r[None, :] * D + kk[:, None]).to(tl.float32)
                x = _dot(vt - pm, tpp, PREC)
                pend = tl.load(p_pre + (b * 16 + 15) * D + kk)
                pw = tl.exp2((pend - ref) * LOG2E)
                sv = (sv + _dot(x, kc, PREC)) * pw[None, :]
            tl.store(p_s + kk[:, None] * D + kk[None, :], sv)
            q0 = tl.load(q + row.to(tl.int64) * s_q + h * D + kk).to(tl.float32)
            qf = tl.math.rsqrt(tl.sum(q0 * q0) + 1e-6) * scale
            o = tl.sum(sv * (q0 * qf)[None, :], axis=1)
            tl.store(
                out + (row * H + h).to(tl.int64) * D + kk, o.to(out.dtype.element_ty)
            )
        idx += tl.num_programs(0)
        row = tl.load(rows + idx // NH, mask=idx < total, other=-1)


@triton.jit
def _orth(w, cw, vi, ci):
    d = tl.sum(vi * w)
    return w - vi * d, cw - ci * d


@triton.jit
def _normalize(w, cw, u_j, j: tl.constexpr, npiv):
    e2 = tl.sum(w * w)
    # The first pivot is kept unless zero, the others above 1e-12 |u_j|^2.
    keep = e2 > tl.where(j == 0, 0.0, 1e-12 * tl.sum(u_j * u_j))
    keep = keep & (j < npiv)
    inv = tl.where(keep, tl.math.rsqrt(tl.where(keep, e2, 1.0)), 0.0)
    return w * inv, cw * inv


@triton.jit
def _kda_sketch_finish_kernel(
    rows, n_rows, slots, meta, heads_all, NHall, frame, phi, f, G, H, scratch,
    D: tl.constexpr, W: tl.constexpr, GCH: tl.constexpr, FRAME_F: tl.constexpr,
    SCRATCH_F: tl.constexpr,
):  # fmt: skip
    # Items (row, sketch head) over every bucket, walked until the first -1
    # row: pivots, the ridge system and phi; the erase history f restarts.
    kk = tl.arange(0, D)
    ww = tl.arange(0, W)
    i4 = tl.arange(0, 4)
    gl = tl.arange(0, GCH)
    total = n_rows * NHall
    idx = tl.program_id(0)
    row = tl.load(rows + idx // NHall, mask=idx < total, other=-1)
    while row >= 0:
        if tl.load(slots + row) > 0:
            info = tl.load(heads_all + idx % NHall)
            h = info & 65535
            m = info >> 16
            npiv = tl.minimum(m, 4)
            slot = tl.load(meta + row).to(tl.int64)
            sh = (slot * H + h) * G
            sc = scratch + idx.to(tl.int64) * SCRATCH_F
            om = frame + h * FRAME_F
            for r0 in range(0, G, 16):
                rr = r0 + tl.arange(0, 16)
                tl.store(
                    f + (sh + rr[:, None]) * W + ww[None, :],
                    tl.zeros([16, W], tl.float32).to(f.dtype.element_ty),
                    mask=rr[:, None] < G,
                )
            safe = tl.load(sc + 1152) / D
            safe = tl.where(safe > 0.0, safe, 1.0)
            inv_s = tl.math.rsqrt(safe)
            u0 = tl.load(sc + kk)
            u1 = tl.load(sc + D + kk)
            u2 = tl.load(sc + 2 * D + kk)
            u3 = tl.load(sc + 3 * D + kk)
            v0, c0 = _normalize(u0, tl.where(i4 == 0, 1.0, 0.0), u0, 0, npiv)
            w, cw = u1, tl.where(i4 == 1, 1.0, 0.0)
            for _ in tl.static_range(2):
                w, cw = _orth(w, cw, v0, c0)
            v1, c1 = _normalize(w, cw, u1, 1, npiv)
            w, cw = u2, tl.where(i4 == 2, 1.0, 0.0)
            for _ in tl.static_range(2):
                w, cw = _orth(w, cw, v0, c0)
                w, cw = _orth(w, cw, v1, c1)
            v2, c2 = _normalize(w, cw, u2, 2, npiv)
            w, cw = u3, tl.where(i4 == 3, 1.0, 0.0)
            for _ in tl.static_range(2):
                w, cw = _orth(w, cw, v0, c0)
                w, cw = _orth(w, cw, v1, c1)
                w, cw = _orth(w, cw, v2, c2)
            v3, c3 = _normalize(w, cw, u3, 3, npiv)
            # zr_j = sum_i C_j[i] Y4_i / sqrt(safe).
            y4 = tl.load(sc + 512 + i4[:, None] * D + kk[None, :])
            zr0 = tl.sum(c0[:, None] * y4, axis=0) * inv_s
            zr1 = tl.sum(c1[:, None] * y4, axis=0) * inv_s
            zr2 = tl.sum(c2[:, None] * y4, axis=0) * inv_s
            zr3 = tl.sum(c3[:, None] * y4, axis=0) * inv_s
            tl.debug_barrier()
            # Per rank g: z_j[g], the residual and the Cholesky sums of
            # I + Z^T diag(1 / (res + 0.1)) Z.
            s0 = tl.zeros([GCH], tl.float32)
            s1 = tl.zeros([GCH], tl.float32)
            s2 = tl.zeros([GCH], tl.float32)
            s3 = tl.zeros([GCH], tl.float32)
            s4 = tl.zeros([GCH], tl.float32)
            s5 = tl.zeros([GCH], tl.float32)
            s6 = tl.zeros([GCH], tl.float32)
            s7 = tl.zeros([GCH], tl.float32)
            s8 = tl.zeros([GCH], tl.float32)
            s9 = tl.zeros([GCH], tl.float32)
            for g0 in range(0, m, GCH):
                gg = g0 + gl
                gm = gg < m
                if m <= 16:
                    wg = tl.load(
                        sc + 1280 + i4[:, None] * D + gg[None, :],
                        mask=gm[None, :],
                        other=0.0,
                    )
                    z0 = tl.sum(c0[:, None] * wg, axis=0) * inv_s
                    z1 = tl.sum(c1[:, None] * wg, axis=0) * inv_s
                    z2 = tl.sum(c2[:, None] * wg, axis=0) * inv_s
                    z3 = tl.sum(c3[:, None] * wg, axis=0) * inv_s
                else:
                    omc = tl.load(om + gg[:, None] * D + kk[None, :], mask=gm[:, None])
                    z0 = tl.sum(omc * zr0[None, :], axis=1)
                    z1 = tl.sum(omc * zr1[None, :], axis=1)
                    z2 = tl.sum(omc * zr2[None, :], axis=1)
                    z3 = tl.sum(omc * zr3[None, :], axis=1)
                z0 = tl.where(gm, z0, 0.0)
                z1 = tl.where(gm, z1, 0.0)
                z2 = tl.where(gm, z2, 0.0)
                z3 = tl.where(gm, z3, 0.0)
                en = tl.load(sc + 1024 + gg, mask=gm, other=0.0)
                res = en / safe - (z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
                res = tl.where(gm & (gg >= npiv), tl.maximum(res, 0.0), 0.0)
                inv_den = 1.0 / (res + 0.1)
                b0 = z0 * inv_den
                b1 = z1 * inv_den
                b2 = z2 * inv_den
                b3 = z3 * inv_den
                s0 += z0 * b0
                s1 += z1 * b0
                s2 += z1 * b1
                s3 += z2 * b0
                s4 += z2 * b1
                s5 += z2 * b2
                s6 += z3 * b0
                s7 += z3 * b1
                s8 += z3 * b2
                s9 += z3 * b3
                tl.store(sc + gg, z0, mask=gm)
                tl.store(sc + D + gg, z1, mask=gm)
                tl.store(sc + 2 * D + gg, z2, mask=gm)
                tl.store(sc + 3 * D + gg, z3, mask=gm)
                tl.store(sc + 1024 + gg, res, mask=gm)
            l00 = tl.sqrt_rn(1.0 + tl.sum(s0))
            r00 = 1.0 / l00
            l10 = tl.sum(s1) * r00
            l11 = tl.sqrt_rn(1.0 + tl.sum(s2) - l10 * l10)
            r11 = 1.0 / l11
            l20 = tl.sum(s3) * r00
            l21 = (tl.sum(s4) - l20 * l10) * r11
            l22 = tl.sqrt_rn(1.0 + tl.sum(s5) - l20 * l20 - l21 * l21)
            r22 = 1.0 / l22
            l30 = tl.sum(s6) * r00
            l31 = (tl.sum(s7) - l30 * l10) * r11
            l32 = (tl.sum(s8) - l30 * l20 - l31 * l21) * r22
            l33 = tl.sqrt_rn(1.0 + tl.sum(s9) - l30 * l30 - l31 * l31 - l32 * l32)
            r33 = 1.0 / l33
            tl.debug_barrier()
            # g_j[g]: the Cholesky solve of the ridge system.
            n0 = zr0
            n1 = zr1
            n2 = zr2
            n3 = zr3
            for g0 in range(0, m, GCH):
                gg = g0 + gl
                gm = gg < m
                z0 = tl.load(sc + gg, mask=gm, other=0.0)
                z1 = tl.load(sc + D + gg, mask=gm, other=0.0)
                z2 = tl.load(sc + 2 * D + gg, mask=gm, other=0.0)
                z3 = tl.load(sc + 3 * D + gg, mask=gm, other=0.0)
                res = tl.load(sc + 1024 + gg, mask=gm, other=0.0)
                inv_den = 1.0 / (res + 0.1)
                y0 = z0 * inv_den * r00
                y1 = (z1 * inv_den - l10 * y0) * r11
                y2 = (z2 * inv_den - l20 * y0 - l21 * y1) * r22
                y3 = (z3 * inv_den - l30 * y0 - l31 * y1 - l32 * y2) * r33
                gm3 = y3 * r33
                gm2 = (y2 - l32 * gm3) * r22
                gm1 = (y1 - l21 * gm2 - l31 * gm3) * r11
                gm0 = (y0 - l10 * gm1 - l20 * gm2 - l30 * gm3) * r00
                fm1 = tl.where(gm, 0.1 * inv_den - 1.0, 0.0)
                omc = tl.load(
                    om + gg[:, None] * D + kk[None, :], mask=gm[:, None], other=0.0
                )
                n0 += tl.sum(omc * (z0 * fm1)[:, None], axis=0)
                n1 += tl.sum(omc * (z1 * fm1)[:, None], axis=0)
                n2 += tl.sum(omc * (z2 * fm1)[:, None], axis=0)
                n3 += tl.sum(omc * (z3 * fm1)[:, None], axis=0)
                tl.store(sc + 512 + gg, gm0, mask=gm)
                tl.store(sc + 512 + D + gg, gm1, mask=gm)
                tl.store(sc + 512 + 2 * D + gg, gm2, mask=gm)
                tl.store(sc + 512 + 3 * D + gg, gm3, mask=gm)
                tl.store(sc + 1024 + gg, res * inv_den, mask=gm)
            tl.debug_barrier()
            # phi_g = a_g Omega_g + sum_j g_j[g] native_j.
            for g0 in range(0, m, GCH):
                gg = g0 + gl
                gm = gg < m
                omc = tl.load(
                    om + gg[:, None] * D + kk[None, :], mask=gm[:, None], other=0.0
                )
                ph = omc * tl.load(sc + 1024 + gg, mask=gm, other=0.0)[:, None]
                ph += tl.load(sc + 512 + gg, mask=gm, other=0.0)[:, None] * n0[None, :]
                ph += tl.load(sc + 640 + gg, mask=gm, other=0.0)[:, None] * n1[None, :]
                ph += tl.load(sc + 768 + gg, mask=gm, other=0.0)[:, None] * n2[None, :]
                ph += tl.load(sc + 896 + gg, mask=gm, other=0.0)[:, None] * n3[None, :]
                tl.store(
                    phi + (sh + gg[:, None]) * D + kk[None, :],
                    ph.to(phi.dtype.element_ty),
                    mask=gm[:, None],
                )
        tl.debug_barrier()
        idx += tl.num_programs(0)
        row = tl.load(rows + idx // NHall, mask=idx < total, other=-1)


@triton.jit
def _kda_dense_flush_kernel(
    rows, slots, meta, heads, state, s_st0, s_st1, dense, s_dn, kr, vr, pr, br,
    s_kr, s_vr, s_pr, s_br, q, s_q, out, scale, H, D: tl.constexpr,
    W: tl.constexpr, BV: tl.constexpr, FOLD: tl.constexpr,
):  # fmt: skip
    # Dense heads: FOLD applies the window's raw rows to the FP32 state (the
    # exact recurrence) and writes the output; then the BF16 state rows.
    item = tl.program_id(0)
    j = tl.program_id(1)
    block = tl.program_id(2)
    row = tl.load(rows + item)
    if row >= 0:
        page = tl.load(slots + row).to(tl.int64)
        if page > 0:
            h = tl.load(heads + j)
            slot = tl.load(meta + row).to(tl.int64)
            kk = tl.arange(0, D)
            vb = block * BV + tl.arange(0, BV)
            sp = state + page * s_st0 + h * s_st1 + vb[:, None] * D + kk[None, :]
            s = tl.load(sp)
            if FOLD:
                ring = h * W
                p_k = kr + page * s_kr + ring * D
                p_v = vr + page * s_vr + ring * D
                p_pre = pr + page * s_pr + ring * D
                p_b = br + page * s_br + ring
                prev = tl.zeros([D], tl.float32)
                for t in range(W):
                    k = tl.load(p_k + t * D + kk).to(tl.float32)
                    k *= tl.math.rsqrt(tl.sum(k * k) + 1e-6)
                    prefix = tl.load(p_pre + t * D + kk)
                    s *= tl.exp(prefix - prev)[None, :]
                    prev = prefix
                    v = tl.load(p_v + t * D + vb).to(tl.float32)
                    delta = tl.load(p_b + t) * (v - tl.sum(s * k[None, :], axis=1))
                    s += delta[:, None] * k[None, :]
                tl.store(sp, s)
                q0 = tl.load(q + row.to(tl.int64) * s_q + h * D + kk).to(tl.float32)
                qh = q0 * (tl.math.rsqrt(tl.sum(q0 * q0) + 1e-6) * scale)
                o = tl.sum(s * qh[None, :], axis=1)
                tl.store(
                    out + (row * H + h).to(tl.int64) * D + vb,
                    o.to(out.dtype.element_ty),
                )
            tl.store(
                dense + slot * s_dn + j * D * D + vb[:, None] * D + kk[None, :],
                s.to(dense.dtype.element_ty),
            )


def _check_scratch(scratch: torch.Tensor, rows: int, sketch: KDASketchArgs) -> None:
    need = rows * sketch.tables.num_sketch_heads * KDA_SKETCH_SCRATCH_F
    assert scratch.dtype == torch.float32 and scratch.numel() >= need


def _check_rows(slots: torch.Tensor, meta: torch.Tensor, null_block_id: int):
    # Slots <= 0 are padding.
    assert null_block_id == 0
    assert slots.is_contiguous() and slots.dtype == torch.int32
    assert meta.is_contiguous() and meta.dtype == torch.int32


def _check_storage(state: torch.Tensor, rings: KDASketchRings, sketch):
    d, w = KDA_SKETCH_HEAD_DIM, sketch.tables.window
    assert state.dtype == torch.float32 and state.shape[2:] == (d, d)
    assert state.stride(2) == d and state.stride(3) == 1
    for ring in rings.tensors():
        assert ring.shape[2] == w and ring[0].is_contiguous()
    for x in (sketch.u, sketch.phi, sketch.f, sketch.dense):
        assert x.is_contiguous() and x.dtype == torch.bfloat16
    assert sketch.f.shape[-1] == w


def _flush_launches(
    state: torch.Tensor,
    rings: KDASketchRings,
    rows: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: KDASketchArgs,
    scratch: torch.Tensor,
    q: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    flush: bool,
) -> None:
    t = sketch.tables
    assert rows.is_contiguous() and rows.dtype == torch.int32
    _check_storage(state, rings, sketch)
    r = rings
    n = rows.numel()
    if t.num_dense_heads and n:
        _kda_dense_flush_kernel[(n, t.num_dense_heads, KDA_SKETCH_HEAD_DIM // 32)](
            rows, slots, meta, t.dense_heads_d, state, state.stride(0),
            state.stride(1), sketch.dense, sketch.dense.stride(0), r.k, r.v,
            r.prefix, r.beta, r.k.stride(0), r.v.stride(0), r.prefix.stride(0),
            r.beta.stride(0), q, q.stride(0) if flush else 0, out, scale,
            t.num_heads, D=KDA_SKETCH_HEAD_DIM, W=t.window, BV=32, FOLD=flush,
            num_warps=4,
        )  # fmt: skip
    if not t.num_sketch_heads:
        return
    _check_scratch(scratch, rows.numel(), sketch)
    sms = _num_sms(state.device)
    rocm = current_platform.is_rocm()
    prec = None if rocm else "tf32x3"
    w = t.window
    # Fold and statistics are separate launches to reduce register pressure.
    blocked = flush and w > KDA_SKETCH_WINDOW
    phases = ((True, False), (False, True)) if flush else ((False, True),)
    for heads, mmax, base in t.flush_groups:
        nh = heads.numel()
        if blocked:
            _kda_sketch_fold_blocked_kernel[(min(n * nh, sms * FOLD_PROGRAMS_PER_SM),)](
                rows, n, slots, t.heads_all_d, base, nh, state, state.stride(0),
                state.stride(1), t.num_heads, r.k, r.v, r.prefix, r.beta,
                r.k.stride(0), r.v.stride(0), r.prefix.stride(0), r.beta.stride(0),
                q, q.stride(0), out, scale, D=KDA_SKETCH_HEAD_DIM, W=w, PREC=prec,
                num_warps=FOLD_WARPS,
            )  # fmt: skip
        for fold, stats in phases:
            if blocked and fold:
                continue
            _kda_sketch_flush_kernel[(min(n * nh, sms * FLUSH_PROGRAMS_PER_SM),)](
                rows, n, slots, meta, t.heads_all_d, base, nh, t.num_sketch_heads,
                state, state.stride(0), state.stride(1), t.frame_gk, sketch.u,
                t.rank_cap, t.num_heads, r.k, r.v, r.prefix, r.beta,
                r.k.stride(0), r.v.stride(0), r.prefix.stride(0), r.beta.stride(0),
                q, q.stride(0) if flush else 0, out, scale, scratch,
                D=KDA_SKETCH_HEAD_DIM, W=w,
                MPAD=mmax, BV=FLUSH_BV, GC=32, FLUSH=fold, PREC=prec,
                FRAME_F=_FRAME_F,
                SCRATCH_F=KDA_SKETCH_SCRATCH_F, BK=FLUSH_BK, STATS=stats,
                num_warps=FLUSH_WARPS, num_stages=FLUSH_STAGES,
            )  # fmt: skip
    nh = t.num_sketch_heads
    _kda_sketch_finish_kernel[(min(n * nh, sms * FINISH_PROGRAMS_PER_SM),)](
        rows, n, slots, meta, t.heads_all_d, nh, t.frame_gk, sketch.phi, sketch.f,
        t.rank_cap, t.num_heads, scratch, D=KDA_SKETCH_HEAD_DIM,
        W=w, GCH=16, FRAME_F=_FRAME_F,
        SCRATCH_F=KDA_SKETCH_SCRATCH_F, num_warps=1,
    )  # fmt: skip


def kda_sketch_triton_cold_build(
    state: torch.Tensor,
    rings: KDASketchRings,
    slots: torch.Tensor,
    meta: torch.Tensor,
    rows: torch.Tensor,
    sketch: KDASketchArgs,
    scratch: torch.Tensor,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Build the sketch of each listed row from its state."""
    if rows.numel() == 0:
        return
    _check_rows(slots, meta, null_block_id)
    _flush_launches(
        state, rings, rows, slots, meta, sketch, scratch, state, state, 1.0, False
    )


def kda_sketch_triton_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    rings: KDASketchRings,
    slots: torch.Tensor,
    meta: torch.Tensor,
    pos: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: KDASketchArgs,
    scratch: torch.Tensor,
    scale: float = KDA_SKETCH_HEAD_DIM**-0.5,
    null_block_id: int = NULL_BLOCK_ID,
    has_flush_rows: bool = True,
) -> None:
    """One SketchSSM decode step of a KDA layer."""
    batch = q.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    h = t.num_heads
    if slots.dim() == 2:
        slots = slots[:, 0]
    _check_rows(slots, meta, null_block_id)
    _check_storage(state, rings, sketch)
    assert pos.is_contiguous() and pos.dtype == torch.int32
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    # Row-strided views are read in place: dense rows of unit inner stride.
    for x in (q, k, v, g):
        assert x.dtype == torch.bfloat16 and x.shape[1:] == (h, KDA_SKETCH_HEAD_DIM)
        assert x.stride(2) == 1 and x.stride(1) == KDA_SKETCH_HEAD_DIM
    assert beta.dtype == torch.bfloat16 and beta.shape[1:] == (h,)
    assert beta.stride(1) == 1
    a_log = A_log.float().contiguous()
    bias = dt_bias.float().contiguous()
    r = rings
    _kda_sketch_step_kernel[(batch, h)](
        q, k, v, g, beta, q.stride(0), k.stride(0), v.stride(0), g.stride(0),
        beta.stride(0), a_log, bias, slots, meta, pos, state, state.stride(0),
        state.stride(1), sketch.u, sketch.phi, sketch.f, t.ranks, r.k, r.v,
        r.beta, r.prefix, r.u_ring, r.d_ring, r.k.stride(0), r.v.stride(0),
        r.beta.stride(0), r.prefix.stride(0), r.u_ring.stride(0),
        r.d_ring.stride(0), out, scale, t.rank_cap, h, sketch.dense,
        sketch.dense.stride(0), t.dense_rows, D=KDA_SKETCH_HEAD_DIM, W=t.window,
        BV=16, BG=STEP_BG, num_warps=STEP_WARPS,
    )  # fmt: skip
    if has_flush_rows:
        _flush_launches(
            state, rings, flush_rows, slots, meta, sketch, scratch, q, out, scale,
            True,
        )  # fmt: skip
