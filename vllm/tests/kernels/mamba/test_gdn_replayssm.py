# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN ReplaySSM: the SketchSSM decode kernels with dense heads (rank 0, no
rotation) against the exact recurrence and the research ReplaySSM kernel."""

import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.gdn.gdn_sketchssm import (
    GDNOfficialReplaySSM,
    GDNSketchSSM,
)
from vllm.model_executor.layers.mamba.ops import gdn_sketchssm_common as common
from vllm.model_executor.layers.mamba.ops import sketchssm_kernels as skk
from vllm.model_executor.layers.mamba.ops.gdn_sketchssm_triton import (
    gdn_sketch_triton_decode,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="GPU")
K = V = 128
W = 16
# Two value heads per key head, and the Qwen3.8 Flash-Next layer (16, 48).
SHAPES = [(2, 4), (16, 48)]


def _cuda_ok(h: int, hv: int) -> bool:
    return skk.gdn_cuda_supported(h, hv, K, V, W, torch.bfloat16, torch.float32)


def _backends(h: int, hv: int):
    no_cuda = pytest.mark.skipif(not _cuda_ok(h, hv), reason="sketchssm package")
    return [pytest.param("cuda", h, hv, marks=no_cuda), ("triton", h, hv)]


DECODE = {"cuda": skk.gdn_cuda_decode, "triton": gdn_sketch_triton_decode}


# The paper's GDN ReplaySSM baseline kernel (fused_recurrent_replayssm.py),
# vendored as the reference.
@triton.jit
def _research_replayssm_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, h0, d_cache, k_cache, g_cache, slots,
    write_pos, scale, s_qkv, s_a, s_b, s_h, s_d, s_k, s_g, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    BC: tl.constexpr, NK: tl.constexpr, BKT: tl.constexpr, L: tl.constexpr,
):  # fmt: skip
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    slot = tl.load(slots + i_n).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if slot <= 0:
        tl.store(p_o, tl.zeros([BV], tl.float32).to(p_o.dtype.element_ty))
        return
    wp = tl.load(write_pos + i_n).to(tl.int64)
    is_flush = wp == L - 1
    valid = o_c < wp
    x = tl.load(a + i_n * s_a + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv)
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * sp
    alpha = tl.exp(g_val)
    beta = tl.sigmoid(tl.load(b + i_n * s_b + i_hv).to(tl.float32))
    beta = beta.to(b.dtype.element_ty).to(tl.float32)
    gs = tl.load(g_cache + slot * s_g + i_hv * L + o_c, mask=valid, other=0.0)
    pre = tl.cumsum(gs, axis=0)
    gtot = tl.sum(gs, axis=0)
    rep = tl.where(valid, tl.exp(gtot - pre), 0.0)
    tot = tl.exp(gtot)
    p_d = d_cache + slot * s_d + (i_hv * L + o_c[None, :]) * V + o_v[:, None]
    ds = tl.load(p_d, mask=valid[None, :], other=0).to(tl.float32)
    ds = (ds * rep[None, :]).to(p_o.dtype.element_ty)
    p_mix = mixed_qkv + i_n * s_qkv
    v = tl.load(p_mix + 2 * H * K + i_hv * V + o_v).to(tl.float32)
    o_kf = tl.arange(0, K)
    qf = tl.load(p_mix + i_h * K + o_kf).to(tl.float32)
    kf = tl.load(p_mix + H * K + i_h * K + o_kf).to(tl.float32)
    q_rn = 1.0 / tl.sqrt(tl.sum(qf * qf) + 1e-6)
    k_rn = 1.0 / tl.sqrt(tl.sum(kf * kf) + 1e-6)
    sq = tl.zeros([BV], tl.float32)
    sk = tl.zeros([BV], tl.float32)
    kq = tl.zeros([1], tl.float32)
    write_k = (not is_flush) and (i_v == 0) and (i_hv == i_h * (HV // H))
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        q_c = tl.load(p_mix + i_h * K + o_kt).to(tl.float32) * q_rn * scale
        k_c = tl.load(p_mix + H * K + i_h * K + o_kt).to(tl.float32) * k_rn
        kq += tl.sum(k_c * q_c)
        h_c = tl.load(h0 + slot * s_h + i_hv * V * K + o_v[:, None] * K + o_kt[None, :])
        p_k = k_cache + slot * s_k + (i_h * L + o_c[:, None]) * K + o_kt[None, :]
        k_all = tl.load(p_k, mask=valid[:, None], other=0).to(p_o.dtype.element_ty)
        h_c = h_c * tot + tl.dot(ds, k_all).to(tl.float32)
        sq += tl.sum(h_c * q_c[None, :], axis=1)
        sk += tl.sum(h_c * k_c[None, :], axis=1)
        if write_k:
            p_ck = k_cache + slot * s_k + (i_h * L + wp) * K + o_kt
            tl.store(p_ck, k_c.to(p_o.dtype.element_ty))
    sq *= alpha
    sk *= alpha
    d_cur = beta * (v - sk)
    tl.store(p_o, (sq + d_cur * tl.sum(kq)).to(p_o.dtype.element_ty))
    if is_flush:
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            k_c = tl.load(p_mix + H * K + i_h * K + o_kt).to(tl.float32) * k_rn
            p_h = h0 + slot * s_h + i_hv * V * K + o_v[:, None] * K + o_kt[None, :]
            p_k = k_cache + slot * s_k + (i_h * L + o_c[:, None]) * K + o_kt[None, :]
            k_all = tl.load(p_k, mask=valid[:, None], other=0).to(p_o.dtype.element_ty)
            h_c = tl.load(p_h) * tot + tl.dot(ds, k_all).to(tl.float32)
            tl.store(p_h, alpha * h_c + d_cur[:, None] * k_c[None, :])
    else:
        p_cd = d_cache + slot * s_d + (i_hv * L + wp) * V + o_v
        tl.store(p_cd, d_cur.to(p_cd.dtype.element_ty))
        if i_v == 0:
            tl.store(g_cache + slot * s_g + i_hv * L + wp, g_val)


def research_decode(mix, a, b, A_log, dt_bias, out, state, dr, kr, gr, slots, pos):
    batch, hv = mix.shape[0], state.shape[1]
    h = kr.shape[1]
    nk = 4
    _research_replayssm_kernel[(V // 64, batch, hv)](
        mix, a, b, A_log, dt_bias, out, state, dr, kr, gr, slots, pos, K**-0.5,
        mix.stride(0), a.stride(0), b.stride(0), state.stride(0), dr.stride(0),
        kr.stride(0), gr.stride(0), H=h, HV=hv, K=K, V=V, BV=64, BC=W, NK=nk,
        BKT=K // nk, L=W, num_warps=4, num_stages=2,
    )  # fmt: skip


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


class Errors:
    """Max absolute and relative (Frobenius, per row) errors."""

    def __init__(self):
        self.abs = self.rel = 0.0

    def add(self, x: torch.Tensor, ref: torch.Tensor) -> None:
        self.abs = max(self.abs, float((x.double() - ref.double()).abs().max()))
        self.rel = max(self.rel, _rel(x.double(), ref.double()))

    def __repr__(self) -> str:
        return f"max abs {self.abs:.3e}, max rel {self.rel:.3e}"


@pytest.mark.parametrize("backend, h, hv", [p for s in SHAPES for p in _backends(*s)])
def test_gdn_replayssm_decode(backend, h, hv):
    """Requests at different window offsets, padding rows and shuffled rows
    over 40 steps (two or three flushes per request), against the exact FP64
    recurrence and the research ReplaySSM kernel."""
    torch.manual_seed(0)
    dev = "cuda"
    nx = 8
    # (slot, sketch row, first step)
    reqs = [(2, 4, 0), (6, 0, 5), (4, 1, 11)]
    steps, rows_per_step = 40, 4
    tables = common.GDNSketchTables(
        torch.zeros(hv, dtype=torch.int32), h, W, dense_rows=False
    ).to(dev)
    sketch = common.GDNSketchArgs.allocate(tables, 5, dev)
    state = torch.randn(nx, hv, V, K, device=dev) * 0.12
    rings = [
        torch.zeros(nx, hv, W, V, device=dev, dtype=torch.bfloat16),
        torch.zeros(nx, h, W, K, device=dev, dtype=torch.bfloat16),
        torch.zeros(nx, hv, W, device=dev),
    ]
    state_r, rings_r = state.clone(), [x.clone() for x in rings]
    exact = {slot: state[slot].double().cpu() for slot, _, _ in reqs}
    A_log = torch.randn(hv, device=dev) * 0.3
    dt_bias = torch.randn(hv, device=dev) * 0.3
    names = "out out_research research_out state state_research research_state"
    err = {k: Errors() for k in names.split()}
    gen = torch.Generator().manual_seed(0)
    for n in range(steps):
        rows = [r for r in reqs if r[2] <= n]
        rows += [(0, 0, 0)] * (rows_per_step - len(rows))
        rows = [rows[i] for i in torch.randperm(rows_per_step, generator=gen)]
        pos = [(n - start) % W if slot else 0 for slot, _, start in rows]
        mix = torch.randn(rows_per_step, (2 * h + hv) * K, device=dev) * 0.25
        a = torch.randn(rows_per_step, hv, device=dev) * 0.5 - 2
        b = torch.randn(rows_per_step, hv, device=dev)
        mix, a, b = (x.bfloat16() for x in (mix, a, b))
        slots = torch.tensor([r[0] for r in rows], dtype=torch.int32, device=dev)
        meta = torch.tensor([r[1] for r in rows], dtype=torch.int32, device=dev)
        wpos = torch.tensor(pos, dtype=torch.int32, device=dev)
        flush = [i for i, (r, p) in enumerate(zip(rows, pos)) if r[0] and p == W - 1]
        flush_rows = torch.tensor(
            flush + [-1] * (rows_per_step - len(flush)), dtype=torch.int32, device=dev
        )
        out = torch.empty(rows_per_step, hv, V, dtype=torch.bfloat16, device=dev)
        out_r = torch.empty_like(out)
        DECODE[backend](
            mix, a, b, A_log, dt_bias, out, state, *rings, slots, wpos, meta,
            flush_rows, sketch, K**-0.5, has_flush_rows=bool(flush),
        )  # fmt: skip
        research_decode(
            mix, a, b, A_log, dt_bias, out_r, state_r, *rings_r, slots, wpos
        )

        # The exact recurrence in FP64 on the BF16 inputs.
        x = mix.double().cpu()
        q = _unit(x[:, : h * K].view(-1, h, K)).repeat_interleave(hv // h, 1)
        k = _unit(x[:, h * K : 2 * h * K].view(-1, h, K))
        k = k.repeat_interleave(hv // h, 1)
        v = x[:, 2 * h * K :].view(-1, hv, V)
        g = -A_log.double().cpu().exp() * F.softplus(a.double().cpu() + dt_bias.cpu())
        beta = b.float().sigmoid().bfloat16().double().cpu()
        for i, (slot, _, _) in enumerate(rows):
            if not slot:
                assert not out[i].any() and not out_r[i].any()
                continue
            s = exact[slot] * g[i].exp()[:, None, None]
            d = beta[i][:, None] * (v[i] - torch.einsum("hvk,hk->hv", s, k[i]))
            s = s + d[:, :, None] * k[i][:, None, :]
            exact[slot] = s
            ref = torch.einsum("hvk,hk->hv", s, q[i]) * K**-0.5
            err["out"].add(out[i].cpu(), ref)
            err["out_research"].add(out[i].cpu(), out_r[i].float().cpu())
            err["research_out"].add(out_r[i].cpu(), ref)
            if pos[i] == W - 1:
                err["state"].add(state[slot].cpu(), s)
                err["state_research"].add(state[slot].cpu(), state_r[slot].cpu())
                err["research_state"].add(state_r[slot].cpu(), s)
    print(
        f"\n{backend} H={h} HV={hv}:",
        *(f"{k}: {e}" for k, e in err.items()),
        sep="\n  ",
    )
    # BF16 outputs and BF16 ring rows (d, k), as in the research kernel.
    assert max(err[k].rel for k in ("out", "out_research")) < 6e-3
    assert max(err[k].rel for k in ("state", "state_research")) < 4e-3


def test_gdn_replayssm_module(monkeypatch):
    """``--use-replayssm`` with VLLM_GDN_REPLAYSSM_KERNEL=sketchssm makes every
    head dense and leaves q/k unrotated; the default is the authors' kernel."""
    default = CacheConfig(use_replayssm=True, replayssm_buffer_len=W)
    default.use_gdn_replayssm = True
    with torch.device("cuda"):
        official = GDNSketchSSM.maybe_create(
            default, 3, 5, 2, 6, K, V, 4, torch.bfloat16, torch.float32
        )
    assert isinstance(official, GDNOfficialReplaySSM)
    monkeypatch.setenv("VLLM_GDN_REPLAYSSM_KERNEL", "sketchssm")
    cache_config = CacheConfig(use_replayssm=True, replayssm_buffer_len=W)
    cache_config.use_gdn_replayssm = True
    assert cache_config.uses_gdn_sketchssm
    with torch.device("cuda"):
        sk = GDNSketchSSM.maybe_create(
            cache_config, 3, 5, 2, 6, K, V, 4, torch.bfloat16, torch.float32
        )
    assert sk is not None and sk.rotation_t is None
    assert not sk.tables.ranks.any()
    mix = torch.randn(3, (2 * 2 + 6) * K, device="cuda", dtype=torch.bfloat16)
    before = mix.clone()
    sk.rotate_(mix)
    assert torch.equal(mix, before)
    plain = CacheConfig(use_replayssm=True, replayssm_buffer_len=W)
    assert (
        GDNSketchSSM.maybe_create(
            plain, 3, 5, 2, 6, K, V, 4, torch.bfloat16, torch.float32
        )
        is None
    )


class _Req:
    def __init__(self, slot: int, prompt: int, chunk: int, start: int):
        self.slot, self.prompt, self.chunk, self.start = slot, prompt, chunk, start
        self.computed = 0


def _layer(vllm_config, kv_cache, weights, h, hv):
    """A minimal object that runs the real ``_forward_core`` bound to it."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
        QwenGatedDeltaNetAttention,
    )

    layer = types.SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=h,
        num_v_heads=hv,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=h * K,
        value_dim=hv * V,
        activation="silu",
        kv_cache=kv_cache,
        A_log=weights[0],
        dt_bias=weights[1],
        conv1d=types.SimpleNamespace(weight=weights[2], bias=weights[3]),
    )
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        layer.sketchssm = GDNSketchSSM.maybe_create(
            vllm_config.cache_config, 0, 0, h, hv, K, V,
            vllm_config.scheduler_config.max_num_seqs, torch.bfloat16, torch.float32,
        )  # fmt: skip
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_sketchssm_decode",
        "_rotate_qk",
        "_forward_core_decode_non_spec",
    ):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))
    return layer


def _forward(layer, meta, mix, a, b) -> torch.Tensor:
    out = torch.zeros(
        mix.shape[0], layer.num_v_heads, V, dtype=mix.dtype, device=mix.device
    )
    ctx = types.SimpleNamespace(attn_metadata={layer.prefix: meta})
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(
            mixed_qkv=mix.clone(), b=b.clone(), a=a.clone(), core_attn_out=out
        )
    return out


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="The GDN prefill uses the CuteDSL chunk kernel (SM10x)",
)
def test_gdn_replayssm_layer_mixed_waves():
    """The real ``_forward_core`` with builder metadata: chunked prefills join
    decode waves, decode rows stay on the window ring (also in mixed waves)
    and match the dense recurrent decode through three flushes."""
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    torch.manual_seed(0)
    h, hv, dev = 2, 4, torch.device("cuda")
    conv_dim = (2 * h + hv) * K
    configs, builders = [], []
    for replay in (True, False):
        cfg = create_vllm_config(
            model_name="Qwen/Qwen3.5-0.8B",
            block_size=16,
            hf_config_override={"linear_key_head_dim": K},
        )
        cfg.additional_config = {"gdn_prefill_backend": "cutedsl"}
        cfg.cache_config.mamba_cache_mode = "none"
        cfg.cache_config.use_replayssm = cfg.cache_config.use_gdn_replayssm = replay
        configs.append(cfg)
        spec = MambaSpec(block_size=16, shapes=((16, 64),), dtypes=(torch.float16,))
        builders.append(GDNAttentionMetadataBuilder(spec, ["l"], cfg, dev))

    nx = 8
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(1, h, hv, K, V, 4)
    shapes = MambaStateShapeCalculator.append_gdn_sketchssm_ring(
        shapes, 1, h, hv, K, V, W
    )
    bf16, fp32 = torch.bfloat16, torch.float32
    dtypes = (bf16, fp32, bf16, bf16, fp32)
    cache = [
        torch.randn(nx, *s, device=dev).to(d) * 0.05 for s, d in zip(shapes, dtypes)
    ]
    weights = (
        torch.randn(hv, device=dev) * 0.1,
        torch.randn(hv, device=dev) * 0.1,
        torch.randn(conv_dim, 1, 4, device=dev, dtype=torch.bfloat16) * 0.1,
        torch.randn(conv_dim, device=dev, dtype=torch.bfloat16) * 0.1,
    )
    replay = _layer(configs[0], [x.clone() for x in cache], weights, h, hv)
    stock = _layer(configs[1], [x.clone() for x in cache[:2]], weights, h, hv)
    assert replay.sketchssm is not None and stock.sketchssm is None

    # (slot, prompt, chunk, first step): chunked prompts join decode waves.
    reqs = [
        _Req(2, 40, 40, 0),
        _Req(5, 23, 16, 0),
        _Req(3, 70, 32, 9),
        _Req(7, 9, 9, 25),
    ]
    out_err = state_err = 0.0
    flushes = mixed_decodes = 0
    for step in range(50):
        live = [r for r in reqs if r.start <= step]
        decodes = [r for r in live if r.computed >= r.prompt]
        prefills = [r for r in live if r.computed < r.prompt]
        rows = decodes + prefills
        query = [1] * len(decodes) + [
            min(r.chunk, r.prompt - r.computed) for r in prefills
        ]
        batch = BatchSpec(
            seq_lens=[r.computed + q for r, q in zip(rows, query)], query_lens=query
        )
        common = create_common_attn_metadata(batch, 16, dev)
        table = common.block_table_tensor.clone()
        table[:, 0] = torch.tensor([r.slot for r in rows], device=dev)
        common = common.replace(
            block_table_tensor=table,
            is_prefilling=torch.tensor([r.computed < r.prompt for r in rows]),
            replayssm_decode_base_cpu=torch.tensor(
                [r.prompt for r in rows], dtype=torch.int32
            ),
            req_idx=np.array([reqs.index(r) for r in rows]),
        )
        metas = [
            b.build(common_prefix_len=0, common_attn_metadata=common) for b in builders
        ]
        assert metas[0].num_decodes == len(decodes)
        n = sum(query)
        mix = torch.randn(n, conv_dim, device=dev, dtype=torch.bfloat16) * 0.3
        a = torch.randn(n, hv, device=dev, dtype=torch.bfloat16) * 0.5 - 1
        b = torch.randn(n, hv, device=dev, dtype=torch.bfloat16)
        out_replay = _forward(replay, metas[0], mix, a, b)
        out_stock = _forward(stock, metas[1], mix, a, b)
        nd = len(decodes)
        torch.testing.assert_close(out_replay[nd:], out_stock[nd:], atol=0, rtol=0)
        if nd:
            out_err = max(
                out_err, _rel(out_replay[:nd].float(), out_stock[:nd].float())
            )
            mixed_decodes += nd if prefills else 0
        for r in decodes:
            if (r.computed - r.prompt) % W == W - 1:
                flushes += 1
                state_err = max(
                    state_err,
                    _rel(replay.kv_cache[1][r.slot], stock.kv_cache[1][r.slot]),
                )
        for r, q in zip(rows, query):
            r.computed += q
    print(
        f"\nlayer: {flushes} flushes, {mixed_decodes} mixed-wave decode rows; "
        f"decode out rel {out_err:.3e}, flushed state rel {state_err:.3e}"
    )
    assert flushes >= 7 and mixed_decodes >= 6
    assert out_err < 1e-2 and state_err < 4e-3
