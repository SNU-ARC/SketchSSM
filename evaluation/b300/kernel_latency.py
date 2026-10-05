# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Per-layer linear-attention decode latency on one GPU, kernels only (Figure 7).

Every recurrent layer of the model gets its own buffers (state, rings, sketch)
and the per-layer ranks and frames of the calibration at mean rank G
(``load_sketchssm_calibration(repo, G, 16)``, as vLLM loads it). One CUDA graph
holds one decode step of all layers for a phase: every row at window position
5 (non-flush) or 15 (flush). A non-flush step has no flush row, so it launches
no flush kernel. 3 warm-up replays, then ``--replays`` timed replays, median
over ``--rounds`` captures; per layer = graph time / layers.

Arms (step functions of ``sketchssm.kernels.tools.benchmark_*``):
  standard   vLLM's full-state decode
  replayssm  Mamba-2: vLLM's ReplaySSM kernel; GDN: the ReplaySSM authors'
             Triton kernel (vLLM fork, ``fused_recurrent_gated_delta_rule_replayssm``);
             KDA: the fork's KDAReplaySSM kernels with the window bookkeeping
             (``kda_bookkeeping.py``) run once per step for all layers
  sketch     SketchSSM at mean rank G
  control    sketch with the flush's sketch and coefficient construction
             compiled out (timing only; the hatch of panel (b))

Example:
    python evaluation/b300/kernel_latency.py --family gdn --arm sketch --g 4 \\
        --out evaluation/b300/results/gdn_sketch_g4.json
"""

import argparse
import glob
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FAMILIES = {
    "mamba2": "SketchSSM/Nemotron-3-Super-NVFP4",
    "gdn": "SketchSSM/Qwen3.8-Flash-Next-NVFP4",
    "kda": "SketchSSM/GLM-5.3-Flash-NVFP4",
}
# Flush build knobs of the control arm: the same flush without the sketch and
# coefficient-map construction.
CONTROL_KNOBS = {
    "mamba2": {"FL_DENSE": 1, "FL_DENSE_KEYMAJOR": 1},
    "gdn": {"W1_NOSKETCH": 1},
    "kda": {"K1_ABLATE": 12},
}
W = 16


def control_config_dir(family: str) -> str:
    """A config folder with the package's tuned files plus the control knobs."""
    import sketchssm.kernels as k

    out = tempfile.mkdtemp(prefix=f"sketchssm_control_{family}_")
    src = Path(k.__file__).parent / "configs" / family
    for f in glob.glob(str(src / "*.json")):
        cfg = json.loads(Path(f).read_text())
        cfg.setdefault("flush", {}).update(CONTROL_KNOBS[family])
        Path(out, Path(f).name).write_text(json.dumps(cfg))
    return out


def layouts(calibration: str, g: float):
    import torch
    from vllm.model_executor.layers.mamba.sketchssm import load_sketchssm_calibration

    from sketchssm.kernels.tools.benchmark_utils import Layout

    c = load_sketchssm_calibration(calibration, g, W)
    out = []
    for f, r in zip(c.frames, c.ranks):
        f = f.float().cuda()
        out.append(Layout(f.contiguous(), f.transpose(-1, -2).contiguous(),
                          r.to(torch.int32).cuda()))  # fmt: skip
    return out, c


def gdn_replay_step(batch, args):
    """The ReplaySSM authors' GDN decode: the FP32 state read every step and
    written on the flush step; rings d (HV, W, V), k (H, W, K) BF16, g (HV, W)."""
    import torch
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_replayssm,
    )

    V = K = 128
    HV, H, w, dev = args.num_v_heads, args.num_k_heads, W, "cuda"
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    state = torch.randn(batch + 1, HV, V, K, device=dev) * 0.1
    mixed = torch.randn(batch, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(batch, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(batch, HV, device=dev, dtype=torch.bfloat16)
    A_log, dt_bias = torch.zeros(HV, device=dev), torch.zeros(HV, device=dev)
    d = torch.randn(batch + 1, HV, w, V, device=dev, dtype=torch.bfloat16) * 0.1
    k = torch.randn(batch + 1, H, w, K, device=dev, dtype=torch.bfloat16) * 0.1
    g = -torch.rand(batch + 1, HV, w, device=dev)
    pos = 5 if args.phase == "nonflush" else w - 1
    write_pos = torch.full((batch,), pos, dtype=torch.int32, device=dev)
    is_flush = (write_pos == w - 1).to(torch.int8)
    out = torch.empty(batch, 1, HV, V, device=dev, dtype=torch.bfloat16)
    return lambda: fused_recurrent_gated_delta_rule_replayssm(
        mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=K**-0.5,
        initial_state=state, d_cache=d, k_cache=k, g_cache=g, out=out,
        ssm_state_indices=slots, write_pos=write_pos, is_flush=is_flush,
        use_qk_l2norm_in_kernel=True,
    )  # fmt: skip


_KDA_ROLE = dict(first=False, last=False)


def kda_replay_steps(batch, args, n_layers):
    """KDAReplaySSM steps of all layers sharing one window bookkeeping (owners,
    positions, slots, work lists): the first layer resolves, the last bumps
    (``kda_bookkeeping``); the ring position is reset to the phase once per
    graph (a memset, timed alone and subtracted)."""
    import torch
    from vllm.model_executor.layers.mamba import kda_replayssm as kr

    H, V, K, dev = 64, 128, 128, "cuda"
    pos = 5 if args.phase == "nonflush" else 15
    reps, steps = [], []
    for _ in range(n_layers):
        rep = kr.KDAReplaySSM(H, batch, torch.device(dev))
        state = torch.randn(batch + 1, H, V, K, device=dev) * 0.1
        ids = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
        q, k, v, g = (torch.randn(batch, H, K, device=dev, dtype=torch.bfloat16) for _ in range(4))
        beta = torch.randn(batch, H, device=dev, dtype=torch.bfloat16)
        A_log, bias = torch.zeros(H, device=dev), torch.zeros(H * K, device=dev)
        _KDA_ROLE.update(first=True, last=True)
        rep.step(state, ids, q, k, v, g, beta, A_log, bias)   # every row takes a slot
        reps.append(rep)
        steps.append(lambda rep=rep, a=(state, ids, q, k, v, g, beta, A_log, bias): rep.step(*a))
    torch.cuda.synchronize()
    for rep in reps[1:]:
        for name in ("owners", "pos", "slots", "old", "old_pos", "fresh", "counts", "work_rows", "work_counts"):
            setattr(rep, name, getattr(reps[0], name))
    sys.path.insert(0, str(HERE))
    import kda_bookkeeping

    cuda = kda_bookkeeping.module()
    aux = [torch.zeros(batch + 2, dtype=torch.int32, device=dev), torch.zeros(batch + 2, dtype=torch.int32, device=dev),
           torch.zeros(1, dtype=torch.int32, device=dev), torch.empty(batch, dtype=torch.int32, device=dev)]

    class _Launch:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return self.fn

    def resolve(ids, owners, slots, old, fresh, B, P, WB, WP, pos_, old_pos, counts, work_rows, work_counts, **_):
        if _KDA_ROLE["first"]:
            cuda.resolve(ids, owners, slots, old, fresh, old_pos, pos_, counts, work_rows, work_counts, *aux, B, P)

    def bump(slots, pos_, B, W_, X, **_):
        if _KDA_ROLE["last"]:
            cuda.bump(slots, pos_, B)

    kr.replay_resolve, kr.replay_bump = _Launch(resolve), _Launch(bump)
    reset = lambda: reps[0].pos.fill_(pos)

    def role(i, f):
        def run():
            _KDA_ROLE.update(first=i == 0, last=i == n_layers - 1)
            if i == 0:
                reset()
            f()
        return run

    return [role(i, f) for i, f in enumerate(steps)], [reset]


# vLLM 0.30 Mamba page of Nemotron 3 Super (FP32 state, BF16 conv state, W=16 rings),
# as the engine reports it (KVCacheConfig) at C=2K and 8K.
MAMBA2_VLLM_PAGE = {"standard": 4259840, "rings": 4562944}
MAMBA2_CONV_BYTES = 3 * (128 * 64 + 2 * 8 * 128) * 2


def build(family, arm, batch, args, lays):
    """Per-layer step callables of one phase, and callables timed alone and
    subtracted."""
    if family == "mamba2":
        from sketchssm.kernels.tools import benchmark_mamba2 as bm

        mode = {"standard": "standard", "replayssm": "replay"}.get(arm, "sketch")
        bm._POOLS.clear()
        if not args.vllm_layout:
            return [bm.mamba2_step(mode, batch, args, lay) for lay in lays], []
        # States as vLLM's hybrid KV pool holds them for Super: one page per block
        # (Standard's Mamba page is padded to the attention page; the rings make
        # ReplaySSM/SketchSSM pages larger), conv state first, six blocks per
        # request (five Mamba groups of eight layers and one attention group),
        # layer j of every group sharing one pool tensor.
        page = MAMBA2_VLLM_PAGE["standard" if arm == "standard" else "rings"]
        return [bm.mamba2_step(mode, batch, args, lay, page_bytes=page, slot_stride=6,
                               offset_bytes=MAMBA2_CONV_BYTES, pool_id=i % 8, group=i // 8)
                for i, lay in enumerate(lays)], []  # fmt: skip
    if family == "gdn":
        from sketchssm.kernels.tools import benchmark_gdn as bg

        if arm == "replayssm":
            return [gdn_replay_step(batch, args) for _ in lays], []
        return [bg.gdn_step("standard" if arm == "standard" else "sketch", batch, args, lay)
                for lay in lays], []  # fmt: skip
    from sketchssm.kernels.tools import benchmark_kda as bk

    if arm == "replayssm":
        return kda_replay_steps(batch, args, len(lays))
    return [bk.kda_step("standard" if arm == "standard" else "sketch", batch, args, lay)
            for lay in lays], []  # fmt: skip


def time_graph(fns, warmup, replays):
    import torch

    for f in fns:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in fns:
            f()
    for _ in range(warmup):
        graph.replay()
    ts = []
    for _ in range(replays):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return ts


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--arm", choices=("standard", "replayssm", "sketch", "control"), required=True)
    p.add_argument("--g", type=float, default=8, help="mean rank (sketch, control)")
    p.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--replays", type=int, default=6)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--contiguous-state", action="store_true",
                   help="mamba2: one contiguous state tensor instead of vLLM's KV-pool placement")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.arm == "control":
        os.environ["SKETCHSSM_KERNELS_CONFIG_DIR"] = control_config_dir(a.family)
    if a.family == "mamba2" and a.arm == "standard":
        # vLLM ships no selective_state_update config for B300; the tuned one is here.
        os.environ.setdefault("VLLM_TUNED_CONFIG_FOLDER", str(HERE / "configs" / "selective_state_update"))
    import torch

    torch.set_default_device("cuda")
    lays, cal = layouts(FAMILIES[a.family], a.g)
    nl = len(lays)
    args = SimpleNamespace(window=W, num_heads=128, head_dim=64, ngroups=8, state_size=128,
                           num_k_heads=16, num_v_heads=48,
                           vllm_layout=not a.contiguous_state)  # fmt: skip
    phases = ["step"] if a.arm == "standard" else ["nonflush", "flush"]
    if a.arm == "control":
        phases = ["flush"]
    rows = []
    for batch in a.batches:
        for phase in phases:
            args.phase = "nonflush" if phase == "step" else phase
            rounds = []
            for _ in range(a.rounds):
                steps, subtract = build(a.family, a.arm, batch, args, lays)
                t = statistics.mean(time_graph(steps, a.warmup, a.replays))
                if subtract:
                    t -= statistics.mean(time_graph(subtract, a.warmup, a.replays))
                rounds.append(t)
                del steps, subtract
                torch.cuda.empty_cache()
            us = statistics.median(rounds)
            rows.append(dict(batch=batch, phase=phase, us_per_layer=us / nl,
                             round_us_per_layer=[x / nl for x in rounds]))  # fmt: skip
            print(f"{a.family} {a.arm} G={a.g} B={batch} {phase}: {us / nl:9.2f} us/layer", flush=True)
    import sketchssm

    ranks = torch.cat([torch.as_tensor(x).flatten() for x in cal.ranks])
    res = dict(family=a.family, arm=a.arm, g=a.g if a.arm in ("sketch", "control") else None,
               calibration=FAMILIES[a.family], layers=nl, window=W, dense_heads=int((ranks == 0).sum()),
               warmup=a.warmup, replays=a.replays, rounds=a.rounds, rows=rows,
               device=torch.cuda.get_device_name(), sketchssm=sketchssm.__version__,
               time=time.strftime("%F %T"))  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
