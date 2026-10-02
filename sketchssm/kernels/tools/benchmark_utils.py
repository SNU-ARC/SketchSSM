# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Shared helpers of the decode benchmarks and the CUDA tuners.

``benchmark_{mamba2,gdn,kda}`` time one decode step of one layer;
``tune_{mamba2,gdn,kda}`` run a coordinate descent over the CUDA build knobs
of a family (``tune_and_save``), timed over all layers of a calibration and a
whole window. Both need vLLM (sketch tables, Triton helpers,
baseline decoders and the calibration loader).
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import torch
from vllm.model_executor.layers.mamba.sketchssm import load_sketchssm_calibration

# Default SketchSSM window: a row flushes on the last of every ``L`` steps.
L = 16
PHASES = ("nonflush", "flush", "mixed")


def ring_phase(args, batch, dev):
    """``(write_pos, is_flush)`` of every row for ``args.phase`` in a window
    of ``args.window`` steps."""
    w = args.window
    phase = {
        "nonflush": torch.full((batch,), 5),
        "flush": torch.full((batch,), w - 1),
        "mixed": torch.arange(batch) % w,
    }[args.phase]
    return phase.to(torch.int32).to(dev), (phase == w - 1).to(torch.int8).to(dev)


def flush_row_list(is_flush: torch.Tensor) -> torch.Tensor:
    """``(batch,)`` int32 indices of the flush rows, then -1 padding."""
    rows = torch.full_like(is_flush, -1, dtype=torch.int32)
    idx = torch.nonzero(is_flush).flatten()
    rows[: idx.numel()] = idx.to(torch.int32)
    return rows


class Layout(NamedTuple):
    """One layer's frames ``(groups, K, K)`` (FP32, and FP32 transposed) and
    per-head ranks, on the default device."""

    frames: torch.Tensor
    frames_t: torch.Tensor
    ranks: torch.Tensor


def make_layout(args, heads, groups, key_dim) -> Layout:
    """The calibration layer's layout, or a synthetic one for another shape:
    the layer's ranks cycled over the heads and scaled to ``key_dim``, with
    random orthogonal frames."""
    calibration = load_sketchssm_calibration(args.calibration)
    frames, ranks = calibration.frames[args.layer], calibration.ranks[args.layer]
    if frames.shape != (groups, key_dim, key_dim) or ranks.shape != (heads,):
        scale = key_dim / frames.shape[-1]
        scaled = (ranks.float() * scale).round().clamp(1, key_dim).to(ranks.dtype)
        ranks = torch.where(ranks > 0, scaled, 0)
        ranks = ranks.repeat(-(-heads // ranks.numel()))[:heads]
        gen = torch.Generator().manual_seed(0)
        noise = torch.randn(groups, key_dim, key_dim, generator=gen, device="cpu")
        frames = torch.linalg.qr(noise)[0]
        print(f"synthetic layout: {heads} heads, {groups} groups, state {key_dim}")
    device = torch.get_default_device()
    return Layout(
        frames.contiguous().to(device),
        frames.transpose(-1, -2).contiguous().to(device),
        ranks.to(torch.int32).to(device),
    )


def time_graph(fn, iters):
    """Mean latency in us of ``fn`` replayed as a CUDA graph."""
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def add_benchmark_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--calibration", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--phase", choices=PHASES, default="mixed")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--window", type=int, default=L, help="window (multiple of 16)")


def run_benchmark(args, family, layout, modes, step) -> None:
    """Print the latency of each mode (``step(mode, batch, args, layout)``)
    per batch size."""
    print(
        f"{family} phase={args.phase} window={args.window} layer={args.layer} "
        f"mean rank {layout.ranks.float().mean():.2f} "
        f"dense {int((layout.ranks == 0).sum())}"
    )
    print(f"{'batch':>6}" + "".join(f"{m + ' us':>12}" for m in modes))
    for batch in args.batch_sizes:
        row = [time_graph(step(m, batch, args, layout), args.iters) for m in modes]
        print(f"{batch:>6}" + "".join(f"{t:>12.1f}" for t in row))


def make_layouts(args, heads, groups, key_dim) -> list[Layout]:
    """The layouts of ``--layer``, or of every layer of the calibration."""
    if args.layer is not None:
        return [make_layout(args, heads, groups, key_dim)]
    layers = len(load_sketchssm_calibration(args.calibration).ranks)
    layouts = []
    for args.layer in range(layers):
        layouts.append(make_layout(args, heads, groups, key_dim))
    args.layer = None
    return layouts


def add_tuner_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--calibration", required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="one layer (default: every layer of the calibration)")  # fmt: skip
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--check-batch", type=int, default=128,
                   help="also compare the tuned knobs with the defaults at this "
                   "batch (0: skip)")  # fmt: skip
    p.add_argument("--objective", choices=("window", "mixed"), default="window",
                   help="window: (W - 1) non-flush steps + 1 flush step; mixed: "
                   "steps with the rows spread over the window")  # fmt: skip
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--window", type=int, default=L, help="window (multiple of 16)")
    p.add_argument("--save-configs", action="store_true")
    p.add_argument("--out-dir", help="default: the family's shipped configs folder")


def objective(args, family, configs, batch=None) -> float:
    """Mean latency in us per layer and decode step of ``configs`` (full
    knob dicts per kind) over all of ``family.layouts``: over a window of
    ``W - 1`` non-flush steps and one flush step, or of mixed steps
    (``--objective``). Every layer has its own buffers, as in the model."""
    family.configure(configs)
    batch = batch or args.batch
    w = args.window
    phases = {"nonflush": w - 1, "flush": 1} if args.objective == "window" else {"mixed": w}
    total = 0.0
    for phase, steps in phases.items():
        args.phase = phase
        layers = [family.step(args, layout, batch) for layout in family.layouts]
        total += steps * time_graph(lambda fns=layers: [f() for f in fns], args.iters)
        del layers
        torch.cuda.empty_cache()
    return total / w / len(family.layouts)


def full(family, knobs) -> dict:
    """``knobs`` per kind over the defaults."""
    return {k: dict(family.defaults[k], **knobs.get(k, {})) for k in family.kinds}


def time_config(args, family, knobs, batch=None) -> float:
    """``objective`` of ``knobs`` (per kind, over the defaults); inf if they
    do not build or run."""
    try:
        return objective(args, family, full(family, knobs), batch)
    except Exception as e:  # a knob combination this shape does not build
        print(f"  {knobs}: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
        torch.cuda.empty_cache()
        return float("inf")


def tune(args, family, kind, tuned, best_us) -> tuple[dict, float]:
    """Coordinate descent over ``family.space[kind]`` from ``tuned`` (knobs
    per kind over the defaults, ``best_us``); the knobs of ``kind`` that
    differ from the defaults, and their objective."""
    best = dict(tuned.get(kind, {}))
    for _ in range(args.passes):
        improved = False
        for knob, values in family.space[kind].items():
            for value in values:
                cand = dict(best, **{knob: value})
                if cand == best:
                    continue
                us = time_config(args, family, dict(tuned, **{kind: cand}))
                if us < best_us * (1 - args.min_gain):
                    best, best_us, improved = cand, us, True
                    print(f"{kind} {best}: {best_us:.2f} us", flush=True)
        if not improved:
            break
    defaults = family.defaults[kind]
    return {k: v for k, v in best.items() if defaults.get(k) != v}, best_us


def diff(family, configs) -> dict:
    """Full knob dicts per kind as their differences from the defaults."""
    return {k: {n: v for n, v in configs[k].items() if family.defaults[k].get(n) != v}
            for k in family.kinds}  # fmt: skip


def tune_and_save(args, family) -> None:
    """Tune each of ``family.kinds`` in turn, starting from the better of the
    defaults and the knobs in effect (``family.in_effect``: the shape's file
    for this window, or its window-16 file); print the config and, with
    ``--save-configs``, write it to ``--out-dir`` (default: the shipped
    configs folder) under ``family.name`` unless it equals the knobs in
    effect or is not better than them."""
    print(f"{len(family.layouts)} layers, batch {args.batch}, window {args.window}, "
          f"objective {args.objective} (us per layer per step)")  # fmt: skip
    current = diff(family, family.in_effect)
    default_us = time_config(args, family, {})
    current_us = time_config(args, family, current)
    print(f"defaults: {default_us:.2f} us; in effect {current}: {current_us:.2f} us",
          flush=True)  # fmt: skip
    config, best_us = (current, current_us) if current_us < default_us else ({}, default_us)
    config = {k: dict(config.get(k, {})) for k in family.kinds}
    for kind in family.kinds:
        config[kind], best_us = tune(args, family, kind, config, best_us)
    name = family.name
    print(name, json.dumps(config))
    print(f"batch {args.batch}: defaults {default_us:.2f}, in effect {current_us:.2f}, "
          f"tuned {best_us:.2f} us", flush=True)  # fmt: skip
    if args.check_batch:
        row = [time_config(args, family, c, args.check_batch) for c in ({}, current, config)]
        print(f"batch {args.check_batch}: defaults {row[0]:.2f}, in effect {row[1]:.2f}, "
              f"tuned {row[2]:.2f} us", flush=True)  # fmt: skip
    if config == current or best_us >= current_us * (1 - args.min_gain):
        print("tuned knobs are the knobs in effect or no better; nothing to save")
        return
    if args.save_configs:
        path = Path(args.out_dir or family.out_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=4) + "\n")
        print(f"saved {path}")
