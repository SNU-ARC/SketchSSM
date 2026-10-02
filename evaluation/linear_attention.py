# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Per-layer recurrent decode latency of a real model in vLLM, by kernel.

Follows the measurement behind the paper's kernel figure: run the model's
decode steps in vLLM's CUDA graphs, trace every kernel with Nsight Systems
(graph nodes included), attribute kernels to engine steps with NVTX, and sum
the recurrent kernels of each step (readout, flush, shared B/C precompute,
basis transform; for KDA also the flush finish and the input copies of the
CUDA step) divided by the number of recurrent layers.

Steps are classified by the ReplaySSM ring: a non-flush step has no flushing
row, a flush step has only flushing rows, a mixed step has both. With
``--stagger 1`` (default 0) requests start decoding at spread-out steps, as in
online serving, so decode steps are mixed.

Example (from the repository root, with vllm/ installed):

    python evaluation/linear_attention.py --arm standard --batch 256 --out runs/std
    python evaluation/linear_attention.py --arm sketchssm --batch 256 \\
        --calibration outputs/nano_frames.pt --out runs/sketch
    python evaluation/linear_attention.py --summarize runs/std runs/sketch
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NANO = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
WINDOW = 16

# Kernel name patterns of the recurrent core, by component (Mamba-2, GDN, then
# KDA).
COMPONENTS = {
    "readout": [
        r"^nf_kernel", r"^_sketch_decode_kernel", r"selective_scan_update",
        r"^_replayssm_output_only_kernel",
        r"\bgdn_step_kernel\b", r"^_gdn_sketch_step_kernel",
        r"^fused_recurrent_gated_delta_rule_(packed_decode|replayssm)_kernel",
        r"\bkda_step_kernel\b", r"^_kda_sketch_step_kernel",
        r"^_kda_replayssm_kernel", r"^fused_recurrent_gated_delta_rule_fwd_kernel",
    ],
    "flush": [
        r"^flush_kernel", r"^_sketch_flush_kernel", r"^_cold_build_kernel",
        r"\bgdn_flush_warp_kernel\b", r"^_gdn_sketch_flush_kernel",
        r"^_gdn_build_kernel",
        r"\bkda_flush_main\b", r"^_kda_sketch_flush_kernel",
    ],
    "finish": [r"\bkda_flush_finish\b", r"^_kda_sketch_finish_kernel"],
    # The .contiguous() copies of q/k/v/g/beta right before the KDA CUDA step.
    "input copy": [],
    "shared B/C": [r"^_bc_pre_kernel", r"^_replayssm_output_only_precompute_kernel"],
    "basis transform": [
        r"^_rot_inplace_kernel", r"^_rotate_groups_kernel", r"^_rotate_qk_kernel",
    ],
}


# Copy kernels attributed to "input copy" when they directly precede one of
# these step kernels.
COPY = r"direct_copy_kernel"
COPY_BEFORE = r"\bkda_step_kernel\b"


def component(name: str) -> str | None:
    for comp, patterns in COMPONENTS.items():
        if any(re.search(p, name) for p in patterns):
            return comp
    return None


# --------------------------------------------------------------------------
# Worker: runs the model with per-step NVTX ranges (under nsys).
# --------------------------------------------------------------------------


def truncated_config(cfg, n: int):
    """Keep the first ``n`` decoder layers (and no MTP layers) of a
    ``layer_types`` model, for a model that does not fit on one GPU."""
    for t in (cfg, getattr(cfg, "text_config", None)):
        if t is not None and hasattr(t, "layer_types"):
            t.num_hidden_layers = n
            for key in ("layer_types", "mlp_layer_types", "indexer_types"):
                if getattr(t, key, None) is not None:
                    setattr(t, key, getattr(t, key)[:n])
            for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
                if hasattr(t, key):
                    setattr(t, key, 0)
    return cfg


def truncate_model(n: int) -> None:
    """Skip the checkpoint weights of layers ``>= n`` and of the MTP head."""
    from vllm.model_executor.model_loader import default_loader as dl

    orig = dl.DefaultModelLoader.get_all_weights

    def get_all_weights(self, *a, **k):
        for name, w in orig(self, *a, **k):
            m = re.search(r"layers\.(\d+)\.", name)
            if (m and int(m.group(1)) >= n) or "mtp" in name:
                continue
            yield name, w

    dl.DefaultModelLoader.get_all_weights = get_all_weights



def worker(args) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from vllm.v1.attention.backends import mamba_attn
    from vllm.v1.worker.gpu import model_runner

    from vllm import LLM, SamplingParams

    steps: list[dict] = []
    current: dict = {}
    orig_execute = model_runner.GPUModelRunner.execute_model

    def execute_model(self, *a, **k):
        nonlocal current
        current = dict(step=len(steps), decode=0, prefill=0, flush=0)
        steps.append(current)
        torch.cuda.nvtx.range_push(f"step {current['step']}")
        try:
            return orig_execute(self, *a, **k)
        finally:
            torch.cuda.nvtx.range_pop()

    orig_meta = mamba_attn.BaseMambaAttentionMetadataBuilder._compute_common_metadata

    def compute_common_metadata(self, *a, **k):
        meta = orig_meta(self, *a, **k)
        if current and "seen" not in current:
            current["seen"] = True
            current["decode"] = int(meta.num_decodes)
            current["prefill"] = int(meta.num_prefills)
            if meta.is_flush_d is not None and meta.num_decodes:
                current["flush"] = int(meta.is_flush_d[: meta.num_decodes].sum())
        return meta

    model_runner.GPUModelRunner.execute_model = execute_model
    mamba_attn.BaseMambaAttentionMetadataBuilder._compute_common_metadata = (
        compute_common_metadata
    )
    try:
        from vllm.v1.attention.backends import gdn_attn
    except ImportError:
        gdn_attn = None
    if gdn_attn is not None:
        orig_gdn_build = gdn_attn.GDNAttentionMetadataBuilder.build

        def gdn_build(self, *a, **k):
            meta = orig_gdn_build(self, *a, **k)
            if current and "seen" not in current:
                current["seen"] = True
                current["decode"] = int(meta.num_decodes)
                current["prefill"] = int(meta.num_prefills)
                wp = getattr(meta, "write_pos_d", None)
                if wp is not None and meta.num_decodes:
                    last = args.window - 1
                    current["flush"] = int((wp[: meta.num_decodes] == last).sum())
            return meta

        gdn_attn.GDNAttentionMetadataBuilder.build = gdn_build
    if args.num_hidden_layers:
        truncate_model(args.num_hidden_layers)

    kw = dict(
        model=args.model, trust_remote_code=True, max_model_len=4096,
        max_num_seqs=args.batch, gpu_memory_utilization=0.95,
        mamba_ssm_cache_dtype="float32", enable_prefix_caching=False,
    )  # fmt: skip
    if args.stagger:
        # Prefills spread over one window, so decode runs start out of phase.
        kw["max_num_batched_tokens"] = max(args.prompt_len, args.batch * args.prompt_len // args.window)
    else:
        kw["max_num_batched_tokens"] = max(8192, args.batch * args.prompt_len)
    if args.num_hidden_layers:
        kw["hf_overrides"] = lambda cfg: truncated_config(cfg, args.num_hidden_layers)
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if args.arm == "replayssm":
        kw.update(use_replayssm=True, replayssm_buffer_len=args.window)
    if args.arm == "sketchssm":
        kw.update(sketchssm=args.calibration, replayssm_buffer_len=args.window)
        if args.sketchssm_mean_rank is not None:
            kw["sketchssm_mean_rank"] = args.sketchssm_mean_rank
    llm = LLM(**kw)
    prompt = list(range(1000, 1000 + args.prompt_len))
    params = SamplingParams(
        temperature=0, max_tokens=args.window * (args.windows + 2), ignore_eos=True
    )
    llm.generate([{"prompt_token_ids": prompt}] * args.batch, params, use_tqdm=False)
    steps.clear()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    llm.generate([{"prompt_token_ids": prompt}] * args.batch, params, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    for s in steps:
        s.pop("seen", None)
    Path(args.out, "steps.json").write_text(json.dumps(steps))


# --------------------------------------------------------------------------
# Analysis of the Nsight Systems export.
# --------------------------------------------------------------------------


def analyze(out: Path, num_layers: int, full_batch: int) -> dict:
    steps = {s["step"]: s for s in json.loads((out / "steps.json").read_text())}
    db = sqlite3.connect(out / "trace.sqlite")
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = [
        (start, end, int(text.split()[1]))
        for start, end, text in db.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'step %'"
        )
        if end is not None
    ]
    ranges.sort()
    launches = dict(
        db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    )
    per_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    import bisect

    starts = [r[0] for r in ranges]
    kernels: dict[int, list] = defaultdict(list)
    for corr, name_id, start, end in db.execute(
        "SELECT correlationId, demangledName, start, end "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        t = launches.get(corr)
        if t is None:
            continue
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > ranges[i][1]:
            continue
        kernels[ranges[i][2]].append((start, end, names[name_id]))
    for idx, ks in kernels.items():
        ks.sort()
        comps = [component(name) or "other" for _, _, name in ks]
        for j, (_, _, name) in enumerate(ks):
            if not re.search(COPY_BEFORE, name):
                continue
            j -= 1
            while j >= 0 and comps[j] == "other" and re.search(COPY, ks[j][2]):
                comps[j] = "input copy"
                j -= 1
        for (start, end, _), comp in zip(ks, comps):
            per_step[idx][comp] += (end - start) / 1e3  # us
    # Steady decode steps with the whole batch decoding.
    phases: dict[str, list[dict[str, float]]] = defaultdict(list)
    for idx, comps in per_step.items():
        s = steps.get(idx)
        if s is None or s["prefill"] or s["decode"] != full_batch:
            continue
        phase = (
            "non-flush" if s["flush"] == 0
            else "flush" if s["flush"] == s["decode"] else "mixed"
        )  # fmt: skip
        phases[phase].append(comps)
    if not phases:
        most = max((s["decode"] for s in steps.values()), default=0)
        raise SystemExit(
            f"no decode step ran the whole batch ({full_batch}); at most {most} "
            "requests decoded together (state memory); lower --batch"
        )
    result = {}
    for phase, rows in phases.items():
        mean = {
            c: sum(r.get(c, 0.0) for r in rows) / len(rows) / num_layers
            for c in [*COMPONENTS, "other"]
        }
        mean["recurrent"] = sum(mean[c] for c in COMPONENTS)
        result[phase] = dict(steps=len(rows), us_per_layer=mean)
    return result


def summarize(dirs: list[str]) -> None:
    runs = {d: json.loads(Path(d, "result.json").read_text()) for d in dirs}
    std = next((r for r in runs.values() if r["arm"] == "standard"), None)
    std_phase = std["phases"].get("non-flush") if std else None
    std_step = std_phase["us_per_layer"]["recurrent"] if std_phase else None
    cols = [("readout", "readout"), ("flush", "flush"), ("finish", "finish"),
            ("in copy", "input copy"), ("B/C pre", "shared B/C"),
            ("rotate", "basis transform")]  # fmt: skip
    print(f"{'run':28s} {'phase':10s} {'steps':>5s} "
          + "".join(f"{c:>8s} " for c, _ in cols)
          + f"{'recur.':>8s} {'vs std':>7s}")  # fmt: skip
    for d, r in runs.items():
        for phase, v in sorted(r["phases"].items()):
            u = v["us_per_layer"]
            ratio = f"{std_step / u['recurrent']:.2f}x" if std_step else "-"
            print(f"{Path(d).name:28s} {phase:10s} {v['steps']:5d} "
                  + "".join(f"{u.get(k, 0.0):8.1f} " for _, k in cols)
                  + f"{u['recurrent']:8.1f} {ratio:>7s}")  # fmt: skip
        ph = r["phases"]
        if std_step and "non-flush" in ph and "flush" in ph:
            w = r.get("window", WINDOW)
            window = (w - 1) * ph["non-flush"]["us_per_layer"]["recurrent"]
            window += ph["flush"]["us_per_layer"]["recurrent"]
            print(f"{'':28s} {f'W={w} mean':10s} {'':5s} " + " " * 9 * len(cols)
                  + f"{window / w:8.1f} {w * std_step / window:6.2f}x")  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=["standard", "replayssm", "sketchssm"])
    p.add_argument("--model", default=NANO)
    p.add_argument("--calibration", help="exported SketchSSM calibration (--arm sketchssm)")
    p.add_argument(
        "--sketchssm-mean-rank", type=float, default=None,
        help="mean rank budget for a portable calibration file (--arm sketchssm)",
    )
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--window", type=int, default=WINDOW,
                   help="SketchSSM / ReplaySSM window W (ring length)")  # fmt: skip
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--stagger", type=int, default=0)
    p.add_argument("--num-layers", type=int, default=0, help="recurrent layers (auto)")
    p.add_argument(
        "--num-hidden-layers", type=int, default=0,
        help="keep only the first N decoder layers (layer_types models)",
    )
    p.add_argument("--out")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--summarize", nargs="+", metavar="RUN_DIR")
    args = p.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if args.arm is None or args.out is None:
        p.error("--arm and --out are required")
    if args.arm == "sketchssm" and not args.calibration:
        p.error("--arm sketchssm needs --calibration")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    cmd = [
        "nsys", "profile", "--force-overwrite=true", "--trace=cuda,nvtx",
        "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop", "-o", str(out / "trace"),
        sys.executable, os.path.abspath(__file__), "--worker",
        *[a for a in sys.argv[1:]],
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    subprocess.run(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
         "-o", str(out / "trace.sqlite"), str(out / "trace.nsys-rep")],
        check=True,
    )  # fmt: skip
    num_layers = args.num_layers
    if not num_layers:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        cfg = getattr(cfg, "text_config", cfg)
        if hasattr(cfg, "layer_types"):
            n = args.num_hidden_layers or len(cfg.layer_types)
            num_layers = cfg.layer_types[:n].count("linear_attention")
        else:
            num_layers = cfg.hybrid_override_pattern.count("M")
    phases = analyze(out, num_layers, args.batch)
    result = dict(
        arm=args.arm, model=args.model, calibration=args.calibration,
        sketchssm_mean_rank=args.sketchssm_mean_rank, batch=args.batch,
        window=args.window, stagger=args.stagger, num_layers=num_layers,
        phases=phases,
        sketch_kernels=(
            "triton" if os.environ.get("VLLM_SKETCHSSM_USE_CUDA") == "0" else "auto"
        ),
    )  # fmt: skip
    Path(out, "result.json").write_text(json.dumps(result, indent=1))
    summarize([str(out)])


if __name__ == "__main__":
    main()
