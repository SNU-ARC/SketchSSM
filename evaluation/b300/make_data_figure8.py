"""results/e2e/*.json -> data/super-e2e-data.json, the input of render_figure8.py (no GPU).

results/e2e/sweep_{arm}_c{C}.json: unprofiled decode cohorts (bench run of the actual model, see README), one
summary per batch and repeat. results/e2e/profile_{arm}_c{C}.json: the per-step kernel breakdown of one profiled
cohort at B=512.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("standard", "replayssm", "g8", "g4")
CONTEXTS = (2048, 8192)


def load(kind, arm, c):
    return json.loads((ROOT / "results" / "e2e" / f"{kind}_{arm}_c{c}.json").read_text())


throughput, breakdown = [], []
for c in CONTEXTS:
    for arm in ARMS:
        runs = {}
        for s in load("sweep", arm, c)["summaries"]:
            runs.setdefault(s["batch"], []).append(s)
        for b, rs in sorted(runs.items()):
            ms = sum(r["model_forward_ms_per_step"] for r in rs) / len(rs)
            wall = sum(r["wall_ms_per_step"] for r in rs) / len(rs)
            throughput.append(dict(context=c, arm=arm, batch=b, repeats=len(rs), model_forward_ms_per_step=ms,
                                   tokens_per_s=b * 1000 / ms, wall_ms_per_step=wall, tokens_per_s_wall=b * 1000 / wall))
    for arm in ARMS:
        if not (ROOT / "results" / "e2e" / f"profile_{arm}_c{c}.json").exists():
            continue
        p = load("profile", arm, c)
        cats = {k: v / 1000 for k, v in p["per_step"]["categories_us"].items()}
        step = p["per_step"]["span_us"] / 1000
        breakdown.append(dict(context=c, batch=p["batch"], arm=arm, step_ms=step, components_ms=dict(
            GEMM=cats["GEMM"], **{"Linear attention": cats["Linear attention"], "Softmax attention": cats["Softmax attention"]},
            Others=step - cats["GEMM"] - cats["Linear attention"] - cats["Softmax attention"])))
std = {(r["context"], r["batch"]): r["tokens_per_s"] for r in throughput if r["arm"] == "standard"}
std_end = {c: max((r for r in throughput if r["arm"] == "standard" and r["context"] == c), key=lambda r: r["batch"]) for c in CONTEXTS}
summary = []
for r in throughput:
    end = max(x["batch"] for x in throughput if x["arm"] == r["arm"] and x["context"] == r["context"])
    if (r["context"], r["batch"]) in std:
        summary.append(dict(context=r["context"], batch=r["batch"], arm=r["arm"], speedup_vs_standard=r["tokens_per_s"] / std[r["context"], r["batch"]]))
    if r["batch"] == end:
        summary.append(dict(context=r["context"], batch="max", arm=r["arm"], max_batch=end, tokens_per_s=r["tokens_per_s"],
                            speedup_vs_standard_max=r["tokens_per_s"] / std_end[r["context"]]["tokens_per_s"]))
data = dict(model="super", scope=(
    "Nemotron 3 Super (NVFP4) on one B300, vLLM fork with SketchSSM (Model Runner V2), util 0.95, no prefix caching. "
    "C=2K uses the compact hybrid KV layout (as the paper) and C=8K the default layout; max_model_len = C + 128 (2304 at 2K). One cohort of B random-token "
    "prompts of C tokens, released into decode together, 128 decode steps; per step = mean model-forward GPU time over steps 16-111. "
    "Throughput: unprofiled, 2 repeats, each arm up to its own capacity endpoint (block pool // 8 * 8). Breakdown: one nsys-profiled "
    "cohort at B=512 for every arm; step = GPU span of the decode step, "
    "Others = the rest (convolution, norms, MoE routing, casts, gaps). Standard uses the B300 selective_state_update config in configs/."),
    throughput=throughput, breakdown=breakdown, summary=summary)
(ROOT / "data" / "super-e2e-data.json").write_text(json.dumps(data, indent=1) + "\n")
for s in summary:
    if s["batch"] == "max":
        print(f"C={s['context']} {s['arm']:9s} B_max={s['max_batch']:4d} {s['tokens_per_s']:8.0f} tok/s  {s['speedup_vs_standard_max']:.2f}x vs Standard at its B_max")
