"""results/*.json (kernel_latency.py) -> data/ (the renderers' input) and data/summary.json (no GPU)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = [("super", "mamba2", "Nemotron 3 Super (Mamba-2)"), ("qwen", "gdn", "Qwen3.8 Flash-Next (GDN)"),
          ("glm", "kda", "GLM 5.3 Flash (KDA)")]
RANKS = (8, 4)
CONTROL_G = 4   # the renderers' single "w/o sketch" reference (the lowest rank, as in the paper)
W = 16


def load(fam, arm, g=None):
    f = ROOT / "results" / (f"{fam}_{arm}" + (f"_g{g}" if g is not None else "") + ".json")
    return {(r["batch"], r["phase"]): r["us_per_layer"] for r in json.loads(f.read_text())["rows"]}


summary = []
for model, fam, title in MODELS:
    std, rep = load(fam, "standard"), load(fam, "replayssm")
    sk = {g: load(fam, "sketch", g) for g in RANKS}
    ctl = {g: load(fam, "control", g) for g in RANKS}
    layers = json.loads((ROOT / "results" / f"{fam}_standard.json").read_text())["layers"]
    rows = []
    for b in (128, 256, 512):
        sw = W * std[b, "step"]
        rw = (W - 1) * rep[b, "nonflush"] + rep[b, "flush"]
        rows.append(dict(batch=b, arm="standard", phase="window", ms_per_layer=sw / 1000))
        for arm, t in [("replayssm", rep)] + [(f"g{g}", sk[g]) for g in RANKS]:
            rows += [dict(batch=b, arm=arm, phase="nonflush", us_per_layer=t[b, "nonflush"]),
                     dict(batch=b, arm=arm, phase="flush", us_per_layer=t[b, "flush"]),
                     dict(batch=b, arm=arm, phase="window", ms_per_layer=((W - 1) * t[b, "nonflush"] + t[b, "flush"]) / 1000)]
        rows.append(dict(batch=b, arm="control", phase="flush", us_per_layer=ctl[CONTROL_G][b, "flush"], control_g=CONTROL_G))
        summary.append(dict(model=model, batch=b, arm="replayssm", window_speedup_vs_standard=sw / rw))
        for g in RANKS:
            w = (W - 1) * sk[g][b, "nonflush"] + sk[g][b, "flush"]
            summary.append(dict(model=model, batch=b, arm=f"g{g}", window_us_per_layer=w, window_speedup_vs_standard=sw / w,
                                window_speedup_vs_replayssm=rw / w,
                                nonflush_speedup_vs_standard=std[b, "step"] / sk[g][b, "nonflush"],
                                flush_construction_overhead_pct=100 * (sk[g][b, "flush"] / ctl[g][b, "flush"] - 1)))
    data = dict(model=model, layers=layers, scope=(
        f"{title} on one B300, kernels only: all {layers} recurrent layers with their own buffers and the calibration's "
        "ranks/frames at mean rank G, W=16. One CUDA graph per phase holds one decode step of every layer (every row at "
        "window position 5 = non-flush, no flush launch; or 15 = flush). Per layer = graph time / layers, median of 5 "
        "captures of 6 replays. (c) window = 15 non-flush + 1 flush; Standard = 16 full-state updates. control = the "
        f"flush without the sketch and coefficient-map construction (timing only), G={CONTROL_G} in the hatch."),
        rows=rows)
    (ROOT / "data" / f"{model}-kernel-batches-data.json").write_text(json.dumps(data, indent=1) + "\n")
(ROOT / "data" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
for s in summary:
    if s["batch"] == 512:
        extra = "" if s["arm"] == "replayssm" else f"  flush overhead {s['flush_construction_overhead_pct']:+.1f}%"
        print(f"{s['model']:6s} B=512 {s['arm']:9s} window speedup vs Standard {s['window_speedup_vs_standard']:.2f}x{extra}")
