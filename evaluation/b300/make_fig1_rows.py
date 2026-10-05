"""results/e2e/profile_*.json -> results/e2e/fig1_breakdown.json: Standard and ReplaySSM decode-step breakdown rows (B=256, 512 and 872)
(context, batch, arm, step_ms, components_ms) for the Figure 1 Super panel (no GPU). Same classes as Figure 8 (b)."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "e2e"
rows = []
for f in sorted(ROOT.glob("profile_*.json")):
    arm, c, b = re.fullmatch(r"profile_(\w+?)_c(\d+)(?:_b(\d+))?\.json", f.name).groups()
    if arm not in ("standard", "replayssm"):
        continue
    p = json.loads(f.read_text())
    cats = {k: v / 1000 for k, v in p["per_step"]["categories_us"].items()}
    step = p["per_step"]["span_us"] / 1000
    comp = {k: cats[k] for k in ("GEMM", "Linear attention", "Softmax attention")}
    comp["Others"] = step - sum(comp.values())
    rows.append(dict(context=int(c), batch=p["batch"], arm=arm, step_ms=step, components_ms=comp,
                     tokens_per_s=p["batch"] * 1000 / step))
rows.sort(key=lambda r: (r["context"], r["batch"], r["arm"]))
(ROOT / "fig1_breakdown.json").write_text(json.dumps(dict(scope=(
    "Nemotron 3 Super on one B300, Figure 8 settings (default KV layout, Model Runner V2, shared FlashInfer autotune, "
    "B300 selective_state_update config, max_model_len = C + 128). One nsys-profiled cohort per row (max_num_seqs = B); "
    "step = GPU span of a decode step over steps 16-111; Others = the rest of the span; tokens_per_s = batch / step."),
    rows=rows), indent=1) + "\n")
for r in rows:
    print(f"C={r['context']} B={r['batch']:4d} {r['arm']:9s} {r['step_ms']:7.2f} ms " +
          " ".join(f"{k[:6]} {v:.2f}" for k, v in r["components_ms"].items()))
