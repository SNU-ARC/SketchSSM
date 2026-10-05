"""Figure 8: Nemotron 3 Super end-to-end decode on one B300, from data/super-e2e-data.json (no GPU).

(a) decode throughput against batch at C=2K and 8K, each arm up to its own capacity endpoint; (b) the profiled
decode step at B=512 in each context, split into GEMM, linear attention, softmax attention
and the rest. Layout of the paper's figure (super-e2e-speedup), with SketchSSM at mean rank 8 and 4.
Writes figures/super-e2e-speedup.{pdf,png,svg}.
"""
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
COLORS = {"blue": "#2178B4", "orange": "#F28E2B", "green": "#308C58", "grey": "#777777", "grid": "#D8D8D8"}
RC = {"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 7, "axes.labelsize": 7,
      "axes.linewidth": .5, "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 6.2,
      "ytick.labelsize": 6.2, "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2.4,
      "ytick.major.size": 2.4, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
      "svg.hashsalt": "sketchssm-super-e2e-b300"}
RANKS = (4, 8)
CURVES = [("standard", "Standard", "#4D4D4D", "x", (0, (1, 1))), ("replayssm", "ReplaySSM", "#756BB1", "o", "-"),
          ("g4", None, "#B84365", "s", "--"), ("g8", None, "#B84365", "^", ":")]
LAYOUT = dict(pos={"standard": 0., "replayssm": 1.6, "g8": 2.95, "g4": 3.85},
              labels={"standard": "Standard", "replayssm": "ReplaySSM", "g8": "8", "g4": "4"},
              pitch=5.6, rank_x=3.4, ctx_x=1.9, width=.68)
COMPONENTS = [("GEMM", COLORS["blue"]), ("Linear attention", COLORS["orange"]), ("Softmax attention", COLORS["green"]),
              ("Others", COLORS["grey"])]


def nice(top, steps=(1, 2, 2.5, 5, 10)):
    """Axis limit and ticks: about five ticks, the limit just above the data."""
    import math
    raw = top / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in steps if s * mag >= raw)
    n = math.ceil(top * 1.05 / step)
    return n * step, [i * step for i in range(n + 1)]


def legend(fig, ax, center):
    y, fontsize, ppf = .915, 7.3, fig.get_figwidth() * 72
    lines = ax.get_lines()
    items = [("text", "Standard"), ("gap", 3), ("line", lines[0]), ("gap", 8), ("text", "ReplaySSM"), ("gap", 3),
             ("line", lines[1]), ("gap", 8), ("text", "SketchSSM:"), ("gap", 3), ("line", lines[2]), ("gap", 2),
             ("text", rf"$\bar G={RANKS[0]}$"), ("gap", 6), ("line", lines[3]), ("gap", 2), ("text", rf"$\bar G={RANKS[1]}$")]
    texts = [fig.text(0, y, v, ha="left", va="center", fontsize=fontsize) for k, v in items if k == "text"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widths, t = [], 0
    for k, v in items:
        if k == "text":
            widths.append(texts[t].get_window_extent(r).width / fig.bbox.width); t += 1
        else:
            widths.append((v if k == "gap" else 9) / ppf)
    x, t = center - sum(widths) / 2, 0
    for (k, v), w in zip(items, widths):
        if k == "text":
            texts[t].set_x(x); t += 1
        elif k == "line":
            fig.add_artist(Line2D([x, x + w / 2, x + w], [y, y, y], transform=fig.transFigure, color=v.get_color(),
                                  linestyle=v.get_linestyle(), linewidth=1, marker=v.get_marker(), markersize=2.8, markevery=[1]))
        x += w


def main():
    d = json.loads((ROOT / "data" / "super-e2e-data.json").read_text())
    plt.style.use("default"); plt.rcParams.update(RC)
    fig = plt.figure(figsize=(6.5, 1.55))
    h = 1.85 * (.84 - .32) * .8 * .9 / 1.55
    axes = [fig.add_axes((l, .28, w, h)) for l, w in [(.078, .18), (.315, .185), (.585, .405)]]
    for ax, c in zip(axes[:2], (2048, 8192)):
        pts_c = [p for p in d["throughput"] if p["context"] == c]
        for arm, label, color, marker, ls in CURVES:
            pts = sorted((p for p in pts_c if p["arm"] == arm), key=lambda p: p["batch"])
            ax.plot([p["batch"] for p in pts], [p["tokens_per_s"] / 1000 for p in pts], color=color, marker=marker,
                    linestyle=ls, linewidth=1, markersize=1.8, label=label)
        bmax = {a: max(p["batch"] for p in pts_c if p["arm"] == a) for a, *_ in CURVES}
        # SketchSSM and ReplaySSM speedup over Standard at B=512 (same batch) and at the capacity endpoint (each arm's own maximum)
        std = {p["batch"]: p["tokens_per_s"] for p in pts_c if p["arm"] == "standard"}
        for arm, dy, color in (("g4", 1, "#B84365"), ("g8", -1, "#B84365"), ("replayssm", -1, "#756BB1")):
            pts = {p["batch"]: p["tokens_per_s"] for p in pts_c if p["arm"] == arm}
            for b, ref in ((512, std.get(512)), (bmax[arm], std[bmax["standard"]])):
                if b in pts and ref:
                    ax.annotate(f"{pts[b] / ref:.2f}\u00d7", (b, pts[b] / 1000), xytext=(0, 3.2 * dy), textcoords="offset points",
                                ha="center", va="bottom" if dy > 0 else "top", fontsize=4.6, color=color)
        # paper style: the limit just above the data, ticks every 10k tokens/s
        top = max(p["tokens_per_s"] for p in pts_c) / 1000
        ymax = math.ceil(top * 1.07); yt = list(range(0, int(ymax) + 1, 10))
        ax.set_xlim(0, max(bmax.values()) * 1.05); ax.set_xticks([1, 256, 512, max(bmax.values())])
        ax.set_ylim(0, ymax); ax.set_yticks(yt)
        ax.set_xlabel("Batch size", fontsize=6.2, labelpad=3); ax.set_title(rf"$C={c // 1024}\mathrm{{K}}$", fontsize=6.7, pad=3)
    axes[0].set_ylabel("Throughput\n(10³ tokens/s)", fontsize=7.3, labelpad=2)
    legend(fig, axes[0], .28)
    ax, L = axes[2], LAYOUT
    contexts = sorted({r["context"] for r in d["breakdown"]})
    rows = {(r["context"], r["arm"]): r for r in d["breakdown"]}
    ticks, labels = [], []
    for i, c in enumerate(contexts):
        off = i * L["pitch"]
        for j, arm in enumerate(L["pos"]):
            ticks.append(off + L["pos"][arm]); labels.append(L["labels"][arm])
            if (c, arm) not in rows:
                continue
            r = rows[c, arm]; bottom = 0
            for name, color in COMPONENTS:
                v = r["components_ms"][name]
                ax.bar(off + L["pos"][arm], v, L["width"], bottom=bottom, color=color, edgecolor="white", linewidth=.25,
                       label=name if i == j == 0 else None)
                bottom += v
        ax.text(off + L["rank_x"], -.30, r"SketchSSM rank $G$", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.4)
        ax.text(off + L["ctx_x"], 1.055, rf"$C={c // 1024}\mathrm{{K}}\quad B={rows[c, 'standard']['batch']}$",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.7)
    ax.set_xticks(ticks, labels, fontsize=5.2); ax.tick_params(axis="x", length=0, pad=3)
    ax.axvline(L["pitch"] - .8, color="#C7C7C7", lw=.6, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(-.7, L["pitch"] + max(L["pos"].values()) + 1.05)
    # paper style: ticks every 20 ms, the limit at the tick above the tallest bar plus headroom
    ymax = 20 * math.ceil(max(r["step_ms"] for r in d["breakdown"]) * 1.1 / 20); yt = list(range(0, ymax + 1, 20))
    ax.set_ylim(0, ymax); ax.set_yticks(yt)
    ax.set_ylabel("Latency (ms/step)", fontsize=7.3, labelpad=2)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles, ["GEMM", "Linear attn.", "Softmax attn.", "Others"], loc="center", bbox_to_anchor=(.783, .915),
               ncol=4, frameon=False, fontsize=7.3, columnspacing=.45, handlelength=.8, handletextpad=.3)
    for a in axes:
        a.tick_params(axis="y", labelsize=7); a.set_axisbelow(True); a.grid(axis="y", color=COLORS["grid"], linewidth=.5)
    fig.text(.2825, .025, "(a) End-to-end throughput", ha="center", fontsize=7)
    fig.text(.7895, .025, "(b) Decode breakdown", ha="center", fontsize=7)
    (ROOT / "figures").mkdir(exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        buf = io.BytesIO()
        meta = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else {}
        fig.savefig(buf, format=suffix, dpi=400, facecolor="white", metadata=meta)
        (ROOT / "figures" / f"super-e2e-speedup.{suffix}").write_bytes(buf.getvalue())
    print("wrote figures/super-e2e-speedup.{pdf,png,svg}")


if __name__ == "__main__":
    main()
