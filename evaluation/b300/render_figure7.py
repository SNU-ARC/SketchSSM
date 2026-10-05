"""Figure 7 in the paper layout from data/ (no GPU): (a) non-flush and (b) flush latency, (c) window latency."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ('Nemotron\nSuper', '(Mamba-2)', [8, 4], 'data/super-kernel-batches-data.json'),
    ('Qwen3.8\nFlash-Next', '(GDN)', [8, 4], 'data/qwen-kernel-batches-data.json'),
    ('GLM 5.3\nFlash', '(KDA)', [8, 4], 'data/glm-kernel-batches-data.json'),
]
COLORS = {'standard': '#4c4c4c', 'replayssm': '#786bb5', 'control': '#e6bdca', 'sketch': '#bd4068'}
WINDOW = 16

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.6,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 3, figsize=(9.94, 5.6))
    fig.subplots_adjust(left=.185, right=.978, bottom=.072, top=.82,
                        wspace=.43, hspace=1.05)
    legend = [Patch(facecolor=COLORS[k], label=label) for k, label in
              [('standard', 'Standard'), ('replayssm', 'ReplaySSM'), ('sketch', r'SketchSSM (numbers: rank $\bar G$)')]]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.57, .998), ncol=3,
               frameon=False, fontsize=10, handlelength=1.2, columnspacing=1.4)
    fig.legend(handles=[Patch(facecolor=COLORS['control'], label='SketchSSM w/o sketch'),
                        Patch(facecolor=COLORS['sketch'], edgecolor='white', hatch='////', label='Sketch/Coefficient Map overhead')],
               loc='upper center', bbox_to_anchor=(.57, .95), ncol=2, frameon=False,
               fontsize=9, handlelength=1.2)
    titles = [r'(a) Non-flush ($W-1$ steps)', '(b) Flush (1 step)', r'(c) Total window ($W$ steps)']
    for ri, (name, family, ranks, source) in enumerate(SOURCES):
        data = json.loads((ROOT / source).read_text())
        rows = {(r['batch'], r['phase'], r['arm']): r for r in data['rows'] if 'annotation' not in r}
        sketch_arms = [f'g{g}' for g in ranks]
        def value(batch, phase, arm):
            if arm == 'standard' and phase != 'window':
                # Standard performs the same full-state read/update every step.
                # Its archived window latency includes exactly W such steps.
                step_ms = rows[batch, 'window', arm]['ms_per_layer'] / WINDOW
                return step_ms * 1000 if phase == 'nonflush' else step_ms
            r = rows[batch, phase, arm]
            return r['ms_per_layer'] if phase == 'window' else r['us_per_layer'] / (1000 if phase == 'flush' else 1)
        for ci, phase in enumerate(['nonflush', 'flush', 'window']):
            ax = axes[ri, ci]
            arms = (['standard', 'replayssm'] if phase == 'window' else
                    ['standard', 'replayssm', 'control'] if phase == 'flush' else ['standard', 'replayssm']) + sketch_arms
            pitch = 1.35 * len(arms) + 1.6
            ticks, labels, peak = [], [], 0
            for bi, batch in enumerate([128, 256, 512]):
                off = bi * pitch
                for ai, arm in enumerate(arms):
                    x = off + 1.35 * ai
                    v = value(batch, phase, arm); peak = max(peak, v)
                    ax.bar(x, v, .8, color=COLORS['sketch' if arm.startswith('g') else arm], linewidth=0)
                    ticks.append(x); labels.append(arm[1:] if arm.startswith('g') else '')
                    if phase == 'flush' and arm.startswith('g'):
                        base = value(batch, phase, 'control')
                        ax.bar(x, max(0, v-base), .8, bottom=base, facecolor='none', edgecolor='white', hatch='////', linewidth=0)
                ax.text(off + 1.35 * (len(arms)-1)/2, 1.09, rf'$B={batch}$', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=10)
                if bi < 2:
                    ax.axvline(off + 1.35 * len(arms) + .3, color='.8', ls=(0,(2,2)), lw=.6)
            off = 2 * pitch
            ref = 'control' if phase == 'flush' else 'standard'
            base = value(512, phase, ref)
            ax.hlines(base, off + 1.35 * arms.index(ref)+.4, off + 1.35 * (len(arms)-1)+.2, color='black', ls=(0,(3,2)), lw=.7)
            targets = ['replayssm', sketch_arms[-1]] if phase == 'window' else [sketch_arms[-1]] if phase == 'nonflush' else []
            for arm in targets:
                x = off + 1.35 * arms.index(arm); v = value(512, phase, arm)
                ax.annotate('', xy=(x,v), xytext=(x,base), arrowprops=dict(arrowstyle='->', lw=.8, shrinkA=0, shrinkB=0))
                ax.text(x+.35, base*(.65 if arm=='replayssm' else .36), f'{base/v:.2f}×', fontsize=8.9,
                        rotation=60, va='center')
            ax.set_xlim(-.8, off+1.35*(len(arms)-1)+1.4)
            ax.set_ylim(0, peak*1.17)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1,2,2.5,5,10]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            ax.set_xticks(ticks, labels, fontsize=8)
            ax.tick_params(axis='x', length=0, pad=3)
            ax.tick_params(axis='y', labelsize=9.6, pad=2)
            ax.set_ylabel('Latency (μs/layer)' if phase=='nonflush' else 'Latency (ms/layer)', fontsize=9, labelpad=2)
            ax.set_axisbelow(True); ax.grid(axis='y', color='.87', lw=.6)
            ax.set_xlabel(titles[ci], fontsize=9.4, labelpad=6)
        pos = axes[ri, 0].get_position(); cy = pos.y0+pos.height/2
        fig.text(.082, cy+.013, name, ha='center', va='center', fontsize=10.5, weight='bold')
        fig.text(.082, cy-.057, family, ha='center', va='center', fontsize=10)
    for ext in ['pdf','png','svg']:
        fig.savefig(ROOT / f'figures/combined-kernel-speedup.{ext}', dpi=250, facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()
