"""nsys sqlite -> per-step kernel inventory (stage 1) and per-phase layer averages (stage 2).

Stage 1 (`extract`): steps = the NVTX ranges `decode/batch_size_<B>` of the measured wave (barrier.py);
kernels are attached to a step through the CUDA runtime launch records (graph launches and eager launches)
issued on the range's thread inside the range. Output: for each step, {kernel short name: [calls, us]}.

Stage 2 (`analyze`): kernels are classed (recurrent components / GEMM / softmax attention / other). The flush
step of each 16-step period is found from the data (offset with the largest flush-component time, checked
over the whole wave). Steady statistics use decode steps 16..111 (six complete windows).
"""
import argparse, bisect, collections, json, re, sqlite3, statistics, sys
from pathlib import Path

# Recurrent (linear-attention) core by component. Public evaluation/linear_attention.py patterns, plus the
# per-step bookkeeping kernels of the SketchSSM / ReplaySSM paths that the public script counts as "other".
COMPONENTS = {
    'readout': [r'^nf_kernel', r'^_sketch_decode_kernel', r'selective_state_update', r'selective_scan_update',
                r'^_replayssm_output_only_kernel', r'gdn_step_kernel', r'^_gdn_sketch_step_kernel',
                r'^fused_recurrent_gated_delta_rule_(packed_decode|replayssm)_kernel', r'kda_step_kernel',
                r'^_kda_sketch_step_kernel', r'^_kda_replayssm_kernel', r'^fused_recurrent_gated_delta_rule_fwd_kernel',
                r'^m2_step_kernel', r'^_?gdn_replay', r'replayssm.*(decode|readout|step)'],
    'flush': [r'^flush_kernel', r'^_sketch_flush_kernel', r'^_cold_build_kernel', r'gdn_flush_warp_kernel',
              r'^_gdn_sketch_flush_kernel', r'^_gdn_build_kernel', r'kda_flush_main', r'^_kda_sketch_flush_kernel',
              r'^flush$', r'replayssm.*flush', r'_flush_kernel$'],
    'finish': [r'kda_flush_finish', r'^_kda_sketch_finish_kernel'],
    'input copy': [],
    'shared B/C': [r'^_bc_pre_kernel', r'^_replayssm_output_only_precompute_kernel', r'replayssm.*precompute'],
    'basis transform': [r'^_rot_inplace_kernel', r'^_rotate_groups_kernel', r'^_rotate_qk_kernel', r'^_window_keys_kernel',
                        r'gdn_sketch_qk', r'_rotate'],
}
COPY, COPY_BEFORE = r'direct_copy_kernel', r'kda_step_kernel'
RECURRENT_HINT = re.compile(r'sketch|replay|flush|gdn|kda|mamba|ssm|selective|delta_rule|rot_|_rot|ring|latch|recurr', re.I)


def component(name):
    for comp, pats in COMPONENTS.items():
        if any(re.search(p, name) for p in pats):
            return comp
    return None


def category(name):
    c = component(name)
    if c:
        return 'Linear attention'
    low = name.lower()
    if 'causal_conv' in low or 'conv1d' in low:
        return 'Others'
    if any(x in low for x in ('attention', 'flash_fwd', 'fmha', 'paged', 'reshape_and_cache', 'append_paged', 'batchdecode',
                              'batchprefill', 'indexer', '_qsa_', 'sparse_attn', 'mla', 'flashinfer::', 'xqa', 'decode_attention',
                              'kv_cache')):
        return 'Softmax attention'
    if any(x in low for x in ('gemm', 'cutlass', 'gemv', 'xmma', 'cublas', 'matmul', 'scaled_mm', 'nvjet', 'splitkreduce', 'bmm_',
                              'moe', 'expert', 'grouped_gemm', 'group_gemm', 'sm100', 'sm90_')) and 'routing' not in low:
        return 'GEMM'
    return 'Others'


def extract(trace, steps_wanted, out, batch=None):
    db = sqlite3.connect(f'file:{trace}?mode=ro', uri=True)
    names = dict(db.execute('SELECT id, value FROM StringIds'))
    ranges = [(s, e, t, txt) for txt, s, e, t in db.execute(
        "SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'decode/batch_size_%' AND end IS NOT NULL ORDER BY start")]
    if batch is not None:
        ranges = [r for r in ranges if r[3] == f'decode/batch_size_{batch}']
    assert len(ranges) >= steps_wanted, ('decode ranges', len(ranges))
    ranges = ranges[-steps_wanted:]
    launches = collections.defaultdict(list)
    for s, cid, tid in db.execute('SELECT start, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start'):
        launches[tid].append((s, cid))
    by_cid = collections.defaultdict(list)
    for s, e, cid, nid, sid in db.execute('SELECT start, end, correlationId, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        by_cid[cid].append((s, e, names[sid], names[nid]))
    steps, full = [], {}
    for i, (s, e, tid, _) in enumerate(ranges):
        lt = launches[tid]
        lo, hi = bisect.bisect_left(lt, (s, -1)), bisect.bisect_left(lt, (e, 1 << 62))
        ks = sorted(k for _, cid in lt[lo:hi] for k in by_cid.get(cid, ()))
        assert ks, ('no kernels in step', i)
        comps = [component(k[2]) for k in ks]
        for j, k in enumerate(ks):
            if re.search(COPY_BEFORE, k[2]):
                j -= 1
                while j >= 0 and comps[j] is None and re.search(COPY, ks[j][2]):
                    comps[j] = 'input copy'; j -= 1
        inv = collections.defaultdict(lambda: [0, 0.0])
        for k, c in zip(ks, comps):
            key = k[2] if c != 'input copy' else '[input copy] ' + k[2]
            inv[key][0] += 1; inv[key][1] += (k[1] - k[0]) / 1e3
            full.setdefault(k[2], k[3][:300])
        steps.append(dict(index=i, range_us=(e - s) / 1e3, span_us=(ks[-1][1] - ks[0][0]) / 1e3, kernels=dict(inv)))
    Path(out).write_text(json.dumps(dict(trace=str(trace), steps=steps, demangled=full)))
    return steps


def analyze(inv_path, layers, out=None, window=16):
    d = json.loads(Path(inv_path).read_text())
    steps = d['steps']
    n = len(steps)
    per = []
    for s in steps:
        comps = collections.defaultdict(float); cats = collections.defaultdict(float)
        for name, (calls, us) in s['kernels'].items():
            c = 'input copy' if name.startswith('[input copy] ') else component(name)
            if c:
                comps[c] += us; cats['Linear attention'] += us
            else:
                cats[category(name)] += us
        per.append(dict(comps=comps, cats=cats, total=sum(cats.values()), span=s['span_us'], range=s['range_us']))
    flush = [p['comps'].get('flush', 0.0) for p in per]
    offset = None
    if max(flush) > 0:
        offset = max(range(window), key=lambda o: statistics.mean(flush[o::window]))
        hi = [f for i, f in enumerate(flush) if i % window == offset]
        lo = [f for i, f in enumerate(flush) if i % window != offset]
        assert min(hi) > 3 * max(lo + [0.0]), ('flush phase not separable', min(hi), max(lo))
    steady = list(range(16, 112)) if n >= 112 else list(range(n))
    sel = {'per_step': steady}
    if offset is not None:
        sel['nonflush'] = [i for i in steady if i % window != offset]
        sel['flush'] = [i for i in steady if i % window == offset]
    parts = {}
    allc = list(COMPONENTS)
    for part, idx in sel.items():
        mean = lambda f: statistics.mean(f(per[i]) for i in idx)
        comps = {c: mean(lambda p: p['comps'].get(c, 0.0)) for c in allc}
        cats = {c: mean(lambda p: p['cats'].get(c, 0.0)) for c in ('GEMM', 'Linear attention', 'Softmax attention', 'Others')}
        parts[part] = dict(n=len(idx), recurrent_us=sum(comps.values()), recurrent_per_layer_us=sum(comps.values()) / layers,
                           components_per_layer_us={c: v / layers for c, v in comps.items()}, categories_us=cats,
                           kernel_sum_us=mean(lambda p: p['total']), span_us=mean(lambda p: p['span']), range_us=mean(lambda p: p['range']))
    if offset is not None:
        parts['window'] = dict(recurrent_per_layer_us=(window - 1) * parts['nonflush']['recurrent_per_layer_us'] + parts['flush']['recurrent_per_layer_us'])
    else:
        parts['window'] = dict(recurrent_per_layer_us=window * parts['per_step']['recurrent_per_layer_us'])
    inv = collections.defaultdict(lambda: [0, 0.0])
    for i in steady:
        for name, (calls, us) in steps[i]['kernels'].items():
            inv[name][0] += calls; inv[name][1] += us
    inventory = {k: dict(calls_per_step=v[0] / len(steady), us_per_step=v[1] / len(steady),
                         component=('input copy' if k.startswith('[input copy] ') else component(k)), category=category(k))
                 for k, v in sorted(inv.items(), key=lambda kv: -kv[1][1])}
    suspicious = {k: v for k, v in inventory.items() if v['component'] is None and RECURRENT_HINT.search(k)}
    res = dict(layers=layers, steps=n, flush_offset=offset, parts=parts, steady_inventory=inventory, unclassified_recurrent_like=suspicious,
               per_step_recurrent_us=[sum(p['comps'].values()) for p in per])
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('extract', 'analyze'))
    p.add_argument('path', type=Path)
    p.add_argument('--steps', type=int, default=127)
    p.add_argument('--batch', type=int)
    p.add_argument('--layers', type=int)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'extract':
        extract(a.path, a.steps, a.out, a.batch)
    else:
        r = analyze(a.path, a.layers, a.out)
        print(json.dumps({k: r[k] for k in ('flush_offset', 'unclassified_recurrent_like')}, indent=1))
        for part, v in r['parts'].items():
            print(part, json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items() if not isinstance(x, dict)}),
                  {k: round(x, 2) for k, x in v.get('components_per_layer_us', {}).items()})
        for k, v in list(r['steady_inventory'].items())[:30]:
            print(f"{v['us_per_step']:10.1f} us {v['calls_per_step']:6.1f}x {str(v['component']):16s} {v['category']:18s} {k[:90]}")
