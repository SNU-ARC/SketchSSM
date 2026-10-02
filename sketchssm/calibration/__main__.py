# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Portable tensor-file entry points; no model imports or GPU initialization."""
import argparse
import json
import hashlib
from pathlib import Path
import torch
from .core.basis import fit
from .core.scoring import score
from .core.allocation import allocate
from .core.export import export_frames
from .core.reuse import reuse
from .bundles import MissingBundleData, load_precomputed
from . import calibration


def main(prog=None, epilog=None):
    try:
        _main(prog, epilog)
    except MissingBundleData as error:
        raise SystemExit(f'error: {error}')


def _main(prog=None, epilog=None):
    p = argparse.ArgumentParser(prog=prog, description=__doc__, epilog=epilog)
    p.add_argument('--threads', type=int, default=2)
    sub = p.add_subparsers(dest='command', required=True)
    from .collection.pipeline import add_arguments
    add_arguments(sub.add_parser('calibrate', help='Collect tokens, native covariance and gradients, then allocate'))
    fit_parser = sub.add_parser('fit-basis', help='Fit one shared Omega per native group')
    fit_parser.add_argument('--covariance', type=Path, required=True)
    fit_parser.add_argument('--groups', type=int, required=True)
    fit_parser.add_argument('--rank', type=int, required=True)
    fit_parser.add_argument('--ridge', type=float, default=0.1)
    scoring = sub.add_parser('score', help='Compute same-token/head paired rank curves')
    scoring.add_argument('--trace', type=Path, required=True)
    scoring.add_argument('--basis', type=Path, required=True)
    alloc = sub.add_parser('allocate', help='Allocate with fixed rank costs; report BF16 inference traffic separately')
    alloc.add_argument('--curves', type=Path, required=True)
    alloc.add_argument('--basis', type=Path, required=True)
    alloc.add_argument('--mean-rank', type=float, required=True)
    alloc.add_argument('--value-dim', type=int, required=True)
    alloc.add_argument('--window', type=int, default=16)
    alloc.add_argument('--erase', action='store_true', help='Include projected erase-factor GW traffic for GDN/KDA')
    alloc.add_argument('--max-rank', type=int,
                       help='Sketch rank cap; default and maximum: the dense crossover rank')
    export = sub.add_parser('export', help='Export ordered shared frames and per-head tables')
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument('--allocation', type=Path)
    source.add_argument('--calibration', type=Path, help='Portable calibration file; requires --mean-rank')
    export.add_argument('--mean-rank', type=float, help='Mean rank to allocate from --calibration')
    export.add_argument('--window', type=int,
                        help='Serving window W for --calibration (default: the calibrated window)')
    reuse_parser = sub.add_parser('reuse', help='Preserve an existing group basis and allocation unchanged')
    reuse_parser.add_argument('--allocation', type=Path, required=True)
    reuse_parser.add_argument('--value-dim', type=int, required=True)
    reuse_parser.add_argument('--window', type=int, default=16)
    reuse_parser.add_argument('--erase', action='store_true')
    precomputed = sub.add_parser('precomputed', help='Load a bundled allocation with its matching shared basis')
    precomputed.add_argument('--bundle', type=Path, required=True)
    precomputed.add_argument('--mean-rank', type=float, required=True)
    pack = sub.add_parser('package', help='Write one portable calibration file from a bundle')
    pack.add_argument('--bundle', type=Path, required=True)
    man = sub.add_parser('manifest', help='Describe a packaged calibration (hash, base model, parity) for publishing')
    man.add_argument('--bundle', type=Path, required=True)
    man.add_argument('--calibration', type=Path, required=True)
    man.add_argument('--precision', help='Weight precision of the base checkpoint, e.g. bf16 or nvfp4')
    for parser in (fit_parser, scoring, alloc, export, reuse_parser, precomputed, pack, man):
        parser.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.threads < 1:
        p.error('--threads must be positive')
    if a.command == 'export' and (a.calibration is None) != (a.mean_rank is None):
        p.error('--calibration and --mean-rank must be given together')
    if a.command == 'export' and a.window is not None and a.calibration is None:
        p.error('--window applies only to --calibration')
    torch.set_num_threads(a.threads)
    if a.command == 'calibrate':
        from .collection.pipeline import run
        run(a.config, a.out, stage=a.stage, resume=a.resume)
        return
    def load(path):
        return torch.load(path, map_location='cpu', weights_only=True)
    if a.command == 'fit-basis':
        out = fit(load(a.covariance), a.groups, a.rank, a.ridge)
    elif a.command == 'score':
        out = score(load(a.trace), load(a.basis))
    elif a.command == 'allocate':
        out = allocate(load(a.curves), load(a.basis), mean_rank=a.mean_rank,
                       value_dim=a.value_dim, window=a.window, erase=a.erase, max_rank=a.max_rank)
    elif a.command == 'reuse':
        out = reuse(load(a.allocation), value_dim=a.value_dim, window=a.window, erase=a.erase)
        out['meta']['source_sha256'] = hashlib.sha256(a.allocation.read_bytes()).hexdigest()
    elif a.command == 'precomputed':
        out = load_precomputed(a.bundle, a.mean_rank)
    elif a.command == 'manifest':
        out = calibration.manifest(a.bundle, a.calibration, a.precision)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + '\n')
        bad = [p['mean_rank'] for p in out['parity'] if not (p['bundle_table_equal'] and p['export_frames_equal'])]
        print(f'Saved {a.out}' + (f'; parity FAILED for mean ranks {bad}' if bad else ''))
        if bad:
            raise SystemExit(1)
        return
    elif a.command == 'package':
        out = calibration.package(a.bundle)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, a.out)
        summary = {k: out[k] for k in ('format', 'schema_version', 'model', 'geometry', 'max_rank',
                                       'basis_fingerprint', 'verified_mean_ranks', 'verified')}
        summary['bytes'] = a.out.stat().st_size
        a.out.with_suffix(a.out.suffix + '.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Saved {a.out} ({summary["bytes"]} bytes)')
        return
    elif a.calibration is not None:
        out = calibration.frames(calibration.load(a.calibration), a.mean_rank, a.window)
    else:
        out = export_frames(load(a.allocation))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    a.out.with_suffix(a.out.suffix + '.json').write_text(json.dumps(out['meta'], indent=2) + '\n')
    print(f'Saved {a.out}')


if __name__ == '__main__':
    main(prog='python -m sketchssm.calibration')
