# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Portable calibration file: one per model, any mean rank selected at load time.

The file stores exactly what `allocate` and `export_frames` consume: the shared
group basis, the Full-Gram paired score sums, geometry and the rank cap. Tables
and frames for a mean rank are derived with those unchanged functions.
"""
import hashlib
import json
import logging
import math
from pathlib import Path
import torch
from .core.allocation import allocate
from .core.basis import fingerprint
from .core.export import export_frames
from .core.traffic import AllocationCost

logger = logging.getLogger(__name__)

FORMAT = 'sketchssm-calibration'
SCHEMA_VERSION = 1
GEOMETRY = ('key_dim', 'value_dim', 'groups', 'window', 'erase')


def rank_key(mean_rank):
    return format(float(mean_rank), 'g')


def table_digest(m_table, dense_table):
    """Identity of a rank/dense table, independent of its container file."""
    h = hashlib.sha256(str(tuple(m_table.shape)).encode())
    h.update(m_table.to(torch.int16).contiguous().numpy().tobytes())
    h.update(dense_table.to(torch.bool).contiguous().numpy().tobytes())
    return h.hexdigest()


def _plain(value):
    """Keep only weights_only-loadable metadata values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported metadata value: {type(value).__name__}')


def _crossover(geometry, window=None):
    """Dense crossover rank (the default rank cap) of a geometry at a window."""
    W = geometry['window'] if window is None else window
    return AllocationCost(geometry['key_dim'], geometry['value_dim'], W, geometry['erase']).crossover


def package(bundle):
    """Build a calibration from a bundle directory and verify every bundle table.

    Each configured mean rank is re-allocated from the packaged contents and must
    reproduce the bundle's rank and dense tables exactly; otherwise this fails.
    """
    from .bundles import load_precomputed, require_data
    import yaml
    require_data(bundle)
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    geometry = {k: config['geometry'][k] for k in GEOMETRY}
    for key, short in (('key_dim', 'K'), ('value_dim', 'V'), ('window', 'W'),
                       ('groups', 'groups'), ('erase', 'erase')):
        if geometry[key] != manifest[short]:
            raise ValueError(f'Geometry mismatch between config and manifest: {key}')
    basis = torch.load(bundle / manifest['basis'], map_location='cpu', weights_only=True)
    curves = torch.load(bundle / manifest['curves'], map_location='cpu', weights_only=True)
    omega = basis['omega'].detach().cpu().contiguous()
    identity = fingerprint(omega)
    if identity != manifest['basis_fingerprint']:
        raise ValueError('Basis fingerprint differs from the bundle manifest')
    model = config.get('model', {})
    # The cap at the calibrated window; omitted in the config, it is the dense crossover.
    cap = config['allocation'].get('max_rank')
    cap = _crossover(geometry) if cap is None else cap
    out = dict(
        format=FORMAT, schema_version=SCHEMA_VERSION,
        model=dict(name=str(model.get('name', manifest['model'])),
                   family=str(model.get('family', manifest['family']))),
        geometry=geometry, max_rank=int(cap),
        omega=omega, basis_fingerprint=identity,
        basis_meta=_plain(basis.get('meta', {})),
        curves=dict(joint_dot_sq_sum=curves['joint_dot_sq_sum'].detach().cpu().contiguous(),
                    joint_nstep=int(curves['joint_nstep']), meta=_plain(curves['meta'])),
        verified_mean_ranks=[], verified={})
    if 'layer_ids' in basis:
        out['layer_ids'] = [int(i) for i in basis['layer_ids']]
    for G in config['allocation']['mean_ranks']:
        key = rank_key(G)
        reference = load_precomputed(bundle, G)
        derived = select(out, G)
        for name in ('m_table', 'dense_table'):
            if derived[name].dtype != reference[name].dtype or not torch.equal(derived[name], reference[name]):
                raise ValueError(f'Mean rank {key}: derived {name} differs from the bundle table')
        out['verified_mean_ranks'].append(G)
        out['verified'][key] = dict(
            dense_heads=int(reference['dense_table'].sum()),
            sketch_heads=int((~reference['dense_table']).sum()),
            table_sha256=table_digest(reference['m_table'], reference['dense_table']))
    validate(out)
    return out


def validate(calibration):
    """Check structure and internal consistency; return the calibration."""
    c = calibration
    if c.get('format') != FORMAT:
        raise ValueError(f'Not a {FORMAT} file')
    if c.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Unsupported calibration schema: {c.get("schema_version")}')
    g = c['geometry']
    omega = c['omega']
    if omega.ndim != 4 or omega.shape[1] != g['groups'] or omega.shape[-1] != g['key_dim']:
        raise ValueError('Basis shape differs from the declared geometry')
    if fingerprint(omega) != c['basis_fingerprint']:
        raise ValueError('Basis fingerprint differs from its contents')
    if c['curves']['meta'].get('basis_fingerprint') != c['basis_fingerprint']:
        raise ValueError('Score curves were measured for a different basis')
    error = c['curves']['joint_dot_sq_sum']
    if error.ndim != 3 or error.shape[0] != omega.shape[0] or error.shape[1] % g['groups']:
        raise ValueError('Score curve shape differs from the basis')
    if 'layer_ids' in c and len(c['layer_ids']) != omega.shape[0]:
        raise ValueError('layer_ids length differs from the layer count')
    return c


def load(path):
    return validate(torch.load(path, map_location='cpu', weights_only=True))


def max_rank(calibration, window=None):
    """Rank cap for allocating at a serving window (default: the calibrated one).

    At the calibrated window this is the stored ``max_rank``. At another window
    it is the dense crossover rank there, unless the stored cap is below the
    calibrated crossover (an explicitly pinned cap), which then still applies.
    The cap never exceeds the calibrated basis rank or score curves.
    """
    c = calibration
    g = c['geometry']
    if window is None or window == g['window']:
        return c['max_rank']
    cap = _crossover(g, window)
    if c['max_rank'] < _crossover(g):
        cap = min(cap, c['max_rank'])
    return min(cap, c['omega'].shape[-2], c['curves']['joint_dot_sq_sum'].shape[-1] - 1)


def select(calibration, mean_rank, window=None):
    """Allocate one mean rank from a calibration; returns an `allocate` result.

    ``window`` is the serving window W (default: the calibrated window). The
    rank cost of erase families (GDN/KDA) is K+V+W, so the allocation depends
    on W; Mamba-2's does not. Verified table digests apply only when the
    allocation problem equals the calibrated one (same rank cost and cap).
    """
    c = calibration
    g = c['geometry']
    if not math.isfinite(float(mean_rank)):
        raise ValueError('Mean rank must be finite')
    W = g['window'] if window is None else window
    if not isinstance(W, int) or isinstance(W, bool) or W < 2:
        raise ValueError('The window must be an integer of at least 2')
    cap = max_rank(c, W)
    basis = dict(omega=c['omega'])
    if 'layer_ids' in c:
        basis['layer_ids'] = c['layer_ids']
    result = allocate(c['curves'], basis, mean_rank=mean_rank, value_dim=g['value_dim'],
                      window=W, erase=g['erase'], max_rank=cap)
    key = rank_key(mean_rank)
    expected = c.get('verified', {}).get(key)
    calibrated = (cap == c['max_rank'] and AllocationCost(g['key_dim'], g['value_dim'], W, g['erase']).rank
                  == AllocationCost(g['key_dim'], g['value_dim'], g['window'], g['erase']).rank)
    if expected is not None and not calibrated:
        logger.warning('Mean rank %s at window %d: the table is unverified; verified tables apply '
                       'to the calibrated window %d.', key, W, g['window'])
        expected = None
    if expected is not None and table_digest(result['m_table'], result['dense_table']) != expected['table_sha256']:
        raise RuntimeError(f'Mean rank {key} does not reproduce its verified table')
    result['meta'].update(model=c['model'], verified=expected is not None, window=W,
                          calibrated_window=g['window'])
    return result


def frames(calibration, mean_rank, window=None):
    """Export ordered frames and per-head tables for one mean rank at a serving window."""
    return export_frames(select(calibration, mean_rank, window))


def manifest(bundle, calibration_path, precision=None):
    """Describe a packaged calibration for publishing next to it.

    Records the file hash, the base checkpoint from the bundle config, and for
    each verified mean rank whether the derived tables equal the bundle tables
    and the exported frames equal the `precomputed` + `export` path.
    """
    from .bundles import load_precomputed
    import yaml
    bundle = Path(bundle).resolve()
    calibration_path = Path(calibration_path)
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    c = load(calibration_path)
    model = config.get('model', {})
    out = dict(format=FORMAT, schema_version=SCHEMA_VERSION, model=c['model'],
               base_model=model.get('checkpoint'), base_model_revision=model.get('revision'),
               repository='https://github.com/SNU-ARC/SketchSSM')
    if precision is not None:
        out['base_model_precision'] = precision
    out['files'] = {calibration_path.name: dict(
        bytes=calibration_path.stat().st_size,
        sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest())}
    out.update(geometry=c['geometry'], max_rank=c['max_rank'],
               layers=int(c['omega'].shape[0]), state_heads=int(c['curves']['joint_dot_sq_sum'].shape[1]),
               omega_shape=list(c['omega'].shape), basis_fingerprint=c['basis_fingerprint'],
               verified_mean_ranks=c['verified_mean_ranks'], parity=[])
    for G in c['verified_mean_ranks']:
        derived = select(c, G)
        reference = load_precomputed(bundle, G)
        table_equal = all(derived[k].dtype == reference[k].dtype and torch.equal(derived[k], reference[k])
                          for k in ('m_table', 'dense_table'))
        ours, theirs = export_frames(derived), export_frames(reference)
        keys = ['frames', 'm_table', 'dense_table'] + (['layer_ids'] if 'layer_ids' in ours else [])
        frames_equal = all(
            torch.equal(torch.as_tensor(ours[k]), torch.as_tensor(theirs[k])) for k in keys)
        out['parity'].append(dict(mean_rank=G, dense_heads=int(derived['dense_table'].sum()),
                                  sketch_heads=int((~derived['dense_table']).sum()),
                                  bundle_table_equal=table_equal, export_frames_equal=frames_equal))
    out['parity_definition'] = (
        'bundle_table_equal: m_table and dense_table derived from the calibration equal the '
        'bundle allocations (including dtype). export_frames_equal: frames, m_table, '
        'dense_table and layer_ids exported from the calibration equal the export of the '
        'bundle allocation.')
    return out
