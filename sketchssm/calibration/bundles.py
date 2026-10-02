# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Load a precomputed table with its matching shared basis."""
import hashlib
import json
from pathlib import Path
import torch
from .core.basis import fingerprint
from .core.reuse import reuse

EXAMPLES = Path(__file__).resolve().parent / 'example'


class MissingBundleData(FileNotFoundError):
    """A bundle's tensor files are not present on disk."""


def missing_files(directory, keys=None):
    """Manifest files absent from a bundle directory (``manifest.json`` itself if it is absent).

    ``keys`` restricts the check to the files named by those manifest entries.
    """
    directory = Path(directory)
    manifest = directory / 'manifest.json'
    if not manifest.is_file():
        return ['manifest.json']
    manifest = json.loads(manifest.read_text())
    names = manifest['files'] if keys is None else [manifest[key] for key in keys]
    return [name for name in names if not (directory / name).is_file()]


def require_data(directory, keys=None):
    """Raise a clear error when a bundle's tensor files are missing."""
    missing = missing_files(directory, keys)
    if not missing:
        return
    directory = Path(directory)
    name = directory.resolve().name
    model = name if (EXAMPLES / name / 'collect.yaml').is_file() else '<model>'
    raise MissingBundleData(
        f'{len(missing)} calibration file(s) are missing from {directory} (first: {missing[0]}). '
        'The bundle data of the provided calibrations is not distributed with this repository. '
        'Regenerate it with: python -m sketchssm.calibration calibrate '
        f'--config sketchssm/calibration/example/{model}/collect.yaml --out <new-directory>, '
        'then pass that directory instead. To serve or export a provided model, use its '
        'calibration.pt from the Hugging Face Hub (see sketchssm/calibration/example/README.md).')


def load_precomputed(directory, mean_rank):
    require_data(directory)
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported bundle schema')
    key = format(float(mean_rank), 'g')
    if key not in manifest['allocations']:
        raise ValueError(f'Rank {key} is not included in this bundle')

    def load(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Bundle paths must stay inside the bundle directory')
        expected = manifest['files'][name]
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                digest.update(chunk)
        if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise ValueError(f'Bundle checksum mismatch: {name}')
        return torch.load(path, map_location='cpu', weights_only=True)

    basis = load(manifest['basis'])
    table = load(manifest['allocations'][key]['file'])
    actual = fingerprint(basis['omega'])
    if actual != manifest['basis_fingerprint'] or actual != table['meta']['basis_fingerprint']:
        raise ValueError('Allocation and basis fingerprints differ')
    result = reuse(dict(table, omega=basis['omega']), value_dim=manifest['V'],
                   window=manifest['W'], erase=manifest['erase'])
    result['meta'].update(model=manifest['model'], mean_rank_budget=float(mean_rank),
                          source_cost=table['meta'].get('source_cost', {}),
                          bundle_allocation_objective=manifest['allocation_objective'])
    if 'layer_ids' in basis:
        result['layer_ids'] = basis['layer_ids']
    return result
