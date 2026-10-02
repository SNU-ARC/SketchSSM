# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Rebuild allocations from a collected model directory into the standard layout.

This entry point consumes saved statistics. Model download and collection are
separate stages and are not performed by this command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
import yaml
from .core.allocation import allocate
from .core.basis import fingerprint
from .adapters import load_adapter
from .bundles import MissingBundleData, require_data


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(source, output, mean_ranks=None, adapter=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be separate directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty; existing calibration data will not be overwritten')
    retained = ('basis', 'curves', 'covariance', 'generation_tokens', 'allocation_tokens')
    require_data(source, retained)
    manifest = json.loads((source / 'manifest.json').read_text())
    config = yaml.safe_load((source / 'config.yaml').read_text())
    if manifest['schema_version'] != 1 or config['schema_version'] != 1:
        raise ValueError('Unsupported calibration schema')
    if config['paired']['objective'] != 'full-gram':
        raise ValueError('Allocation requires Full-Gram paired scores')
    geometry = config['geometry']
    declared_family = config.get('model', {}).get('family', manifest['family'])
    if declared_family != manifest['family']:
        raise ValueError('Configured model family differs from collected data')
    selection = adapter if adapter is not None else config.get('adapter', 'auto')
    selected = load_adapter(selection, family=declared_family)
    selected.validate_geometry(geometry)
    config['adapter'] = selected.family if selection in (None, 'auto') else selection
    for cfg_key, manifest_key in [('key_dim', 'K'), ('value_dim', 'V'),
                                  ('window', 'W'), ('groups', 'groups'), ('erase', 'erase')]:
        if geometry[cfg_key] != manifest[manifest_key]:
            raise ValueError(f'Geometry mismatch: {cfg_key}')
    ranks = list(config['allocation']['mean_ranks'] if mean_ranks is None else mean_ranks)
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError('Provide a nonempty list of distinct mean ranks')
    config['allocation']['mean_ranks'] = ranks
    for key in retained:
        name = manifest[key]
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Input paths must remain inside the source directory')
        info = manifest['files'][name]
        if path.stat().st_size != info['bytes'] or checksum(path) != info['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    basis = torch.load(source / manifest['basis'], map_location='cpu', weights_only=True)
    curves = torch.load(source / manifest['curves'], map_location='cpu', weights_only=True)
    if fingerprint(basis['omega']) != manifest['basis_fingerprint']:
        raise ValueError('Manifest basis fingerprint differs from collected basis')
    if basis['omega'].shape[1] != geometry['groups'] or basis['omega'].shape[-1] != geometry['key_dim']:
        raise ValueError('Basis geometry differs from the configuration')
    # Solve before creating the output, so invalid rank budgets leave no partial bundle.
    tables = {format(float(G), 'g'): allocate(curves, basis, mean_rank=G,
        value_dim=geometry['value_dim'], window=geometry['window'], erase=geometry['erase'],
        max_rank=config['allocation'].get('max_rank')) for G in ranks}
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for key in retained:
        name = manifest[key]
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        files[name] = manifest['files'][name]
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    allocations = {}
    (output / 'allocations').mkdir(exist_ok=True)
    for G, table in tables.items():
        del table['omega']  # Stored once in basis/omega.pt.
        name = f'allocations/g{G}.pt'
        torch.save(table, output / name)
        files[name] = dict(bytes=(output / name).stat().st_size, sha256=checksum(output / name))
        allocations[G] = dict(file=name, dense_heads=table['meta']['dense_heads'],
                              max_rank=table['meta']['max_sketch_rank'])
    result = {key: manifest[key] for key in ('schema_version', 'model', 'family',
              'K', 'V', 'W', 'groups', 'erase', 'basis_fingerprint', *retained)}
    result.update(allocations=allocations, files=files, allocation_objective='full-gram',
                  allocation_policy='fixed_rank_cost', inference_pivots=4,
                  source_manifest_sha256=checksum(source / 'manifest.json'),
                  collection_reused=True, adapter=config['adapter'])
    # A manifest is written last and marks a complete, loadable output.
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mean-ranks', type=float, nargs='+')
    parser.add_argument('--adapter', help='mamba2, gdn, kda, auto or importable module:AdapterClass')
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        result = rebuild(args.source, args.out, args.mean_ranks, adapter=args.adapter)
    except MissingBundleData as error:
        parser.exit(1, f'error: {error}\n')
    print(f'Saved {len(result["allocations"])} allocations to {args.out}')


if __name__ == '__main__':
    main()
