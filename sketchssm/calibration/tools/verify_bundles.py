# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Verify packaged calibration data and reproduce every saved allocation on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sketchssm.calibration.core.allocation import allocate
from sketchssm.calibration.bundles import MissingBundleData, load_precomputed, require_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    catalog = json.loads((args.root / 'catalog.json').read_text())
    total = 0
    for entry in catalog['models']:
        root = args.root / entry['id']
        try:
            require_data(root)
        except MissingBundleData as error:
            raise SystemExit(f'error: {error}')
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            path = root / name
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8 << 20), b''):
                    digest.update(chunk)
            if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
        def load(name):
            return torch.load(root / name, map_location='cpu', weights_only=True)
        basis = load(manifest['basis'])
        curves = load(manifest['curves'])
        tokens = {**load(manifest['generation_tokens']), **load(manifest['allocation_tokens'])}
        for name, shape in [('prompt_token_ids', (520, 512)),
                            ('generated_token_ids', (520, 256)),
                            ('paired_validation_token_ids', (64, 257))]:
            if tuple(tokens[name].shape) != shape:
                raise ValueError(f'Unexpected token shape: {name}')
        for G in entry['ranks']:
            old = load_precomputed(root, G)
            new = allocate(curves, basis, mean_rank=G, value_dim=manifest['V'],
                           window=manifest['W'], erase=manifest['erase'],
                           max_rank=manifest['allocations'][str(G)]['max_rank'])
            for key in ('m_table', 'dense_table', 'omega'):
                if not torch.equal(new[key], old[key]):
                    raise ValueError(f'{entry["id"]} G{G}: {key} differs')
            total += 1
            print(f'{entry["id"]} G{G}: exact match', flush=True)
    print(f'All {total} allocations match; all bundle checksums verified.')


if __name__ == '__main__':
    main()
