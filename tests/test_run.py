# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Saved-statistics runs preserve inputs and produce reloadable model directories."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
import yaml
from sketchssm.calibration.core.basis import fingerprint
from sketchssm.calibration.bundles import load_precomputed
from sketchssm.calibration.run import checksum, rebuild


class RunTests(unittest.TestCase):
    def test_rebuild_preserves_sources_and_reloads_allocation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            omega = torch.eye(4)[:2].reshape(1, 1, 2, 4)
            fp = fingerprint(omega)
            payloads = {
                'basis/omega.pt': dict(omega=omega),
                'statistics/paired_scores.pt': dict(joint_nstep=1,
                    joint_dot_sq_sum=torch.tensor([[[100., 100., 100.], [1., .1, 0.]]]),
                    meta=dict(coefficient_model='full-gram', basis_fingerprint=fp)),
                'statistics/covariance.pt': {},
                'data/generation_tokens.pt': {},
                'data/allocation_tokens.pt': {},
            }
            for name, data in payloads.items():
                (source / name).parent.mkdir(exist_ok=True)
                torch.save(data, source / name)
            manifest = dict(schema_version=1, model='tiny', family='mamba2',
                K=4, V=4, W=16, groups=1, erase=False, basis_fingerprint=fp,
                basis='basis/omega.pt', curves='statistics/paired_scores.pt',
                covariance='statistics/covariance.pt',
                generation_tokens='data/generation_tokens.pt', allocation_tokens='data/allocation_tokens.pt',
                files={n: dict(bytes=(source / n).stat().st_size, sha256=checksum(source / n)) for n in payloads})
            (source / 'manifest.json').write_text(json.dumps(manifest))
            config = dict(schema_version=1, geometry=dict(key_dim=4, value_dim=4,
                window=16, groups=1, erase=False), paired=dict(objective='full-gram'),
                allocation=dict(mean_ranks=[1.5], max_rank=2))
            (source / 'config.yaml').write_text(yaml.safe_dump(config))
            hashes = {p.relative_to(source): checksum(p) for p in source.rglob('*') if p.is_file()}
            result = rebuild(source, output)
            self.assertEqual(list(result['allocations']), ['1.5'])
            self.assertEqual(result['adapter'], 'mamba2')
            loaded = load_precomputed(output, 1.5)
            self.assertEqual(loaded['m_table'].tolist(), [[0, 1]])
            for name, digest in hashes.items():
                self.assertEqual(checksum(source / name), digest)
            for name in payloads:
                self.assertEqual(checksum(output / name), checksum(source / name))
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                rebuild(source, output)
            with self.assertRaisesRegex(ValueError, 'separate'):
                rebuild(source, source / 'nested_output')
            with self.assertRaisesRegex(ValueError, 'differs'):
                rebuild(source, root / 'wrong_adapter', adapter='gdn')
            self.assertFalse((root / 'wrong_adapter').exists())
            (source / 'data/generation_tokens.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                rebuild(source, root / 'bad')
            self.assertFalse((root / 'bad').exists())
