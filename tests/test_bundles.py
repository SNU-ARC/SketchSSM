# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Reject changed data or mismatched bases before reusing a bundled table."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from sketchssm.calibration.core.basis import fingerprint
from sketchssm.calibration.bundles import load_precomputed


class BundleTests(unittest.TestCase):
    def test_bundle_integrity_and_exact_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            omega = torch.eye(4)[:2].reshape(1, 1, 2, 4)
            table = dict(m_table=torch.tensor([[0, 1]], dtype=torch.int16),
                         dense_table=torch.tensor([[True, False]]),
                         meta=dict(basis_fingerprint=fingerprint(omega)))
            torch.save(dict(omega=omega), root / 'basis.pt')
            torch.save(table, root / 'g1.pt')
            manifest = dict(schema_version=1, model='test', V=4, W=16, erase=False,
                            basis='basis.pt', basis_fingerprint=fingerprint(omega),
                            allocation_objective='full-gram', allocations={'1': {'file': 'g1.pt'}})
            def write_manifest():
                manifest['files'] = {p.name: dict(bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.glob('*.pt')}
                (root / 'manifest.json').write_text(json.dumps(manifest))
            write_manifest()
            result = load_precomputed(root, 1)
            for key in ('m_table', 'dense_table'):
                self.assertEqual(result[key].dtype, table[key].dtype)
                self.assertTrue(torch.equal(result[key], table[key]))
            self.assertTrue(torch.equal(result['omega'], omega))
            with self.assertRaisesRegex(ValueError, 'not included'):
                load_precomputed(root, 2)
            table['meta']['basis_fingerprint'] = 'wrong'
            torch.save(table, root / 'g1.pt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_precomputed(root, 1)
            write_manifest()
            with self.assertRaisesRegex(ValueError, 'fingerprints'):
                load_precomputed(root, 1)
