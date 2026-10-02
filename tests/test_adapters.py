# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Check recurrence adapters against direct boundary-state updates."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from sketchssm.calibration.adapters import load_adapter


class AdapterTests(unittest.TestCase):
    def test_effective_queries_match_independent_state_updates(self):
        torch.manual_seed(98)
        torch.set_num_threads(2)
        B, T, H, K, V, W = 2, 8, 3, 5, 4, 4
        for family in ('mamba2', 'gdn', 'kda'):
            with self.subTest(family=family):
                adapter = load_adapter(family)
                query = torch.randn(B, T, H, K, dtype=torch.float64)
                decay_shape = query.shape if family == 'kda' else query.shape[:-1]
                decay = torch.rand(decay_shape, dtype=torch.float64) * .5 + .4
                key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1)
                beta = torch.rand(B, T, H, dtype=torch.float64)
                extra = dict(key=key, beta=beta) if adapter.erase else {}
                effective = adapter.effective_queries(query, decay=decay, window=W, **extra)
                for t in range(T):
                    if t % W == 0:
                        initial = torch.randn(B, H, V, K, dtype=torch.float64)
                        state = initial.clone()
                    a = decay[:, t] if family == 'kda' else decay[:, t, :, None]
                    state = state * a[..., None, :]
                    if adapter.erase:
                        memory = (state @ key[:, t, :, :, None])[..., 0]
                        state = state - beta[:, t, :, None, None] * memory[..., None] * key[:, t, :, None, :]
                    direct = state @ query[:, t, :, :, None]
                    reconstructed = initial @ effective[:, t, :, :, None]
                    torch.testing.assert_close(direct, reconstructed, rtol=1e-12, atol=1e-12)

    def test_selection_and_bad_geometry_fail_explicitly(self):
        self.assertEqual(load_adapter('auto', family='gdn').family, 'gdn')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            load_adapter('guess-from-model-name')
        with self.assertRaisesRegex(ValueError, 'Auto requires'):
            load_adapter('auto')
        with self.assertRaisesRegex(ValueError, 'differs'):
            load_adapter('mamba2', family='gdn')
        with self.assertRaisesRegex(ValueError, 'erase=True'):
            load_adapter('kda').validate_geometry(dict(key_dim=128, value_dim=128,
                groups=64, window=16, erase=False))
        q = torch.ones(2, 4, 3, 5)
        with self.assertRaisesRegex(ValueError, 'Decay shape'):
            load_adapter('kda').effective_queries(q, decay=torch.ones(2, 4, 3),
                key=q, beta=torch.ones(2, 4, 3), window=4)
        with self.assertRaisesRegex(ValueError, 'complete windows'):
            load_adapter('mamba2').effective_queries(q, decay=torch.ones(2, 4, 3), window=3)

    def test_external_adapter_requires_no_core_edits(self):
        name = 'test_external_sketch_adapter'
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, name + '.py').write_text(
                'from sketchssm.calibration.adapters.mamba2 import Mamba2Adapter\n'
                'class CustomAdapter(Mamba2Adapter):\n'
                '    family = "custom_scalar"\n')
            sys.path.insert(0, temporary)
            importlib.invalidate_caches()
            try:
                custom = load_adapter(name + ':CustomAdapter', family='custom_scalar')
                q = torch.ones(4, 2, 3)
                result = custom.effective_queries(q, decay=torch.full((4, 2), .5), window=2)
                torch.testing.assert_close(result[:, 0, 0], torch.tensor([.5, .25, .5, .25]))
            finally:
                sys.path.remove(temporary)
                sys.modules.pop(name, None)
