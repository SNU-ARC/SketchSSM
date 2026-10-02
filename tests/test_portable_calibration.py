# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""The portable calibration file reproduces bundle tables and exported frames."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
import yaml
from sketchssm.calibration import calibration
from sketchssm.calibration.bundles import MissingBundleData, load_precomputed, missing_files
from sketchssm.calibration.core.basis import fingerprint
from sketchssm.calibration.core.export import export_frames
from sketchssm.calibration.run import checksum, rebuild

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'sketchssm' / 'calibration' / 'example'
BUNDLES = ('nemotron_nano', 'nemotron_super', 'qwen_flash_next', 'glm_flash')
MEAN_RANKS = [1.5, 2.5]


def make_bundle(root):
    """Collected-statistics source with two layers and two groups, rebuilt into a bundle."""
    generator = torch.Generator().manual_seed(5)
    source, bundle = root / 'source', root / 'bundle'
    omega = torch.linalg.qr(torch.randn(2, 2, 6, 6, generator=generator)).Q[..., :4, :].contiguous()
    fp = fingerprint(omega)
    steps = torch.rand(2, 4, 5, generator=generator, dtype=torch.float64)
    curves = steps.flip(-1).cumsum(-1).flip(-1) * 10  # Nonincreasing in rank.
    payloads = {
        'basis/omega.pt': dict(omega=omega, layer_ids=[1, 3], meta=dict(groups=2)),
        'statistics/paired_scores.pt': dict(joint_nstep=4, joint_dot_sq_sum=curves,
            grad_sq_sum=torch.ones(2, 4, dtype=torch.float64),
            meta=dict(coefficient_model='full-gram', basis_fingerprint=fp)),
        'statistics/covariance.pt': dict(head_scov=torch.ones(1)),
        'data/generation_tokens.pt': dict(tokens=torch.arange(3)),
        'data/allocation_tokens.pt': dict(tokens=torch.arange(3)),
    }
    for name, data in payloads.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, source / name)
    manifest = dict(schema_version=1, model='tiny', family='mamba2',
        K=6, V=4, W=16, groups=2, erase=False, basis_fingerprint=fp,
        basis='basis/omega.pt', curves='statistics/paired_scores.pt',
        covariance='statistics/covariance.pt',
        generation_tokens='data/generation_tokens.pt', allocation_tokens='data/allocation_tokens.pt',
        files={n: dict(bytes=(source / n).stat().st_size, sha256=checksum(source / n)) for n in payloads})
    (source / 'manifest.json').write_text(json.dumps(manifest))
    config = dict(schema_version=1, model=dict(name='Tiny', family='mamba2'),
        geometry=dict(key_dim=6, value_dim=4, window=16, groups=2, erase=False),
        paired=dict(objective='full-gram'), allocation=dict(mean_ranks=MEAN_RANKS, max_rank=2))
    (source / 'config.yaml').write_text(yaml.safe_dump(config))
    rebuild(source, bundle)
    return bundle


def assert_same_export(test, actual, expected):
    for key in ('frames', 'm_table', 'dense_table'):
        test.assertEqual(actual[key].dtype, expected[key].dtype, key)
        test.assertTrue(torch.equal(actual[key], expected[key]), key)
    test.assertEqual(actual.get('layer_ids'), expected.get('layer_ids'))


class PortableCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_package_contents_and_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_bundle(root)
            packaged = calibration.package(bundle)
            torch.save(packaged, root / 'calibration.pt')
            loaded = calibration.load(root / 'calibration.pt')
            self.assertEqual(loaded['format'], 'sketchssm-calibration')
            self.assertEqual(loaded['model'], dict(name='Tiny', family='mamba2'))
            self.assertEqual(loaded['geometry'], dict(key_dim=6, value_dim=4, groups=2, window=16, erase=False))
            self.assertEqual(loaded['max_rank'], 2)
            self.assertEqual(loaded['layer_ids'], [1, 3])
            self.assertEqual(loaded['verified_mean_ranks'], MEAN_RANKS)
            self.assertEqual(set(loaded['curves']), {'joint_dot_sq_sum', 'joint_nstep', 'meta'})
            basis = torch.load(bundle / 'basis/omega.pt', weights_only=True)
            self.assertEqual(loaded['omega'].dtype, basis['omega'].dtype)
            self.assertTrue(torch.equal(loaded['omega'], basis['omega']))
            for G in MEAN_RANKS:
                table = torch.load(bundle / f'allocations/g{G:g}.pt', weights_only=True)
                derived = calibration.select(loaded, G)
                for key in ('m_table', 'dense_table'):
                    self.assertEqual(derived[key].dtype, table[key].dtype)
                    self.assertTrue(torch.equal(derived[key], table[key]))
                self.assertTrue(derived['meta']['verified'])
                assert_same_export(self, calibration.frames(loaded, G),
                                   export_frames(load_precomputed(bundle, G)))
            self.assertFalse(calibration.select(loaded, 2)['meta']['verified'])
            described = calibration.manifest(bundle, root / 'calibration.pt', 'bf16')
            self.assertEqual(described['files']['calibration.pt']['bytes'],
                             (root / 'calibration.pt').stat().st_size)
            self.assertEqual([p['mean_rank'] for p in described['parity']], MEAN_RANKS)
            self.assertTrue(all(p['bundle_table_equal'] and p['export_frames_equal']
                                for p in described['parity']))

    def test_rejects_inconsistent_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = make_bundle(Path(temporary))
            packaged = calibration.package(bundle)
            with self.assertRaisesRegex(ValueError, 'Not a'):
                calibration.validate(dict(packaged, format='other'))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                calibration.validate(dict(packaged, omega=packaged['omega'] * 2))
            changed = dict(packaged['curves'], joint_dot_sq_sum=packaged['curves']['joint_dot_sq_sum'].flip(1))
            with self.assertRaisesRegex(RuntimeError, 'verified table'):
                for G in MEAN_RANKS:
                    calibration.select(dict(packaged, curves=changed), G)
            table = torch.load(bundle / 'allocations/g1.5.pt', weights_only=True)
            table['m_table'] = table['m_table'].flip(1)
            table['dense_table'] = table['m_table'] == 0
            torch.save(table, bundle / 'allocations/g1.5.pt')
            manifest = json.loads((bundle / 'manifest.json').read_text())
            manifest['files']['allocations/g1.5.pt'] = dict(
                bytes=(bundle / 'allocations/g1.5.pt').stat().st_size,
                sha256=checksum(bundle / 'allocations/g1.5.pt'))
            (bundle / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'differs from the bundle table'):
                calibration.package(bundle)

    def test_missing_data_names_the_regeneration_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_bundle(root)
            (bundle / 'allocations/g2.5.pt').unlink()
            for call in (lambda: calibration.package(bundle), lambda: load_precomputed(bundle, 1.5)):
                with self.assertRaisesRegex(MissingBundleData, r'not distributed.*sketchssm.calibration calibrate'):
                    call()
            (root / 'source' / 'statistics/covariance.pt').unlink()
            with self.assertRaisesRegex(MissingBundleData, 'statistics/covariance.pt'):
                rebuild(root / 'source', root / 'rebuilt')
            result = subprocess.run(
                [sys.executable, '-m', 'sketchssm.calibration', 'precomputed', '--bundle', str(bundle),
                 '--mean-rank', '1.5', '--out', str(root / 'table.pt')],
                cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('collect.yaml', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_command_line_matches_existing_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_bundle(root)
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            def run(*args):
                subprocess.run([sys.executable, '-m', 'sketchssm.calibration', *map(str, args)],
                               check=True, cwd=ROOT, env=env, capture_output=True)
            run('package', '--bundle', bundle, '--out', root / 'calibration.pt')
            summary = json.loads((root / 'calibration.pt.json').read_text())
            self.assertEqual(summary['verified_mean_ranks'], MEAN_RANKS)
            run('export', '--calibration', root / 'calibration.pt', '--mean-rank', 2.5, '--out', root / 'new.pt')
            run('precomputed', '--bundle', bundle, '--mean-rank', 2.5, '--out', root / 'table.pt')
            run('export', '--allocation', root / 'table.pt', '--out', root / 'old.pt')
            assert_same_export(self, torch.load(root / 'new.pt', weights_only=True),
                               torch.load(root / 'old.pt', weights_only=True))


@unittest.skipIf(any(missing_files(EXAMPLES / b) for b in BUNDLES),
                 'calibration bundle data is not present (it is not distributed; see '
                 'sketchssm/calibration/example/README.md to regenerate it)')
def make_erase_calibration(window=16):
    """GDN-like portable calibration (erase, K=V=16) with verified tables at its window."""
    generator = torch.Generator().manual_seed(11)
    omega = torch.linalg.qr(torch.randn(2, 2, 16, 16, generator=generator)).Q[..., :8, :].contiguous()
    fp = fingerprint(omega)
    steps = torch.rand(2, 4, 9, generator=generator, dtype=torch.float64)
    geometry = dict(key_dim=16, value_dim=16, groups=2, window=window, erase=True)
    c = dict(format=calibration.FORMAT, schema_version=calibration.SCHEMA_VERSION,
             model=dict(name='tiny-gdn', family='gdn'), geometry=geometry,
             max_rank=calibration._crossover(geometry), omega=omega, basis_fingerprint=fp,
             curves=dict(joint_dot_sq_sum=steps.flip(-1).cumsum(-1).flip(-1), joint_nstep=1,
                         meta=dict(coefficient_model='full-gram', basis_fingerprint=fp)),
             verified_mean_ranks=MEAN_RANKS, verified={})
    for G in MEAN_RANKS:
        table = calibration.select(c, G)
        c['verified'][calibration.rank_key(G)] = dict(
            table_sha256=calibration.table_digest(table['m_table'], table['dense_table']))
    return calibration.validate(c)


class ServingWindowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_erase_family_allocates_at_the_serving_window(self):
        c = make_erase_calibration()
        self.assertEqual((calibration._crossover(c['geometry']), calibration._crossover(c['geometry'], 32)), (5, 4))
        for G in MEAN_RANKS:
            calibrated = calibration.select(c, G, window=16)
            self.assertTrue(calibrated['meta']['verified'])
            self.assertTrue(torch.equal(calibrated['m_table'], calibration.select(c, G)['m_table']))
            with self.assertLogs('sketchssm.calibration.calibration', 'WARNING') as logs:
                wide = calibration.select(c, G, window=32)
            self.assertIn('unverified', logs.output[0])
            meta = wide['meta']
            self.assertFalse(meta['verified'])
            self.assertEqual((meta['window'], meta['calibrated_window']), (32, 16))
            self.assertEqual(meta['max_sketch_rank'], 4)
            self.assertEqual(meta['allocation_cost']['rank'], 16 + 16 + 32)
            self.assertLessEqual(int(wide['m_table'].max()), 4)
            self.assertLessEqual(meta['allocation_cost']['used_total'],
                                 meta['allocation_cost']['budget_per_head'] * 8 + 1e-6)
            exported = calibration.frames(c, G, window=32)
            self.assertTrue(torch.equal(exported['m_table'], wide['m_table'].long()))
        # A cap pinned below the calibrated crossover still applies at other windows.
        self.assertEqual(calibration.max_rank(dict(c, max_rank=3), 32), 3)
        self.assertEqual(calibration.max_rank(dict(c, max_rank=3), 8), 3)
        self.assertEqual(calibration.max_rank(c, 8), 6)
        with self.assertRaisesRegex(ValueError, 'window'):
            calibration.select(c, 2.5, window=1)

    def test_mamba2_tables_do_not_depend_on_the_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = make_bundle(Path(temporary))
            c = calibration.package(bundle)
            # Without allocation.max_rank the packaged cap is the crossover (2 here).
            config = yaml.safe_load((bundle / 'config.yaml').read_text())
            del config['allocation']['max_rank']
            (bundle / 'config.yaml').write_text(yaml.safe_dump(config))
            self.assertEqual(calibration.package(bundle)['max_rank'], c['max_rank'])
        self.assertEqual(c['max_rank'], 2)
        for G in MEAN_RANKS:
            wide = calibration.select(c, G, window=64)
            self.assertTrue(wide['meta']['verified'])
            self.assertTrue(torch.equal(wide['m_table'], calibration.select(c, G)['m_table']))


class PublicBundleParityTests(unittest.TestCase):
    """Every configured mean rank of the four public bundles (under a minute on CPU)."""

    def test_package_reproduces_bundle_tables_and_frames(self):
        torch.set_num_threads(8)
        for name in BUNDLES:
            bundle = EXAMPLES / name
            packaged = calibration.package(bundle)
            ranks = yaml.safe_load((bundle / 'config.yaml').read_text())['allocation']['mean_ranks']
            self.assertEqual(packaged['verified_mean_ranks'], ranks)
            for G in ranks:
                with self.subTest(model=name, mean_rank=G):
                    table = torch.load(bundle / f'allocations/g{G:g}.pt', weights_only=True)
                    selected = calibration.select(packaged, G)
                    for key in ('m_table', 'dense_table'):
                        self.assertEqual(selected[key].dtype, table[key].dtype)
                        self.assertTrue(torch.equal(selected[key], table[key]))
                    assert_same_export(self, export_frames(selected), export_frames(load_precomputed(bundle, G)))


if __name__ == '__main__':
    unittest.main()
