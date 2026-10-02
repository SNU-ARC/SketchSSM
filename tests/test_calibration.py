# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Independent numerical checks for the portable offline pipeline (CPU only)."""
import itertools
import unittest
import numpy as np
import torch
from sketchssm.calibration.core.basis import fit, fingerprint, solve
from sketchssm.calibration.core.scoring import score
from sketchssm.calibration.core.allocation import allocate
from sketchssm.calibration.core.export import export_frames
from sketchssm.calibration.core.traffic import Traffic, AllocationCost
from sketchssm.calibration.core.reuse import reuse
from sketchssm.calibration.core._solver import solve as allocate_options


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(2)

    def make_basis(self):
        raw = torch.randn(1, 4, 6, 6, dtype=torch.float64)
        E = raw.transpose(-1, -2) @ raw
        return fit(dict(head_scov=E, head_qcov=E.flip(1),
                        head_cov_windows=torch.tensor([2]),
                        head_cov_queries=torch.tensor([8])), 2, 4)

    def test_group_pooling_precedes_fit(self):
        raw = torch.randn(1, 4, 6, 6, dtype=torch.float64)
        E = raw.transpose(-1, -2) @ raw
        C = E.flip(1)
        basis = fit(dict(head_scov=E, head_qcov=C,
                        head_cov_windows=torch.tensor([2]), head_cov_queries=torch.tensor([8])), 2, 4)
        expected = solve(E.reshape(1, 2, 2, 6, 6).mean(2) / 2,
                         C.reshape(1, 2, 2, 6, 6).mean(2) / 8, 4)[0]
        torch.testing.assert_close(basis['omega'], expected, rtol=0, atol=0)
        self.assertFalse(basis['meta']['gradients_used'])

    def test_full_gram_paired_scores_match_least_squares(self):
        basis = self.make_basis()
        state = torch.randn(1, 2, 2, 4, 5, 6, dtype=torch.float64)
        query = torch.randn(1, 2, 2, 4, 6, 4, dtype=torch.float64)
        grad = torch.randn(1, 2, 2, 4, 5, 4, dtype=torch.float64)
        result = score(dict(state=state, effective_query=query, gradient=grad), basis)
        errors = torch.zeros(4, 5, dtype=torch.float64)
        dots = torch.zeros_like(errors)
        for b, n, h in itertools.product(range(2), range(2), range(4)):
            S = state[0, b, n, h]; target = S @ query[0, b, n, h]
            U = S @ basis['omega'][0, h // 2].double().T
            for rank in range(5):
                residual = target if rank == 0 else target - U[:, :rank] @ torch.linalg.lstsq(U[:, :rank], target).solution
                errors[h, rank] += residual.square().sum()
                dots[h, rank] += (grad[0, b, n, h] * residual).sum(0).square().sum()
        torch.testing.assert_close(result['output_error_sum'][0], errors, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(result['joint_dot_sq_sum'][0], dots, atol=1e-9, rtol=1e-9)
        self.assertEqual(result['joint_nstep'], 16)

    def test_dependent_prefix_does_not_invent_a_direction(self):
        rows = torch.tensor([[1., 0., 0.], [1., 0., 0.]]).reshape(1, 1, 2, 3)
        trace = dict(state=torch.eye(3).reshape(1, 1, 1, 1, 3, 3),
                     effective_query=torch.ones(1, 1, 1, 1, 3, 2),
                     gradient=torch.ones(1, 1, 1, 1, 3, 2))
        result = score(trace, dict(omega=rows))
        self.assertEqual(result['output_error_sum'][0, 0].tolist(), [6., 4., 4.])

    def test_dense_fallback_and_frame_export(self):
        basis = dict(omega=torch.eye(4)[:2].reshape(1, 1, 2, 4), layer_ids=[3])
        curves = dict(joint_dot_sq_sum=torch.tensor([[[100., 100., 100.], [1., .1, 0.]]]),
                      joint_nstep=1, meta=dict(coefficient_model='full-gram', basis_fingerprint=fingerprint(basis['omega'])))
        result = allocate(curves, basis, mean_rank=1.5, value_dim=4, max_rank=2)
        self.assertEqual(result['m_table'].tolist(), [[0, 1]])
        self.assertEqual(result['dense_table'].tolist(), [[True, False]])
        cost = result['meta']['allocation_cost']
        self.assertLessEqual(cost['used_total'], cost['budget_per_head'] * 2)
        export = export_frames(result)
        self.assertEqual(export['layer_ids'], [3])
        R = export['frames'].double()
        torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(4).double().expand_as(R))
        self.assertEqual(export['meta']['coefficient_map_dtype'], 'bfloat16')
        wrong = dict(curves, meta=dict(curves['meta'], coefficient_model='p4'))
        with self.assertRaisesRegex(ValueError, 'Full-Gram'):
            allocate(wrong, basis, mean_rank=1.5, value_dim=4, max_rank=2)
        wrong = dict(curves, meta=dict(curves['meta'], basis_fingerprint='different'))
        with self.assertRaisesRegex(ValueError, 'basis'):
            allocate(wrong, basis, mean_rank=1.5, value_dim=4, max_rank=2)

    def test_max_rank_defaults_to_the_dense_crossover(self):
        omega = torch.linalg.qr(torch.randn(8, 8))[0][:6].reshape(1, 1, 6, 8)
        error = torch.rand(1, 4, 7).sort(-1, descending=True).values.double()
        curves = dict(joint_dot_sq_sum=error, joint_nstep=1,
                      meta=dict(coefficient_model='full-gram', basis_fingerprint=fingerprint(omega)))
        for erase in (False, True):
            crossover = AllocationCost(8, 8, 16, erase).crossover
            self.assertEqual(crossover, 2 if erase else 4)
            default = allocate(curves, dict(omega=omega), mean_rank=1.5, value_dim=8, erase=erase)
            self.assertEqual(default['meta']['max_sketch_rank'], crossover)
            self.assertFalse(default['meta']['restricted_candidates'])
            pinned = allocate(curves, dict(omega=omega), mean_rank=1.5, value_dim=8, erase=erase,
                              max_rank=crossover)
            self.assertTrue(torch.equal(pinned['m_table'], default['m_table']))
            lower = allocate(curves, dict(omega=omega), mean_rank=1.5, value_dim=8, erase=erase,
                             max_rank=1)
            self.assertEqual(lower['meta']['max_sketch_rank'], 1)
            self.assertLessEqual(int(lower['m_table'].max()), 1)
            with self.assertRaisesRegex(ValueError, 'crossover'):
                allocate(curves, dict(omega=omega), mean_rank=1.5, value_dim=8, erase=erase,
                         max_rank=crossover + 1)

    def test_solver_bound_against_exhaustive_choices(self):
        rng = np.random.default_rng(13)
        cost = np.array([3., 6., 9., 15.])
        for _ in range(8):
            objective = np.c_[rng.uniform(0, 20, (3, 3)), np.zeros(3)]
            budget = 25.
            chosen, report = allocate_options(objective, cost, budget)
            optimum = min(sum(objective[h, c] for h, c in enumerate(cs))
                          for cs in itertools.product(range(4), repeat=3) if sum(cost[c] for c in cs) <= budget)
            self.assertLessEqual(cost[chosen].sum(), budget + 1e-8)
            self.assertLessEqual(report['lower_bound'], optimum + 1e-8)
            self.assertGreaterEqual(report['objective'] + 1e-8, optimum)

    def test_traffic_bytes_and_dense_crossover(self):
        cost = Traffic(128, 80)
        self.assertEqual(AllocationCost(128, 80).crossover, 49)
        self.assertEqual(cost.sketch_nonflush * 4, 2 * (128 + 80))
        report = cost.report(cost.dense_nonflush)
        self.assertEqual(report['read_reduction'], 1)
        self.assertAlmostEqual(report['access_reduction'], 2 / (1 + 1 / 16))
        self.assertEqual(Traffic(128, 128, erase=True).sketch_nonflush * 4, 2 * (128 + 128 + 16))

    def test_reuse_preserves_tables_and_reports_actual_precision(self):
        source = dict(omega=torch.eye(4)[:2].reshape(1, 1, 2, 4),
                      m_table=torch.tensor([[0, 1]], dtype=torch.int16),
                      dense_table=torch.tensor([[True, False]]), meta={})
        out = reuse(source, value_dim=4)
        for key in ('omega', 'm_table', 'dense_table'):
            self.assertEqual(source[key].dtype, out[key].dtype)
            torch.testing.assert_close(source[key], out[key], rtol=0, atol=0)
        self.assertEqual(out['meta']['allocation_policy'], 'reuse_unchanged')
        self.assertEqual(out['meta']['traffic']['state_dtype'], 'float32')
        self.assertEqual(out['meta']['traffic']['sketch_dtype'], 'bfloat16')
        self.assertAlmostEqual(out['meta']['traffic']['state_read'], 4 + 15 / 16 * 40)

    def test_all_dense_group_exports_identity(self):
        basis = self.make_basis()
        result = export_frames(dict(omega=basis['omega'], m_table=torch.zeros(1, 4),
                                    dense_table=torch.ones(1, 4, dtype=torch.bool), meta={}))
        torch.testing.assert_close(result['frames'], torch.eye(6).expand(1, 2, 6, 6))

    def test_frames_preserve_every_selected_prefix(self):
        basis = self.make_basis()
        result = export_frames(dict(omega=basis['omega'], m_table=torch.tensor([[1, 3, 2, 4]]),
                                    dense_table=torch.zeros(1, 4, dtype=torch.bool), meta={}))
        for group in range(2):
            for rank in range(1, (3, 4)[group] + 1):
                omega = basis['omega'][0, group, :rank].double().T
                Q = torch.linalg.qr(omega).Q
                F = result['frames'][0, group, :rank].double().T
                torch.testing.assert_close(Q @ Q.T, F @ F.T, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
