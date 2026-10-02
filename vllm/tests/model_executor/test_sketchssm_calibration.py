# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading and validation of SketchSSM calibrations."""

from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.mamba import sketchssm_calibration as calib
from vllm.model_executor.layers.mamba.sketchssm import load_sketchssm_calibration


@pytest.fixture(autouse=True)
def _clear_cache():
    load_sketchssm_calibration.cache_clear()
    yield
    load_sketchssm_calibration.cache_clear()


def _save(tmp_path, data, name="calibration.pt"):
    torch.save(data, tmp_path / name)
    return str(tmp_path / name)


def _frames(**overrides):
    g = torch.Generator().manual_seed(0)
    ranks = torch.tensor([[0, 1, 5, 3], [2, 0, 0, 0]])
    frames = torch.linalg.qr(torch.randn(2, 2, 8, 8, generator=g))[0]
    return dict(frames=frames, m_table=ranks, dense_table=ranks == 0) | overrides


def _portable(L=3, groups=2, heads=8, K=16, V=8, max_rank=4):
    g = torch.Generator().manual_seed(1)
    omega = torch.linalg.qr(torch.randn(L, groups, K, K, generator=g))[0]
    omega = omega[..., : max_rank + 2, :].contiguous().float()
    scale = torch.rand(L, heads, 1, generator=g, dtype=torch.float64) + 0.1
    error = scale * torch.exp(-0.5 * torch.arange(max_rank + 3)) * 64
    fp = calib.fingerprint(omega)
    return dict(
        format="sketchssm-calibration",
        schema_version=1,
        geometry=dict(key_dim=K, value_dim=V, groups=groups, window=16, erase=False),
        max_rank=max_rank,
        omega=omega,
        basis_fingerprint=fp,
        curves=dict(
            joint_dot_sq_sum=error,
            joint_nstep=64,
            meta=dict(coefficient_model="full-gram", basis_fingerprint=fp),
        ),
        layer_ids=[1, 4, 6],
    )


def test_exported_frames(tmp_path):
    calibration = load_sketchssm_calibration(
        _save(tmp_path, _frames(layer_ids=torch.tensor([3, 7])))
    )
    assert calibration.num_layers == 2
    assert calibration.layer_index(recurrent_idx=0, model_layer_idx=7) == 1
    with pytest.raises(ValueError, match="no entry for layer 5"):
        calibration.layer_index(recurrent_idx=0, model_layer_idx=5)


@pytest.mark.parametrize(
    ("data", "mean_rank", "match"),
    [
        (_frames(frames=torch.ones(2, 2, 8, 8)), None, "not orthogonal"),
        (_frames(dense_table=torch.zeros(2, 4, dtype=torch.bool)), None, "rank-0"),
        (_frames(m_table=torch.full((2, 4), 9)), None, r"lie in \[0, K\]"),
        (_frames(), 4.0, "already fixed"),
        (_portable() | dict(basis_fingerprint="0"), 2.0, "fingerprint"),
    ],
)
def test_rejects_bad_files(tmp_path, data, mean_rank, match):
    with pytest.raises(ValueError, match=match):
        load_sketchssm_calibration(_save(tmp_path, data), mean_rank)


def test_portable_mean_rank_allocation(tmp_path):
    c, mean_rank = _portable(), 2.5
    path = _save(tmp_path, c)
    calibration = load_sketchssm_calibration(path, mean_rank)
    ranks, frames = calibration.ranks, calibration.frames
    assert ranks.shape == (3, 8) and frames.shape == (3, 2, 16, 16)
    assert calibration.layer_ids == (1, 4, 6)
    rank_cost, dense_cost = 16 + 8, 16 * 8
    used = (ranks.sum() * rank_cost + (ranks == 0).sum() * dense_cost).item()
    assert used <= rank_cost * mean_rank * ranks.numel()
    # Each group frame starts with the basis prefix up to its largest rank.
    for li in range(3):
        for gi in range(2):
            m = int(ranks[li, gi * 4 : (gi + 1) * 4].max())
            prefix, basis = frames[li, gi, :m].double(), c["omega"][li, gi, :m].double()
            torch.testing.assert_close(
                basis @ prefix.T @ prefix, basis, atol=1e-5, rtol=0
            )
    # The mean rank defaults to 10, and the serving window reaches the allocation.
    default = load_sketchssm_calibration(path)
    assert torch.equal(default.ranks, load_sketchssm_calibration(path, 8.0).ranks)
    spy = mock.patch.object(
        calib, "frames_for_mean_rank", wraps=calib.frames_for_mean_rank
    )
    with spy as frames_for_mean_rank:
        wide = load_sketchssm_calibration(path, mean_rank, 32)
        assert load_sketchssm_calibration(path, mean_rank, 32) is wide
    assert [call.args[2] for call in frames_for_mean_rank.call_args_list] == [32]
    # A verified rank must reproduce its table.
    c["verified"] = {calib.rank_key(mean_rank): dict(table_sha256="0" * 64)}
    with pytest.raises(RuntimeError, match="verified table"):
        calib.frames_for_mean_rank(c, mean_rank)
