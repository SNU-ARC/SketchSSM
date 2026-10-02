# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Export portable ordered frames; serving layouts are built by engine adapters."""
import torch
from .frames import q_full_from


def export_frames(allocation):
    omega = allocation['omega'].double()
    m = allocation['m_table'].long()
    dense = allocation['dense_table'].bool()
    if omega.ndim != 4:
        raise ValueError('Expected basis layout (layers, groups, rank, K)')
    L, groups, available, K = omega.shape
    if groups < 1 or m.ndim != 2 or m.shape[0] != L or dense.shape != m.shape or m.shape[1] % groups:
        raise ValueError('Invalid allocation geometry')
    if not torch.equal(m == 0, dense) or (m < 0).any() or (m > available).any():
        raise ValueError('Use rank 0 only for dense fallback; sketch ranks must be calibrated')
    heads_per_group = m.shape[1] // groups
    frames = []
    for li in range(L):
        layer = []
        for gi in range(groups):
            rank = int(m[li, gi * heads_per_group:(gi + 1) * heads_per_group].max())
            R = q_full_from(omega[li, gi, :rank])
            torch.testing.assert_close(R @ R.T, torch.eye(K, dtype=R.dtype, device=R.device), atol=1e-8, rtol=1e-8)
            layer.append(R.float())
        frames.append(torch.stack(layer))
    out = dict(frames=torch.stack(frames), m_table=m, dense_table=dense,
        meta=dict(allocation=allocation['meta'], coordinates='original',
                  frame_layout='L,group,K,K', rank_layout='L,head', inference_pivots=4,
                  state_dtype='float32', sketch_dtype='bfloat16', coefficient_map_dtype='bfloat16'))
    if 'layer_ids' in allocation:
        out['layer_ids'] = allocation['layer_ids']
    return out
