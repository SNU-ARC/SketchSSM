# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
'Build ordered orthogonal frames without changing any nested prefix span.'
import torch

def q_full_from(rows: torch.Tensor) -> torch.Tensor:
    """Complete ordered prefix rows to an orthogonal state rotation."""
    state_dim = rows.shape[1]
    augmented = torch.cat(
        [rows.T.double(), torch.eye(state_dim, dtype=torch.float64, device=rows.device)], dim=1)
    q, _ = torch.linalg.qr(augmented)
    return q.T.contiguous()
