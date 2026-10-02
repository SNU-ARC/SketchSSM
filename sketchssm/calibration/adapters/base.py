# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Recurrence contract independent of model names and serving engines."""
from abc import ABC, abstractmethod
import torch


class CalibrationAdapter(ABC):
    """Convert normalized recurrence inputs to boundary-state effective queries.

    Inputs use (..., T, H, K) with one entry per state head, already expanded
    from native query/key groups. The engine binding supplies queries/keys after
    its own normalization and scaling, multiplicative decay (not log decay),
    and beta after its gate. This interface never modifies the model forward.
    """

    family: str
    erase: bool

    def validate_geometry(self, geometry):
        for key in ('key_dim', 'value_dim', 'groups', 'window'):
            value = geometry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if geometry['window'] < 2:
            raise ValueError('window must be at least 2')
        if type(geometry['erase']) is not bool or geometry['erase'] != self.erase:
            raise ValueError(f'{self.family} requires erase={self.erase}')

    @abstractmethod
    def effective_queries(self, query, *, decay, window=16, key=None, beta=None):
        """Return (..., T, H, K), relative to each W-token window start."""


def check_inputs(query, decay, window, *, channel_decay=False, key=None, beta=None, erase=False):
    if query.ndim < 3 or min(query.shape) < 1:
        raise ValueError('Expected nonempty query shape (..., T, H, K)')
    if isinstance(window, bool) or not isinstance(window, int) or window < 2 or query.shape[-3] % window:
        raise ValueError('Supply complete windows with integer window >= 2')
    if decay.shape != (query.shape if channel_decay else query.shape[:-1]):
        raise ValueError('Decay shape differs from the adapter contract')
    tensors = [query, decay]
    if erase:
        if key is None or beta is None or key.shape != query.shape or beta.shape != query.shape[:-1]:
            raise ValueError('Erase requires state-head keys and beta with matching shapes')
        tensors.extend([key, beta])
    elif key is not None or beta is not None:
        raise ValueError('This recurrence has no erase key/beta')
    if any(not x.is_floating_point() or x.device != query.device or not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Inputs must be finite floating tensors on the same device')
    if (decay < 0).any():
        raise ValueError('Supply multiplicative decay, not log decay')
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in tensors) else torch.float32
    return tuple(x.to(dtype) for x in tensors)


def ordered_erase_queries(query, decay, key, beta, window, *, channel_decay):
    """Reverse-apply ordered transitions without assuming factors commute."""
    outputs = []
    for t in range(query.shape[-3]):
        x = query[..., t, :, :]
        for s in range(t, t // window * window - 1, -1):
            k = key[..., s, :, :]
            x = x - beta[..., s, :, None] * k * (k * x).sum(-1, keepdim=True)
            a = decay[..., s, :, :] if channel_decay else decay[..., s, :, None]
            x = a * x
        outputs.append(x)
    return torch.stack(outputs, dim=-3)
