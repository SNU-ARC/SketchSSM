# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Mamba-2 scalar-decay calibration adapter."""
from .base import CalibrationAdapter, check_inputs


class Mamba2Adapter(CalibrationAdapter):
    family = 'mamba2'
    erase = False

    def effective_queries(self, query, *, decay, window=16, key=None, beta=None):
        query, decay = check_inputs(query, decay, window, key=key, beta=beta)
        windows = decay.reshape(*decay.shape[:-2], -1, window, decay.shape[-1])
        cumulative = windows.cumprod(dim=-2).reshape_as(decay)
        return query * cumulative[..., None]
