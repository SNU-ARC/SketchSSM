# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""GDN scalar-decay and ordered rank-one erase calibration adapter."""
from .base import CalibrationAdapter, check_inputs, ordered_erase_queries


class GDNAdapter(CalibrationAdapter):
    family = 'gdn'
    erase = True

    def effective_queries(self, query, *, decay, window=16, key=None, beta=None):
        query, decay, key, beta = check_inputs(query, decay, window, key=key, beta=beta, erase=True)
        return ordered_erase_queries(query, decay, key, beta, window, channel_decay=False)
