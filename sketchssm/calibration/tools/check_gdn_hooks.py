# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
from unittest.mock import patch

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from sketchssm.calibration.collection.hooks.gdn import GDNJointCollector

torch.set_num_threads(2)
torch.manual_seed(37)
c = Qwen3_5TextConfig(
    hidden_size=32,
    num_hidden_layers=1,
    layer_types=["linear_attention"],
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=6,
    linear_conv_kernel_dim=4,
)
with (
    patch.object(m, "FusedRMSNormGated", None),
    patch.object(m, "chunk_gated_delta_rule", None),
    patch.object(m, "causal_conv1d_fn", None),
):
    mixer = m.Qwen3_5GatedDeltaNet(c, 0).float().eval()
    basis = torch.linalg.qr(torch.randn(1, 2, 8, 4), mode="reduced").Q.transpose(-1, -2)
    paired = GDNJointCollector(
        [mixer],
        basis,
        window=4,
        warmup=8,
        mmax=4,
        rank_tol=1e-7,
        recurrence_atol=1e-4,
        recurrence_rtol=1e-4,
    )
    try:
        x = torch.randn(1, 16, 32, requires_grad=True)
        mixer(x).float().square().mean().backward()
        paired.finish_sequence(8)
        assert (
            paired.nseq == 1 and torch.isfinite(paired.sums["joint_dot_sq_sum"]).all()
        )
        assert paired.sums["grad_sq_sum"].sum() > 0
        print(
            "GDN tiny real HF module forward/backward PASS", paired.recurrence_max_rel
        )
    finally:
        paired.close()
