# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Small real Mamba-2 module gate without model weights or GPU."""

import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from sketchssm.calibration.collection.hooks.mamba2 import JointCollector


def main():
    torch.manual_seed(31)
    torch.set_num_threads(2)
    config = NemotronHConfig(
        hidden_size=32,
        num_hidden_layers=1,
        hybrid_override_pattern="M",
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=8,
        n_groups=2,
        conv_kernel=4,
        chunk_size=8,
        use_mamba_kernels=False,
    )
    mixer = NemotronHMamba2Mixer(config, layer_idx=0).float().eval()
    mixer.time_step_limit = (0.0, float("inf"))
    basis = torch.linalg.qr(torch.randn(1, 2, 8, 4), mode="reduced").Q.transpose(-1, -2)
    collector = JointCollector(
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
        collector.finish_sequence(8)
        assert (
            collector.nseq == 1
            and torch.isfinite(collector.sums["joint_dot_sq_sum"]).all()
        )
        assert collector.sums["grad_sq_sum"].sum() > 0
        print(
            "Mamba2 real HF tiny module forward/backward PASS",
            collector.recurrence_max_rel,
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
