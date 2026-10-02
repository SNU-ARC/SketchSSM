# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Tune the SketchSSM Mamba-2 CUDA build knobs for this GPU and a layer shape.

Coordinate descent over the non-flush kernel knobs, then the flush kernel
knobs, from the better of this architecture's defaults
(``mamba2.default_configs``) and the knobs in effect. Each candidate is timed
over every layer of the calibration with its own ranks (or ``--layer``), over
a window of W - 1 non-flush steps and one flush step (``--objective``).
Every candidate is a separate NVRTC build (cached on disk). Knobs change scheduling
only, so outputs stay bitwise identical. ``--save-configs`` writes the knobs
that differ from the defaults to the file ``mamba2.tuned_config`` loads.

Example:
    python -m sketchssm.kernels.tools.tune_mamba2 \\
        --calibration nano_frames.pt --head-dim 80 --save-configs
"""

import argparse

import torch

from sketchssm.kernels import _runtime as rt
from sketchssm.kernels import mamba2 as skm
from sketchssm.kernels.tools import benchmark_mamba2 as bm
from sketchssm.kernels.tools.benchmark_utils import (
    add_tuner_args,
    make_layouts,
    tune_and_save,
)

SPACE = {
    "nf": {
        "NF_MINB": [4, 6, 8, 10, 12, 16],
        "NF_UROWS": [1, 2, 4, 8],
        "NF_UBATCH": [1, 2, 4, 8],
        "NF_HEADS": [2, 4, 8, 16],
        "NF_SMAPS": [1, 2, 4],
        "NF_PF_ROWS": [0, 2, 4, 8, 16],
        "NF_PF_FULL": [0, 1, 2],
        "NF_PF_BULK": [0, 1],
    },
    "flush": {
        "MINB": [4, 6, 8, 9, 10, 12, 16],
        "WARPS": [1, 2, 4],
        "FL_STAGES": [2, 3, 4],
        "FL_HALF": [0, 1],
        "FL_PF_BLOCKS": [0, 1, 2, 3, 4],
        "FL_EARLY": [0, 1],
        "FL_CSMEM": [0, 1],
        "QREG": [0, 1],
        "FL_Q0SMEM": [0, 1],
        "FL_OUTSMEM": [0, 1],
    },
}


class Mamba2:
    space = SPACE
    kinds = ("nf", "flush")

    def __init__(self, args):
        shape = (args.head_dim, args.num_heads // args.ngroups, args.state_size,
                 args.window)  # fmt: skip
        self.defaults = dict(zip(self.kinds, skm.default_configs()))
        self.in_effect = dict(zip(self.kinds, skm.tuned_config(*shape)))
        self.layouts = make_layouts(args, args.num_heads, args.ngroups, args.state_size)
        self.name = skm.config_file_name(*shape)
        self.out_dir = rt.CONFIGS / skm.FAMILY

    def configure(self, configs):
        skm.tuned_config = lambda *shape: (configs["nf"], configs["flush"])
        # A candidate that does not build must fail, not fall back.
        skm.window16_fallback = lambda *shape: False
        skm._nf_ext.cache_clear()
        skm._flush_ext.cache_clear()

    def step(self, args, layout, batch):
        return bm.mamba2_step("sketch", batch, args, layout)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_tuner_args(p)
    bm.add_shape_args(p)
    args = p.parse_args()
    torch.set_default_device("cuda")
    tune_and_save(args, Mamba2(args))


if __name__ == "__main__":
    main()
