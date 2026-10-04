# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Tune the SketchSSM Gated DeltaNet CUDA build knobs for this GPU and a shape.

Coordinate descent over the step kernel knobs, then the flush kernel knobs,
from the better of the architecture defaults (``gdn.default_config``) and the
knobs in effect. Each candidate is timed over every layer of the calibration
with its own ranks (or ``--layer``), over a window of W - 1 non-flush steps
and one flush step (``--objective``). Every candidate is a
separate NVRTC build (cached on disk). Knobs change scheduling only, so outputs stay
bitwise identical. ``--save-configs`` writes the knobs that differ from the
defaults to the file ``gdn.tuned_config`` loads.

Example:
    python -m sketchssm.kernels.tools.tune_gdn \\
        --calibration qwen_frames.pt --num-k-heads 16 --num-v-heads 32
"""

import argparse

import torch

from sketchssm.kernels import _runtime as rt
from sketchssm.kernels import gdn as skg
from sketchssm.kernels.tools import benchmark_gdn as bm
from sketchssm.kernels.tools.benchmark_utils import (
    add_tuner_args,
    make_layouts,
    tune_and_save,
)

# Scheduling knobs only. The step's SKETCH_BF16,
# NF_FFMA2 and NF_ABLATE and the flush's W1_NOSKETCH, W1_NOSK and
# W1_ABLATE change the math or skip work, so they are not tuned.
# ROWS_PER_PROGRAM is the flush launch's rows per CTA.
SPACE = {
    "step": {
        "NF_MINB": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        "NF_UROWS": [0, 2, 4, 6, 8],
        "NF_AG_FG": [0, 16, 24, 32, 48, 64],
        "NF_FS_FG": [0, 16, 24, 32],
    },
    "flush": {
        "W1_MINB": [1, 2, 3, 4, 5, 6],
        "ROWS_PER_PROGRAM": [1, 2, 4, 8, 16],
    },
}


class GDN:
    space = SPACE
    kinds = ("step", "flush")

    def __init__(self, args):
        h, hv, w = args.num_k_heads, args.num_v_heads, args.window
        self.defaults = dict(zip(self.kinds, skg.default_config(h, hv, w)))
        self.in_effect = dict(zip(self.kinds, skg.tuned_config(h, hv, w)))
        self.layouts = make_layouts(args, hv, h, bm.K)
        self.name = skg.config_file_name(h, hv, w)
        self.out_dir = rt.CONFIGS / skg.FAMILY

    def configure(self, configs):
        skg.tuned_config = lambda *shape: (configs["step"], configs["flush"])
        skg._step_ext.cache_clear()
        skg._flush_ext.cache_clear()

    def step(self, args, layout, batch):
        return bm.gdn_step("sketch", batch, args, layout)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_tuner_args(p)
    bm.add_shape_args(p)
    args = p.parse_args()
    torch.set_default_device("cuda")
    tune_and_save(args, GDN(args))


if __name__ == "__main__":
    main()
