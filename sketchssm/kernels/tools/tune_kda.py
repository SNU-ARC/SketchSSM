# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Tune the SketchSSM Kimi Delta Attention CUDA build knobs for this GPU.

Coordinate descent over the step kernel knobs, then the flush kernel knobs,
from the better of the architecture defaults (``kda.default_config``) and the
knobs in effect, timed over every layer of the calibration (or ``--layer``)
and a whole window (``--objective``). Every candidate is a
separate NVRTC build (cached on disk). Knobs change scheduling only, so outputs stay
bitwise identical. ``--save-configs`` writes the knobs that differ from the
defaults to the file ``kda.tuned_config`` loads.

Example:
    python -m sketchssm.kernels.tools.tune_kda \\
        --calibration glm_frames.pt --window 64
"""

import argparse

import torch

from sketchssm.kernels import _runtime as rt
from sketchssm.kernels import kda as skk
from sketchssm.kernels.tools import benchmark_kda as bm
from sketchssm.kernels.tools.benchmark_utils import (
    add_tuner_args,
    make_layouts,
    tune_and_save,
)

# Scheduling knobs only. The step's S8_PREFETCH is an
# L2 prefetch; the flush's K1_ABLATE skips work, so it is not tuned.
# K1_MINB<M> is the main kernel's minimum CTAs per SM (register cap and
# persistent grid size) of rank bucket M (K1_MINB sets 32/64/128), K2_MINB
# the finish's.
SPACE = {
    "step": {
        "S7_MINB": [2, 3, 4, 5, 6, 7, 8],
        "NW": [1, 2, 3, 4, 5],
        "S8_PREFETCH": [0, 1],
    },
    "flush": {
        "K1_MINB": [1, 2, 3, 4],
        "K1_MINB16": [1, 2, 3, 4, 5, 6],
        "K1_MINB32": [1, 2, 3, 4],
        "K1_MINB64": [1, 2, 3, 4],
        "K1_MINB128": [1, 2, 3, 4],
        "K2_MINB": [2, 4, 6, 8, 12, 16],
    },
}


class KDA:
    space = SPACE
    kinds = ("step", "flush")

    def __init__(self, args):
        self.defaults = dict(zip(self.kinds, skk.default_config(bm.H, args.window)))
        self.in_effect = dict(zip(self.kinds, skk.tuned_config(bm.H, args.window)))
        self.layouts = make_layouts(args, bm.H, bm.H, bm.K)
        self.name = skk.config_file_name(bm.H, args.window)
        self.out_dir = rt.CONFIGS / skk.FAMILY

    def configure(self, configs):
        skk.tuned_config = lambda *shape: (configs["step"], configs["flush"])
        skk._step_ext.cache_clear()
        skk._flush_ext.cache_clear()

    def step(self, args, layout, batch):
        return bm.kda_step("sketch", batch, args, layout)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_tuner_args(p)
    args = p.parse_args()
    torch.set_default_device("cuda")
    tune_and_save(args, KDA(args))


if __name__ == "__main__":
    main()
