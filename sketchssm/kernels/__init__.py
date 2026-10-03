# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""CUDA decode kernels of SketchSSM (Mamba-2, Gated DeltaNet, KDA).

Each kernel is specialized per layer shape, window and build knobs. A
specialization is taken precompiled (AOT, ``build.py``) when available, else
compiled at run time with NVRTC and cached on disk.
"""

from . import gdn, kda, mamba2
from ._runtime import Support, set_aot_dirs, set_cache_dir, set_config_dirs
from .gdn import gdn_decode, gdn_supported
from .kda import kda_cold_build, kda_decode, kda_supported
from .mamba2 import mamba2_decode, mamba2_supported

# Bumped on a breaking change to the API below.
API_VERSION = 2

__all__ = [
    "API_VERSION",
    "Support",
    "gdn",
    "gdn_decode",
    "gdn_supported",
    "kda",
    "kda_cold_build",
    "kda_decode",
    "kda_supported",
    "mamba2",
    "mamba2_decode",
    "mamba2_supported",
    "set_aot_dirs",
    "set_cache_dir",
    "set_config_dirs",
]
