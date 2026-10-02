# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Shared runtime of the families: config, cache and precompiled-kernel
folders, and the target GPU (architecture, name, shared memory)."""

import functools
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("sketchssm.kernels")

CSRC = Path(__file__).parent / "csrc"
CONFIGS = Path(__file__).parent / "configs"
# Precompiled kernels installed with the package (e.g. by vLLM's build).
AOT_PACKAGE_DIR = Path(__file__).parent / "aot"

_config_dirs: list[str] = []
_aot_dirs: list[str] = []
_cache_dir: str | None = None
_logged: set[str] = set()


@dataclass(frozen=True)
class Support:
    """Whether a layer can use the CUDA kernels; truthy when it can.

    ``source`` says where its kernels come from: "aot" (all precompiled),
    "nvrtc" (at least one compiled at run time) or None (unsupported).
    """

    ok: bool
    reason: str = ""
    source: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def set_config_dirs(dirs: Sequence[str | None]) -> None:
    """Folders searched for tuned-config files after
    ``SKETCHSSM_KERNELS_CONFIG_DIR`` and before the shipped configs."""
    _config_dirs[:] = [d for d in dirs if d]


def set_cache_dir(path: str | os.PathLike) -> None:
    """Folder of the NVRTC builds (default ``SKETCHSSM_KERNELS_CACHE_DIR`` or
    ``~/.cache/sketchssm/kernels``)."""
    global _cache_dir
    _cache_dir = str(path)


def set_aot_dirs(dirs: Sequence[str | os.PathLike | None]) -> None:
    """Folders of precompiled kernels, searched after
    ``SKETCHSSM_KERNELS_AOT_DIR`` and before the package's ``aot/``."""
    from . import _kernel

    _aot_dirs[:] = [str(d) for d in dirs if d]
    _kernel.reset_aot_index()


def aot_dirs() -> list[Path]:
    env = os.environ.get("SKETCHSSM_KERNELS_AOT_DIR")
    user = [d for d in env.split(os.pathsep) if d] if env else []
    return [Path(d) for d in (*user, *_aot_dirs)] + [AOT_PACKAGE_DIR]


def cache_dir() -> Path:
    if _cache_dir is not None:
        return Path(_cache_dir)
    env = os.environ.get("SKETCHSSM_KERNELS_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "sketchssm" / "kernels"


def config_folders(family: str) -> list[Path]:
    """Folders searched for a family's tuned-config files, in order."""
    env = os.environ.get("SKETCHSSM_KERNELS_CONFIG_DIR")
    user = [env] if env else []
    return [Path(d) for d in (*user, *_config_dirs)] + [CONFIGS / family]


def find_config(family: str, name: str) -> Path | None:
    for folder in config_folders(family):
        if (folder / name).exists():
            return folder / name
    return None


def read_config(path: Path) -> dict:
    return json.loads(path.read_text())


def log_once(level: int, msg: str, *args) -> None:
    key = msg % args
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def normalize_device_name(name: str) -> str:
    """A GPU name as in the tuned-config file names."""
    return re.sub(r"[\s/]+", "_", name)


# Shared memory (per-CTA opt-in maximum, per-SM capacity) by compute
# capability, for building without a GPU.
ARCH_SMEM = {
    (8, 0): (163 * 1024, 164 * 1024),
    (8, 6): (99 * 1024, 100 * 1024),
    (8, 7): (163 * 1024, 164 * 1024),
    (8, 9): (99 * 1024, 100 * 1024),
    (9, 0): (227 * 1024, 228 * 1024),
    (10, 0): (227 * 1024, 228 * 1024),
    (10, 3): (227 * 1024, 228 * 1024),
    (11, 0): (227 * 1024, 228 * 1024),
    (12, 0): (99 * 1024, 100 * 1024),
    (12, 1): (99 * 1024, 100 * 1024),
}
# The GPU whose tuned files an architecture build uses (see ``Target``).
REFERENCE_DEVICE = {
    (9, 0): "NVIDIA_H100_80GB_HBM3",
    (10, 0): "NVIDIA_B300_SXM6_AC",
    (10, 3): "NVIDIA_B300_SXM6_AC",
}


@dataclass(frozen=True)
class Target:
    """The GPU the build knobs are resolved for: this GPU at run time, or an
    architecture (with its reference GPU's tuned files) ahead of time."""

    major: int
    minor: int
    device_name: str
    smem_per_block_optin: int
    smem_per_sm: int

    @property
    def capability(self) -> tuple[int, int]:
        return (self.major, self.minor)

    @property
    def cc(self) -> int:
        return 10 * self.major + self.minor

    @property
    def ffma2(self) -> int:
        """Packed FP32 FMA needs sm_100; the kernels have scalar fallbacks."""
        return int(self.major >= 10)

    @staticmethod
    def for_arch(arch: str, device_name: str | None = None) -> "Target":
        """An architecture's target without a GPU (``9.0a``, ``sm_100f``, ...).
        ``device_name`` (default: the reference GPU of the architecture, the
        H100 for sm_90) selects the tuned-config files."""
        from ._kernel import arch_capability

        cap = arch_capability(arch)
        optin, per_sm = ARCH_SMEM.get(cap, ARCH_SMEM.get((cap[0], 0), (99 * 1024, 100 * 1024)))
        name = device_name if device_name is not None else REFERENCE_DEVICE.get(cap, f"sm_{cap[0]}{cap[1]}")
        return Target(cap[0], cap[1], normalize_device_name(name), optin, per_sm)


@functools.cache
def _current(device: int) -> Target:
    import torch

    props = torch.cuda.get_device_properties(device)
    return Target(
        props.major, props.minor, normalize_device_name(props.name),
        props.shared_memory_per_block_optin, props.shared_memory_per_multiprocessor,
    )  # fmt: skip


def current_target() -> Target:
    import torch

    return _current(torch.cuda.current_device())


def device_name() -> str:
    """This GPU's name as in the tuned-config file names."""
    return current_target().device_name


def capability() -> tuple[int, int]:
    return current_target().capability


def ffma2() -> int:
    return current_target().ffma2


def cuda_available(min_capability: int) -> Support:
    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return Support(False, "no CUDA device")
    if current_target().cc < min_capability:
        return Support(False, f"needs sm_{min_capability}+")
    return Support(True)


def kernel_support(specs, what: str = "") -> Support:
    """Support of a layer whose kernels are ``specs`` (precompiled or NVRTC)."""
    from . import _kernel

    cap = capability()
    sources = []
    for spec in specs:
        src = _kernel.source_of(spec, cap)
        if src is None:
            return Support(False, _kernel.unavailable_reason(spec, cap))
        sources.append(f"{spec.kind}: {src}")
    source = "aot" if all(s.endswith("aot") for s in sources) else "nvrtc"
    reason = ", ".join(sources) + (f" ({what})" if what else "")
    return Support(True, reason, source)
