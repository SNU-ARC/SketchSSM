# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Shared setup of the CUDA kernel tests.

The tests build the storage the kernels read (sketches, tables, rings) with
vLLM's containers, so they need vLLM and a CUDA GPU.
"""

import functools

import pytest
import torch

collect_ignore_glob: list[str] = []
try:
    import vllm  # noqa: F401
except ImportError:
    collect_ignore_glob.append("test_*.py")


# The quick suite (``-m "not slow"``): one small W = 16 case per family and
# the config lookup. Everything else is marked slow.
QUICK = {
    "test_mamba2.py::test_mamba2_decode[shape2-128-16]",
    "test_gdn.py::test_gdn_gate[widths1-False-dtype1-3]",
    "test_kda.py::test_kda_gate[16]",
    "test_kda.py::test_kda_config_lookup",
}


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="Requires a CUDA GPU")
    for item in items:
        if item.nodeid.split("/")[-1] not in QUICK:
            item.add_marker(pytest.mark.slow)
        if not torch.cuda.is_available():
            item.add_marker(skip)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """A tuned-config folder searched alone: the shipped configs and any
    folders set with ``set_config_dirs`` are hidden."""
    from sketchssm.kernels import _runtime as rt

    monkeypatch.setenv("SKETCHSSM_KERNELS_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(rt, "CONFIGS", tmp_path / "none")
    monkeypatch.setattr(rt, "_config_dirs", [])
    return tmp_path


@pytest.fixture
def clear_caches():
    """Clears the given ``functools.cache``s now and after the test."""
    cached: list[functools._lru_cache_wrapper] = []

    def clear(*fns):
        cached.extend(fns)
        for fn in fns:
            fn.cache_clear()

    yield clear
    for fn in cached:
        fn.cache_clear()
