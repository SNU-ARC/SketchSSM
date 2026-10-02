# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""One stage per subprocess; engine-dependent imports stay inside GPU stages."""

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

from ..core.allocation import allocate
from ..core.basis import fingerprint, fit
from .config import STAGES, read_config
from .storage import atomic_json, atomic_torch, checksum, load


def allocations(config, root):
    basis = load(root / "basis/omega.pt")
    curves = load(root / "statistics/paired_scores.pt")
    geo = config["geometry"]
    settings = config["allocation"]
    entries = {}
    for rank in settings["mean_ranks"]:
        result = allocate(
            curves,
            basis,
            mean_rank=rank,
            value_dim=geo["value_dim"],
            window=geo["window"],
            erase=geo["erase"],
            max_rank=settings.get("max_rank"),
        )
        del result["omega"]
        label = format(float(rank), "g")
        name = f"allocations/g{label}.pt"
        atomic_torch(result, root / name)
        entries[label] = dict(
            file=name,
            dense_heads=result["meta"]["dense_heads"],
            max_rank=result["meta"]["max_sketch_rank"],
        )
    paths = dict(
        generation_tokens="data/generation_tokens.pt",
        allocation_tokens="data/allocation_tokens.pt",
        covariance="statistics/covariance.pt",
        basis="basis/omega.pt",
        curves="statistics/paired_scores.pt",
    )
    names = list(paths.values()) + [e["file"] for e in entries.values()]
    files = {
        name: dict(bytes=(root / name).stat().st_size, sha256=checksum(root / name))
        for name in names
    }
    manifest = dict(
        schema_version=1,
        model=config["model"]["name"],
        family=config["model"]["family"],
        K=geo["key_dim"],
        V=geo["value_dim"],
        W=geo["window"],
        groups=geo["groups"],
        erase=geo["erase"],
        basis_fingerprint=fingerprint(basis["omega"]),
        allocations=entries,
        files=files,
        allocation_objective="full-gram",
        allocation_policy="fixed_rank_cost",
        inference_pivots=4,
        collection_reused=False,
        adapter=config["adapter"],
        **paths,
    )
    atomic_json(manifest, root / "manifest.json")


def import_engine():
    """Resolve the installed vLLM ahead of the project's vllm/ source directory.

    The project root is on sys.path (PYTHONPATH and the working directory). Its
    vllm/ subdirectory would otherwise shadow an editable vLLM install, which is
    resolved by an import hook rather than a path entry, as an empty namespace
    package, here and in spawned engine processes.
    """
    if importlib.util.find_spec("vllm").origin is not None:
        return
    root = Path(__file__).resolve().parents[3]
    saved = sys.path[:]
    sys.path[:] = [p for p in saved if Path(p or os.getcwd()).resolve() != root]
    try:
        spec = importlib.util.find_spec("vllm")
    finally:
        sys.path[:] = saved
    if spec is None or spec.origin is None:
        raise ImportError("vLLM is not installed in this interpreter")
    base = str(Path(spec.origin).parents[1])
    sys.path.insert(0, base)
    os.environ["PYTHONPATH"] = os.pathsep.join([base, os.environ.get("PYTHONPATH", "")])


def execute(config, root, stage, identity):
    torch.set_num_threads(config["runtime"].get("cpu_threads", 2))
    if stage in ("generate", "covariance"):
        import_engine()
    if stage == "generate":
        from .engine import generate

        generate(config, root, identity)
    elif stage == "covariance":
        from .engine import covariance

        covariance(config, root, identity)
    elif stage == "basis":
        spec = config["basis"]
        out = fit(
            load(root / "statistics/covariance.pt"),
            config["geometry"]["groups"],
            spec["rank"],
            spec["ridge"],
        )
        atomic_torch(out, root / "basis/omega.pt")
    elif stage == "paired":
        from .paired import collect

        collect(config, root, identity)
    elif stage == "allocate":
        allocations(config, root)
    else:
        raise ValueError(stage)


def main():
    # Selecting an interpreter does not activate its environment's executables.
    # Build tools (e.g. ninja) used by native kernels must follow that interpreter.
    bins = [str(Path(sys.executable).parent), str(Path(sys.base_prefix) / 'bin')]
    os.environ['PATH'] = os.pathsep.join(bins + [os.environ.get('PATH', '')])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    execute(read_config(args.config), args.out, args.stage, args.identity)


if __name__ == "__main__":
    main()
