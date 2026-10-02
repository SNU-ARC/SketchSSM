# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Precompile (AOT) kernel specializations to cubins.

Runs from a source checkout without installing the package, with only torch
and NVRTC (no nvcc, no GPU)::

    python <src>/sketchssm/kernels/build.py --arch "9.0a;10.0f" --out <dir>
    python -m sketchssm.kernels.build --arch 9.0a --out <dir> --layers my.json

Output: ``<dir>/manifest.json`` and ``<dir>/<arch>/<kind>_<key>.cubin``. The
runtime looks precompiled kernels up in ``$SKETCHSSM_KERNELS_AOT_DIR``, the
folders given to ``set_aot_dirs`` and the package's ``aot/`` folder, before
compiling with NVRTC.
"""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if not __package__:
    # Run as a script: import this folder as a package under a private name,
    # so that the relative imports below work without installing anything.
    import importlib.util

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_sketchssm_kernels_build", _here / "__init__.py",
        submodule_search_locations=[str(_here)],
    )  # fmt: skip
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _pkg
    _spec.loader.exec_module(_pkg)
    __package__ = _spec.name

from . import _kernel as kn  # noqa: E402
from . import _runtime as rt  # noqa: E402
from . import gdn, kda, mamba2  # noqa: E402

# The default set: the layer shapes of the published models, at tensor
# parallel degrees 1, 2, 4 and 8 and windows 16, 32 and 64.
DEFAULT_LAYERS = [
    # Nemotron Nano 9B v2 and Nemotron 3 Super (Mamba-2)
    dict(family="mamba2", num_heads=128, head_dim=80, state_size=128, n_groups=8),
    dict(family="mamba2", num_heads=128, head_dim=64, state_size=128, n_groups=8),
    # Qwen3.5-9B and Qwen3.8 Flash-Next (GDN)
    dict(family="gdn", num_k_heads=16, num_v_heads=32),
    dict(family="gdn", num_k_heads=16, num_v_heads=48),
    # GLM 5.3 Flash (KDA)
    dict(family="kda", num_heads=64),
]
DEFAULT_TP = (1, 2, 4, 8)
# Oldest architecture of each family (KDA's flush uses TMA).
MIN_CC = {"mamba2": 80, "gdn": 80, "kda": 90}
DEFAULT_WINDOWS = (16, 32, 64)


def _split(layer: dict, tp: int) -> dict | None:
    """A layer's shape on one of ``tp`` tensor-parallel ranks."""
    layer = dict(layer)
    keys = {"mamba2": ("num_heads", "n_groups"), "gdn": ("num_k_heads", "num_v_heads"),
            "kda": ("num_heads",)}[layer["family"]]  # fmt: skip
    for k in keys:
        if layer[k] % tp:
            return None
        layer[k] //= tp
    return layer


def layer_specs(layer: dict, window: int, target: rt.Target) -> list[tuple[kn.Spec, bool]]:
    fam = layer["family"]
    if fam == "mamba2":
        return mamba2.aot_specs(layer["num_heads"], layer["head_dim"], layer["state_size"],
                                layer["n_groups"], window, target)  # fmt: skip
    if fam == "gdn":
        return gdn.aot_specs(layer["num_k_heads"], layer["num_v_heads"], window, target)
    if fam == "kda":
        return kda.aot_specs(layer["num_heads"], window, target)
    raise ValueError(f"unknown family {fam!r}")


def _clear_caches() -> None:
    for fn in (mamba2.tuned_config, mamba2.window16_fallback, gdn.tuned_config, kda.tuned_config):
        fn.cache_clear()


def collect(archs, layers, tps, windows, device_names, specs_files) -> dict:
    """{(key, arch): (spec, required)} of everything to build."""
    todo: dict[tuple[str, str], tuple[kn.Spec, bool]] = {}

    def add(spec: kn.Spec, arch: str, required: bool) -> None:
        k = (spec.key, arch)
        old = todo.get(k)
        todo[k] = (spec, required or (old is not None and old[1]))

    for arch in archs:
        names = device_names if device_names else [None]
        targets = [rt.Target.for_arch(arch, name) for name in names]
        # The architecture defaults too (GPUs of the arch without tuned files).
        targets.append(rt.Target.for_arch(arch, f"{arch}_defaults"))
        for target in targets:
            _clear_caches()
            for layer in layers:
                if target.cc < MIN_CC[layer["family"]]:
                    continue
                for tp in tps:
                    shape = _split(layer, tp)
                    if shape is None:
                        continue
                    for w in layer.get("windows", windows):
                        for spec, required in layer_specs(shape, w, target):
                            add(spec, arch, required)
    for path in specs_files:
        for f in sorted(Path(path).rglob("*.json")) if Path(path).is_dir() else [Path(path)]:
            d = json.loads(f.read_text())
            for entry in d if isinstance(d, list) else [d]:
                if "spec" not in entry:
                    continue
                spec = kn.Spec.from_json(entry["spec"])
                for arch in archs:
                    add(spec, arch, True)
    return todo


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True,
                   help='architectures, ";"- or ","-separated: 9.0a, 10.0f, 10.0a, 10.3a, 12.0f, '
                        "90a, sm_100a, ...")  # fmt: skip
    p.add_argument("--out", required=True, help="output folder (manifest.json, <arch>/*.cubin)")
    p.add_argument("--clean", action="store_true", help="empty --out first (default: add to it)")
    p.add_argument("--layers", nargs="*", default=[],
                   help="JSON files with extra layers ([{family, shape..., windows?}])")  # fmt: skip
    p.add_argument("--no-default", action="store_true", help="skip the default model set")
    p.add_argument("--configs", nargs="*", default=[],
                   help="tuned-config folders searched first (like set_config_dirs)")
    p.add_argument("--device-name", nargs="*", default=[],
                   help="GPU names whose tuned files to build (default: the arch's reference GPU)")
    p.add_argument("--tp", default=",".join(map(str, DEFAULT_TP)), help="tensor-parallel degrees")
    p.add_argument("--windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    p.add_argument("--specs", nargs="*", default=[],
                   help="exact specializations: JSON files or folders, e.g. an NVRTC cache "
                        "(<cache>/nvrtc), which also holds stride-specialized Mamba-2 "
                        "non-flush kernels")  # fmt: skip
    p.add_argument("--nvrtc", help="libnvrtc to use (default: torch's CUDA wheel, then CUDA_HOME)")
    p.add_argument("--cuda-include", help="CUDA headers matching --nvrtc")
    p.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument("--list", action="store_true", help="print the specializations, build nothing")
    args = p.parse_args(argv)

    if args.nvrtc:
        os.environ["SKETCHSSM_KERNELS_NVRTC"] = args.nvrtc
    if args.cuda_include:
        os.environ["SKETCHSSM_KERNELS_CUDA_INCLUDE"] = args.cuda_include
    rt.set_config_dirs(args.configs)
    archs = sorted({kn.normalize_arch(a) for a in args.arch.replace(",", ";").split(";") if a.strip()})
    layers = [] if args.no_default else list(DEFAULT_LAYERS)
    for f in args.layers:
        layers += json.loads(Path(f).read_text())
    tps = [int(t) for t in args.tp.split(",")]
    windows = [int(w) for w in args.windows.split(",")]
    todo = collect(archs, layers, tps, windows, args.device_name, args.specs)
    if args.list:
        for (key, arch), (spec, required) in sorted(todo.items(), key=lambda t: (t[0][1], t[1][0].kind)):
            print(arch, spec.kind, key, "" if required else "(optional)", dict(spec.defines))
        print(f"{len(todo)} specializations")
        return 0

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / kn.MANIFEST
    entries: dict[tuple[str, str], dict] = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("source_digest") == kn.source_digest():
            entries = {(e["key"], e["arch"]): e for e in old["kernels"]}
    nv = kn._nvrtc.nvrtc()
    print(f"sketchssm.kernels build: {len(todo)} specializations for {', '.join(archs)} "
          f"with NVRTC {nv.version[0]}.{nv.version[1]} ({nv.path})", flush=True)  # fmt: skip

    def build(item):
        (key, arch), (spec, required) = item
        if (key, arch) in entries and (out / entries[(key, arch)]["file"]).exists():
            return item, None, None
        try:
            return item, kn.compile_spec(spec, arch), None
        except Exception as e:  # noqa: BLE001
            return item, None, e

    failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for ((key, arch), (spec, required)), art, err in pool.map(build, todo.items()):
            if err is not None:
                level = "error" if required else "skipped (optional)"
                print(f"{level}: {arch} {spec.kind} {key}: {str(err).splitlines()[0]}", flush=True)
                failed += required
                continue
            if art is None:
                continue
            rel = f"{arch}/{spec.kind}_{key}.cubin"
            (out / arch).mkdir(exist_ok=True)
            (out / rel).write_bytes(art.cubin)
            entries[(key, arch)] = dict(
                kind=spec.kind, key=key, arch=arch, file=rel, lowered=art.lowered,
                probes=art.probes, bytes=len(art.cubin), spec=spec.to_json(),
            )  # fmt: skip
    manifest = dict(
        format=kn.MANIFEST_FORMAT, version=kn.MANIFEST_VERSION,
        source_digest=kn.source_digest(), nvrtc=f"{nv.version[0]}.{nv.version[1]}",
        flags=list(kn.BASE_FLAGS),
        kernels=sorted(entries.values(), key=lambda e: (e["arch"], e["kind"], e["key"])),
    )  # fmt: skip
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, manifest_path)
    size = sum(e["bytes"] for e in entries.values())
    print(f"{len(entries)} kernels, {size / 2**20:.1f} MiB in {out} "
          f"({time.time() - t0:.0f} s, {failed} failed)", flush=True)  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
