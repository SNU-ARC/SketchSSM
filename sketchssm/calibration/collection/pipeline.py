# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Isolated collection stages with checked resume and a portable final bundle."""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .config import STAGES, read_config
from .storage import atomic_json, checksum, digest

OUTPUTS = {
    "generate": ("data/generation_tokens.pt", "data/allocation_tokens.pt"),
    "covariance": ("statistics/covariance.pt",),
    "basis": ("basis/omega.pt",),
    "paired": ("statistics/paired_scores.pt",),
    "allocate": ("manifest.json",),
}
INPUTS = {
    "generate": (),
    "covariance": ("data/generation_tokens.pt",),
    "basis": ("statistics/covariance.pt",),
    "paired": ("data/allocation_tokens.pt", "basis/omega.pt"),
    "allocate": (
        "data/generation_tokens.pt",
        "data/allocation_tokens.pt",
        "statistics/covariance.pt",
        "basis/omega.pt",
        "statistics/paired_scores.pt",
    ),
}


def run(config_path, output, stage=None, resume=False):
    cfg = read_config(config_path)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(
            "Output is not empty; use --resume to verify and continue this run"
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = {
            str(p.relative_to(Path(__file__).parents[1])): checksum(p)
            for p in sorted(Path(__file__).parents[1].rglob("*.py"))
        }
        identity = digest(cfg)
        config_file = output / "config.yaml"
        if config_file.exists():
            if digest(yaml.safe_load(config_file.read_text())) != identity:
                raise ValueError(
                    "Configuration changed; use a separate output directory"
                )
        else:
            config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for current in [stage] if stage else STAGES:
            deps = {name: checksum(output / name) for name in INPUTS[current]}
            signature = digest(
                {"config": identity, "inputs": deps, "implementation": code}
            )
            marker = output / "progress" / f"{current}.json"
            if marker.exists():
                prior = json.loads(marker.read_text())
                if prior["signature"] != signature:
                    raise ValueError(f"{current}: input identity changed")
                if prior["status"] == "complete":
                    for name, expected in prior["outputs"].items():
                        if checksum(output / name) != expected:
                            raise ValueError(f"Artifact changed: {name}")
                    print(f"{current}: verified complete", flush=True)
                    continue
            atomic_json({"status": "running", "signature": signature}, marker)
            runtime = cfg["runtime"]
            python = runtime.get(
                f"{current}_python", runtime.get("python", sys.executable)
            )
            # A fresh process releases the entire engine before the next stage.
            cmd = [
                python,
                "-m",
                "sketchssm.calibration.collection.worker",
                "--config",
                str(config_file),
                "--out",
                str(output),
                "--stage",
                current,
                "--identity",
                signature,
            ]
            try:
                env = os.environ.copy()
                for key in list(env):
                    if key.startswith(
                        (
                            "NS_",
                            "NESTED_SSM",
                            "HEADOMEGA",
                            "FZV2_",
                            "GLM_KDA_",
                            "Q38NEXT_",
                            "VLLM_USE_REPLAY",
                            "VLLM_REPLAY",
                            "SUPER_",
                            "NANO_",
                        )
                    ):
                        env.pop(key)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
                env.update(
                    {k: str(v) for k, v in runtime.get("environment", {}).items()}
                )
                subprocess.run(cmd, check=True, env=env)
                names = list(OUTPUTS[current])
                if current == "allocate":
                    manifest = json.loads((output / "manifest.json").read_text())
                    names.extend(manifest["files"])
                hashes = {name: checksum(output / name) for name in names}
                atomic_json(
                    {"status": "complete", "signature": signature, "outputs": hashes},
                    marker,
                )
            except BaseException as exc:
                atomic_json(
                    {"status": "failed", "signature": signature, "error": str(exc)},
                    marker,
                )
                raise
    return output


def add_arguments(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--resume", action="store_true")
