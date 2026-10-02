# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Create a small original-weight Nano checkpoint for integration tests only.

Keeps the first N blocks, embeddings and output head without changing tensor
values or precision. This is not a model for accuracy or final calibration.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "nemotron_h" or config.get("quantization_config"):
        raise ValueError("This integration fixture expects original BF16 Nano weights")
    if not 0 < args.layers < config["num_hidden_layers"]:
        raise ValueError("Choose a positive proper subset of the model's layers")
    args.out.mkdir(parents=True, exist_ok=False)
    original_layers = config["num_hidden_layers"]
    config["num_hidden_layers"] = args.layers
    config["hybrid_override_pattern"] = config["hybrid_override_pattern"][: args.layers]
    config.pop("auto_map", None)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    wanted = {}
    for name, filename in index["weight_map"].items():
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None or int(match[1]) < args.layers:
            wanted.setdefault(filename, []).append(name)
    weights = {}
    for filename, names in wanted.items():
        with safe_open(source / filename, framework="pt", device="cpu") as reader:
            for name in names:
                weights[name] = reader.get_tensor(name).contiguous()
    save_file(weights, args.out / "model.safetensors", metadata={"format": "pt"})
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        if (source / name).exists():
            shutil.copyfile(source / name, args.out / name)
    audit = {
        "purpose": "reduced-layer integration gate, not accuracy or final calibration",
        "source": str(source),
        "original_layers": original_layers,
        "retained_layers": args.layers,
        "pattern": config["hybrid_override_pattern"],
        "tensors": len(weights),
        "bytes": sum(t.numel() * t.element_size() for t in weights.values()),
        "weights_modified": False,
        "dtype_conversion": False,
    }
    (args.out / "subset_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
