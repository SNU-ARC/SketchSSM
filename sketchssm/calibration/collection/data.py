# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Common raw WikiText token selection; generation and gradient data are disjoint."""

import torch


def prepare(config):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    m = config["model"]
    tok = AutoTokenizer.from_pretrained(
        m["checkpoint"],
        revision=m.get("tokenizer_revision") or m.get("revision"),
        trust_remote_code=False,
    )
    dataset = config["dataset"]

    def tokens(split):
        ds = load_dataset(
            "Salesforce/wikitext" if dataset["name"] == "wikitext" else dataset["name"],
            dataset.get("subset"),
            split=split,
            revision=dataset.get("revision"),
        )
        return tok("\n\n".join(ds["text"]), add_special_tokens=False)["input_ids"]

    g, p = config["generation"], config["paired"]
    train = tokens(g["split"])
    count = g["sequences"] * g["prompt_tokens"]
    if len(train) < count:
        raise ValueError("Generation corpus too short")
    prompts = torch.tensor(train[:count], dtype=torch.long).reshape(
        g["sequences"], g["prompt_tokens"]
    )
    validation = tokens(p["split"])
    T = p["sequence_length"]
    start = p["block_start"] * T
    if len(validation) < start + p["sequences"] * T + 1:
        raise ValueError("Paired corpus too short")
    blocks = torch.tensor(
        [
            validation[start + i * T : start + (i + 1) * T + 1]
            for i in range(p["sequences"])
        ]
    )
    return prompts, blocks
