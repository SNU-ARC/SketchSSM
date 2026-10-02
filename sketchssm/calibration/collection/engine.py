# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Native vLLM generation and teacher-forced covariance collection."""

import torch

from .data import prepare
from .storage import atomic_json, atomic_torch, load


def create_engine(config, capture=False):
    from vllm import LLM

    model, runtime, gen = config["model"], config["runtime"], config["generation"]
    kwargs = dict(
        model=model["checkpoint"],
        revision=model.get("revision"),
        tokenizer_revision=model.get("tokenizer_revision"),
        dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        seed=gen["seed"],
        max_model_len=gen["prompt_tokens"] + gen["new_tokens"] + 1,
        max_num_seqs=runtime["batch_size"],
        gpu_memory_utilization=0.8,
    )
    options = runtime.get("engine_kwargs", {})
    forbidden = {
        "model",
        "revision",
        "tokenizer_revision",
        "mamba_ssm_cache_dtype",
        "worker_extension_cls",
        "logits_processors",
        "enforce_eager",
        "enable_prefix_caching",
        "max_num_seqs",
        "enable_replayssm",
        "use_replayssm",
        "quantization",
    }
    if forbidden.intersection(options):
        raise ValueError(
            f"Calibration-critical engine options cannot be overridden: {forbidden.intersection(options)}"
        )
    if options.get("async_scheduling") or options.get("speculative_config"):
        raise ValueError(
            "Capture requires synchronous scheduling without speculative decoding"
        )
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if options.get(key, 1) != 1:
            raise ValueError("Built-in capture bindings require single-rank execution")
    kwargs.update(options)
    if capture:
        family = model["family"]
        kwargs["worker_extension_cls"] = runtime.get(
            "covariance_worker",
            f"sketchssm.calibration.collection.native.{family}.Worker",
        )
        if runtime["teacher_force"] == "logits_processor":
            kwargs["logits_processors"] = [
                "sketchssm.calibration.collection.forced_tokens:SavedTokens"
            ]
        else:
            kwargs["enable_trace_replay"] = True
    return LLM(**kwargs)


def close_engine(engine):
    shutdown = getattr(engine.llm_engine, "shutdown", None)
    if shutdown:
        shutdown()
    elif hasattr(engine.llm_engine, "engine_core"):
        engine.llm_engine.engine_core.shutdown()


def generate(config, root, identity):
    from vllm import SamplingParams

    prompt_path = root / "data" / "prepared.pt"
    if prompt_path.exists():
        prepared = load(prompt_path)
        if prepared["identity"] != identity:
            raise ValueError("Prepared data identity changed")
        prompts, paired = prepared["prompts"], prepared["paired"]
    else:
        prompts, paired = prepare(config)
        atomic_torch(
            dict(identity=identity, prompts=prompts, paired=paired), prompt_path
        )
    atomic_torch(
        dict(
            paired_validation_token_ids=paired,
            meta=dict(
                split=config["paired"]["split"],
                block_start=config["paired"]["block_start"],
            ),
        ),
        root / "data/allocation_tokens.pt",
    )
    batch = config["runtime"]["batch_size"]
    gen = config["generation"]
    fragments = root / "progress" / "generation"
    results = []
    engine = None
    try:
        for offset in range(0, len(prompts), batch):
            path = fragments / f"{offset:06d}.pt"
            count = min(batch, len(prompts) - offset)
            if path.exists():
                saved = load(path)
                if saved["identity"] != identity or saved["tokens"].shape != (
                    count,
                    gen["new_tokens"],
                ):
                    raise ValueError("Generation checkpoint mismatch")
                rows = saved["tokens"]
            else:
                if engine is None:
                    engine = create_engine(config)
                inputs = [
                    dict(prompt_token_ids=row.tolist())
                    for row in prompts[offset : offset + batch]
                ]
                params = SamplingParams(
                    temperature=0.0,
                    max_tokens=gen["new_tokens"],
                    min_tokens=gen["new_tokens"],
                    ignore_eos=True,
                    seed=gen["seed"],
                )
                outputs = engine.generate(inputs, params, use_tqdm=False)
                rows = torch.tensor(
                    [o.outputs[0].token_ids for o in outputs], dtype=torch.long
                )
                if rows.shape != (count, gen["new_tokens"]):
                    raise RuntimeError("Incomplete generated sequences")
                atomic_torch(dict(identity=identity, tokens=rows), path)
            results.append(rows)
            print(f"generate: {offset + count}/{len(prompts)}", flush=True)
        atomic_torch(
            dict(
                prompt_token_ids=prompts,
                generated_token_ids=torch.cat(results),
                meta=dict(
                    native_forward=True,
                    generation=config["generation"],
                    identity=identity,
                ),
            ),
            root / "data/generation_tokens.pt",
        )
    finally:
        if engine is not None:
            close_engine(engine)


def covariance(config, root, identity):
    from vllm import SamplingParams

    source = load(root / "data/generation_tokens.pt")
    prompts, generated = source["prompt_token_ids"], source["generated_token_ids"]
    path = root / "statistics/covariance.pt"
    start = 0
    if path.exists():
        prior = load(path)
        if prior["meta"]["identity"] != identity:
            raise ValueError("Covariance checkpoint identity changed")
        start = prior["processed_sequences"]
    if start == len(prompts):
        return
    engine = create_engine(config, capture=True)
    try:
        audit = engine.collective_rpc(
            "install_covariance", args=(config, str(path) if start else None)
        )
        atomic_json(audit, root / "progress/native_audit.json")
        batch, gen = config["runtime"]["batch_size"], config["generation"]
        for offset in range(start, len(prompts), batch):
            engine.collective_rpc("reset_covariance_slots")
            inputs = [
                dict(prompt_token_ids=row.tolist())
                for row in prompts[offset : offset + batch]
            ]
            params = []
            for row in generated[offset : offset + batch]:
                forcing = (
                    {"extra_args": {"saved_tokens": row.tolist()}}
                    if config["runtime"]["teacher_force"] == "logits_processor"
                    else {"trace_decode_token_ids": row.tolist() + [0]}
                )
                params.append(
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=gen["new_tokens"] + 1,
                        min_tokens=gen["new_tokens"] + 1,
                        ignore_eos=True,
                        **forcing,
                        seed=gen["seed"],
                        detokenize=False,
                    )
                )
            outputs = engine.generate(inputs, params, use_tqdm=False)
            if len(outputs) != len(inputs):
                raise RuntimeError("Missing native outputs")
            for i, output in enumerate(outputs):
                if list(output.prompt_token_ids) != prompts[offset + i].tolist():
                    raise RuntimeError("Native engine reordered or changed prompts")
                if len(output.outputs[0].token_ids) != gen["new_tokens"] + 1:
                    raise RuntimeError("Missing sentinel output")
                if (
                    output.outputs[0].token_ids[: gen["new_tokens"]]
                    != generated[offset + i].tolist()
                ):
                    raise RuntimeError(
                        "Native teacher forcing failed; no covariance committed"
                    )
            completed = offset + len(inputs)
            path.parent.mkdir(parents=True, exist_ok=True)
            engine.collective_rpc(
                "save_covariance",
                args=(
                    str(path),
                    completed,
                    len(inputs),
                    dict(
                        identity=identity,
                        native_forward=True,
                        window=config["geometry"]["window"],
                    ),
                ),
            )
            print(f"covariance: {completed}/{len(prompts)}", flush=True)
    finally:
        close_engine(engine)
