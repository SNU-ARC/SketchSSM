# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Decode throughput of one arm on synthetic prompts, per batch size.

For each batch size B, B random-token prompts of ``--input-len`` tokens are
submitted at once (``max_num_seqs`` = the largest B) and each generates exactly
N = ``--output-len`` tokens (``ignore_eos``). The same requests are also run
with one output token, so the prefill cancels out:

    decode tok/s = B * (N - 1) / (time(N tokens) - time(1 token))

Each batch size is timed ``--repeats`` times after a warm-up run, and the
median is reported. All requests start together, so the ReplaySSM / SketchSSM
windows of all rows are aligned (W - 1 non-flush steps, then one flush step).

Example (from the repository root, with vllm/ installed):

    python evaluation/throughput.py --arm standard --batch 128 256 --out runs/tput/standard
    python evaluation/throughput.py --arm sketchssm --sketchssm-mean-rank 8 \\
        --batch 128 256 --out runs/tput/sketchssm
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

NANO = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
CALIBRATION = "SketchSSM/Nemotron-Nano-9B-v2-BF16"


def engine_kwargs(args) -> dict:
    kw = {
        "model": args.model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": args.input_len + args.output_len,
        "max_num_seqs": max(args.batch),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "mamba_ssm_cache_dtype": "float32",
        "enable_prefix_caching": False,
        "seed": 0,
    }
    if args.arm == "sketchssm":
        kw["sketchssm"] = args.calibration
        if args.sketchssm_mean_rank is not None:
            kw["sketchssm_mean_rank"] = args.sketchssm_mean_rank
        kw["replayssm_buffer_len"] = args.window
    elif args.arm == "replayssm":
        kw["use_replayssm"] = True
        kw["replayssm_buffer_len"] = args.window
    return kw


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", choices=["standard", "replayssm", "sketchssm"], required=True)
    p.add_argument("--model", default=NANO)
    p.add_argument("--calibration", default=CALIBRATION, help="SketchSSM calibration")
    p.add_argument("--sketchssm-mean-rank", type=float, default=None)
    p.add_argument("--window", type=int, default=16, help="ReplaySSM / SketchSSM window W")
    p.add_argument("--batch", type=int, nargs="+", default=[128, 256])
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-num-batched-tokens", type=int, default=65536)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # SketchSSM needs Model Runner V2 and Triton ReplaySSM needs V1.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0" if args.arm == "replayssm" else "1"

    import numpy as np

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.platforms import current_platform

    llm = LLM(**engine_kwargs(args))
    vocab = llm.get_tokenizer().vocab_size
    rng = np.random.default_rng(0)

    def timed(prompts, n):
        params = SamplingParams(max_tokens=n, ignore_eos=True, temperature=0)
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        assert all(len(o.outputs[0].token_ids) == n for o in outs)
        return elapsed

    results = []
    for batch in args.batch:
        ids = rng.integers(1000, vocab - 1000, size=(batch, args.input_len))
        prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in ids]
        timed(prompts, args.output_len)  # warm-up
        rates = []
        for _ in range(args.repeats):
            t1 = timed(prompts, 1)
            tn = timed(prompts, args.output_len)
            rates.append(batch * (args.output_len - 1) / (tn - t1))
        row = {"batch": batch, "decode_tok_s": statistics.median(rates), "runs": rates}
        results.append(row)
        print(f"{args.arm} batch {batch}: {row['decode_tok_s']:,.0f} decode tok/s", flush=True)

    summary = {
        "arm": args.arm,
        "model": args.model,
        "calibration": args.calibration if args.arm == "sketchssm" else None,
        "sketchssm_mean_rank": args.sketchssm_mean_rank,
        "gpu": current_platform.get_device_name(0),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
