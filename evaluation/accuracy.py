# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Accuracy of one arm on IFEval, MATH-500, HumanEval and MBPP, as avg@K.

lm-evaluation-harness 0.4.12 with its vLLM backend and the settings of the
decode demo: chat template, few-shot examples as multi-turn chat, system
instruction ``/no_think``, MATH-500 with ``max_gen_toks`` 1024 (lm-eval's
default is 256), and K samples per prompt (T=0.6, top-p 0.95, seeds
``--seed`` ... ``--seed + K - 1``). Accuracy per task is the mean over problems
of the per-problem mean over the K samples, with the standard error over
problems.

Example (from the repository root, with vllm/ installed):

    python evaluation/accuracy.py --arm standard --out runs/acc/standard
    python evaluation/accuracy.py --arm replayssm --out runs/acc/replayssm
    python evaluation/accuracy.py --arm sketchssm --sketchssm-mean-rank 8 \\
        --out runs/acc/sketchssm
"""

import argparse
import json
import os
from pathlib import Path

NANO = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
CALIBRATION = "SketchSSM/Nemotron-Nano-9B-v2-BF16"
TASKS = ["ifeval", "minerva_math500", "humaneval_instruct", "mbpp_instruct"]
# Headline metric per task: (result key, per-sample key, filter, label).
METRICS = {
    "ifeval": (
        "prompt_level_strict_acc,none", "prompt_level_strict_acc", "none",
        "IFEval prompt strict",
    ),
    "minerva_math500": ("math_verify,none", "math_verify", "none", "MATH-500 math-verify"),
    "humaneval_instruct": (
        "pass@1,create_test", "pass@1", "create_test", "HumanEval (instruct) pass@1",
    ),
    "mbpp_instruct": (
        "pass_at_1,extract_code", "pass_at_1", "extract_code", "MBPP (instruct) pass@1",
    ),
}  # fmt: skip


def engine_kwargs(args) -> dict:
    kw = {
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "mamba_ssm_cache_dtype": "float32",
        "enable_prefix_caching": False,
        "batch_size": "auto",
        "seed": args.seed,
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


def task_manager(args):
    """lm-eval's task manager with MATH-500's max_gen_toks raised."""
    from lm_eval.tasks import TaskManager

    class Manager(TaskManager):
        def load(self, *a, **kw):
            loaded = super().load(*a, **kw)
            for name, task in loaded["tasks"].items():
                if str(name) == "minerva_math500":
                    task.set_config(
                        key="generation_kwargs",
                        value={"max_gen_toks": args.math_max_gen_toks},
                        update=True,
                    )
            return loaded

    return Manager()


def correctness(results: dict) -> dict[tuple[str, int], float]:
    """(task, doc_id) -> the sample's score under the task's headline metric."""
    out = {}
    for task, samples in results.get("samples", {}).items():
        if task not in METRICS:
            continue
        _, metric, flt, _ = METRICS[task]
        for s in samples:
            if s.get("filter", "none") == flt and metric in s:
                out[(task, int(s["doc_id"]))] = float(s[metric])
    return out


def avg_at_k(runs: list[dict[tuple[str, int], float]], task: str) -> tuple[float, float]:
    per_doc: dict[int, list[float]] = {}
    for run in runs:
        for (t, doc), c in run.items():
            if t == task:
                per_doc.setdefault(doc, []).append(c)
    m = [sum(v) / len(v) for v in per_doc.values()]
    mean = sum(m) / len(m)
    var = sum((x - mean) ** 2 for x in m) / max(len(m) - 1, 1)
    return mean, (var / len(m)) ** 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", choices=["standard", "replayssm", "sketchssm"], required=True)
    p.add_argument("--model", default=NANO)
    p.add_argument("--calibration", default=CALIBRATION, help="SketchSSM calibration")
    p.add_argument("--sketchssm-mean-rank", type=float, default=None)
    p.add_argument("--window", type=int, default=16, help="ReplaySSM / SketchSSM window W")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--samples", type=int, default=4, help="K in avg@K")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--math-max-gen-toks", type=int, default=1024)
    p.add_argument("--batch", type=int, default=256, help="max_num_seqs")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--limit", type=float, default=None, help="docs per task (smoke tests)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    # SketchSSM needs Model Runner V2 and Triton ReplaySSM needs V1.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0" if args.arm == "replayssm" else "1"

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    lm = VLLM(pretrained=args.model, **engine_kwargs(args))
    tasks = args.tasks.split(",")
    runs, first = [], None
    for k in range(args.samples):
        gen = f"temperature={args.temperature},top_p={args.top_p},seed={args.seed + k}"
        if args.temperature > 0:
            gen = "do_sample=True," + gen
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=args.limit,
            gen_kwargs=gen,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            system_instruction="/no_think",
            log_samples=True,
            confirm_run_unsafe_code=True,
            task_manager=task_manager(args),
        )
        runs.append(correctness(results))
        first = first or results

    accuracy = {}
    for task in tasks:
        if task in METRICS:
            mean, se = avg_at_k(runs, task)
            accuracy[task] = {
                "metric": f"{METRICS[task][3]} avg@{args.samples}",
                "value": mean,
                "stderr": se,
            }
    values = [a["value"] for a in accuracy.values()]
    ses = [a["stderr"] for a in accuracy.values()]
    summary = {
        "arm": args.arm,
        "model": args.model,
        "calibration": args.calibration if args.arm == "sketchssm" else None,
        "sketchssm_mean_rank": args.sketchssm_mean_rank,
        "samples": args.samples,
        "accuracy": accuracy,
        "average": {
            "value": sum(values) / len(values),
            "stderr": sum(s * s for s in ses) ** 0.5 / len(ses),
        },
        "versions": first.get("versions"),
        "n-shot": first.get("n-shot"),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    rows = [f"| {a['metric']} | {100 * a['value']:.1f} ± {100 * a['stderr']:.1f} |"
            for a in accuracy.values()]  # fmt: skip
    avg = summary["average"]
    rows.append(f"| Average | {100 * avg['value']:.1f} ± {100 * avg['stderr']:.1f} |")
    table = "\n".join([f"| {args.arm} | accuracy (%) |", "|---|---|", *rows])
    (out / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
