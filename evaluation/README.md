# Evaluation

Three scripts compare a model in vLLM without SketchSSM (`standard`), with
ReplaySSM (`replayssm`) and with SketchSSM (`sketchssm`). All of them use the
bundled vLLM (`vllm/`, installed) and, for SketchSSM, a calibration from the
Hugging Face Hub or a local file. The default model is Nemotron Nano 9B v2
with the calibration `SketchSSM/Nemotron-Nano-9B-v2-BF16` (`--model`,
`--calibration` and `--sketchssm-mean-rank` for others).

| script | measures |
|---|---|
| `accuracy.py` | IFEval, MATH-500, HumanEval and MBPP accuracy as avg@K (lm-eval 0.4.12) |
| `throughput.py` | decode throughput on synthetic prompts at chosen batch sizes |
| `linear_attention.py` | per-layer recurrent (linear-attention) decode latency, by kernel (Nsight Systems) |

SketchSSM runs on vLLM's Model Runner V2 and the Triton ReplaySSM on Model
Runner V1; the scripts select the runner per arm, and Standard uses V2.

## Accuracy

```bash
pip install "lm-eval[math]==0.4.12" langdetect immutabledict nltk
python evaluation/accuracy.py --arm standard  --out runs/acc/standard
python evaluation/accuracy.py --arm replayssm --out runs/acc/replayssm
python evaluation/accuracy.py --arm sketchssm --sketchssm-mean-rank 8 --out runs/acc/sketchssm
```

lm-eval's own prompts and scoring through its vLLM backend, with the chat
template, few-shot examples as multi-turn chat and the system instruction
`/no_think`: IFEval (prompt-level strict), MATH-500 (4-shot, math-verify,
`max_gen_toks` raised from 256 to 1024), HumanEval instruct and MBPP instruct
(3-shot, pass@1). Each prompt is sampled K = `--samples` times (default 4; T=0.6,
top-p 0.95, one seed per sample); a task's accuracy is the mean over problems
of the per-problem mean over the K samples, ± the standard error over problems.
HumanEval and MBPP execute the generated code. `results.md` and `results.json`
are written to `--out`.

## Decode throughput

```bash
python evaluation/throughput.py --arm standard  --batch 128 256 --out runs/tput/standard
python evaluation/throughput.py --arm replayssm --batch 128 256 --out runs/tput/replayssm
python evaluation/throughput.py --arm sketchssm --sketchssm-mean-rank 8 --batch 128 256 \
    --out runs/tput/sketchssm
```

For each batch size B, B random-token prompts of `--input-len` tokens (default
1024) are submitted at once, and each generates exactly `--output-len` tokens
(default 512). The same requests are also run with a single output token, and
the difference of the two times is the decode time, so

    decode tok/s = B × (output_len − 1) / (time(output_len) − time(1))

reported as the median of `--repeats` runs (default 3) after a warm-up. All
requests start together, so the windows of all rows are aligned (W − 1
non-flush steps, then one flush step); in online serving, rows are spread over
the window.

Nemotron Nano 9B v2 on one RTX PRO 6000 Blackwell (input 1024, output 512
tokens; SketchSSM mean rank 8, W=16):

| batch | Standard | + ReplaySSM | + SketchSSM | SketchSSM vs ReplaySSM |
|---|---|---|---|---|
| 128 | 3,159 tok/s | 4,351 tok/s (1.38x) | 6,493 tok/s (2.06x) | 1.49x |
| 256 | 3,687 tok/s | 5,405 tok/s (1.47x) | 9,541 tok/s (2.59x) | 1.77x |

## Per-layer linear-attention latency

```bash
python evaluation/linear_attention.py --arm standard  --batch 256 --out runs/b256/standard
python evaluation/linear_attention.py --arm replayssm --batch 256 --out runs/b256/replayssm
python evaluation/linear_attention.py --arm sketchssm --batch 256 \
    --calibration outputs/nano_frames.pt --out runs/b256/sketchssm
VLLM_SKETCHSSM_USE_CUDA=0 python evaluation/linear_attention.py --arm sketchssm \
    --batch 256 --calibration outputs/nano_frames.pt --out runs/b256/sketchssm_triton
python evaluation/linear_attention.py --summarize runs/b256/*
```

This runs the model's decode steps in vLLM's CUDA graphs under Nsight Systems
(`nsys` required), attributes every kernel (graph nodes included) to its engine
step, and reports the recurrent part of each step per layer, by component:

| component | kernels |
|---|---|
| readout | SketchSSM non-flush read (`nf_kernel`, `_sketch_decode_kernel`), ReplaySSM readout, Standard state update |
| flush | SketchSSM flush + sketch rebuild (`flush_kernel`, Triton `_sketch_flush_kernel`, `_cold_build_kernel`) |
| shared B/C | per-group B·C precompute of the ring (`_bc_pre_kernel`, ReplaySSM precompute) |
| basis transform | B/C rotation into the calibration frames (`_rot_inplace_kernel`) |

Projections, convolution, attention and sampling are reported as `other` and
excluded from the recurrent total. Steps are grouped by the ReplaySSM ring:
non-flush steps (no row flushes), flush steps (every row flushes) and mixed
steps. By default all requests start decoding together, so a window is 15
non-flush steps and one flush step; the window total is compared with 16
Standard steps. `--stagger 1` spreads request starts over a window, as in
online serving, which makes decode steps mixed.

With a portable calibration file (`outputs/calibrations/<model>/calibration.pt`),
pass the rank budget with `--sketchssm-mean-rank`. The recurrent layer count is
read from the model config.

For one layer's decode step with synthetic inputs, see the kernel
microbenchmarks in [`sketchssm/kernels/`](../sketchssm/kernels/README.md).
