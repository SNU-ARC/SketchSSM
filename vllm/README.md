<!-- markdownlint-disable MD001 MD041 -->
<!-- ===================== SketchSSM fork ===================== -->
# vLLM with SketchSSM (research fork)

> This directory is [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0)
> v0.30.0, upstream commit
> [`ced6857af`](https://github.com/vllm-project/vllm/commit/ced6857af), plus the
> SketchSSM commits. SketchSSM keeps the exact recurrent state and an exact
> flush every 16 decode steps (ReplaySSM's window), but on the steps in between
> reads a compact per-request sketch of the state instead of the full state. The
> per-layer frames and per-head sketch ranks come from the offline calibration
> in the parent repository.

## Try it out

```bash
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
pip install sketchssm   # the CUDA kernels; without them vLLM uses its Triton kernels

# A calibration from the Hugging Face Hub; pick the rank budget at load time.
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --trust-remote-code \
    --sketchssm SketchSSM/Nemotron-Nano-9B-v2-BF16 --sketchssm-mean-rank 8 \
    --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching
```

`--sketchssm` also takes a local portable calibration file (`python -m
sketchssm.calibration package` in the parent repository) or frames exported for
one budget (`python -m sketchssm.calibration export`, then no
`--sketchssm-mean-rank`).

| flag | meaning & constraints |
|---|---|
| `--sketchssm PATH_OR_REPO` | enables SketchSSM with a calibration: a local file, a directory or a Hugging Face repo id holding `calibration.pt`. Model Runner V2, FP32 SSM cache, no Mamba prefix caching, TP=1, no speculative decoding. |
| `--sketchssm-mean-rank R` | rank budget (mean sketch rank per state head, default 8) for a portable calibration; any value works, the per-head ranks and frames are allocated when the model loads. |
| `--replayssm-buffer-len W` | SketchSSM window (flush every W steps, default 16); the CUDA kernels take any multiple of 16. Tuned knobs are per window (`--window` in the tuners); without a file for W the W=16 knobs, then the GPU's defaults, are used. |
| `VLLM_SKETCHSSM_USE_CUDA=0` | force the portable Triton kernels. |

`--use-replayssm` alone is upstream ReplaySSM for Mamba-2; Gated DeltaNet and
KDA layers use their window rings only for SketchSSM. See
[`docs/features/sketchssm.md`](docs/features/sketchssm.md).

| Model | Calibration (Hugging Face) | Collected with |
|---|---|---|
| Nemotron Nano 9B v2 | `SketchSSM/Nemotron-Nano-9B-v2-BF16` | BF16 `nvidia/NVIDIA-Nemotron-Nano-9B-v2` |
| Nemotron 3 Super | `SketchSSM/Nemotron-3-Super-NVFP4` | NVFP4 `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| Qwen3.8 Flash-Next | `SketchSSM/Qwen3.8-Flash-Next-NVFP4` | NVFP4 `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| GLM 5.3 Flash | `SketchSSM/GLM-5.3-Flash-NVFP4` | NVFP4 `RedHatAI/GLM-5.3-Flash-NVFP4` |
| Qwen3.5 9B | `SketchSSM/Qwen3.5-9B-BF16` | BF16 `Qwen/Qwen3.5-9B` |

The CUDA kernels live in the parent repository's
[`sketchssm/kernels/`](../sketchssm/kernels/) package; vLLM uses them when it is
installed and its own Triton kernels otherwise. Their build knobs default to the
paper's B300 configuration (H100 and B300 files are included) and can be tuned per GPU,
layer shape and window with the package's tools, for example:

```bash
python -m sketchssm.kernels.tools.tune_mamba2 \
    --calibration nano_g6.pt --layer 13 --head-dim 80 --save-configs
python -m sketchssm.kernels.tools.benchmark_mamba2 \
    --calibration nano_g6.pt --layer 13 --head-dim 80 --phase mixed
```

Outputs do not depend on the knobs.

Model-level measurements (every recurrent kernel of real decode steps, as in
the paper) are in the parent repository's `benchmarks/`.

## Status

| model family | kernels | shapes |
|---|---|---|
| Mamba-2 (Nemotron Nano / Super) | the paper's CUDA kernels and a portable Triton fallback | state size 64, 128 or 256 |
| Gated DeltaNet (Qwen3.5, Qwen3.8 Flash-Next) | the paper's CUDA kernels and a portable Triton fallback | 1 to 8 value heads per key head, head dim 128 |
| KDA (GLM 5.3 Flash) | the paper's CUDA kernels and a portable Triton fallback | head dim 128 |

A full vLLM build precompiles the CUDA kernels (`cmake/external_projects/sketchssm.cmake`);
other shapes are compiled at run time with NVRTC, without nvcc. The kernels
give bit-identical results to the paper's kernels.

## Core implementation files

- [`vllm/model_executor/layers/mamba/sketchssm.py`](vllm/model_executor/layers/mamba/sketchssm.py)
  and [`sketchssm_calibration.py`](vllm/model_executor/layers/mamba/sketchssm_calibration.py):
  calibration loading and the mean-rank allocation.
- Per-layer objects the model layers call:
  [`mamba2_sketchssm.py`](vllm/model_executor/layers/mamba/mamba2_sketchssm.py),
  [`gdn/gdn_sketchssm.py`](vllm/model_executor/layers/mamba/gdn/gdn_sketchssm.py),
  [`kda_sketchssm.py`](vllm/model_executor/layers/mamba/kda_sketchssm.py).
- [`vllm/model_executor/layers/mamba/ops/`](vllm/model_executor/layers/mamba/ops/):
  `sketchssm_mamba2*.py`, `gdn_sketchssm_*.py` and `kda_sketchssm_*.py` (storage,
  rotation, cold build and the portable Triton decode) and `sketchssm_kernels.py`
  (selects the CUDA kernels of the `sketchssm.kernels` package).
- [`../sketchssm/kernels/`](../sketchssm/kernels/): the paper's CUDA kernels.

The upstream vLLM README follows.

---
<!-- ========================================================== -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
