# SketchSSM

SketchSSM speeds up the decode of recurrent (state space and linear attention) layers. Each request decodes in windows of `W` steps (`--replayssm-buffer-len`, 16 by default). The last step of a window flushes the window into the exact recurrent state and rebuilds a compact per-request sketch of it; the other steps read the sketch instead of the full state, except for dense heads.

The per-head sketch ranks and orthogonal frames come from an offline calibration of the model ([SketchSSM](https://github.com/SNU-ARC/SketchSSM)).

!!! warning
    SketchSSM is experimental.

## Usage

```bash
vllm serve <model> --sketchssm /path/to/calibration.pt
```

`--sketchssm` takes a file, a directory, or a Hugging Face repository id that contains `calibration.pt`. The file holds either:

- **Exported frames for one rank budget:** `frames[L, G, K, K]`, `m_table[L, H]` (sketch rank per head, 0 = dense), and optionally `dense_table[L, H]` and `layer_ids`.
- **A portable calibration:** the per-group basis and rank-score curves. `--sketchssm-mean-rank` sets the mean sketch rank per head (default 8); ranks and frames are allocated at load time for the serving window.

## Constraints

SketchSSM is available for models that implement `SupportsSketchSSM`. It requires Model Runner V2, `--mamba-backend triton` and `--mamba-ssm-cache-dtype float32`. It does not support tensor parallelism, Mamba prefix caching, speculative decoding, stochastic rounding of the SSM state, or KV connectors.

## Kernels

Mamba-2, Gated DeltaNet and KDA layers have Triton kernels. On supported NVIDIA GPUs, faster CUDA kernels are used where the layer shape allows; they are vendored at build time or come from the optional `sketchssm` package, and are compiled just-in-time (with `nvcc` and `ninja`) when not prebuilt. Set `VLLM_SKETCHSSM_USE_CUDA=0` to always use the Triton kernels.
