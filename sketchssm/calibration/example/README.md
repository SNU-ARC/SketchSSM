# Provided calibration data

Each model directory describes one collected calibration bundle: a group-shared
basis, reusable calibration statistics, the original calibration token IDs, and
matching per-head tables. The tables retain the established Full-Gram
allocation policy.

The bundle data itself (the `.pt` files below, about 3.7 GB for the four
models) is not distributed with this repository. Each directory keeps the
recipe and the expected results: `config.yaml` (the collection configuration,
including the pinned checkpoint and revision), `collect.yaml` (the same recipe
as input to `calibrate`) and `manifest.json` (geometry, budgets, dense-head
counts and the size and SHA-256 of every collected file). `catalog.json`,
`allocation_parity.json` and `basis_parity.json` record the shared catalog and
parity results.

| Directory | Model | Budgets in the paper | Allocation rank cap |
| --- | --- | --- | --- |
| `nemotron_nano` | Nemotron Nano 9B v2 | 4, 6, 10, 21 | 49 |
| `nemotron_super` | Nemotron 3 Super | 2, 3, 5, 9, 20 | 42 |
| `qwen_flash_next` | Qwen3.8 Flash-Next | 3, 4, 7, 11, 26 | 60 |
| `glm_flash` | GLM 5.3 Flash | 3, 4, 7, 12, 28 | 60 |
| `qwen3_5_9b` | Qwen3.5 9B | 3, 4, 7, 11, 26 | 60 |

These budgets only select which allocation tables are stored and checked; a
calibration serves any mean rank.

## Use the Hugging Face calibration

For serving and for exporting frames, the portable `calibration.pt` of each model
is enough. It is packaged from the bundle and contains the shared basis and the
allocation scores, so any mean rank can be derived without the bundle data:

| Directory | Hugging Face repository |
| --- | --- |
| `nemotron_nano` | [`SketchSSM/Nemotron-Nano-9B-v2-BF16`](https://huggingface.co/SketchSSM/Nemotron-Nano-9B-v2-BF16) |
| `nemotron_super` | [`SketchSSM/Nemotron-3-Super-NVFP4`](https://huggingface.co/SketchSSM/Nemotron-3-Super-NVFP4) |
| `qwen_flash_next` | [`SketchSSM/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/SketchSSM/Qwen3.8-Flash-Next-NVFP4) |
| `glm_flash` | [`SketchSSM/GLM-5.3-Flash-NVFP4`](https://huggingface.co/SketchSSM/GLM-5.3-Flash-NVFP4) |
| `qwen3_5_9b` | [`SketchSSM/Qwen3.5-9B-BF16`](https://huggingface.co/SketchSSM/Qwen3.5-9B-BF16) |

```bash
hf download SketchSSM/Nemotron-3-Super-NVFP4 calibration.pt --local-dir outputs/super
python -m sketchssm.calibration export --calibration outputs/super/calibration.pt --mean-rank 5 --out outputs/super_g5_frames.pt
```

Pass the repository id or the file to vLLM with `--sketchssm`; see the
[top-level README](../../../README.md).

## Regenerate the bundle data

The bundles were collected with the checkpoints and revisions pinned in each
`config.yaml`. Running `calibrate` with the model's `collect.yaml` repeats the
collection (a GPU and the model weights are required; see the
[collection guide](../collection/README.md) for the runtime settings):

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/nemotron_super/collect.yaml --out outputs/nemotron_super
```

Compare the result with the model's `manifest.json`, which records the
dense-head count of every budget and the size and SHA-256 of every original
file. Commands that read bundle tensors (`precomputed`, `package`,
`sketchssm.calibration.run`, `sketchssm.calibration.tools.verify_bundles`) report this command
when the files are missing.

## Use a saved table

From the project root, with a collected directory:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/nemotron_super --mean-rank 5 --out outputs/super_g5.pt
python -m sketchssm.calibration export --allocation outputs/super_g5.pt --out outputs/super_g5_frames.pt
```

The loader verifies file checksums and the basis fingerprint, then combines the
shared basis with the selected table without changing its tensor values. Export
produces portable frames; a serving-engine adapter is still required for inference.

To reproduce the table from its saved scores instead:

```bash
python -m sketchssm.calibration allocate --basis outputs/nemotron_super/basis/omega.pt --curves outputs/nemotron_super/statistics/paired_scores.pt --mean-rank 5 --value-dim 64 --max-rank 42 --out outputs/super_g5_rebuilt.pt
```

Use V=80/cap=49 for Nano and V=128/cap=60/`--erase` for Qwen and GLM.
All bundles use W=16. This path needs only CPU, the basis and curves; it does not
load model weights or repeat generation/gradient collection.

## Bundle files

| File | Contents |
| --- | --- |
| `basis/omega.pt` | Ordered group Omega in original state coordinates |
| `allocations/g*.pt` | Per-head ranks and dense flags; references the shared basis |
| `statistics/paired_scores.pt` | Full-Gram paired scores, output-error sums, gradient-norm sums, and per-sequence output-error/paired-score sums |
| `statistics/covariance.pt` | Per-head state/query covariance sums and observation counts, retaining the collected precision |
| `data/generation_tokens.pt` and `data/allocation_tokens.pt` | Model-specific prompt, generated-continuation and paired-validation token IDs |
| `manifest.json` | Geometry, the paper's budgets, basis fingerprint, file sizes and SHA-256 checksums |

Raw per-token gradient vectors and full state/query traces are **not** included:
the collectors retained their accumulated statistics. The saved curves suffice
to reproduce allocation, but changing the basis requires new paired scoring.
Refitting a basis from the covariance alone does not make existing score curves
valid for that new basis.

## Collection recipe

The calibration has two separate passes, both based on WikiText-2 raw v1:

1. **Basis statistics:** 520 training prompts of 512 tokens, with 256 generated
   continuation tokens each. Generation uses raw token IDs without a chat
   template, temperature 0, seed 0 and ignored EOS. The saved continuations are
   then replayed to collect covariance statistics. With W=16 and the first
   generated window excluded, each layer has 7,800 boundary-state observations
   and 124,800 effective-query observations. Gradients do not weight Omega.
2. **Allocation statistics:** 64 validation sequences of 256 tokens, starting
   at token-block index 320, with 128 warmup tokens each. The last 128 positions
   contribute 8,192 token observations per head. For each position/head, the
   Full-Gram output residual is paired with the NLL gradient of the raw readout;
   `(gradient^T residual)^2` is accumulated. This pass uses validation text,
   rather than the generated continuations from the first pass.

`data/generation_tokens.pt` stores `[520,512]` prompt IDs and `[520,256]`
generated IDs. `data/allocation_tokens.pt` stores `[64,257]` validation blocks
including the next-token label. These are the actual
saved token IDs; use the matching model tokenizer rather than retokenizing them
with another model's tokenizer.

Nano collected basis statistics through readout reconstruction hooks;
Super/Qwen/GLM used native NVFP4 observers. The latter models' gradient collectors
used frozen packed weights and a temporary per-operation BF16 gradient estimator,
not a native NVFP4 backward pass. The portable collection launcher and model
bindings are described in the [collection guide](../collection/README.md).

## Verification

The four bundles occupy about 3.67 GiB, mostly covariance tensors, in 39 `.pt`
files. Their sizes and SHA-256 checksums are recorded in the manifests in this
directory. With the collected data placed in each model directory using the
layout above, verify all checksums and reproduce all 19 tables on CPU:

```bash
python -m sketchssm.calibration.tools.verify_bundles --root sketchssm/calibration/example
```

Collected data in these directories is ignored by Git.
