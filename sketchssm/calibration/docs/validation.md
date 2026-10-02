# Calibration validation

What the offline calibration was checked against, and how to rerun the checks.

## Numerical tests (CPU)

```bash
python -m pytest tests --ignore=tests/kernels
```

The tests compare the recurrence adapters with independent state updates and
cover adapter selection, the integrity of saved statistics and bundles, group
covariance pooling, Full-Gram residuals against an independent least-squares
solve, paired per-head gradient scores, dependent sketch columns, the dense
fallback, ordered frame export, traffic accounting, and the rank allocation
against exhaustive search on small problems.

## Reproducing the published calibrations

- **Rank allocation.** Rebuilding the Full-Gram allocations of the paper's
  models from their saved statistics reproduces every per-head rank and dense
  choice exactly:

  | Model | Mean rank budgets | Heads per table |
  | --- | --- | --- |
  | Nemotron Nano 9B v2 | 4, 6, 10, 21 | 3,456 |
  | Nemotron 3 Super | 2, 3, 5, 9, 20 | 5,120 |
  | Qwen3.8 Flash-Next | 3, 4, 7, 11, 26 | 1,728 |
  | GLM 5.3 Flash | 3, 4, 7, 12, 28 | 2,176 |

  [`example/allocation_parity.json`](../example/allocation_parity.json) records
  the comparison.
- **Basis.** Refitting each model's basis from its collected covariance with
  the packaged fitter reproduces the saved FP32 frames bit for bit (Nano
  27×8×79×128, Super 40×8×63×128, Qwen 36×16×78×128, GLM 34×64×60×128);
  [`example/basis_parity.json`](../example/basis_parity.json) records it.
- **Reuse.** Loading a calibration and exporting a budget keeps every basis,
  rank and dense tensor and its dtype without rerunning the allocation.
- **Packaging.** The command-line pipeline runs end to end on synthetic tensors
  from an installed wheel, outside this repository.

## Statistics collection

- The Mamba-2, GDN and KDA covariance observers match independently updated
  boundary states and ordered transition products, including KV-cache page
  changes within one request (Mamba-2, GDN).
- Linear-input gradients of packed (NVFP4) layers match an explicit BF16
  reconstruction exactly.
- Stage resume skips verified completed outputs and rejects changed inputs.
- On small Transformers modules, the Mamba-2 and GDN readout recurrences have
  relative errors of 3.9e-8 and 1.3e-7, with finite per-head curves and nonzero
  readout gradients.
- The KDA statistics kernel (CUDA) matches an independent tensor recurrence:
  relative boundary-query decomposition error 2.5e-7, maximum state difference
  1.2e-7, maximum output difference 1.5e-8.

These module checks can be rerun after installing the model dependencies:

```bash
python -m sketchssm.calibration.tools.check_mamba2_hooks
python -m sketchssm.calibration.tools.check_gdn_hooks
python -m sketchssm.calibration.tools.check_kda_collection  # needs a CUDA device
```

## End-to-end collection on a reduced Nemotron Nano

The full collection pipeline was run on the first four blocks (two Mamba-2 and
two MLP blocks) of `nvidia/NVIDIA-Nemotron-Nano-9B-v2` in BF16 on one H100:
two WikiText prompts (512 prompt and 256 generated tokens), teacher-forced
capture of 30 windows (W=16) per Mamba-2 layer with FP32 states, a group basis
of shape (2, 8, 79, 128), paired Full-Gram statistics on two validation blocks
of 256 tokens, and the allocation tables for mean ranks 4, 6, 10 and 21. All
256 state heads had nonzero gradients and all saved statistics were finite.
This checks the collection pipeline, not full-model accuracy.
