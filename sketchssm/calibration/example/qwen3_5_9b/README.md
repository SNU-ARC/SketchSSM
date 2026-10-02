# Qwen3.5 9B calibration example

This directory holds the recipe and the expected results of the Qwen3.5 9B
calibration: Full-Gram allocation tables for mean ranks 3, 4, 7, 11, 26. The collected
data (`data/`, `statistics/`, `basis/`, `allocations/`) is not distributed with
this repository; `manifest.json` records the dense-head count of every budget
and the size and SHA-256 of every file.

## Use the Hugging Face calibration

[`SketchSSM/Qwen3.5-9B-BF16`](https://huggingface.co/SketchSSM/Qwen3.5-9B-BF16) holds the packaged
`calibration.pt` (shared basis and allocation scores). Serve it with
`--sketchssm SketchSSM/Qwen3.5-9B-BF16`, or export frames for any mean rank:

```bash
hf download SketchSSM/Qwen3.5-9B-BF16 calibration.pt --local-dir outputs/qwen3_5_9b_hub
python -m sketchssm.calibration export --calibration outputs/qwen3_5_9b_hub/calibration.pt --mean-rank 3 --out outputs/qwen3_5_9b_g3_frames.pt
```

## Regenerate the data

The data was collected with the
BF16 weights [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. `collect.yaml` pins the same checkpoint
and revision; from the project root:

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/qwen3_5_9b/collect.yaml --out outputs/qwen3_5_9b
```

On one H100 80GB the five stages took 19 minutes (generation 4, covariance 7,
paired gradients 8). The basis and allocation scores depend on these weights, so collect with your
own checkpoint for other weights. `config.yaml` records the recipe and
geometry of the original collection.

## Use or rebuild an allocation

With a collected directory, load a saved table, or recalculate all configured
mean-rank tables on CPU without repeating generation or gradients:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/qwen3_5_9b --mean-rank 3 --out outputs/qwen3_5_9b_allocation.pt
python -m sketchssm.calibration.run --source outputs/qwen3_5_9b --out outputs/qwen3_5_9b_rebuilt
```

See [the collection recipe](../README.md) and [new-model instructions](../../docs/new_model.md).
