# Qwen3.8 Flash-Next calibration example

This directory holds the recipe and the expected results of the Qwen3.8 Flash-Next
calibration: Full-Gram allocation tables for mean ranks 3, 4, 7, 11, 26. The collected
data (`data/`, `statistics/`, `basis/`, `allocations/`) is not distributed with
this repository; `manifest.json` records the dense-head count of every budget
and the size and SHA-256 of every file.

## Use the Hugging Face calibration

[`SketchSSM/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/SketchSSM/Qwen3.8-Flash-Next-NVFP4) holds the packaged
`calibration.pt` (shared basis and allocation scores). Serve it with
`--sketchssm SketchSSM/Qwen3.8-Flash-Next-NVFP4`, or export frames for any mean rank:

```bash
hf download SketchSSM/Qwen3.8-Flash-Next-NVFP4 calibration.pt --local-dir outputs/qwen_flash_next_hub
python -m sketchssm.calibration export --calibration outputs/qwen_flash_next_hub/calibration.pt --mean-rank 3 --out outputs/qwen_flash_next_g3_frames.pt
```

## Regenerate the data

The data was collected with the
NVFP4 weights [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. `collect.yaml` pins the same checkpoint
and revision; from the project root:

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/qwen_flash_next/collect.yaml --out outputs/qwen_flash_next
```

The basis and allocation scores depend on these weights, so collect with your
own checkpoint for other weights. `config.yaml` records the recipe and
geometry of the original collection.

## Use or rebuild an allocation

With a collected directory, load a saved table, or recalculate all configured
mean-rank tables on CPU without repeating generation or gradients:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/qwen_flash_next --mean-rank 3 --out outputs/qwen_flash_next_allocation.pt
python -m sketchssm.calibration.run --source outputs/qwen_flash_next --out outputs/qwen_flash_next_rebuilt
```

See [the collection recipe](../README.md) and [new-model instructions](../../docs/new_model.md).
