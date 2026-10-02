# Nemotron 3 Super calibration example

This directory holds the recipe and the expected results of the Nemotron 3 Super
calibration: Full-Gram allocation tables for mean ranks 2, 3, 5, 9, 20. The collected
data (`data/`, `statistics/`, `basis/`, `allocations/`) is not distributed with
this repository; `manifest.json` records the dense-head count of every budget
and the size and SHA-256 of every file.

## Use the Hugging Face calibration

[`SketchSSM/Nemotron-3-Super-NVFP4`](https://huggingface.co/SketchSSM/Nemotron-3-Super-NVFP4) holds the packaged
`calibration.pt` (shared basis and allocation scores). Serve it with
`--sketchssm SketchSSM/Nemotron-3-Super-NVFP4`, or export frames for any mean rank:

```bash
hf download SketchSSM/Nemotron-3-Super-NVFP4 calibration.pt --local-dir outputs/nemotron_super_hub
python -m sketchssm.calibration export --calibration outputs/nemotron_super_hub/calibration.pt --mean-rank 2 --out outputs/nemotron_super_g2_frames.pt
```

## Regenerate the data

The data was collected with the
NVFP4 weights [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) at revision
`ff433f5493e25d631c9f12b5d55c674229923d02`. `collect.yaml` pins the same checkpoint
and revision; from the project root:

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/nemotron_super/collect.yaml --out outputs/nemotron_super
```

The basis and allocation scores depend on these weights, so collect with your
own checkpoint for other weights. `config.yaml` records the recipe and
geometry of the original collection.

## Use or rebuild an allocation

With a collected directory, load a saved table, or recalculate all configured
mean-rank tables on CPU without repeating generation or gradients:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/nemotron_super --mean-rank 2 --out outputs/nemotron_super_allocation.pt
python -m sketchssm.calibration.run --source outputs/nemotron_super --out outputs/nemotron_super_rebuilt
```

See [the collection recipe](../README.md) and [new-model instructions](../../docs/new_model.md).
