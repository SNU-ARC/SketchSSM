# Nemotron Nano 9B v2 calibration example

This directory holds the recipe and the expected results of the Nemotron Nano 9B v2
calibration: Full-Gram allocation tables for mean ranks 4, 6, 10, 21. The collected
data (`data/`, `statistics/`, `basis/`, `allocations/`) is not distributed with
this repository; `manifest.json` records the dense-head count of every budget
and the size and SHA-256 of every file.

## Use the Hugging Face calibration

[`SketchSSM/Nemotron-Nano-9B-v2-BF16`](https://huggingface.co/SketchSSM/Nemotron-Nano-9B-v2-BF16) holds the packaged
`calibration.pt` (shared basis and allocation scores). Serve it with
`--sketchssm SketchSSM/Nemotron-Nano-9B-v2-BF16`, or export frames for any mean rank:

```bash
hf download SketchSSM/Nemotron-Nano-9B-v2-BF16 calibration.pt --local-dir outputs/nemotron_nano_hub
python -m sketchssm.calibration export --calibration outputs/nemotron_nano_hub/calibration.pt --mean-rank 4 --out outputs/nemotron_nano_g4_frames.pt
```

## Regenerate the data

The data was collected with the
BF16 weights [`nvidia/NVIDIA-Nemotron-Nano-9B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) at revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`. `collect.yaml` pins the same checkpoint
and revision; from the project root:

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/nemotron_nano/collect.yaml --out outputs/nemotron_nano
```

The basis and allocation scores depend on these weights, so collect with your
own checkpoint for other weights. `config.yaml` records the recipe and
geometry of the original collection.

## Use or rebuild an allocation

With a collected directory, load a saved table, or recalculate all configured
mean-rank tables on CPU without repeating generation or gradients:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/nemotron_nano --mean-rank 4 --out outputs/nemotron_nano_allocation.pt
python -m sketchssm.calibration.run --source outputs/nemotron_nano --out outputs/nemotron_nano_rebuilt
```

See [the collection recipe](../README.md) and [new-model instructions](../../docs/new_model.md).
