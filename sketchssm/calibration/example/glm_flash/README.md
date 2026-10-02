# GLM 5.3 Flash calibration example

This directory holds the recipe and the expected results of the GLM 5.3 Flash
calibration: Full-Gram allocation tables for mean ranks 3, 4, 7, 12, 28. The collected
data (`data/`, `statistics/`, `basis/`, `allocations/`) is not distributed with
this repository; `manifest.json` records the dense-head count of every budget
and the size and SHA-256 of every file.

## Use the Hugging Face calibration

[`SketchSSM/GLM-5.3-Flash-NVFP4`](https://huggingface.co/SketchSSM/GLM-5.3-Flash-NVFP4) holds the packaged
`calibration.pt` (shared basis and allocation scores). Serve it with
`--sketchssm SketchSSM/GLM-5.3-Flash-NVFP4`, or export frames for any mean rank:

```bash
hf download SketchSSM/GLM-5.3-Flash-NVFP4 calibration.pt --local-dir outputs/glm_flash_hub
python -m sketchssm.calibration export --calibration outputs/glm_flash_hub/calibration.pt --mean-rank 3 --out outputs/glm_flash_g3_frames.pt
```

## Regenerate the data

The data was collected with the
NVFP4 weights [`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) at revision
`36c184c6cda000a481711306df5adde42f63321a`. `collect.yaml` pins the same checkpoint
and revision; from the project root:

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/glm_flash/collect.yaml --out outputs/glm_flash
```

The basis and allocation scores depend on these weights, so collect with your
own checkpoint for other weights. `config.yaml` records the recipe and
geometry of the original collection.

## Use or rebuild an allocation

With a collected directory, load a saved table, or recalculate all configured
mean-rank tables on CPU without repeating generation or gradients:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/glm_flash --mean-rank 3 --out outputs/glm_flash_allocation.pt
python -m sketchssm.calibration.run --source outputs/glm_flash --out outputs/glm_flash_rebuilt
```

See [the collection recipe](../README.md) and [new-model instructions](../../docs/new_model.md).
