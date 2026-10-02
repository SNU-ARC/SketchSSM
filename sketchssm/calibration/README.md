# Offline calibration

Use this directory to understand the calibration procedure, inspect the
recipes of the provided calibrations, and produce a matching basis and per-head
allocation.

- **New model:** follow [the step-by-step guide](docs/new_model.md).
- **Existing model:** use its calibration from the [Hugging Face Hub](https://huggingface.co/SketchSSM) (see
  [Portable calibration file](#portable-calibration-file)), or regenerate it from
  one of the folders under `example/` below.
- **Implement an adapter:** use [the tensor and numerical contracts](docs/file_formats.md).

Install the calibration tools from a checkout of this repository (collecting
also needs vLLM, see [`vllm/`](../../vllm/README.md)):

```bash
python -m pip install ".[calibration]"
```

Run the commands below as `python -m sketchssm.calibration ...` or
`sketchssm-calibrate ...`.

Select `adapter: mamba2`, `adapter: gdn` or `adapter: kda` explicitly in
`config.yaml`. A custom adapter can be supplied as `my_package:MyAdapter`
without editing the common core. See [adapter selection and extension](adapters/README.md).

## What calibration computes

1. Generate continuation tokens from WikiText-2 training prompts.
2. Replay those tokens and collect boundary-state and effective-query covariances.
3. Pool count-normalized covariances inside each native group and fit one shared Omega.
4. On separate validation text, pair the Full-Gram output residual with the
   same token/head's readout gradient to obtain rank score curves.
5. Allocate a rank or dense fallback to each state head under each requested budget.

Omega is shared by group; rank allocation remains per head. Gradients weight
allocation scores, not Omega. Full-Gram scores are independent of the P4
approximation used during inference. Allocation uses the established fixed
rank/dense weights; inference traffic is reported separately.

## Existing examples

| Folder | Family |
| --- | --- |
| [nemotron_nano](example/nemotron_nano/README.md) | Mamba-2 |
| [nemotron_super](example/nemotron_super/README.md) | Mamba-2 |
| [qwen_flash_next](example/qwen_flash_next/README.md) | GDN |
| [glm_flash](example/glm_flash/README.md) | KDA |
| [qwen3_5_9b](example/qwen3_5_9b/README.md) | GDN |

Any mean rank budget can be served from a calibration; the budgets in each
folder's `config.yaml` are the ones used in the paper, for which the allocation
tables are stored and checked.

Each folder holds the recipe and the expected results of one provided
calibration: `README.md`, `config.yaml` (the configuration the data was
collected with, including the pinned checkpoint and revision), `collect.yaml`
(the same recipe as input to `calibrate`) and `manifest.json` (geometry,
the paper's budgets, dense-head counts and the SHA-256 of every output file). The collected
tensors (about 3.7 GB for the four models) are not distributed with the
repository. Regenerate a model's data with its `collect.yaml`:

```bash
python -m sketchssm.calibration calibrate --config sketchssm/calibration/example/nemotron_super/collect.yaml --out outputs/nemotron_super
```

A collected directory has the following layout:

```text
<model>/
  data/
    generation_tokens.pt
    allocation_tokens.pt
  statistics/
    covariance.pt
    paired_scores.pt
  basis/
    omega.pt
  allocations/
    g*.pt
```

Generation tokens and validation tokens are separate inputs. Stored paired
statistics include gradient-norm and joint residual/gradient sums; they are
not raw per-token gradient vectors. See [data details](example/README.md).

## Reuse or rebuild an allocation

Run these commands from the project root on a collected directory, here
`outputs/nemotron_super` from the command above. Reuse a table unchanged:

```bash
python -m sketchssm.calibration precomputed --bundle outputs/nemotron_super --mean-rank 5 --out outputs/super_g5.pt
```

Rebuild all configured tables from collected statistics into a new directory:

```bash
python -m sketchssm.calibration.run --source outputs/nemotron_super --out outputs/nemotron_super_rebuilt
```

Or request different budgets using `--mean-ranks 3 5 9`. The output includes the
same calibration data and basis, a configuration and a manifest, plus the new
allocations. The command verifies its inputs and refuses to overwrite a
nonempty output directory. This is a CPU operation; generation, covariance
capture and gradients are not repeated. These commands report the
regeneration command when a directory's tensor files are missing.

## Portable calibration file

A calibration file holds one model's shared basis and allocation scores, so that
the per-head table and frames for any mean rank can be derived when the model is
served. Package a collected directory, then export frames for a chosen mean rank:

```bash
python -m sketchssm.calibration package --bundle outputs/nemotron_super --out outputs/super/calibration.pt
python -m sketchssm.calibration export --calibration outputs/super/calibration.pt --mean-rank 5 --out outputs/super_g5_frames.pt
```

`package` re-allocates every mean rank configured in the bundle from the
packaged contents and fails unless each result equals the bundle's
`allocations/g*.pt` table. Those ranks are recorded as verified. `export
--calibration` runs the unchanged `allocate` and `export_frames` functions; for a
verified rank its frames and tables equal those of `precomputed` followed by
`export --allocation`. Other mean ranks use the same allocation rule, but no
stored table exists to compare them with. The file contains no covariance,
gradient statistics or tokens.

| Key | Type | Contents |
| --- | --- | --- |
| `format`, `schema_version` | str, int | `"sketchssm-calibration"`, 1 |
| `model` | dict | `name`, `family` (`mamba2`, `gdn` or `kda`) |
| `geometry` | dict | `key_dim`, `value_dim`, `groups`, `window`, `erase` |
| `max_rank` | int | Allocation rank cap at the calibrated window: the dense crossover rank unless `allocation.max_rank` pinned a lower cap |
| `omega` | float32 L,groups,R,K | Bundle basis, unchanged |
| `basis_fingerprint`, `basis_meta` | str, dict | Basis identity and fitting metadata |
| `curves` | dict | `joint_dot_sq_sum` (float64 L,H,R+1), `joint_nstep` (int), `meta` (includes `coefficient_model` and `basis_fingerprint`) |
| `layer_ids` | optional list | Original recurrent-layer identifiers, when the basis provides them |
| `verified_mean_ranks` | list | Mean ranks checked against bundle tables |
| `verified` | dict | Per rank (`format(float(G), 'g')`): `dense_heads`, `sketch_heads`, `table_sha256` |

The file loads with `torch.load(..., weights_only=True)`. Use
`sketchssm.calibration.calibration.load`, `select` and `frames` from Python.
`select` also checks a verified rank against its `table_sha256`, the SHA-256 of
the table shape, its int16 `m_table` bytes and its bool `dense_table` bytes.

`select(calibration, mean_rank, window=None)` and `frames(...)` (CLI:
`export --calibration ... --mean-rank R [--window W]`) allocate for a serving
window W, by default the calibrated `geometry.window`. With erase (GDN/KDA) a
rank costs K+V+W, so the table depends on W, and the rank cap at W is the
dense crossover there unless `max_rank` was pinned below the calibrated
crossover. Verified digests apply only to the calibrated allocation problem:
at another W of an erase family the digest check is skipped and a warning
states that the table is unverified. Mamba-2 tables do not depend on W.

| Model | Weights | Calibration | Size |
| --- | --- | --- | --- |
| Nemotron Nano 9B v2-BF16 | [`nvidia/NVIDIA-Nemotron-Nano-9B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | [`SketchSSM/Nemotron-Nano-9B-v2-BF16`](https://huggingface.co/SketchSSM/Nemotron-Nano-9B-v2-BF16) | 11 MB |
| Nemotron 3 Super-NVFP4 | [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | [`SketchSSM/Nemotron-3-Super-NVFP4`](https://huggingface.co/SketchSSM/Nemotron-3-Super-NVFP4) | 13 MB |
| Qwen3.8 Flash-Next-NVFP4 | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | [`SketchSSM/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/SketchSSM/Qwen3.8-Flash-Next-NVFP4) | 24 MB |
| GLM 5.3 Flash-NVFP4 | [`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | [`SketchSSM/GLM-5.3-Flash-NVFP4`](https://huggingface.co/SketchSSM/GLM-5.3-Flash-NVFP4) | 68 MB |
| Qwen3.5 9B-BF16 | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | [`SketchSSM/Qwen3.5-9B-BF16`](https://huggingface.co/SketchSSM/Qwen3.5-9B-BF16) | 16 MB |

A calibration depends on the weights it was collected with; the repository
name states their precision.

Each repository contains `calibration.pt` packaged from the collected data
that the matching folder under `example/` describes, a `manifest.json` with its
SHA-256 and parity results, and a model card. Because the file holds the basis
and the allocation scores, any mean rank can be exported from it without the
collected data:

```bash
hf download SketchSSM/Nemotron-3-Super-NVFP4 calibration.pt --local-dir outputs/super
python -m sketchssm.calibration export --calibration outputs/super/calibration.pt --mean-rank 7 --out outputs/super_g7_frames.pt
```

## Collect a new model

Copy `example/<model>/collect.yaml`, set the original checkpoint and compatible
runtime, then run:

```bash
python -m sketchssm.calibration calibrate --config my_model.yaml --out sketchssm/calibration/outputs/my_model
```

Add `--resume` to continue verified partial work, or select
`--stage generate|covariance|basis|paired|allocate`. The stages run in separate
processes. See [collection configuration and extension](collection/README.md).

The stages are:

1. `generate`: generate continuations from WikiText-2 prompts.
2. `covariance`: replay them and collect state and query covariances.
3. `basis`: fit one shared sketch basis per head group.
4. `paired`: score each rank on validation text with output-error and gradient pairs.
5. `allocate`: assign each head a rank or a dense fallback for every requested budget.

Then package the result into one portable file, serve it, and optionally share
it on the Hugging Face Hub with a manifest (the file hash, the base checkpoint
and, for every calibrated budget, the parity of the derived tables and frames):

```bash
python -m sketchssm.calibration package --bundle outputs/my_model --out outputs/my_model/calibration.pt
vllm serve <your-model> --sketchssm outputs/my_model/calibration.pt --sketchssm-mean-rank 6 \
  --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching

python -m sketchssm.calibration manifest --bundle outputs/my_model \
  --calibration outputs/my_model/calibration.pt --precision bf16 --out outputs/my_model/manifest.json
hf upload <user>/<repo> outputs/my_model/calibration.pt calibration.pt   # serve with --sketchssm <user>/<repo>
hf upload <user>/<repo> outputs/my_model/manifest.json manifest.json
```

To regenerate a provided calibration, run `calibrate` with its
`example/<model>/collect.yaml` (the checkpoints and revisions are pinned in the
matching `config.yaml`); see [the calibration data guide](example/README.md).

Generation and covariance use native vLLM. Packed NVFP4 gradient estimation
uses temporary per-operation BF16 weight reconstruction from that checkpoint;
ordinary BF16/FP16/FP32 models use their normal differentiable forward.
Whole-model expansion and replacement BF16 checkpoints are not used.

## Validation status

The calibration reproduces the published allocations and bases exactly, and the
collection pipeline was run end to end on a reduced Nemotron Nano. See
[`docs/validation.md`](docs/validation.md) for what was checked and how to rerun it.

## Small tensor-only example

After installing the package, run this pipeline from the project root without
model weights, vLLM or a GPU:

```bash
python -m sketchssm.calibration.tools.make_toy_data --out outputs/toy
python -m sketchssm.calibration fit-basis --covariance outputs/toy/covariance.pt --groups 2 --rank 4 --out outputs/toy/basis.pt
python -m sketchssm.calibration score --trace outputs/toy/trace.pt --basis outputs/toy/basis.pt --out outputs/toy/curves.pt
python -m sketchssm.calibration allocate --curves outputs/toy/curves.pt --basis outputs/toy/basis.pt --mean-rank 2 --value-dim 8 --max-rank 4 --out outputs/toy/allocation.pt
python -m sketchssm.calibration export --allocation outputs/toy/allocation.pt --out outputs/toy/frames.pt
```

Export produces portable frames; a serving-engine adapter is required to use
them for inference. Run the numerical tests with:

```bash
python -m unittest discover -s tests -v
```
