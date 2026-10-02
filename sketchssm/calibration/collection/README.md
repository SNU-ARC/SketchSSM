# Collection pipeline

`python -m sketchssm.calibration calibrate --config model.yaml --out outputs/model`
executes five isolated stages:

| Stage | Model execution | Output |
| --- | --- | --- |
| `generate` | Native vLLM, original checkpoint, raw WikiText training prompts | Continuation and disjoint validation token IDs |
| `covariance` | Native vLLM, saved continuations teacher-forced through unchanged full-state updates | Per-head state/query covariance sums and counts |
| `basis` | CPU | Group-shared Omega |
| `paired` | Frozen checkpoint, differentiable readout hooks, validation NLL | Per-head Full-Gram error/gradient curves |
| `allocate` | CPU | Rank/dense tables and complete bundle manifest |

Pass `--stage <name>` to run one stage or `--resume` to continue. Completed
stages and partial generation/paired batches are preserved. Configuration,
implementation, and dependency hashes must match. Change the output directory
when changing the model, corpus, basis, scoring procedure or implementation.
The implementation hash covers every `.py` file in `sketchssm/calibration/`, so
after any code change (including a fix for a failed stage) `--resume` refuses
the directory with "input identity changed"; start a new output directory.
One lock prevents concurrent writers to the same output. A fresh subprocess
per stage releases engine resources before the gradient model is loaded.

## Recurrence and engine bindings

`observers/` implements the online covariance arithmetic. `native/` observes
specific engine entry points without replacing their updates. `hooks/` pairs
raw recurrent readouts and their gradients. Generation, corpus selection,
checkpointing, NLL loss and output layout are common.

| Family | Native binding entry point | Gradient binding |
| --- | --- | --- |
| Mamba-2 | `vllm.model_executor.layers.mamba.mamba_mixer2` / `MambaMixer2` | Nemotron-H Mamba-2 projections and scan |
| GDN | `vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn` | Separate QKV/a/b projections and gated delta recurrence |
| KDA | `vllm.models.glm5next.nvidia.kda` / `Glm5NextLinearAttention` | GLM text model and FLA KDA |

These are explicit engine ABIs, not a promise that every vLLM release is
interchangeable. A different engine layout needs a binding. The KDA paired
statistics kernel currently requires K=V=128; unsupported geometry is rejected.
Capture uses one device, eager execution and synchronous scheduling, without
speculative decoding or ReplaySSM. Per-layer counts detect missed capture paths.

For an engine adaptation, set `runtime.covariance_worker` to an importable
worker-extension class implementing `install_covariance(config, resume)`,
`reset_covariance_slots()`, and `save_covariance(path, completed, batch_size, meta)`.
A recurrence adapter alone does not supply an engine ABI or model weight loader.

## Original weights and gradient estimation

Set `gradient.loader` explicitly:

- `ordinary`: the original floating-point checkpoint, normal differentiable
  forward; quantized checkpoints are rejected.
- `super_nvfp4`: Nemotron-H ModelOpt checkpoint with packed linears/experts.
- `qwen_nvfp4`: Flash-Next packed experts and FP8 embedding checkpoint layout.
- `qwen3_5_nvfp4`: dense Qwen3.5-architecture compressed-tensors checkpoint with
  NVFP4 and FP8 per-channel linears (for example `unsloth/Qwen3.8-27B-NVFP4`).
- `glm_nvfp4`: GLM compressed-tensors packed checkpoint layout.
- `module:factory`: return `GradientModel(model, logits_callable, audit_dict)`
  for another checkpoint layout.

The packed loaders retain packed weight storage. Each linear operation
reconstructs its weight temporarily in BF16, discards that view, and reconstructs
it again when propagating input gradients. This reproduces the packed gradient
estimator; it is **not native NVFP4 backward**. They cannot be used for forward
covariance collection. No whole-model BF16 conversion or replacement checkpoint
is selected automatically.

For a different HF module layout, set `gradient.binding` to a class accepting
`(model, basis, config)`, with `start()`, `result(measured_tokens)`, and `close()`.
It must return CPU per-layer/head sums with `output_error_sum`,
`joint_dot_sq_sum`, and `grad_sq_sum`, using the same-token raw-readout gradient.

## Environments

Install SketchSSM into each selected environment. `runtime.python` selects a
common interpreter; `generate_python`, `covariance_python`, and `paired_python`
can select separate environments. Relative model paths are resolved against
the input YAML. No engine checkout or model path is embedded in package code.
The numerical dependencies are sufficient for saved-statistics operations;
collection also requires the matching vLLM/Transformers/model kernels.

For Mamba-2 gradient collection, install the scan and causal-convolution
kernels supported by the chosen Transformers release. For example, the tested
Transformers 5.13.0 environment accepts `kernels>=0.15.2,<0.16`; installing a
newer incompatible loader silently leaves its fast scan unavailable. The
collector rejects a missing CUDA scan instead of changing its computation.

Set engine-specific options under `runtime.engine_kwargs`. Calibration-critical
state precision, eager capture, checkpoint identity, and batch size cannot be
overridden there. `runtime.environment` supplies explicit process-local engine
settings such as the GLM model runner selection.

## Small original-weight integration test

Before a full collection run, a few original-weight blocks can verify the
pipeline on a smaller GPU. For an original BF16 Nemotron Nano checkpoint:

```bash
python -m sketchssm.calibration.tools.make_nano_layer_subset \
  --source /path/to/original-nano-checkpoint \
  --out /path/to/nano-four-block-test --layers 4
```

Copy `sketchssm/calibration/example/nemotron_nano/collect.yaml` to a test config.
Point `model.checkpoint` at this local subset and set local checkpoint revisions
to null. Set `gradient.loader: ordinary`, `generation.sequences: 2`,
`paired.sequences: 2`, and `runtime.batch_size: 2`; select compatible native
and gradient Python environments. Keep the original head dimensions, W=16,
group basis and Full-Gram scoring. Then use the normal command:

```bash
python -m sketchssm.calibration calibrate \
  --config /path/to/test-config.yaml --out outputs/nano-four-block-test
```

The helper retains the first blocks, embeddings, normalization and output
head with their original tensor values and dtypes. Its small-bundle results
are for integration checks only: removing layers changes the model, and two
sequences do not establish calibration quality. Do not replace the provided
full-model example artifacts with these test outputs.

GDN/Mamba use the `logits_processor` teacher-forcing binding. The historical
GLM binding uses `trace_decode`, which requires the engine's trace replay
extension. The N+1 sentinel output makes the Nth generated token enter the
recurrent state update; it is excluded from covariance statistics.

Full-model collection with the relocated package must still be validated in
each selected engine environment before results are reported. The included
unit tests and small module gates do not establish full-model compatibility.
