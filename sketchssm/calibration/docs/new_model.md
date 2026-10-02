# Calibrate a new model

This guide defines the complete workflow and the files that each stage must
produce. The common launcher connects native generation/covariance and paired-gradient
collection to the numerical stages. Select a compatible engine and model
binding; a model identifier alone does not guarantee compatibility. Start with
`example/<model>/collect.yaml` and the [collection guide](../collection/README.md).

## 1. Create one model directory and record its configuration

Choose a fresh output directory such as `sketchssm/calibration/outputs/my_model`.
Use a nearby `example/<model>/config.yaml` as a recipe reference and record:

- Pinned model and tokenizer repository identifiers/revisions; preserve the
  model's intended weight storage rather than replacing a quantized checkpoint.
- Recurrent family, recurrent layer order, native query/key groups, state-head
  count, K, V and window length W.
- Basis rank and ridge, and the requested mean-rank budgets. The candidate
  rank cap is automatic: `allocation.max_rank` may be omitted and defaults to
  the dense crossover rank `min(K, K*V // (K + V + W))` (drop `+ W` without
  erase), the largest cap `allocate` accepts (60 for GDN/KDA with K=V=128 and
  W=16, 49 for Nano, 42 for Super; the provided examples state it explicitly).
  Set it only to restrict candidates to a lower cap. `basis.rank` must lie
  between the cap and K; the examples use 60-79. For the same family and
  geometry, reuse that example's values.
- Input dataset, tokenization, prompt/generation lengths and validation selection.

Set `adapter: mamba2`, `gdn`, or `kda`. For a custom recurrence, use an
importable `module:AdapterClass`; see [adapter extension](../adapters/README.md).
Explicit selection checks the declared family and geometry. It does not bypass
validation of the model's engine-specific capture and backward bindings.

The included examples use W=16, group-shared Omega and per-head allocation.
Do not infer the group count from the state-head count: Q/K groups may be shared
by several state heads. A model with unequal layer geometries needs separate
compatible tensor groups; the current core expects uniform geometry within a pack.

## 2. Generate the basis-calibration data

Prepare 520 raw prompts of 512 tokens from WikiText-2 raw v1's training split.
Use the model's tokenizer, without a chat template. Generate 256 continuation
tokens per prompt with temperature 0, seed 0 and ignored EOS.

Save the actual prompt and continuation token IDs to
`data/generation_tokens.pt`. Preserve sequence order and record model/tokenizer
revisions and the prompt-selection recipe. Resuming must reuse saved outputs
rather than silently generating different continuations for completed prompts.

The four examples record their prompt-selection recipe in `config.yaml`, and
their manifests record the SHA-256 of the original token files (the token IDs
themselves are not distributed). Do not infer a dataset selection merely from
the number of prompts; a new collection must record its own selection
deterministically.

## 3. Collect boundary-state and effective-query covariances

Teacher-force the saved continuations through the full-state model, with
SketchSSM disabled. Keep its forward recurrence unchanged. At each retained
window capture the state *before* the first update in that window, and the
effective query for each read relative to that boundary state.

For the standard recipe, discard the first generated W=16 window. Each sequence
then contributes 15 boundary states and 240 effective queries. Across 520
sequences, every layer/head must have 7,800 state observations and 124,800
query observations. Ensure the last generated token actually enters the model;
returning a sampled token alone does not mean its state transition was captured.

Accumulate per-head covariance sums and separate observation counts into
`statistics/covariance.pt`. Pooling across groups happens in the basis fitter,
not by discarding individual head statistics during capture.

The effective query includes the complete boundary-to-read transition:
scalar decay for Mamba-2, ordered erase/decay factors for GDN, and the appropriate
channel-wise decay plus erase factors for KDA. Using the raw query or a
Mamba-2 decay shortcut for GDN/KDA is incorrect.

## 4. Fit the shared basis

```bash
python -m sketchssm.calibration fit-basis --covariance sketchssm/calibration/outputs/my_model/statistics/covariance.pt --groups <native-groups> --rank <basis-rank> --out sketchssm/calibration/outputs/my_model/basis/omega.pt
```

The fitter normalizes each covariance by its own count, pools heads within
native groups and solves the state-weighted eigenproblem (ridge 0.1 by default).
The resulting fingerprint identifies the exact ordered basis. Omega is not
weighted by gradients.

## 5. Measure paired error and gradient statistics

Use separate WikiText-2 validation text: 64 blocks of 256 input tokens,
starting at block index 320, plus the next-token labels. Save the 257-token
blocks in `data/allocation_tokens.pt`. Use a 128-token warmup and score the
remaining 128 positions per block, yielding 8,192 observations per head.

Differentiate the sum of next-token NLL losses with respect to the raw
recurrent readout, before its output gate/normalization. Keep token/head axes
separate. Pair that gradient with the Full-Gram reconstruction residual for
the *same* token/head; accumulate `(gradient^T residual)^2`. A product of
separately averaged error and gradient magnitudes is not equivalent.

A quantized model's native forward support does not establish native backward
support. Document and validate the gradient implementation separately. Retain
packed storage; never substitute a fully dequantized model implicitly.

For adapters exporting small selected-window traces, the numerical scorer is:

```bash
python -m sketchssm.calibration score --trace <paired-trace.pt> --basis sketchssm/calibration/outputs/my_model/basis/omega.pt --out sketchssm/calibration/outputs/my_model/statistics/paired_scores.pt
```

The [trace format](file_formats.md) specifies boundary state, effective query
and readout-gradient axes. Full model traces can be large, so production
collectors should accumulate the same statistics while each sequence/layer is
available. The example bundles hold accumulated statistics, not these full
traces. A changed basis requires new scoring.

## 6. Allocate ranks

```bash
python -m sketchssm.calibration allocate --curves sketchssm/calibration/outputs/my_model/statistics/paired_scores.pt --basis sketchssm/calibration/outputs/my_model/basis/omega.pt --mean-rank <G> --value-dim <V> --out <allocation.pt>
```

Add `--erase` for GDN/KDA and set `--window` if not 16. `--max-rank` defaults
to the dense crossover rank; pass a lower value only to restrict candidates. Repeat for the requested
mean budgets. Keep the same fixed allocation-cost policy across runtime
storage choices. Rank zero in the saved table denotes dense fallback; it is
not a zero-cost sketch candidate.

Collected model directories store Omega once in `basis/omega.pt` and tables
in `allocations/g*.pt`, linked by basis fingerprints. Standalone CLI allocation
files include Omega for convenience. `manifest.json` records the common layout,
geometry, budgets and file checksums; it is written after all required files
are complete. `run.py` demonstrates creating this layout from a collected bundle.

## 7. Validate before inference

- Verify observation counts, finite covariances and scores, and head/group order.
- Check full-state read = boundary read + within-window contribution against
  the model's unmodified recurrent path. Check the effective-query read
  independently against explicitly updated boundary state.
- Verify nonzero finite readout gradients and that warmup/input/label positions align.
- Check feasible allocation costs, dense sentinels and matching basis fingerprints.
- Export frames and validate the engine adapter independently before reporting accuracy.

Calibration completion does not by itself validate an inference kernel. The
portable package exports group frames and head tables; the engine must still
handle state layout, exact updates and request lifecycles correctly.
