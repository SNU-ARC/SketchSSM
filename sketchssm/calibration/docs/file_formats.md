# Offline calibration

This module consumes calibration tensors and produces a group basis and per-head
rank allocation. It contains no serving engine, model loader, scheduler, or
full-model dequantization path.

## Pipeline

1. **Fit Omega.** Normalize each head's state/query covariance by its observation
   count, pool the normalized covariances inside each native group, and solve the
   state-weighted generalized eigenproblem. Do not average fitted eigenvectors
   or weight Omega by gradients.
2. **Score ranks.** For every window/head, form the output sketch U from the
   full state and the group basis. Compute the ideal least-squares projection of
   the boundary-state read into each prefix span of U. Pair its residual e with
   the raw readout gradient g from the same token and head. Accumulate (g^T e)^2;
   never replace this by a product of independently averaged magnitudes.
3. **Allocate.** Choose a positive sketch rank or FP32 dense fallback for each
   head under the fixed mean-rank allocation budget. Omega remains shared by native group.
4. **Export.** Complete each group's selected prefix to an ordered orthogonal
   frame and save it with the per-head rank/dense tables.

All models use **Full-Gram allocation scores**. P4 is an inference coefficient
approximation, not a scoring option here. Rank caps are explicit and must be
covered by the measured curves and basis; no cap is silently inferred from a
model name.

## Commands

See the executable example in the [root README](../../../README.md). Run
`python -m sketchssm.calibration --help` for subcommands. Every stage writes a
`.pt` checkpoint and a `.pt.json` metadata sidecar. The CLI reads tensor/basic-
container checkpoints with `torch.load(weights_only=True)` on CPU.

The functions `basis.fit`, `scoring.score`, `allocation.allocate`, and
`export.export_frames` also accept Python dictionaries directly.

## Tensor contracts

K is the state/key dimension, V the output/value dimension, H the state-head
count, L the recurrent-layer count, B the sequence count, N the number of
retained windows, and W the window length. Unlike the K-by-V notation often
used on paper, **state tensors here are stored V-by-K**.

### Covariance input

| Key | Shape | Meaning |
| --- | --- | --- |
| `head_scov` | L,H,K,K | Sum of boundary-state S^T S |
| `head_qcov` | L,H,K,K | Sum of effective-query outer products |
| `head_cov_windows` | L | Boundary observation count per head in each layer |
| `head_cov_queries` | L | Query observation count per head in each layer |
| `layer_ids` | optional list | Original recurrent-layer identifiers |

Counts must be positive. All heads in a layer must have the same sample count.
Native groups are contiguous and must divide H. The fitter uses the established
relative ridge of 0.1 on the state metric. Basis rows are stored as FP32 offline;
BF16 U/C storage applies to the inference read representation, not to the
covariance solve or calibration statistics.

### Basis

`omega[L,groups,R,K]` stores ordered basis vectors as rows in original state
coordinates. A content fingerprint ties scoring curves to the exact basis,
without embedding absolute file paths. Changing the basis requires re-scoring.

### Paired trace input

| Key | Shape |
| --- | --- |
| `state` | L,B,N,H,V,K |
| `effective_query` | L,B,N,H,K,W |
| `gradient` | L,B,N,H,V,W |

`state` is the full state at the **start** of each retained window.
`gradient` is the loss gradient with respect to the raw state-read output for
that token/head. The reconstruction target is S q_eff; exact within-window
buffer contributions are not part of its residual.

The caller supplies already selected, complete windows, with warmup and prefill
excluded as required by its collection recipe. Mamba-2 effective queries include
scalar decay; GDN and KDA include ordered erase transitions, with channel-wise
decay for KDA. Raw queries are not interchangeable with these effective queries.

Scoring uses every supplied calibration query, matching the Full-Gram collector.
It measures ideal prefix-space reconstruction, not runtime P4, BF16 rounding,
or the exact-flush output. Orthogonalizing U is a stable way to compute this
projection without explicitly inverting its full Gram matrix. Dependent columns
are rejected using the established relative residual threshold 1e-5.

### Allocation input/output

`joint_dot_sq_sum[L,H,R+1]` contains paired residual/gradient sums.
`joint_nstep` is the per-head token count used to normalize them.
Rank zero in a score curve is the zero-readout diagnostic; it is **not** an
available zero-cost sketch. The allocator requires the scoring basis fingerprint
and `meta.coefficient_model = "full-gram"`.

Output keys are `omega`, `m_table[L,H]`, `dense_table[L,H]`, and `meta`.
In `m_table`, zero denotes dense fallback; positive integers are sketch ranks.
The deterministic Lagrangian solver reports its feasible objective, lower bound
and relative gap. It does not claim an exact global optimum.

## Preserve existing tables

A collected bundle (see the [provided calibration data](../example/README.md),
which is regenerated with `calibrate`) holds matching bases and allocations,
plus the saved covariance, gradient/error statistics and calibration tokens.
Use `precomputed --bundle ... --mean-rank ... --out ...` to load a table
directly, with integrity checks.

Use the `reuse` subcommand or `reuse.reuse` to retain an existing group basis,
rank table and dense table exactly, including tensor dtypes. No scoring,
allocation or frame reconstruction runs on this path. The CLI records the source
checkpoint SHA-256 and recalculates only the inference traffic report. It does
not edit the source file or claim a new allocation objective.

## Portable calibration file

`package --bundle` writes the basis, the allocation score sums, geometry and
rank cap into one file, after checking that every configured mean rank
reproduces its bundle table exactly. `export --calibration FILE --mean-rank G`
then runs `allocate` and `export_frames` on its contents. See the
[key table](../README.md#portable-calibration-file).

## Allocation costs and inference traffic

Use `--erase` for GDN/KDA and omit it for Mamba-2. Let e be 1 for erase models
and 0 otherwise. The allocator keeps the established normalized weights:

```
sketch rank G: G (K+V+eW)
dense fallback: KV
mean variable budget: mean_rank (K+V+eW)
```

The common flush term KV/W can be added to both the allocation budget and costs
without affecting the table. The candidate crossover in these allocation units
is `min(K, floor(KV / (K+V+eW)))`. It is the default rank cap (`max_rank`);
an explicit lower cap restricts candidates.
For Nano K128/V80 this crossover is 49. These fixed optimization weights preserve
the established allocation policy. They are not a BF16 byte-budget constraint.

Actual inference uses BF16 sketches/maps and FP32 dense state. Its non-flush
read costs are `2 G (K+V+eW)` and `4KV` bytes respectively. Reports evaluate
these costs on the selected table, with one full-state flush read per window:

```
state_read = 4KV/W + (W-1)/W * mean(actual non-flush read bytes)
state_write = 4KV/W
```

Read reduction uses Standard's `4KV`; total state-access reduction uses
Standard's `8KV`. Ring traffic, construction writes, weights and other runtime
accesses are outside this state-memory metric. BF16 storage changes this report,
not the allocation weights or saved dense-head selection. The inference byte
crossover is therefore different from the allocation candidate crossover.

## Export boundary

Exported frames have shape `L,groups,K,K` and FP32 storage. Their leading rows
preserve the selected basis prefix subspaces. All-dense groups use identity
frames. U and C depend on the runtime state and are **not** offline artifacts;
engine adapters construct them in BF16 and maintain FP32 full state. Each adapter
must implement its own ring storage, state layout and request lifecycle.
