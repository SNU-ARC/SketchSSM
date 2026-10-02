# Recurrence adapters

Select the recurrence explicitly in `config.yaml`:

```yaml
adapter: gdn
model:
  family: gdn
```

Built-in names are `mamba2`, `gdn`, and `kda`. `auto` uses declared family
metadata only; it never guesses from a model repository's name. Unknown families
require an explicit adapter. A conflicting family or erase setting is an error.
The saved-statistics runner also accepts `--adapter gdn` and records the
resolved selection in its output configuration and manifest. Changing this flag
does not transform or recollect existing statistics.

## Current interface

Each adapter supplies `family`, `erase`, `validate_geometry(geometry)`, and
`effective_queries(query, decay=..., key=..., beta=..., window=...)`.
The numerical core and data distribution do not import a model or an engine.

| Input | Mamba-2 | GDN | KDA |
| --- | --- | --- | --- |
| query | `(...,T,H,K)` | `(...,T,H,K)` | `(...,T,H,K)` |
| decay | `(...,T,H)` | `(...,T,H)` | `(...,T,H,K)` |
| key | none | same shape as query | same shape as query |
| beta | none | `(...,T,H)` | `(...,T,H)` |

H is the **state-head** count. An engine binding expands shared Q/K groups to
these heads, preserves model-specific normalization/scaling and converts log
decay to multiplicative decay before calling the adapter. Beta is the already
gated update strength; adapters do not apply another sigmoid. The first input
token is a window start; only complete windows are accepted. Mamba-2 includes
the current token's decay. GDN/KDA apply erase transitions in reverse order to
the query, with KDA's channel decay in the correct order relative to erase.

These are reference calibration tensor transformations, not inference kernels.
Calculations use FP32 (FP64 if any input is FP64). The model's native forward
path remains unchanged. The result can be rearranged into the common
`effective_query[L,B,N,H,K,W]` scoring layout.

## Add an adapter without editing the core

Put a class in an importable Python module and specify `module:Class`:

```yaml
adapter: my_calibration.adapters:MyAdapter
model:
  family: my_recurrence
```

Subclass `sketchssm.calibration.adapters.CalibrationAdapter`, declare `family`
and boolean `erase`, and implement `effective_queries`. A complete working
extension that shares the Mamba-2 recurrence is:

```python
from sketchssm.calibration.adapters.mamba2 import Mamba2Adapter

class MyAdapter(Mamba2Adapter):
    family = "my_recurrence"
```

For a genuinely different recurrence, override `effective_queries` and verify
`S_boundary @ q_effective` against independently updated boundary state. Never
reuse a scalar-decay implementation for a recurrence with erase transitions.
The adapter is user-supplied Python code and is imported only when explicitly
selected. Its class takes no constructor arguments in this interface.

The current allocator's cost model covers Mamba-2/GDN/KDA sketch representations.
A new representation with different memory costs needs a corresponding cost
model change; selecting a custom recurrence alone does not establish its costs.

## Engine binding boundary

Selecting a recurrence does **not** establish compatibility with an arbitrary
model implementation. The remaining model-side binding must expose:

- Ordered recurrent layers, native groups, state-head geometry and state layout.
- Boundary states and normalized/scaled recurrence inputs without changing the
  model's forward, including prefill/decode positions and selected windows.
- The raw readout before output gate/normalization, and the NLL gradient at that
  same position, retaining token/head alignment.

The [collection package](../collection/README.md) provides common generation
and stage orchestration, native capture bindings and packed-weight gradient
loaders. These interfaces remain separate from the recurrence math. Validate
the selected engine and model binding before claiming end-to-end calibration
support for a new model.
