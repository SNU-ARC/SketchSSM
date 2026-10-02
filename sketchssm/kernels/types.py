# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""The sketch, table and ring containers the kernels read.

The package does not own them: the caller (e.g. vLLM) allocates them in its
storage layout. These protocols list exactly the attributes the wrappers read.
"""

from collections.abc import Sequence
from typing import Protocol

import torch


class Mamba2Tables(Protocol):
    ranks: torch.Tensor
    ag_offsets: torch.Tensor
    u_offsets: torch.Tensor
    w_offsets: torch.Tensor
    dense_rows: torch.Tensor


class Mamba2Sketch(Protocol):
    """Per-request sketch of a Mamba-2 layer (first dim: request index)."""

    u: torch.Tensor  # (reqs, u_rows, head_dim) BF16
    w: torch.Tensor  # (reqs, w_rows, state_size) BF16
    ag: torch.Tensor  # (reqs, 5, ag_cols) FP32
    tables: Mamba2Tables


class GDNTables(Protocol):
    ranks: torch.Tensor
    layout: torch.Tensor  # (HV, 4) int32
    rank_cap: int
    window: int


class GDNSketch(Protocol):
    """Per-request sketch of a GDN layer (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    fs: torch.Tensor
    beta: torch.Tensor
    current_d: torch.Tensor
    current_k: torch.Tensor
    tables: GDNTables


class KDATables(Protocol):
    num_heads: int
    window: int
    rank_cap: int
    frame_gk: torch.Tensor
    ranks: torch.Tensor
    heads_step: torch.Tensor
    heads_all: torch.Tensor
    # (heads, rank bucket, head base) of each flush launch.
    flush_groups: Sequence[tuple[torch.Tensor, int, int]]
    num_sketch_heads: int


class KDASketch(Protocol):
    """Per-request sketch of a KDA layer (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    f: torch.Tensor
    tables: KDATables


class KDARings(Protocol):
    """Window rings of a KDA layer (first dim: state slot). Every ring is
    dense within a slot and all share one slot stride in bytes."""

    k: torch.Tensor
    v: torch.Tensor
    prefix: torch.Tensor
    beta: torch.Tensor
    u_ring: torch.Tensor
    d_ring: torch.Tensor

    @property
    def window(self) -> int: ...
