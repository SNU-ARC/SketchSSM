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
    # (HV, 4) int32 (u_off, phi_off, fs_off, FG). A dense head (rank 0) with
    # u_off >= 0 keeps BF16 rows of its full state at u_off, read on non-flush
    # steps; with u_off -1 (ReplaySSM) it reads the FP32 state.
    layout: torch.Tensor
    rank_cap: int
    window: int
    # Optional: False when no head has a rank 0 < m < K (no coefficient maps to
    # rebuild, e.g. ReplaySSM); the flush then skips its finish launch.
    # has_sketch_heads: bool


class GDNSketch(Protocol):
    """Per-request sketch of a GDN layer (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    fs: torch.Tensor
    beta: torch.Tensor  # (reqs, HV, W) FP32
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
    # Dense heads (rank 0): their ids in dense-row order (device int32), the
    # dense row of each head or -1 (device int32, (H,)), and their count.
    dense_heads_d: torch.Tensor
    dense_rows: torch.Tensor
    num_dense_heads: int


class KDASketch(Protocol):
    """Per-request sketch of a KDA layer (first dim: request index)."""

    u: torch.Tensor
    phi: torch.Tensor
    f: torch.Tensor
    dense: torch.Tensor  # (reqs, num_dense_heads, V, K) BF16 state rows
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
