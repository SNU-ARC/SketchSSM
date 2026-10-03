# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM of Kimi Delta Attention layers (exact W=16 window)."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.kda_replayssm_triton import (
    KDA_REPLAY_HEAD_DIM,
    KDA_REPLAY_LOWER_BOUND,
    KDA_REPLAY_WINDOW,
    replay_acquire,
    replay_bump,
    replay_flush,
    replay_partial_flush,
    replay_prefill_resolve,
    replay_read,
    replay_release,
    replay_resolve,
    replay_step,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.triton_utils import triton

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


class KDAReplaySSM:
    """W=16 ring pool of one KDA layer, separate from the KV cache.

    Decode reads the native FP32 page as the window checkpoint and writes it
    only on the flush step. Ring slots are owned by physical pages, so the
    runner must release finished/preempted pages (``release_finished``) before
    reuse and fold continuing streams (``before_prefill``) before a prefill.
    """

    def __init__(self, heads: int, capacity: int, device: torch.device):
        if heads < 1 or capacity < 1:
            raise ValueError("Positive head count and slot capacity required")
        self.heads, self.capacity = heads, capacity
        f32, i32 = torch.float32, torch.int32
        ring = (capacity, heads, KDA_REPLAY_WINDOW, KDA_REPLAY_HEAD_DIM)
        vec = (capacity, heads, KDA_REPLAY_HEAD_DIM)
        self.pos = torch.zeros(capacity, device=device, dtype=i32)
        self.k = torch.zeros(ring, device=device, dtype=torch.bfloat16)
        self.v = torch.zeros(ring, device=device, dtype=torch.bfloat16)
        self.log_a = torch.zeros(ring, device=device, dtype=f32)
        self.beta = torch.zeros(ring[:3], device=device, dtype=f32)
        self.prefix = torch.zeros(vec, device=device, dtype=f32)
        self.direct_decay = torch.zeros(ring, device=device, dtype=f32)
        self.delta = torch.zeros(ring, device=device, dtype=f32)
        self.query_scaled = torch.empty(vec, device=device, dtype=f32)
        self.key_scaled = torch.empty_like(self.query_scaled)
        self.rhs = torch.empty_like(self.query_scaled)
        self.replay_out = torch.empty_like(self.query_scaled)
        self.scalars = torch.empty(capacity, heads, 2, device=device, dtype=f32)
        self.owners = torch.full((capacity,), -1, device=device, dtype=i32)
        self.slots = torch.empty(capacity, device=device, dtype=i32)
        self.old = torch.empty_like(self.slots)
        self.old_pos = torch.empty_like(self.slots)
        self.fresh = torch.empty(capacity, device=device, dtype=torch.bool)
        self.counts = torch.zeros(1, device=device, dtype=torch.int64)
        self.work_rows = torch.empty(2 * capacity, device=device, dtype=i32)
        self.work_counts = torch.zeros(2, device=device, dtype=i32)
        self.flush_programs = min(
            8 * torch.cuda.get_device_properties(device).multi_processor_count,
            capacity * heads,
        )

    @classmethod
    def maybe_create(
        cls,
        vllm_config: "VllmConfig",
        heads: int,
        head_dim: int,
        lower_bound: float | None,
        prefill_backend: str,
    ) -> "KDAReplaySSM | None":
        if not vllm_config.cache_config.use_replayssm:
            return None
        if (
            head_dim != KDA_REPLAY_HEAD_DIM
            or lower_bound != KDA_REPLAY_LOWER_BOUND
            or vllm_config.model_config.dtype != torch.bfloat16
            or prefill_backend != "flashkda"
        ):
            raise NotImplementedError(
                f"ReplaySSM for KDA needs head dim {KDA_REPLAY_HEAD_DIM}, gate "
                f"lower bound {KDA_REPLAY_LOWER_BOUND}, BF16 activations and "
                "the FlashKDA prefill backend"
            )
        capacity = max(
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        return cls(heads, capacity, torch.get_default_device())

    def reset(self) -> None:
        """Discard capture/warmup ownership without writing physical pages."""
        self.owners.fill_(-1)
        self.pos.zero_()
        self.counts.zero_()

    def release_finished(self, physical_ids: torch.Tensor) -> None:
        if physical_ids.numel():
            released = (self.owners[:, None] == physical_ids[None, :]).any(1)
            self.owners.masked_fill_(released, -1)
            self.pos.masked_fill_(released, 0)

    def before_prefill(
        self,
        state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> None:
        """Commit continuing rows, discard stale ownership for new prompts."""
        batch = indices.numel()
        if not batch:
            return
        slots = torch.empty_like(indices, dtype=torch.int32)
        flush_slots = torch.empty_like(slots)
        replay_prefill_resolve[(1,)](
            indices,
            has_initial_state,
            self.owners,
            slots,
            flush_slots,
            batch,
            self.capacity,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.capacity),
        )
        replay_partial_flush[(batch, self.heads, 4)](
            flush_slots,
            self.pos,
            self.owners,
            state,
            self.k,
            self.v,
            self.log_a,
            self.beta,
            self.heads,
            32,
            *state.stride(),
            num_warps=4,
        )
        replay_release[(batch,)](slots, self.owners)

    def step(
        self,
        state: torch.Tensor,
        indices: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """One decode token per row; ``[B, H, 128]`` raw post-conv inputs."""
        if state.dtype != torch.float32 or state.shape[1:] != (
            self.heads,
            KDA_REPLAY_HEAD_DIM,
            KDA_REPLAY_HEAD_DIM,
        ):
            raise ValueError("Native FP32 [page, head, V128, K128] state required")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError("Raw post-convolution k/v must be BF16")
        batch = q.shape[0]
        if batch > self.capacity:
            raise ValueError("Decode batch exceeds configured slot capacity")
        if not batch:
            return torch.empty_like(v)
        q, k, v, gate, beta = (x.contiguous() for x in (q, k, v, gate, beta))
        a_log, bias = a_log.float().contiguous(), bias.float().contiguous()
        slots = self.slots[:batch]
        strides = state.stride()
        replay_resolve[(1,)](
            indices,
            self.owners,
            slots,
            self.old,
            self.fresh,
            batch,
            self.capacity,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(self.capacity),
            self.pos,
            self.old_pos,
            self.counts,
            self.work_rows,
            self.work_counts,
            num_warps=4,
        )
        replay_acquire[(self.flush_programs,)](
            indices,
            slots,
            self.old,
            self.old_pos,
            self.fresh,
            self.owners,
            self.pos,
            state,
            self.k,
            self.v,
            self.log_a,
            self.beta,
            self.heads,
            *strides,
            self.work_rows,
            self.work_counts,
            num_warps=4,
        )
        out = torch.empty_like(v)
        replay_step[(self.heads, batch)](
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            bias,
            slots,
            self.pos,
            self.k,
            self.v,
            self.log_a,
            self.beta,
            self.prefix,
            self.direct_decay,
            self.delta,
            self.query_scaled,
            self.key_scaled,
            self.rhs,
            self.replay_out,
            self.scalars,
            self.heads,
            num_warps=1,
        )
        replay_read[(4, self.heads, batch)](
            indices,
            slots,
            self.pos,
            state,
            self.delta,
            self.query_scaled,
            self.key_scaled,
            self.rhs,
            self.replay_out,
            self.scalars,
            out,
            self.heads,
            32,
            *strides,
            num_warps=4,
        )
        replay_flush[(self.flush_programs,)](
            indices,
            q,
            slots,
            state,
            self.prefix,
            self.direct_decay,
            self.key_scaled,
            self.rhs,
            self.scalars,
            self.delta,
            out,
            self.work_rows,
            self.work_counts,
            self.capacity,
            self.heads,
            32,
            *strides,
            num_warps=1,
            num_stages=1,
        )
        replay_bump[(1,)](
            slots, self.pos, batch, KDA_REPLAY_WINDOW, triton.next_power_of_2(batch)
        )
        return out

    def forward(
        self,
        state: torch.Tensor,
        metadata: "GDNAttentionMetadata",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        prefill: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Decode rows stay in the window; only prefill rows reach ``prefill``.

        Inputs are post-convolution ``[1, T, H, D]`` in decode-first order.
        Decode rows keep their ring and position on mixed steps.
        """
        nd = metadata.num_decodes
        if metadata.num_decode_tokens != nd:
            raise ValueError("KDA ReplaySSM decode requires one token per request")
        if metadata.num_spec_decodes:
            raise ValueError("KDA ReplaySSM does not support speculative decoding")
        if metadata.num_prefills:
            ids = metadata.prefill_state_indices
            initial = metadata.prefill_has_initial_state
            cu = metadata.prefill_query_start_loc
            assert ids is not None and initial is not None and cu is not None
            self.before_prefill(state, ids, initial)
            initial_state = gather_initial_states(state, ids, initial)
            end = nd + metadata.num_prefill_tokens
            _, final_state = prefill(
                q=q[:, nd:end],
                k=k[:, nd:end],
                v=v[:, nd:end],
                g=gate[:, nd:end],
                beta=beta[:, nd:end],
                initial_state=initial_state,
                cu_seqlens=cu,
                out=out[:, nd:end],
            )
            scatter_states(state, final_state, ids)
        if nd:
            ids = metadata.non_spec_state_indices_tensor
            assert ids is not None
            out[:, :nd].copy_(
                self.step(
                    state,
                    ids[:nd],
                    q[0, :nd],
                    k[0, :nd],
                    v[0, :nd],
                    gate[0, :nd],
                    beta[0, :nd],
                    a_log,
                    bias,
                ).unsqueeze(0)
            )
