# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""GDN paired gradient hooks and ordered erase-transition reference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...core._statistics import _mgs_prefix, _rank_statistics
from .mamba2 import JointCollector


def full_transition_query(
    query: torch.Tensor,
    keys: list[torch.Tensor],
    alpha: list[torch.Tensor],
    beta: list[torch.Tensor],
) -> torch.Tensor:
    """Apply ``M_start:t^T`` to a GDN query.

    Shapes are query/key ``(B,H,K)`` and alpha/beta ``(B,H)``.  GDN's
    transition is symmetric, but its factors do not commute, so the transpose
    product is applied from the current token back to the window boundary.
    """
    if not len(keys) == len(alpha) == len(beta):
        raise ValueError("transition ring lengths disagree")
    result = query
    for key_s, alpha_s, beta_s in zip(reversed(keys), reversed(alpha), reversed(beta)):
        projection = (key_s * result).sum(-1)
        result = alpha_s[..., None] * (
            result - beta_s[..., None] * key_s * projection[..., None]
        )
    return result


@dataclass
class GDNLayerRecipe:
    output: torch.Tensor
    basis_q: torch.Tensor
    output_error_sum: torch.Tensor
    rejected_by_rank: torch.Tensor
    recurrence_max_abs: float
    recurrence_max_rel: float
    replayed_kernel_max_abs: float
    replayed_kernel_max_rel: float
    decomposition_max_abs: float
    decomposition_max_rel: float
    decomposition_first_max_abs: float
    decomposition_first_max_rel: float
    rank0_max_abs: float
    output_monotonic_max: float
    energy_identity_max_rel: float
    group_accounting_max_abs: float


class GDNJointCollector(JointCollector):
    """Qwen GDN specialization of the common paired-statistics collector."""

    def __init__(self, mixers, basis, **kwargs):
        mixers = list(mixers)
        for mixer in mixers:
            if hasattr(mixer, "num_heads") and mixer.num_heads != mixer.num_v_heads:
                raise RuntimeError("existing num_heads disagrees with num_v_heads")
            mixer.num_heads = mixer.num_v_heads
            mixer.n_groups = mixer.num_k_heads
            mixer.ssm_state_size = mixer.head_k_dim
            mixer.head_dim = mixer.head_v_dim
        super().__init__(mixers, basis, **kwargs)
        self.rank0_max_abs = 0.0
        self.output_monotonic_max = 0.0
        self.energy_identity_max_rel = 0.0
        self.group_accounting_max_abs = 0.0

    def _norm_backward_hook(self, li):

        def hook(_module, grad_input, _grad_output):
            recipe = self.pending_recipes[li]
            self.pending_recipes[li] = None
            if recipe is None:
                raise RuntimeError(f"layer {li}: backward ran without paired recipe")
            grad = grad_input[0]
            if grad is None:
                raise RuntimeError(f"layer {li}: raw readout gradient is None")
            batch, nwin, window, heads, value_dim = recipe.output.shape
            if grad.numel() % (batch * heads * value_dim):
                raise RuntimeError(
                    f"layer {li}: gradient {tuple(grad.shape)} cannot map to B={batch}, H={heads}, V={value_dim}"
                )
            total_tokens = grad.numel() // (batch * heads * value_dim)
            stop = self.warmup + nwin * window
            if total_tokens < stop:
                raise RuntimeError(
                    f"layer {li}: gradient has T={total_tokens}, needs {stop}"
                )
            paired_grad = (
                grad.reshape(batch, total_tokens, heads, value_dim)[
                    :, self.warmup : stop
                ]
                .float()
                .reshape(batch, nwin, window, heads, value_dim)
            )
            stats = _rank_statistics(paired_grad, recipe.output, recipe.basis_q)
            torch.testing.assert_close(
                stats["output_error_sum"],
                recipe.output_error_sum,
                rtol=0.005,
                atol=2e-05,
            )
            rank0 = recipe.output.double().square().sum((0, 1, 2, 4)).cpu()
            rank0_abs = float((stats["output_error_sum"][:, 0] - rank0).abs().max())
            if rank0_abs > 1e-06 * max(float(rank0.max()), 1.0):
                raise RuntimeError(
                    f"layer {li}: rank-zero energy mismatch {rank0_abs:.3e}"
                )
            repeat = self.mixers[li].num_v_heads // self.mixers[li].num_k_heads
            grouped = (
                stats["output_error_sum"]
                .reshape(self.mixers[li].num_k_heads, repeat, -1)
                .sum(1)
            )
            independently_grouped = torch.stack(
                [
                    stats["output_error_sum"][g * repeat : (g + 1) * repeat].sum(0)
                    for g in range(self.mixers[li].num_k_heads)
                ]
            )
            group_abs = float((grouped - independently_grouped).abs().max())
            if group_abs != 0.0:
                raise RuntimeError(
                    f"layer {li}: group/head accounting mismatch {group_abs:.3e}"
                )
            for key in (
                "output_error_sum",
                "output_error_sq_sum",
                "joint_dot_sum",
                "joint_dot_abs_sum",
                "joint_dot_sq_sum",
                "scalar_grad_output_error_sum",
                "grad_sq_sum",
            ):
                self.sums[key][li] += stats[key]
            self.sums["rank_deficient_by_column"][li] += recipe.rejected_by_rank
            self.max_energy_identity_error = max(
                self.max_energy_identity_error,
                float(stats["max_energy_identity_error"]),
            )
            self.recurrence_max_abs = max(
                self.recurrence_max_abs, recipe.recurrence_max_abs
            )
            self.recurrence_max_rel = max(
                self.recurrence_max_rel, recipe.recurrence_max_rel
            )
            self.replayed_kernel_max_abs = max(
                self.replayed_kernel_max_abs, recipe.replayed_kernel_max_abs
            )
            self.replayed_kernel_max_rel = max(
                self.replayed_kernel_max_rel, recipe.replayed_kernel_max_rel
            )
            self.decomposition_max_abs = max(
                self.decomposition_max_abs, recipe.decomposition_max_abs
            )
            self.decomposition_max_rel = max(
                self.decomposition_max_rel, recipe.decomposition_max_rel
            )
            self.decomposition_first_max_abs = max(
                self.decomposition_first_max_abs, recipe.decomposition_first_max_abs
            )
            self.decomposition_first_max_rel = max(
                self.decomposition_first_max_rel, recipe.decomposition_first_max_rel
            )
            self.rank0_max_abs = max(
                self.rank0_max_abs, rank0_abs, recipe.rank0_max_abs
            )
            self.output_monotonic_max = max(
                self.output_monotonic_max, recipe.output_monotonic_max
            )
            self.energy_identity_max_rel = max(
                self.energy_identity_max_rel, recipe.energy_identity_max_rel
            )
            self.group_accounting_max_abs = max(
                self.group_accounting_max_abs,
                group_abs,
                recipe.group_accounting_max_abs,
            )

        return hook

    @torch.no_grad()
    def _build_recipe(self, li, input_hidden, raw_scan) -> GDNLayerRecipe:
        import importlib

        qwen = importlib.import_module(type(self.mixers[li]).__module__)
        mixer = self.mixers[li]
        batch, total_tokens, _ = input_hidden.shape
        heads = mixer.num_v_heads
        groups = mixer.num_k_heads
        repeat = heads // groups
        key_dim = mixer.head_k_dim
        value_dim = mixer.head_v_dim
        if groups * repeat != heads:
            raise RuntimeError(
                f"layer {li}: H={heads} is not divisible by groups={groups}"
            )
        expected_numel = batch * total_tokens * heads * value_dim
        if raw_scan.numel() != expected_numel:
            raise RuntimeError(
                f"layer {li}: raw norm input {tuple(raw_scan.shape)} has {raw_scan.numel()} values, expected {expected_numel}"
            )
        raw_scan = raw_scan.reshape(batch, total_tokens, heads, value_dim)
        nwin = (total_tokens - self.warmup) // self.window
        if nwin <= 0:
            raise RuntimeError(
                f"T={total_tokens}, warmup={self.warmup} has no complete window"
            )
        stop = self.warmup + nwin * self.window
        mixed = mixer.in_proj_qkv(input_hidden).transpose(1, 2)
        conv = getattr(mixer, "causal_conv1d_fn", qwen.causal_conv1d_fn)
        if conv is None:
            mixed = F.silu(mixer.conv1d(mixed)[:, :, :total_tokens]).transpose(1, 2)
        else:
            mixed = conv(
                mixed,
                mixer.conv1d.weight.squeeze(1),
                mixer.conv1d.bias,
                mixer.activation,
            ).transpose(1, 2)
        query_group, key_group, value = torch.split(
            mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1
        )
        query_group = query_group.reshape(batch, total_tokens, groups, key_dim)
        key_group = key_group.reshape(batch, total_tokens, groups, key_dim)
        value = value.reshape(batch, total_tokens, heads, value_dim)
        beta = mixer.in_proj_b(input_hidden).sigmoid()
        log_decay = -mixer.A_log.float().exp() * F.softplus(
            mixer.in_proj_a(input_hidden).float() + mixer.dt_bias
        )
        query_norm = qwen.l2norm(query_group, dim=-1, eps=1e-06)
        key_norm = qwen.l2norm(key_group, dim=-1, eps=1e-06)
        query_repeated = query_norm.repeat_interleave(repeat, dim=2)
        key_repeated = key_norm.repeat_interleave(repeat, dim=2)
        _, state = qwen.torch_chunk_gated_delta_rule(
            query_repeated[:, : self.warmup],
            key_repeated[:, : self.warmup],
            value[:, : self.warmup],
            g=log_decay[:, : self.warmup],
            beta=beta[:, : self.warmup],
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        if state is None or state.shape != (batch, heads, key_dim, value_dim):
            raise RuntimeError(
                f"layer {li}: unexpected prefix state {getattr(state, 'shape', None)}"
            )
        state = state.float()
        query = query_repeated.float() / math.sqrt(key_dim)
        key = key_repeated.float()
        value = value.float()
        beta = beta.float()
        alpha = log_decay.exp().float()
        dense_outputs = []
        output_windows = []
        all_q_basis = []
        current_outputs = []
        ring_keys: list[torch.Tensor] = []
        ring_alpha: list[torch.Tensor] = []
        ring_beta: list[torch.Tensor] = []
        window_state0 = None
        boundary_state = None
        replay_state = None
        basis_group = self.basis_orig[li].to(state.device)
        if basis_group.shape == (heads, self.mmax, key_dim):
            basis_head = basis_group
        elif basis_group.shape == (groups, self.mmax, key_dim):
            basis_head = basis_group.repeat_interleave(repeat, dim=0)
        else:
            raise RuntimeError(f"Invalid basis shape {basis_group.shape}")
        rejected_total = torch.zeros(heads, self.mmax, dtype=torch.long)
        effective_max_abs = 0.0
        effective_max_rel = 0.0
        decomposition_max_abs = 0.0
        decomposition_max_rel = 0.0
        decomposition_first_max_abs = 0.0
        decomposition_first_max_rel = 0.0
        for token in range(self.warmup, stop):
            window_pos = (token - self.warmup) % self.window
            if window_pos == 0:
                window_state0 = state.clone()
                boundary_state = window_state0.clone()
                replay_state = torch.zeros_like(state)
                ring_keys = []
                ring_alpha = []
                ring_beta = []
                current_outputs = []
                latch_matrix = torch.einsum("bhkv,hmk->bhvm", window_state0, basis_head)
                q_basis, rejected = _mgs_prefix(latch_matrix, self.rank_tol)
                rejected_total += rejected.sum(0).long().cpu()
                all_q_basis.append(q_basis)
            key_t = key[:, token]
            query_t = query[:, token]
            value_t = value[:, token]
            beta_t = beta[:, token]
            alpha_t = alpha[:, token]
            ring_keys.append(key_t)
            ring_alpha.append(alpha_t)
            ring_beta.append(beta_t)
            decayed = state * alpha_t[..., None, None]
            memory = torch.einsum("bhkv,bhk->bhv", decayed, key_t)
            delta = beta_t[..., None] * (value_t - memory)
            state = decayed + key_t[..., None] * delta[..., None, :]
            full_output = torch.einsum("bhkv,bhk->bhv", state, query_t)
            dense_outputs.append(full_output)
            replay_decayed = replay_state * alpha_t[..., None, None]
            replay_memory = torch.einsum("bhkv,bhk->bhv", replay_decayed, key_t)
            replay_delta = beta_t[..., None] * (value_t - replay_memory)
            replay_state = (
                replay_decayed + key_t[..., None] * replay_delta[..., None, :]
            )
            replay_output = torch.einsum("bhkv,bhk->bhv", replay_state, query_t)
            effective_query = full_transition_query(
                query_t, ring_keys, ring_alpha, ring_beta
            )
            boundary_output = torch.einsum(
                "bhkv,bhk->bhv", window_state0, effective_query
            )
            decomposition = full_output - (boundary_output + replay_output)
            dec_abs = float(decomposition.abs().max())
            dec_rel = float(decomposition.norm() / full_output.norm().clamp_min(1e-30))
            decomposition_max_abs = max(decomposition_max_abs, dec_abs)
            decomposition_max_rel = max(decomposition_max_rel, dec_rel)
            if window_pos == 0:
                decomposition_first_max_abs = max(decomposition_first_max_abs, dec_abs)
                decomposition_first_max_rel = max(decomposition_first_max_rel, dec_rel)
            boundary_decayed = boundary_state * alpha_t[..., None, None]
            boundary_memory = torch.einsum("bhkv,bhk->bhv", boundary_decayed, key_t)
            boundary_state = (
                boundary_decayed
                - key_t[..., None] * (beta_t[..., None] * boundary_memory)[..., None, :]
            )
            forward_boundary_output = torch.einsum(
                "bhkv,bhk->bhv", boundary_state, query_t
            )
            effective_diff = forward_boundary_output - boundary_output
            effective_max_abs = max(
                effective_max_abs, float(effective_diff.abs().max())
            )
            effective_max_rel = max(
                effective_max_rel,
                float(effective_diff.norm() / boundary_output.norm().clamp_min(1e-30)),
            )
            current_outputs.append(boundary_output)
            if len(current_outputs) == self.window:
                output_windows.append(torch.stack(current_outputs, dim=1))
        dense = torch.stack(dense_outputs, dim=1)
        official = raw_scan[:, self.warmup : stop].float()
        recurrent_diff = dense - official
        recurrence_abs = float(recurrent_diff.abs().max())
        recurrence_rel = float(recurrent_diff.norm() / official.norm().clamp_min(1e-30))
        if (
            recurrence_abs > self.recurrence_atol
            and recurrence_rel > self.recurrence_rtol
        ):
            raise RuntimeError(
                f"layer {li}: recurrent GDN disagrees with official chunk path: max_abs={recurrence_abs:.3e}, rel={recurrence_rel:.3e}"
            )
        if effective_max_abs > 0.002 and effective_max_rel > 2e-05:
            raise RuntimeError(
                f"layer {li}: full-transition query ordering mismatch: max_abs={effective_max_abs:.3e}, rel={effective_max_rel:.3e}"
            )
        if decomposition_max_abs > 0.003 and decomposition_max_rel > 3e-05:
            raise RuntimeError(
                f"layer {li}: boundary+replay decomposition mismatch: max_abs={decomposition_max_abs:.3e}, rel={decomposition_max_rel:.3e}"
            )
        output = torch.stack(output_windows, dim=1)
        basis_q = torch.stack(all_q_basis, dim=1)
        projected = torch.einsum("buhvm,buwhv->buwhm", basis_q, output)
        total = output.square().sum(-1)
        retained = projected.square().cumsum(-1)
        residual = (total[..., None] - retained).clamp_min(0.0)
        output_error = torch.cat([total[..., None], residual], dim=-1)
        monotonic_max = float(
            (output_error[..., 1:] - output_error[..., :-1]).clamp_min(0).max()
        )
        energy_rel = float(
            (total[..., None] - (retained + residual))
            .abs()
            .div(total[..., None].clamp_min(1e-20))
            .max()
        )
        output_error_sum = output_error.double().sum((0, 1, 2)).cpu()
        grouped_a = output_error_sum.reshape(groups, repeat, -1).sum(1)
        grouped_b = torch.stack(
            [
                output_error_sum[g * repeat : (g + 1) * repeat].sum(0)
                for g in range(groups)
            ]
        )
        group_abs = float((grouped_a - grouped_b).abs().max())
        return GDNLayerRecipe(
            output=output,
            basis_q=basis_q,
            output_error_sum=output_error_sum,
            rejected_by_rank=rejected_total,
            recurrence_max_abs=recurrence_abs,
            recurrence_max_rel=recurrence_rel,
            replayed_kernel_max_abs=effective_max_abs,
            replayed_kernel_max_rel=effective_max_rel,
            decomposition_max_abs=decomposition_max_abs,
            decomposition_max_rel=decomposition_max_rel,
            decomposition_first_max_abs=decomposition_first_max_abs,
            decomposition_first_max_rel=decomposition_first_max_rel,
            rank0_max_abs=0.0,
            output_monotonic_max=monotonic_max,
            energy_identity_max_rel=energy_rel,
            group_accounting_max_abs=group_abs,
        )
