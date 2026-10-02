# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Readout and gradient hooks; model construction is supplied by the caller."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...core._statistics import _mgs_prefix, _rank_statistics
from .mamba_scan import scan


@dataclass
class LayerRecipe:
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


class JointCollector:
    def __init__(
        self,
        mixers,
        basis: torch.Tensor,
        window: int,
        warmup: int,
        mmax: int,
        rank_tol: float,
        recurrence_atol: float,
        recurrence_rtol: float,
    ):
        self.mixers = list(mixers)
        self.window = int(window)
        self.warmup = int(warmup)
        self.mmax = int(mmax)
        self.rank_tol = float(rank_tol)
        self.recurrence_atol = float(recurrence_atol)
        self.recurrence_rtol = float(recurrence_rtol)
        self.pending_inputs: list[torch.Tensor | None] = [None] * len(self.mixers)
        self.pending_recipes: list[LayerRecipe | None] = [None] * len(self.mixers)
        self.handles = []
        L = len(self.mixers)
        if not L:
            raise ValueError("at least one mixer is required")
        H = int(self.mixers[0].num_heads)
        G = int(self.mixers[0].n_groups)
        N = int(self.mixers[0].ssm_state_size)
        P = int(self.mixers[0].head_dim)
        if H % G:
            raise ValueError(f"H={H} is not divisible by G={G}")
        for li, mixer in enumerate(self.mixers):
            shape = (
                int(mixer.num_heads),
                int(mixer.n_groups),
                int(mixer.ssm_state_size),
                int(mixer.head_dim),
            )
            if shape != (H, G, N, P):
                raise ValueError(f"layer {li} mixer geometry {shape} != {(H, G, N, P)}")
        if (
            basis.ndim != 4
            or basis.shape[0] != L
            or basis.shape[1] != G
            or basis.shape[-1] != N
        ):
            raise ValueError(
                f"Expected group basis (layers={L}, groups={G}, rank, K={N}); got {tuple(basis.shape)}"
            )
        if self.mmax > basis.shape[2]:
            raise ValueError("Scored rank exceeds the calibrated group basis")
        self.basis_orig = basis[:, :, : self.mmax].float()
        R = self.mmax + 1
        zcurve = lambda: torch.zeros(L, H, R, dtype=torch.float64)
        self.sums = {
            "output_error_sum": zcurve(),
            "output_error_sq_sum": zcurve(),
            "joint_dot_sum": zcurve(),
            "joint_dot_abs_sum": zcurve(),
            "joint_dot_sq_sum": zcurve(),
            "scalar_grad_output_error_sum": zcurve(),
            "grad_sq_sum": torch.zeros(L, H, dtype=torch.float64),
            "rank_deficient_by_column": torch.zeros(L, H, self.mmax, dtype=torch.long),
        }
        self.sequence_sums = {"output_error_sum": [], "joint_dot_sq_sum": []}
        self._sequence_start = {
            key: self.sums[key].clone() for key in self.sequence_sums
        }
        self.nstep = 0
        self.nseq = 0
        self.max_energy_identity_error = 0.0
        self.recurrence_max_abs = 0.0
        self.recurrence_max_rel = 0.0
        self.replayed_kernel_max_abs = 0.0
        self.replayed_kernel_max_rel = 0.0
        self.decomposition_max_abs = 0.0
        self.decomposition_max_rel = 0.0
        self.decomposition_first_max_abs = 0.0
        self.decomposition_first_max_rel = 0.0
        self.rank0_max_abs = 0.0
        self.output_monotonic_max = 0.0
        self.energy_identity_max_rel = 0.0
        self.group_accounting_max_abs = 0.0
        for li, mixer in enumerate(self.mixers):
            self.handles.append(
                mixer.register_forward_pre_hook(self._mixer_pre_hook(li), with_kwargs=True)
            )
            self.handles.append(
                mixer.norm.register_forward_pre_hook(self._norm_pre_hook(li))
            )
            self.handles.append(
                mixer.norm.register_full_backward_hook(self._norm_backward_hook(li))
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _mixer_pre_hook(self, li):

        def hook(_module, args, kwargs):
            if self.pending_inputs[li] is not None:
                raise RuntimeError(f"layer {li}: stale pending mixer input")
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError(f"layer {li}: missing mixer hidden_states")
            self.pending_inputs[li] = hidden.detach()

        return hook

    def _norm_pre_hook(self, li):

        def hook(_module, args):
            hidden = self.pending_inputs[li]
            self.pending_inputs[li] = None
            if hidden is None:
                raise RuntimeError(f"layer {li}: norm ran without mixer input")
            if self.pending_recipes[li] is not None:
                raise RuntimeError(f"layer {li}: prior recipe was not consumed")
            raw = args[0]
            recipe = self._build_recipe(li, hidden, raw.detach())
            self.pending_recipes[li] = recipe

        return hook

    def _norm_backward_hook(self, li):

        def hook(_module, grad_input, _grad_output):
            recipe = self.pending_recipes[li]
            self.pending_recipes[li] = None
            if recipe is None:
                raise RuntimeError(f"layer {li}: backward ran without paired recipe")
            grad = grad_input[0]
            if grad is None:
                raise RuntimeError(f"layer {li}: raw readout gradient is None")
            B, T, HP = grad.shape
            H = self.mixers[li].num_heads
            P = HP // H
            nwin = recipe.output.shape[1]
            stop = self.warmup + nwin * self.window
            paired_grad = (
                grad[:, self.warmup : stop].float().reshape(B, nwin, self.window, H, P)
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
            G = int(self.mixers[li].n_groups)
            repeat = H // G
            grouped = stats["output_error_sum"].reshape(G, repeat, -1).sum(1)
            independently_grouped = torch.stack(
                [
                    stats["output_error_sum"][g * repeat : (g + 1) * repeat].sum(0)
                    for g in range(G)
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
    def _build_recipe(self, li, input_hidden, raw_scan) -> LayerRecipe:
        mixer = self.mixers[li]
        Bsz, T, _ = input_hidden.shape
        H, P = (mixer.num_heads, mixer.head_dim)
        G, N = (mixer.n_groups, mixer.ssm_state_size)
        if H % G:
            raise RuntimeError(f"layer {li}: H={H} not divisible by G={G}")
        if raw_scan.shape != (Bsz, T, H * P):
            raise RuntimeError(
                f"layer {li}: raw scan {tuple(raw_scan.shape)} != {(Bsz, T, H * P)}"
            )
        if self.warmup >= T:
            raise RuntimeError(f"warmup={self.warmup} must be less than T={T}")
        nwin = (T - self.warmup) // self.window
        if nwin <= 0:
            raise RuntimeError(
                f"T={T}, warmup={self.warmup} contains no complete W={self.window} window"
            )
        stop = self.warmup + nwin * self.window
        import importlib

        nh = importlib.import_module(type(mixer).__module__)
        projected = mixer.in_proj(input_hidden)
        _, hidden_B_C, dt_raw = projected.split(
            [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1
        )
        conv = getattr(nh, "causal_conv1d_fn", None)
        if conv is None:
            hidden_B_C = mixer.act(
                mixer.conv1d(hidden_B_C.transpose(1, 2))[..., :T].transpose(1, 2)
            )
        else:
            hidden_B_C = conv(
                hidden_B_C.transpose(1, 2),
                mixer.conv1d.weight.squeeze(1),
                mixer.conv1d.bias,
                activation=mixer.activation,
            ).transpose(1, 2)
        x_raw, Bgate_raw, Cgate_raw = hidden_B_C.split(
            [
                mixer.intermediate_size,
                mixer.n_groups * mixer.ssm_state_size,
                mixer.n_groups * mixer.ssm_state_size,
            ],
            dim=-1,
        )
        x_grouped = x_raw.view(Bsz, T, H, P)
        B_grouped = Bgate_raw.view(Bsz, T, G, N)
        C_grouped = Cgate_raw.view(Bsz, T, G, N)
        scan_replayed = scan(
            nh,
            x_grouped,
            dt_raw,
            -torch.exp(mixer.A_log.float()),
            B_grouped,
            C_grouped,
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            dt_limit=mixer.time_step_limit,
        ).reshape(Bsz, T, H * P)
        replay_diff = (scan_replayed.float() - raw_scan.float()).abs()
        replay_abs = float(replay_diff.max())
        replay_rel = float(
            replay_diff.norm() / raw_scan.float().norm().clamp_min(1e-30)
        )
        if replay_abs > 0.02 and replay_rel > 0.0002:
            raise RuntimeError(
                f"layer {li}: replayed official chunk scan disagrees with hook input: max_abs={replay_abs:.3e}, rel={replay_rel:.3e}"
            )
        _, prefill_state = scan(
            nh,
            x_grouped[:, : self.warmup],
            dt_raw[:, : self.warmup],
            -torch.exp(mixer.A_log.float()),
            B_grouped[:, : self.warmup],
            C_grouped[:, : self.warmup],
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            dt_limit=mixer.time_step_limit,
            return_final_states=True,
        )
        x = x_grouped.float()
        Bgate = B_grouped.float().repeat_interleave(H // G, dim=2)
        Cgate = C_grouped.float().repeat_interleave(H // G, dim=2)
        dt = F.softplus(dt_raw.float() + mixer.dt_bias.float()[None, None, :])
        A = -torch.exp(mixer.A_log.float())
        decay = torch.exp(dt * A[None, None, :])
        state = prefill_state.float()
        dense = torch.empty(
            Bsz, stop - self.warmup, H, P, device=x.device, dtype=torch.float32
        )
        boundary_states = []
        output_windows = []
        basis = self.basis_orig[li].to(x.device)
        if tuple(basis.shape) not in ((G, self.mmax, N), (H, self.mmax, N)):
            raise RuntimeError(f"Invalid basis shape {tuple(basis.shape)}")
        repeat = H // G
        basis_head = (
            basis.repeat_interleave(repeat, dim=0) if basis.shape[0] == G else basis
        )
        rejected_total = torch.zeros(H, self.mmax, dtype=torch.long)
        current_Q = None
        alpha = None
        current_out = []
        all_Q = []
        decomposition_max_abs = 0.0
        decomposition_max_rel = 0.0
        decomposition_first_max_abs = 0.0
        decomposition_first_max_rel = 0.0
        replay_state = None
        for t in range(self.warmup, stop):
            window_pos = (t - self.warmup) % self.window
            if window_pos == 0:
                boundary_states.append(state.clone())
                U = torch.einsum("bhpn,hmn->bhpm", state, basis_head)
                if basis.shape[0] == G:
                    U_group = torch.einsum(
                        "bgrpn,gmn->bgrpm", state.view(Bsz, G, H // G, P, N), basis
                    ).reshape(Bsz, H, P, self.mmax)
                    torch.testing.assert_close(U, U_group, rtol=1e-05, atol=1e-07)
                current_Q, rejected = _mgs_prefix(U, self.rank_tol)
                rejected_total += rejected.sum(0).long().cpu()
                all_Q.append(current_Q)
                alpha = torch.ones(Bsz, H, device=x.device, dtype=torch.float32)
                replay_state = torch.zeros_like(state)
                current_out = []
            a = decay[:, t]
            state = (
                state * a[:, :, None, None]
                + x[:, t, :, :, None] * (dt[:, t, :, None] * Bgate[:, t])[:, :, None, :]
            )
            dense[:, t - self.warmup] = (
                torch.einsum("bhpn,bhn->bhp", state, Cgate[:, t])
                + x[:, t] * mixer.D.float()[None, :, None]
            )
            alpha = alpha * a
            replay_state = (
                replay_state * a[:, :, None, None]
                + x[:, t, :, :, None] * (dt[:, t, :, None] * Bgate[:, t])[:, :, None, :]
            )
            qtilde = Cgate[:, t] * alpha[:, :, None]
            o = torch.einsum("bhpn,bhn->bhp", boundary_states[-1], qtilde)
            replay_o = torch.einsum("bhpn,bhn->bhp", replay_state, Cgate[:, t])
            full_state_o = torch.einsum("bhpn,bhn->bhp", state, Cgate[:, t])
            decomposition_diff = (full_state_o - (o + replay_o)).abs()
            decomposition_max_abs = max(
                decomposition_max_abs, float(decomposition_diff.max())
            )
            decomposition_max_rel = max(
                decomposition_max_rel,
                float(decomposition_diff.norm() / full_state_o.norm().clamp_min(1e-30)),
            )
            if window_pos == 0:
                decomposition_first_max_abs = max(
                    decomposition_first_max_abs, float(decomposition_diff.max())
                )
                decomposition_first_max_rel = max(
                    decomposition_first_max_rel,
                    float(
                        decomposition_diff.norm() / full_state_o.norm().clamp_min(1e-30)
                    ),
                )
            current_out.append(o)
            if len(current_out) == self.window:
                output_windows.append(torch.stack(current_out, dim=1))
        dense_flat = dense.reshape(Bsz, stop - self.warmup, H * P)
        diff = (dense_flat - scan_replayed[:, self.warmup : stop].float()).abs()
        max_abs = float(diff.max())
        max_rel = float(
            diff.norm()
            / scan_replayed[:, self.warmup : stop].float().norm().clamp_min(1e-30)
        )
        if max_abs > self.recurrence_atol and max_rel > self.recurrence_rtol:
            raise RuntimeError(
                f"layer {li}: explicit recurrence disagrees with official chunk scan: max_abs={max_abs:.3e}, rel={max_rel:.3e}"
            )
        if decomposition_first_max_abs > 0.05 and decomposition_first_max_rel > 0.002:
            raise RuntimeError(
                f"layer {li}: first-step recurrent state read does not decompose into boundary and replay terms: max_abs={decomposition_first_max_abs:.3e}, rel={decomposition_first_max_rel:.3e}"
            )
        output = torch.stack(output_windows, dim=1)
        basis_q = torch.stack(all_Q, dim=1)
        qo = torch.einsum("buhpm,buwhp->buwhm", basis_q, output)
        total = output.square().sum(-1)
        retained = qo.square().cumsum(-1)
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
        grouped_a = output_error_sum.reshape(G, repeat, -1).sum(1)
        grouped_b = torch.stack(
            [output_error_sum[g * repeat : (g + 1) * repeat].sum(0) for g in range(G)]
        )
        group_abs = float((grouped_a - grouped_b).abs().max())
        return LayerRecipe(
            output=output,
            basis_q=basis_q,
            output_error_sum=output_error_sum,
            rejected_by_rank=rejected_total,
            recurrence_max_abs=max_abs,
            recurrence_max_rel=max_rel,
            replayed_kernel_max_abs=replay_abs,
            replayed_kernel_max_rel=replay_rel,
            decomposition_max_abs=decomposition_max_abs,
            decomposition_max_rel=decomposition_max_rel,
            decomposition_first_max_abs=decomposition_first_max_abs,
            decomposition_first_max_rel=decomposition_first_max_rel,
            rank0_max_abs=0.0,
            output_monotonic_max=monotonic_max,
            energy_identity_max_rel=energy_rel,
            group_accounting_max_abs=group_abs,
        )

    def finish_sequence(self, n_steps: int):
        stale = [i for i, x in enumerate(self.pending_recipes) if x is not None]
        if stale:
            raise RuntimeError(f"recipes not consumed after backward: {stale}")
        for key in self.sequence_sums:
            delta = self.sums[key] - self._sequence_start[key]
            self.sequence_sums[key].append(delta.float())
            self._sequence_start[key].copy_(self.sums[key])
        self.nseq += 1
        self.nstep += int(n_steps)

    def state_dict(self):
        payload = dict(self.sums)
        for key, values in self.sequence_sums.items():
            payload[f"per_sequence_{key}"] = torch.stack(values) if values else None
        return payload

    def load_state_dict(self, payload: dict):
        """Restore an atomically saved calibration checkpoint."""
        for key, target in self.sums.items():
            if key not in payload:
                raise RuntimeError(f"resume checkpoint is missing {key}")
            source = torch.as_tensor(payload[key])
            if source.shape != target.shape:
                raise RuntimeError(
                    f"resume {key} shape {tuple(source.shape)} != {tuple(target.shape)}"
                )
            target.copy_(source)
        self.nseq = int(payload["joint_nseq"])
        self.nstep = int(payload["joint_nstep"])
        self.max_energy_identity_error = float(
            payload.get("joint_max_energy_identity_error", 0.0)
        )
        self.recurrence_max_abs = float(payload.get("joint_recurrence_max_abs", 0.0))
        self.recurrence_max_rel = float(payload.get("joint_recurrence_max_rel", 0.0))
        self.replayed_kernel_max_abs = float(
            payload.get("joint_replayed_kernel_max_abs", 0.0)
        )
        self.replayed_kernel_max_rel = float(
            payload.get("joint_replayed_kernel_max_rel", 0.0)
        )
        self.decomposition_max_abs = float(
            payload.get("joint_decomposition_max_abs", 0.0)
        )
        self.decomposition_max_rel = float(
            payload.get("joint_decomposition_max_rel", 0.0)
        )
        self.decomposition_first_max_abs = float(
            payload.get("joint_decomposition_first_max_abs", 0.0)
        )
        self.decomposition_first_max_rel = float(
            payload.get("joint_decomposition_first_max_rel", 0.0)
        )
        self.rank0_max_abs = float(payload.get("joint_rank0_max_abs", 0.0))
        self.output_monotonic_max = float(
            payload.get("joint_output_monotonic_max", 0.0)
        )
        self.energy_identity_max_rel = float(
            payload.get("joint_energy_identity_max_rel", 0.0)
        )
        self.group_accounting_max_abs = float(
            payload.get("joint_group_accounting_max_abs", 0.0)
        )
        for key in self.sequence_sums:
            saved = payload.get(f"per_sequence_{key}")
            if saved is None or len(saved) != self.nseq:
                raise RuntimeError(f"resume checkpoint has invalid per-sequence {key}")
            self.sequence_sums[key] = [x.clone() for x in saved]
            self._sequence_start[key].copy_(self.sums[key])
