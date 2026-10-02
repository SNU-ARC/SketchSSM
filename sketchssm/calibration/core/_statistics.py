# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
'Paired output error and same-token, per-head output-gradient statistics.'
import torch

def _mgs_prefix(U: torch.Tensor, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Twice-reorthogonalized MGS that preserves every nested prefix.

    Args:
        U: (..., P, M) sketch matrices.
        tol: relative residual-norm threshold for accepting a column.

    Returns:
        Q: (..., P, M), with zero columns for numerically dependent inputs.
        rejected: (..., M) boolean dependency mask.
    """
    if U.ndim < 2:
        raise ValueError(f"U must have at least two axes, got {tuple(U.shape)}")
    scale = U.norm(dim=-2).amax(dim=-1).clamp_min(1e-30)
    qcols = []
    rejected = []
    for j in range(U.shape[-1]):
        v = U[..., j]
        if qcols:
            Qp = torch.stack(qcols, dim=-1)
            for _ in range(2):
                coef = torch.einsum("...pi,...p->...i", Qp, v)
                v = v - torch.einsum("...pi,...i->...p", Qp, coef)
        nv = v.norm(dim=-1)
        keep = nv / scale > tol
        qcols.append(torch.where(
            keep[..., None], v / nv.clamp_min(1e-30)[..., None], 0.0))
        rejected.append(~keep)
    return torch.stack(qcols, dim=-1), torch.stack(rejected, dim=-1)

def _rank_statistics(
    grad: torch.Tensor,
    output: torch.Tensor,
    basis_q: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all rank-prefix statistics from paired gradients and outputs.

    Shapes:
      grad, output: (B, U, W, H, P)
      basis_q:      (B, U, H, P, M)
    Returned rank curves have shape (H, M+1); grad_sq has shape (H,).
    """
    if grad.shape != output.shape:
        raise ValueError(f"gradient {grad.shape} != output {output.shape}")
    if basis_q.shape[:-2] != output.shape[:2] + output.shape[3:4] \
            or basis_q.shape[-2] != output.shape[-1]:
        raise ValueError(
            f"basis {basis_q.shape} is incompatible with output {output.shape}")

    qo = torch.einsum("buhpm,buwhp->buwhm", basis_q, output)
    total = output.square().sum(-1)
    retained = qo.square().cumsum(-1)
    output_error = torch.cat(
        [total[..., None], (total[..., None] - retained).clamp_min(0.0)], dim=-1)

    go = (grad * output).sum(-1)
    gq = torch.einsum("buwhp,buhpm->buwhm", grad, basis_q)
    removed = (gq * qo).cumsum(-1)
    dot = torch.cat([go[..., None], go[..., None] - removed], dim=-1)
    reduce_axes = (0, 1, 2)
    return {
        "output_error_sum": output_error.double().sum(reduce_axes).cpu(),
        "output_error_sq_sum": output_error.double().square().sum(reduce_axes).cpu(),
        "joint_dot_sum": dot.double().sum(reduce_axes).cpu(),
        "joint_dot_abs_sum": dot.double().abs().sum(reduce_axes).cpu(),
        "joint_dot_sq_sum": dot.double().square().sum(reduce_axes).cpu(),

        "scalar_grad_output_error_sum": (
            grad.square().sum(-1)[..., None] * output_error
        ).double().sum(reduce_axes).cpu(),
        "grad_sq_sum": grad.double().square().sum((0, 1, 2, 4)).cpu(),
        "max_energy_identity_error":
            (total[..., None] - (retained + output_error[..., 1:])).abs().max().cpu(),
    }
