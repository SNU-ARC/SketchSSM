# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"KDA ordered transitions and Full-Gram paired calibration statistics."

import torch
import triton
import triton.language as tl


@triton.jit
def _states(
    Q,
    K,
    V,
    A,
    B,
    Starts,
    Boundary,
    Output,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    h, vi = tl.program_id(0), tl.program_id(1)
    kk = tl.arange(0, 128)
    vv = vi * 8 + tl.arange(0, 8)
    state = tl.full((8, 128), 0, tl.float32)
    boundary = tl.full((8, 128), 0, tl.float32)
    for t in range(T):
        if t % W == 0:
            tl.store(
                Starts + ((t // W * H + h) * 128 + vv[:, None]) * 128 + kk[None, :],
                state,
            )
            boundary = state
        q = tl.load(Q + (t * H + h) * 128 + kk)
        k = tl.load(K + (t * H + h) * 128 + kk)
        a = tl.load(A + (t * H + h) * 128 + kk)
        v = tl.load(V + (t * H + h) * 128 + vv).to(tl.float32)
        b = tl.load(B + t * H + h)
        state *= a[None, :]
        state += b * (v - tl.sum(state * k[None, :], 1))[:, None] * k[None, :]
        boundary *= a[None, :]
        boundary -= b * tl.sum(boundary * k[None, :], 1)[:, None] * k[None, :]
        tl.store(Output + (t * H + h) * 128 + vv, tl.sum(state * q[None, :], 1))
        tl.store(Boundary + (t * H + h) * 128 + vv, tl.sum(boundary * q[None, :], 1))


@triton.jit
def _queries(Q, K, A, B, X, H: tl.constexpr, W: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    kk = tl.arange(0, 128)
    x = tl.load(Q + (t * H + h) * 128 + kk)
    for s in range(t, t // W * W - 1, -1):
        k = tl.load(K + (s * H + h) * 128 + kk)
        a = tl.load(A + (s * H + h) * 128 + kk)
        b = tl.load(B + s * H + h)
        x = a * (x - b * k * tl.sum(k * x))
    tl.store(X + (t * H + h) * 128 + kk, x)


@torch.no_grad()
def features(q, k, v, g, beta, window=16):
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("Expected one (1,T,H,128) sequence")
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        raise ValueError("KDA statistics require equal K=V=128 and matching heads")
    q, k, v, g = [x[0].float().contiguous() for x in (q, k, v, g)]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 128**0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    a = g.exp()
    beta = beta[0].float().contiguous()
    t, h, _ = q.shape
    if t % window:
        raise ValueError("Only complete windows can enter calibration")
    starts = torch.empty(t // window, h, 128, 128, device=q.device)
    boundary, output, x = [torch.empty_like(q) for _ in range(3)]
    with torch.accelerator.device_index(q.device.index):
        _states[(h, 16)](q, k, v, a, beta, starts, boundary, output, t, h, window)
        _queries[(t, h)](q, k, a, beta, x, h, window)
    reconstructed = torch.einsum(
        "uhvk,uwhk->uwhv", starts, x.reshape(-1, window, h, 128)
    )
    rel = (
        reconstructed.flatten() - boundary.flatten()
    ).norm() / boundary.norm().clamp_min(1e-30)
    if not torch.isfinite(rel) or rel > 2e-5:
        raise ValueError(f"Full-transition decomposition failed: {rel.item()}")
    return starts, x.reshape(-1, window, h, 128), reconstructed, output, float(rel)


def prefix_qr(u, tol=1e-7):
    """Twice-reorthogonalized MGS; rejected directions remain zero columns."""
    scale = u.norm(dim=-2).amax(-1).clamp_min(1e-30)
    columns, rejected = [], []
    for j in range(u.shape[-1]):
        v = u[..., j]
        if columns:
            q = torch.stack(columns, -1)
            for _ in range(2):
                v = v - (q @ (q.transpose(-1, -2) @ v[..., None]))[..., 0]
        norm = v.norm(dim=-1)
        keep = norm > tol * scale
        columns.append(
            torch.where(keep[..., None], v / norm.clamp_min(1e-30)[..., None], 0)
        )
        rejected.append(~keep)
    return torch.stack(columns, -1), torch.stack(rejected, -1)


def paired_curves(gradient, output, q):
    """Shapes: gradient/output (U,W,H,V), Q (U,H,V,M)."""
    qo = torch.einsum("uhvm,uwhv->uwhm", q, output)
    gq = torch.einsum("uhvm,uwhv->uwhm", q, gradient)
    energy = output.square().sum(-1)
    residual = torch.cat(
        [energy[..., None], (energy[..., None] - qo.square().cumsum(-1)).clamp_min(0)],
        -1,
    )
    dot = (gradient * output).sum(-1)
    dots = torch.cat([dot[..., None], dot[..., None] - (gq * qo).cumsum(-1)], -1)
    return {
        "output_error_sum": residual.double().sum((0, 1)).cpu(),
        "joint_dot_sq_sum": dots.double().square().sum((0, 1)).cpu(),
        "scalar_grad_output_error_sum": (
            gradient.square().sum(-1)[..., None] * residual
        )
        .double()
        .sum((0, 1))
        .cpu(),
        "joint_dot_sum": dots.double().sum((0, 1)).cpu(),
        "energy_identity_error": float(
            (energy[..., None] - qo.square().cumsum(-1) - residual[..., 1:]).abs().max()
        ),
    }
