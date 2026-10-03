# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""SketchSSM Kimi Delta Attention (KDA) decode with CUDA kernels.

The step and flush kernels come from the SketchSSM paper; they live in
``csrc/kda/`` and are specialized per window and build knobs (precompiled when
available, else compiled with NVRTC). The flush kernels also build a
request's sketch from its state after prefill (``kda_cold_build``).
"""

import functools
import logging
from pathlib import Path

import torch

from . import _driver as drv
from . import _kernel as kn
from . import _runtime as rt
from ._runtime import Support, Target
from .types import KDARings, KDASketch

FAMILY = "kda"
HEAD_DIM = 128
WINDOW = 16
LOWER_BOUND = -5.0
MAX_HEADS = 128
# FP32 scratch floats per (row, sketch head) between a flush and its finish.
SCRATCH_FLOATS = 1792

# Default build knobs.
_STEP_CONFIG = dict(S7_MINB=6, NW=4, S8_PREFETCH=0)
_FLUSH_CONFIG = dict(K1_MINB=3, K1_ABLATE=0)
# sm_90 flush occupancy: lower CTAs/SM avoid register spills.
_FLUSH_CONFIG_SM90 = dict(K1_MINB=2, K1_MINB16=3)


def window_supported(window: int) -> bool:
    return window >= 16 and window % 16 == 0


def _target(target: Target | None) -> Target:
    return target if target is not None else rt.current_target()


def config_file_name(num_heads: int, window: int = WINDOW,
                     target: Target | None = None) -> str:  # fmt: skip
    """Tuned-config file name of a layer shape on this GPU."""
    w = "" if window == WINDOW else f",window={window}"
    return f"kda,num_heads={num_heads}{w},device_name={_target(target).device_name}.json"


def default_config(
    num_heads: int,
    window: int = WINDOW,
    target: Target | None = None,
) -> tuple[dict, dict]:
    """Step and flush knobs without a tuned config for ``target`` (default:
    the current GPU)."""
    flush = dict(_FLUSH_CONFIG)
    if _target(target).cc < 100:
        flush.update(_FLUSH_CONFIG_SM90)
    return dict(_STEP_CONFIG), flush


def _config_file(num_heads: int, window: int, target: Target | None = None) -> Path | None:
    return rt.find_config(FAMILY, config_file_name(num_heads, window, target))


def _config_path(num_heads: int, window: int, target: Target | None = None) -> Path | None:
    """The window's own tuned file, else the shape's window-16 file."""
    path = _config_file(num_heads, window, target)
    if path is None and window != WINDOW:
        path = _config_file(num_heads, WINDOW, target)
    return path


@functools.cache
def tuned_config(num_heads: int, window: int = WINDOW,
                 target: Target | None = None) -> tuple[dict, dict]:  # fmt: skip
    """Step and flush build knobs for a layer shape on this GPU (or
    ``target``).

    A JSON file ``{"step": {...}, "flush": {...}}`` named by
    ``config_file_name`` in a config folder overrides ``default_config``.
    """
    path = _config_path(num_heads, window, target)
    if path is not None:
        rt.log_once(
            logging.INFO, "Using SketchSSM KDA CUDA config %s for window %d", path,
            window,
        )  # fmt: skip
        return _file_config(num_heads, window, path, target)
    rt.log_once(
        logging.INFO,
        "No tuned SketchSSM KDA CUDA config %s; using the default build knobs.",
        config_file_name(num_heads, window, target),
    )
    return default_config(num_heads, window, target)


def _file_config(num_heads: int, window: int, path: Path,
                 target: Target | None = None) -> tuple[dict, dict]:  # fmt: skip
    step, flush = default_config(num_heads, window, target)
    raw = rt.read_config(path)
    step.update(raw.get("step", {}))
    flush.update(raw.get("flush", {}))
    return step, flush




def kda_supported(
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    window: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
    lower_bound: float = LOWER_BOUND,
) -> Support:
    """Whether a KDA layer can use the SketchSSM CUDA kernels (TMA: sm_90+),
    and where they come from (``Support.source``)."""
    shape_ok = (
        0 < num_heads <= MAX_HEADS
        and head_k_dim == HEAD_DIM
        and head_v_dim == HEAD_DIM
        and window_supported(window)
        and lower_bound == LOWER_BOUND
        and activation_dtype == torch.bfloat16
        and state_dtype == torch.float32
    )
    if not shape_ok:
        return Support(False, "unsupported layer shape, window, gate or dtype")
    support = rt.cuda_available(90)
    if not support:
        return support
    step, flush = tuned_config(num_heads, window)
    return rt.kernel_support([step_spec(step, window), flush_spec(flush, window)])


# ── specializations ──

_MMAX = (16, 32, 64, 128)
_STEP_PROBES = (("nw", "NW"), ("window", "KDA_W"))
_FLUSH_PROBES = (
    ("threads", "NTHREADS"),
    ("nblk", "NBLK"),
    ("scratch", "SCRATCH_F"),
    ("ablate", "K1_ABLATE"),
    ("window", "KDA_W"),
    ("dense_smem", "DN_BYTES"),
    *((f"smem{m}", f"Smem<{m}, false>::BYTES") for m in _MMAX),
    *((f"smemb{m}", f"Smem<{m}, true>::BYTES") for m in _MMAX),
    *((f"minb{m}", f"K1_MINB_OF({m})") for m in _MMAX),
    *((f"minbb{m}", f"K1_MINB_LB({m}, true)") for m in _MMAX),
)


def _main_name(mmax: int, flush: bool) -> str:
    return f"kda_flush_main<{mmax}, __nv_bfloat16, {'true' if flush else 'false'}>"


_FINISH = "kda_flush_finish<__nv_bfloat16>"
_DENSE = ("kda_dense_flush<false>", "kda_dense_flush<true>")


def _window_define(config: dict, window: int) -> dict:
    # The kernels' window define (KDA_W, default 16).
    return config if window == WINDOW else dict(config, KDA_W=window)


def step_spec(config: dict, window: int) -> kn.Spec:
    return kn.Spec.make("kda_step", "kda/kda_sketch_step.cuh", _window_define(config, window),
                        ["kda_step_kernel"], (), _STEP_PROBES)  # fmt: skip


def flush_spec(config: dict, window: int) -> kn.Spec:
    names = [_main_name(m, f) for m in _MMAX for f in (False, True)] + [_FINISH, *_DENSE]
    return kn.Spec.make("kda_flush", "kda/kda_sketch_flush.cuh", _window_define(config, window),
                        names, (), _FLUSH_PROBES)  # fmt: skip


_SPEC = {0: step_spec, 1: flush_spec}


def _build_ext(kind: int, num_heads: int, window: int) -> kn.Kernel:
    """Load kernel ``kind`` (0: step, 1: flush). Knobs borrowed from the
    window-16 file that do not build at ``window`` fall back to the defaults."""
    config = tuned_config(num_heads, window)[kind]
    try:
        return kn.load(_SPEC[kind](config, window))
    except Exception as e:
        default = default_config(num_heads, window)[kind]
        path = _config_path(num_heads, window)
        borrowed = (
            window != WINDOW
            and path is not None
            and _config_file(num_heads, window) is None
            and config == _file_config(num_heads, window, path)[kind]
        )
        if not borrowed or config == default:
            raise
        rt.logger.warning(
            "SketchSSM KDA: the window-16 knobs %s do not build at window %d "
            "(%s); using the architecture defaults %s",
            config, window, type(e).__name__, default,
        )  # fmt: skip
        return kn.load(_SPEC[kind](default, window))


@functools.cache
def _step_ext(num_heads: int, window: int = WINDOW) -> kn.Kernel:
    return _build_ext(0, num_heads, window)


@functools.cache
def _flush_ext(num_heads: int, window: int = WINDOW) -> kn.Kernel:
    return _build_ext(1, num_heads, window)


def scratch_numel(max_rows: int, num_sketch_heads: int) -> int:
    """FP32 scratch floats between a flush and its finish for ``max_rows``."""
    return max(1, max_rows * num_sketch_heads * SCRATCH_FLOATS)


def _check_scratch(scratch: torch.Tensor, rows: int, sketch: KDASketch) -> None:
    need = rows * sketch.tables.num_sketch_heads * SCRATCH_FLOATS
    assert scratch.dtype == torch.float32 and scratch.numel() >= need


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


_ring_strides: dict[tuple, tuple[tuple, int]] = {}


def _ring_page_stride(rings, H: int, W: int) -> int:
    """The rings' shared slot stride in bytes (the Mamba cache page): each is
    dense [slot][H][W][128] ([slot][H][W] for beta) within a slot, and slots
    are 16-byte aligned. Checked once per set of ring tensors (the cache's
    rings are allocated once)."""
    key = (*map(id, rings), H, W)
    hit = _ring_strides.get(key)
    if hit is not None and all(a is b for a, b in zip(hit[0], rings)):
        return hit[1]
    RS = rings[0].stride(0) * rings[0].element_size()
    for r in rings:
        ok = r.shape[1] == H and r.shape[2] == W and (
            r.shape[3] == 128 and r.stride(3) == 1 and r.stride(2) == 128
            and r.stride(1) == W * 128 if r.dim() == 4
            else r.dim() == 3 and r.stride(2) == 1 and r.stride(1) == W
        )  # fmt: skip
        _check(ok, "kda sketch: rings must be dense [slot][H][W][128] / [slot][H][W] "
                   f"within a slot (W = {W})")  # fmt: skip
        _check(r.stride(0) * r.element_size() == RS,
               "kda sketch: rings must share one slot stride (bytes)")  # fmt: skip
        _check(r.data_ptr() % 16 == 0, "kda sketch: rings must be 16-byte aligned")
    _check(RS % 16 == 0, "kda sketch: the ring slot stride must be a multiple of 16 bytes")
    if len(_ring_strides) > 256:
        _ring_strides.clear()
    _ring_strides[key] = (tuple(rings), RS)
    return RS


def _input_row_stride(x: torch.Tensor, B: int, H: int, vec: bool, name: str) -> int:
    """A step input may be a row-strided view: rows of [H][128] (beta: [H])
    with dense heads and unit inner stride; returns the row stride."""
    _check(x.dtype == torch.bfloat16 and x.shape[0] == B, f"kda step: bf16 input rows: {name}")
    if vec:
        ok = x.dim() == 3 and x.shape[1] == H and x.shape[2] == 128 and x.stride(2) == 1 \
            and x.stride(1) == 128  # fmt: skip
    else:
        ok = x.dim() == 2 and x.shape[1] == H and x.stride(1) == 1
    _check(ok, f"kda step: {name} must be [B][H][128] (beta [B][H]) with dense rows of unit "
               "inner stride")  # fmt: skip
    dense = H * 128 if vec else H
    rs = x.stride(0) if B > 1 else dense
    _check(rs >= dense and (B - 1) * rs + H * 128 <= 0xFFFFFFFF,
           f"kda step: {name} row stride out of range")  # fmt: skip
    _check(not vec or (rs % 4 == 0 and x.data_ptr() % 8 == 0),
           f"kda step: {name} rows must be 8-byte aligned")  # fmt: skip
    return rs


_STEP_SIG = drv.Signature(
    ["Q", "i"] + ["Q"] * 5 + ["5I"] + ["Q"] * 6 + ["q", "q"] + ["Q"] * 10 + ["q", "Q", "i", "i", "f"]
    + ["Q", "q", "Q"]
)
_MAIN_SIG = drv.Signature(
    ["i", "i", "i", f"{drv.TENSOR_MAP_BYTES}s", "Q", "q", "q", "Q", "i", "Q", "Q", "128i",
     "i", "i"] + ["Q"] * 7 + ["q", "Q", "q", "Q", "f", "Q"]
)  # fmt: skip
_FINISH_SIG = drv.Signature(["Q", "i", "Q", "128i", "i", "i", "i"] + ["Q"] * 5)
_DENSE_SIG = drv.Signature(
    ["Q", "i", "Q", "Q", "Q", "i", "Q", "q", "q", "Q", "q"] + ["Q"] * 4
    + ["q", "Q", "q", "Q", "i", "f"]
)  # fmt: skip


def _step_launch(k: kn.Kernel, heads, q, kk, v, gate, beta, a_log, bias, slots, meta, pos,
                 state, S0, S1, latch, phi, ranks, kr, vr, br, prefix_r, fr, ur, dr, out,
                 H, G, scale, dense, dense_rows) -> None:  # fmt: skip
    """The step launch (``step`` of the former C++ launcher)."""
    W = k.probes["window"]
    _check(G % 8 == 0, "kda step: padded rank must be a multiple of 8")
    _check(latch.dtype == torch.bfloat16 and ur.dtype == torch.bfloat16,
           "kda step s7: bf16 sketch storage and bf16 d/u rings")  # fmt: skip
    _check(kr.dtype == torch.bfloat16 and vr.dtype == torch.bfloat16 and dr.dtype == torch.bfloat16
           and br.dtype == torch.float32 and prefix_r.dtype == torch.float32,
           "kda step: bf16 k / v / d rings, fp32 beta / prefix rings")  # fmt: skip
    _check(all(t.dtype == torch.int32 and t.is_contiguous() for t in (slots, meta, pos, heads)),
           "kda step: contiguous int32 heads, slots, meta and pos")  # fmt: skip
    _check(state.dtype == torch.float32 and state.stride(2) == 128 and state.stride(3) == 1,
           "kda step: fp32 [page][H][128][128] state")  # fmt: skip
    _check(dense.dtype == torch.bfloat16 and dense.is_contiguous()
           and dense.shape[2:] == (128, 128) and dense_rows.dtype == torch.int32
           and dense_rows.is_contiguous(),
           "kda step: bf16 [reqs][dense heads][128][128] dense state rows, int32 dense row table")  # fmt: skip
    RS = _ring_page_stride((kr, vr, br, prefix_r, ur, dr), H, W)
    B, NH = q.shape[0], heads.shape[0]
    if NH == 0 or B == 0:
        return
    strides = (
        _input_row_stride(q, B, H, True, "q"), _input_row_stride(kk, B, H, True, "k"),
        _input_row_stride(v, B, H, True, "v"), _input_row_stride(gate, B, H, True, "gate"),
        _input_row_stride(beta, B, H, False, "beta"),
    )  # fmt: skip
    nw = k.probes["nw"]
    args = (
        heads.data_ptr(), NH, q.data_ptr(), kk.data_ptr(), v.data_ptr(), gate.data_ptr(),
        beta.data_ptr(), *strides, a_log.data_ptr(), bias.data_ptr(), slots.data_ptr(),
        meta.data_ptr(), pos.data_ptr(), state.data_ptr(), S0, S1, latch.data_ptr(),
        phi.data_ptr(), ranks.data_ptr(), kr.data_ptr(), vr.data_ptr(), br.data_ptr(),
        prefix_r.data_ptr(), fr.data_ptr(), ur.data_ptr(), dr.data_ptr(), RS,
        out.data_ptr(), H, G, scale, dense.data_ptr(), dense.stride(0), dense_rows.data_ptr(),
    )  # fmt: skip
    drv.launch(k.function("kda_step_kernel", _STEP_SIG), _STEP_SIG, (B, (NH + nw - 1) // nw, 1),
               (32 * nw, 1, 1), 0, drv.current_stream(q.device.index), args)  # fmt: skip


_main_configured: dict[int, int] = {}  # flush main function -> CTAs per SM it fits
_tensor_maps: dict[tuple, bytes] = {}
_head_tables: dict[int, tuple[torch.Tensor, list[int]]] = {}


def _head_table(heads: torch.Tensor) -> list[int]:
    """The packed head table of an immutable CPU int32 vector (cached per
    tensor)."""
    hit = _head_tables.get(id(heads))
    if hit is not None and hit[0] is heads:
        return hit[1]
    _check(heads.device.type == "cpu" and heads.dtype == torch.int32 and heads.is_contiguous()
           and heads.numel() <= 128,
           "head metadata must be an immutable CPU int32 vector of length<=128")  # fmt: skip
    packed = heads.tolist()
    packed += [0] * (128 - len(packed))
    if len(_head_tables) > 1024:
        _head_tables.clear()
    _head_tables[id(heads)] = (heads, packed)
    return packed


def _state_map(state: torch.Tensor, S0: int, S1: int, half: bool) -> bytes:
    """The state TMA descriptor (rows of 128 floats over every state page)."""
    key = (state.data_ptr(), state.shape[0], state.shape[1], S0, S1, half)
    tmap = _tensor_maps.get(key)
    if tmap is None:
        span = (state.shape[0] - 1) * S0 + (state.shape[1] - 1) * S1 + 16384
        if half:
            _check(S0 % 128 == 0 and S1 % 128 == 0,
                   "half-key TMA requires row-aligned native state pages")  # fmt: skip
            tmap = drv.encode_tensor_map_tiled(state.data_ptr(), [32, 4, span // 128],
                                               [128, 512], [32, 2, 16])  # fmt: skip
        else:
            _check(S0 % 32 == 0 and S1 % 32 == 0,
                   "state TMA requires 128-byte aligned native state pages")  # fmt: skip
            tmap = drv.encode_tensor_map_tiled(state.data_ptr(), [32, span // 32], [128],
                                               [32, 64])  # fmt: skip
        if len(_tensor_maps) > 64:
            _tensor_maps.clear()
        _tensor_maps[key] = tmap
    return tmap


def _main_function(k: kn.Kernel, mmax: int, flush: bool, smem: int, device: int) -> tuple[int, int]:
    """The main kernel of a rank bucket, opted in to its shared memory, and
    the CTAs per SM it fits."""
    p = k.probes
    fn = k.function(_main_name(mmax, flush), _MAIN_SIG)
    occ = _main_configured.get(fn)
    if occ is None:
        W = p["window"]
        optin = drv.device_attribute(device, drv.DEV_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        _check(smem <= optin, f"kda flush: window {W}, rank bucket {mmax}: the main kernel needs "
                              f"{smem} bytes of shared memory per block, above this GPU's limit "
                              f"of {optin} (cudaDevAttrMaxSharedMemoryPerBlockOptin)")  # fmt: skip
        drv.set_max_dynamic_smem(fn, smem)
        occ = drv.occupancy(fn, p["threads"], smem)
        _check(occ >= 1, f"kda flush: window {W}, rank bucket {mmax}: the main kernel "
                         f"({p['threads']} threads, {smem} bytes of shared memory) does not fit "
                         "one CTA per SM (registers / shared memory per SM)")  # fmt: skip
        _main_configured[fn] = occ
    return fn, occ


def _flush_launches(
    state: torch.Tensor,
    rings: KDARings,
    rows: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    sketch: KDASketch,
    scratch: torch.Tensor,
    q: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    flush: bool,
) -> None:
    """The flush (or cold build): the dense heads' fold and BF16 state rows,
    ``flush_main`` per rank bucket, then ``flush_finish`` (the former C++
    launchers)."""
    t = sketch.tables
    _dense_launch(state, rings, rows, slots, meta, sketch, q, out, scale, flush)
    if not t.num_sketch_heads:
        return
    assert rows.is_contiguous() and rows.dtype == torch.int32
    _check_scratch(scratch, rows.numel(), sketch)
    _check_window(rings, sketch)
    k = _flush_ext(t.num_heads, t.window)
    p = k.probes
    H, G, NHall = t.num_heads, t.rank_cap, t.num_sketch_heads
    S0, S1 = state.stride(0), state.stride(1)
    frame, widths, latch = t.frame_gk, t.ranks, sketch.u
    kr, vr, prefix_r, beta_r = rings.k, rings.v, rings.prefix, rings.beta
    _check(state.dtype == torch.float32 and state.stride(2) == 128 and state.stride(3) == 1,
           "kda flush: fp32 [page][H][128][128] state with dense inner strides")  # fmt: skip
    _check(frame.is_contiguous() and frame.dtype == torch.float32,
           "frame: fp32 [H][128][128] G-major storage")  # fmt: skip
    _check(latch.dtype == torch.bfloat16, "kda flush: bf16 sketch storage")
    _check(all(x.dtype == torch.int32 and x.is_contiguous() for x in (rows, slots, meta)),
           "kda flush: contiguous int32 rows, slots and meta")  # fmt: skip
    cap = rows.numel()
    if cap == 0:
        return
    _check(kr.dtype == torch.bfloat16 and vr.dtype == torch.bfloat16
           and prefix_r.dtype == torch.float32 and beta_r.dtype == torch.float32,
           "kda flush: bf16 k / v rings, fp32 prefix / beta rings")  # fmt: skip
    RS = _ring_page_stride((kr, vr, prefix_r, beta_r), H, p["window"])
    QS = 0
    if flush:
        _check(q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[1] == H
               and q.shape[2] == 128 and q.stride(2) == 1 and q.stride(1) == 128
               and q.data_ptr() % 8 == 0,
               "kda flush: q must be bf16 [B][H][128] with dense rows, 8-byte aligned")  # fmt: skip
        QS = q.stride(0) if q.shape[0] > 1 else H * 128
        _check(QS >= H * 128 and QS % 4 == 0,
               "kda flush: q row stride must be >= H * 128 and a multiple of 4")  # fmt: skip
        _check(out.dtype == torch.bfloat16 and out.is_contiguous() and out.shape[0] == q.shape[0]
               and out.numel() == q.shape[0] * H * 128,
               "kda flush: out must be contiguous bf16 [B][H][128]")  # fmt: skip
    _check(scratch.numel() >= cap * NHall * p["scratch"], "kda flush: scratch too small")
    device = state.device.index
    stream = drv.current_stream(device)
    sms = drv.device_attribute(device, drv.DEV_MULTIPROCESSOR_COUNT)
    blocked = flush and p["nblk"] > 1
    ptrs = (state.data_ptr(), rows.data_ptr(), slots.data_ptr(), meta.data_ptr(),
            frame.data_ptr(), widths.data_ptr(), latch.data_ptr(), kr.data_ptr(),
            vr.data_ptr(), prefix_r.data_ptr(), beta_r.data_ptr(), q.data_ptr(),
            out.data_ptr(), scratch.data_ptr())  # fmt: skip
    for heads, mmax, base in t.flush_groups:
        _check(mmax in _MMAX, "kda flush: mmax must be 16/32/64/128")
        smem = p[f"smemb{mmax}"] if blocked else p[f"smem{mmax}"]
        tmap = _state_map(state, S0, S1, mmax == 16 and not blocked)
        fn, occ = _main_function(k, mmax, flush, smem, device)
        NH = heads.shape[0]
        table = _head_table(heads)
        per_sm = min(p[f"minbb{mmax}"], occ) if blocked else p[f"minb{mmax}"]
        args = (
            NH, base, NHall, tmap, ptrs[0], S0, S1, ptrs[1], cap, ptrs[2], ptrs[3], *table,
            H, G, *ptrs[4:11], RS, ptrs[11], QS, ptrs[12], scale, ptrs[13],
        )  # fmt: skip
        drv.launch(fn, _MAIN_SIG, (min(cap * NH, sms * per_sm), 1, 1), (p["threads"], 1, 1),
                   smem, stream, args)  # fmt: skip
    # The finish.
    if p["ablate"] & 8:
        return
    phi, fr, heads_all = sketch.phi, sketch.f, t.heads_all
    _check(phi.dtype == torch.bfloat16 and fr.dtype == torch.bfloat16,
           "kda flush: bf16 sketch storage")  # fmt: skip
    n_all = heads_all.shape[0]
    if n_all == 0:
        return
    table = _head_table(heads_all)
    args = (
        ptrs[1], cap, ptrs[3], *table, n_all, H, G, ptrs[4], ptrs[5], phi.data_ptr(),
        fr.data_ptr(), ptrs[13],
    )  # fmt: skip
    drv.launch(k.function(_FINISH, _FINISH_SIG), _FINISH_SIG,
               (min((cap * n_all + 3) // 4, sms * 16), 1, 1), (128, 1, 1), 0, stream, args)


def _dense_launch(state, rings, rows, slots, meta, sketch, q, out, scale, flush) -> None:
    """Dense heads of the listed rows: (flush) fold the window into the FP32
    state exactly and write the output; then their BF16 state rows."""
    t = sketch.tables
    nd, cap = t.num_dense_heads, rows.numel()
    if nd == 0 or cap == 0:
        return
    dense, heads = sketch.dense, t.dense_heads_d
    _check(dense.dtype == torch.bfloat16 and dense.is_contiguous() and dense.shape[1] >= nd
           and dense.shape[2:] == (128, 128),
           "kda flush: bf16 [reqs][dense heads][128][128] dense state rows")  # fmt: skip
    _check(heads.dtype == torch.int32 and heads.is_contiguous() and heads.numel() == nd,
           "kda flush: int32 dense head list")  # fmt: skip
    _check(state.dtype == torch.float32 and state.stride(2) == 128 and state.stride(3) == 1,
           "kda flush: fp32 [page][H][128][128] state with dense inner strides")  # fmt: skip
    _check(all(x.dtype == torch.int32 and x.is_contiguous() for x in (rows, slots, meta)),
           "kda flush: contiguous int32 rows, slots and meta")  # fmt: skip
    k = _flush_ext(t.num_heads, t.window)
    H = t.num_heads
    RS, QS = 0, 0
    kr, vr, prefix_r, beta_r = rings.k, rings.v, rings.prefix, rings.beta
    if flush:
        RS = _ring_page_stride((kr, vr, prefix_r, beta_r), H, k.probes["window"])
        _check(q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[1] == H
               and q.stride(2) == 1 and q.stride(1) == 128 and q.data_ptr() % 8 == 0
               and out.dtype == torch.bfloat16 and out.is_contiguous(),
               "kda flush: bf16 [B][H][128] q (dense rows) and contiguous out")  # fmt: skip
        QS = q.stride(0) if q.shape[0] > 1 else H * 128
    device = state.device.index
    sms = drv.device_attribute(device, drv.DEV_MULTIPROCESSOR_COUNT)
    fn = k.function(_DENSE[int(flush)], _DENSE_SIG)
    smem = k.probes["dense_smem"]
    occ = _main_configured.get(fn)
    if occ is None:
        drv.set_max_dynamic_smem(fn, smem)
        occ = _main_configured[fn] = max(1, drv.occupancy(fn, 128, smem))
    args = (
        rows.data_ptr(), cap, slots.data_ptr(), meta.data_ptr(), heads.data_ptr(), nd,
        state.data_ptr(), state.stride(0), state.stride(1), dense.data_ptr(), dense.stride(0),
        kr.data_ptr(), vr.data_ptr(), prefix_r.data_ptr(), beta_r.data_ptr(), RS, q.data_ptr(),
        QS, out.data_ptr(), H, scale,
    )  # fmt: skip
    drv.launch(fn, _DENSE_SIG, (min(cap * nd, sms * occ), 1, 1), (128, 1, 1), smem,
               drv.current_stream(device), args)  # fmt: skip


def _check_window(rings: KDARings, sketch: KDASketch) -> None:
    w = sketch.tables.window
    assert rings.window == w and sketch.f.shape[-1] == w


def _check_rows(slots: torch.Tensor, meta: torch.Tensor, null_block_id: int):
    # The kernels treat slots <= 0 as padding.
    assert null_block_id == 0
    assert slots.is_contiguous() and slots.dtype == torch.int32
    assert meta.is_contiguous() and meta.dtype == torch.int32


def kda_cold_build(
    state: torch.Tensor,
    rings: KDARings,
    slots: torch.Tensor,
    meta: torch.Tensor,
    rows: torch.Tensor,
    sketch: KDASketch,
    scratch: torch.Tensor,
    null_block_id: int = 0,
) -> None:
    """Build the sketch (and the dense heads' BF16 state rows) of each listed
    row from its state.

    The rows' next decode step must be at window position 0.

    Args:
        state: ``(slots, H, V, K)`` FP32 state.
        rings: the layer's window rings (not read).
        slots: ``(n,)`` int32 state slots.
        meta: ``(n,)`` int32 request rows.
        rows: int32 indices into ``slots`` / ``meta`` to build, then -1
            padding.
        sketch: the layer's per-request sketch.
        scratch: at least ``scratch_numel(rows.numel(), num_sketch_heads)``
            FP32 floats.
    """
    if rows.numel() == 0:
        return
    _check_rows(slots, meta, null_block_id)
    _flush_launches(
        state, rings, rows, slots, meta, sketch, scratch, rings.k, rings.k, 1.0,
        False,
    )  # fmt: skip


def kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    rings: KDARings,
    slots: torch.Tensor,
    meta: torch.Tensor,
    pos: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: KDASketch,
    scratch: torch.Tensor,
    scale: float = HEAD_DIM**-0.5,
    null_block_id: int = 0,
    has_flush_rows: bool = True,
) -> None:
    """One SketchSSM decode step of a KDA layer.

    Every row runs the step (ring append; sketch heads read their sketch,
    dense heads their BF16 state rows and the window, keeping FP32 replay
    rows in their u / d ring bytes). Only rows at ``pos == W - 1`` change
    the FP32 state: they fold the window into it exactly, rebuild the sketch
    and the dense heads' BF16 state rows, and write their output.
    q and k are l2-normalized in the kernels, the gate is
    ``-5 sigmoid(exp(A_log) (g + dt_bias))`` and beta is ``sigmoid(beta)``.

    Args:
        q, k, v, g: ``(batch, H, 128)`` BF16, possibly row-strided views
            (dense within a row, 8-byte aligned row stride).
        beta: ``(batch, H)`` BF16, pre-sigmoid.
        A_log: ``(H,)`` FP32.
        dt_bias: ``(H * 128,)`` FP32.
        out: ``(batch, H, 128)`` contiguous BF16 output.
        state: ``(slots, H, V, K)`` FP32 state.
        rings: the window rings, indexed by state slot.
        slots: ``(batch,)`` int32 state slots; ``null_block_id`` (0) is
            padding.
        meta: ``(batch,)`` int32 persistent request indices.
        pos: ``(batch,)`` int32 window positions in ``[0, W - 1]``.
        flush_rows: ``(batch,)`` int32 rows at position ``W - 1``, then -1
            padding.
        sketch: the layer's per-request sketch.
        scratch: at least ``scratch_numel(batch, num_sketch_heads)`` FP32
            floats.
        has_flush_rows: False skips the flush launches.
    """
    batch = q.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    h = t.num_heads
    if slots.dim() == 2:
        slots = slots[:, 0]
    _check_rows(slots, meta, null_block_id)
    assert pos.is_contiguous() and pos.dtype == torch.int32
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    _check_window(rings, sketch)
    a_log = A_log.float().contiguous()
    bias = dt_bias.float().contiguous()
    _step_launch(
        _step_ext(h, t.window), t.heads_step, q, k, v, g, beta, a_log, bias, slots,
        meta, pos, state, state.stride(0), state.stride(1), sketch.u, sketch.phi,
        t.ranks, rings.k, rings.v, rings.beta, rings.prefix, sketch.f, rings.u_ring,
        rings.d_ring, out, h, t.rank_cap, scale, sketch.dense, t.dense_rows,
    )  # fmt: skip
    if has_flush_rows:
        _flush_launches(
            state, rings, flush_rows, slots, meta, sketch, scratch, q, out, scale,
            True,
        )  # fmt: skip


def aot_specs(num_heads: int, window: int, target: Target) -> list[tuple[kn.Spec, bool]]:
    """Specializations of a layer to precompile for ``target``, each with
    whether it must build (knobs borrowed from the window-16 file may not
    build at a longer window; the defaults are then added)."""
    out = []
    borrowed = window != WINDOW and _config_file(num_heads, window, target) is None
    for kind, config in enumerate(tuned_config(num_heads, window, target)):
        default = default_config(num_heads, window, target)[kind]
        out.append((_SPEC[kind](config, window), not borrowed or config == default))
        if borrowed and config != default:
            out.append((_SPEC[kind](default, window), True))
    return out
