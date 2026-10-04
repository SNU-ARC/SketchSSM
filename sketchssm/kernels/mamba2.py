# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""SketchSSM Mamba-2 decode with the CUDA kernels of the SketchSSM paper.

The non-flush read and the fused flush + sketch rebuild live in
``csrc/mamba2/``. Each is specialized per layer shape (head count, group
size, head dim, state size and a window that is a multiple of 16) and build
knobs; the non-flush kernel also bakes in its 27 activation and ring strides.
Specializations come precompiled when available, else from NVRTC.
"""

import functools
import logging
import math
from collections.abc import Callable

import torch

from . import _driver as drv
from . import _kernel as kn
from . import _runtime as rt
from ._runtime import Support, Target
from .types import Mamba2Sketch

FAMILY = "mamba2"
STATE_SIZES = (64, 128, 256)
# The kernels replay the window in 16-step tiles.
WINDOW_TILE = 16

# Default build knobs on sm_100 and newer. Packed FP32 FMA (FFMA2) needs
# sm_100; both kernels have scalar fallbacks.
_NF_CONFIG = dict(
    NF_MINB=16, NF_UROWS=2, NF_UBATCH=4, NF_ABLATE=0, NF_HEADS=4, NF_SMAPS=2,
    UW_BF16=1, NF_LEAN=1, NF_FFMA2=1, NF_FASTSP=0, NF_CONST_STRIDES=1,
    NF_MMA=0, NF_LANES8=0, NF_ROWS=1, NF_PF_ROWS=8, NF_PF_FULL=2, NF_PF_BULK=0,
)  # fmt: skip
_FLUSH_CONFIG = dict(
    WARPS=1, MINB=9, QREG=0, PREFETCH=1, UNROLL=1, UW_BF16=1, FL_STAGES=2,
    FL_HALF=1, FL_PF_ROWS=0, FL_GRID_HF=0, FL_PF_BLOCKS=2, FL_EARLY=0,
    FL_CSMEM=1, FL_XSMEM=0, FL_PF_HEADS=0, FL_Q0SMEM=0, FL_OUTSMEM=0,
    FL_STATS_FFMA2=0, FL_ROW_LIST=1,
)  # fmt: skip
# Overrides of the defaults before sm_100. The tuned files in
# configs/mamba2/ hold their differences from these defaults.
_SM90_FLUSH_OVERRIDES = dict(MINB=4)


def _target(target: Target | None) -> Target:
    return target if target is not None else rt.current_target()


def default_configs(target: Target | None = None) -> tuple[dict, dict]:
    """Default non-flush and flush build knobs of an architecture (default:
    this GPU's)."""
    if _target(target).major >= 10:
        return dict(_NF_CONFIG), dict(_FLUSH_CONFIG)
    return dict(_NF_CONFIG), dict(_FLUSH_CONFIG, **_SM90_FLUSH_OVERRIDES)


def _shape_flush_defaults(flush: dict, heads_per_group: int, state_size: int) -> dict:
    """The flush defaults of a layer shape: CTAs of up to eight heads of one
    group, which rotate their window keys once, with the frame fragments in
    shared memory for state sizes up to 128."""
    warps = math.gcd(8, heads_per_group)
    if warps > 1:
        flush.update(WARPS=warps, MINB=max(1, 8 // warps),
                     FL_FRAG_SMEM=int(state_size <= 128 and warps >= 4))
    return flush


def config_file_name(
    head_dim: int,
    heads_per_group: int,
    state_size: int = 128,
    window: int = 16,
    target: Target | None = None,
) -> str:
    """Tuned-knob file of a layer shape on this GPU (see ``tuned_config``)."""
    state = "" if state_size == 128 else f"state_size={state_size},"
    win = "" if window == 16 else f"window={window},"
    return (
        f"head_dim={head_dim},heads_per_group={heads_per_group},{state}{win}"
        f"device_name={_target(target).device_name}.json"
    )


def _config_file(head_dim, heads_per_group, state_size, window, target=None):
    name = config_file_name(head_dim, heads_per_group, state_size, window, target)
    return rt.find_config(FAMILY, name)


@functools.cache
def window16_fallback(
    head_dim: int,
    heads_per_group: int,
    state_size: int,
    window: int,
    target: Target | None = None,
) -> bool:
    """Whether ``tuned_config`` of a window other than 16 takes the knobs of
    the window-16 file (there is no file for the window itself)."""
    return (
        window != 16
        and _config_file(head_dim, heads_per_group, state_size, window, target) is None
        and _config_file(head_dim, heads_per_group, state_size, 16, target) is not None
    )


def _tuned_path(head_dim, heads_per_group, state_size, window, target=None):
    path = _config_file(head_dim, heads_per_group, state_size, window, target)
    if path is None and window16_fallback(
        head_dim, heads_per_group, state_size, window, target
    ):
        path = _config_file(head_dim, heads_per_group, state_size, 16, target)
    return path


@functools.cache
def large_batch_config(
    head_dim: int,
    heads_per_group: int,
    state_size: int = 128,
    window: int = 16,
    target: Target | None = None,
) -> tuple[int, dict] | None:
    """``(min_rows, flush knobs)`` of the flush of steps with at least
    ``min_rows`` flush rows, from the ``"flush_large"`` overrides (applied on
    top of ``"flush"``) and ``"flush_large_min_batch"`` of the tuned file, if
    any. Wide CTAs that rotate the window keys once for many heads pay off
    when many rows flush; with few, their heads run one after the other."""
    path = _tuned_path(head_dim, heads_per_group, state_size, window, target)
    raw = rt.read_config(path) if path is not None else {}
    if "flush_large" not in raw:
        return None
    flush = tuned_config(head_dim, heads_per_group, state_size, window, target)[1]
    return int(raw["flush_large_min_batch"]), dict(flush, **raw["flush_large"])


@functools.cache
def tuned_config(
    head_dim: int,
    heads_per_group: int,
    state_size: int = 128,
    window: int = 16,
    target: Target | None = None,
) -> tuple[dict, dict]:
    """Non-flush and flush build knobs for a layer shape on this GPU (or
    ``target``).

    The defaults are ``default_configs()`` of the architecture, with the flush
    CTA width adapted to the shape (``_shape_flush_defaults``). A JSON file
    ``{"nf": {...}, "flush": {...}}`` named by ``config_file_name`` in a
    config folder (``_runtime.config_folders``) overrides them. The lookup
    takes the file of the layer shape and window, then that of the shape at
    window 16 (``window16_fallback``; if its knobs do not build or fit at this
    window, the build falls back to the defaults), then the defaults.
    """
    nf, flush = default_configs(target)
    flush = _shape_flush_defaults(flush, heads_per_group, state_size)
    path = _tuned_path(head_dim, heads_per_group, state_size, window, target)
    if path is not None:
        raw = rt.read_config(path)
        rt.log_once(
            logging.INFO, "Using SketchSSM CUDA config %s for window %d", path, window
        )
        return dict(nf, **raw.get("nf", {})), dict(flush, **raw.get("flush", {}))
    rt.log_once(
        logging.INFO,
        "No tuned SketchSSM CUDA config %s; using the default build knobs.",
        config_file_name(head_dim, heads_per_group, state_size, window, target),
    )
    return nf, flush


# ── specializations ──

# Host-side constants of each kernel, read from its cubin.
_NF_PROBES = (
    ("strides_bytes", "sizeof(NfStrides)"),
    ("const_strides", "NF_CONST_STRIDES"),
    ("uw_bf16", "UW_BF16"),
    ("heads", "NF_HEADS"),
    ("lanes8", "NF_LANES8"),
    ("rows", "NF_ROWS"),
)
_FLUSH_PROBES = (
    ("strides_bytes", "sizeof(FlushStrides)"),
    ("dyn_smem", "FL_DYN_SMEM"),
    ("warps", "WARPS"),
    ("hpw", "FL_HPW"),
    ("uw_bf16", "UW_BF16"),
    ("stages", "FL_STAGES"),
    ("csmem", "FL_CSMEM"),
    ("dense", "FL_DENSE"),
    ("dense_t", "FL_DENSE_T"),
    ("row_list", "FL_ROW_LIST"),
    ("grid_hf", "FL_GRID_HF"),
)


def _nf_name(config: dict) -> str:
    if config.get("NF_LANES8"):
        return "nf_kernel8"
    return "nf_kernel_rows" if config.get("NF_ROWS", 1) > 1 else "nf_kernel"


def make_spec(name: str, config: dict) -> kn.Spec:
    """The specialization of kernel source ``name`` (``mamba2_sketch_nf`` or
    ``mamba2_sketch_flush``) with ``config`` as its defines."""
    if name == "mamba2_sketch_nf":
        return kn.Spec.make(
            "mamba2_nf", f"{FAMILY}/{name}.cuh", config, (), probes=_NF_PROBES
        )
    return kn.Spec.make(
        "mamba2_flush", f"{FAMILY}/{name}.cuh", config, (), probes=_FLUSH_PROBES
    )


class _Built:
    """A loaded Mamba-2 kernel (non-flush or flush)."""

    def __init__(self, name: str, config: dict, kernel: kn.Kernel):
        self.name = name
        self.config = config
        self.kernel = kernel
        self.source = kernel.source
        self.probes = kernel.probes
        self.fn_name = _nf_name(config) if name == "mamba2_sketch_nf" else "flush_kernel"
        self.sig = _NF_SIG if name == "mamba2_sketch_nf" else _FLUSH_SIG
        self._configured: set[int] = set()

    def function(self) -> int:
        fn = self.kernel.function(self.fn_name, self.sig)
        if fn not in self._configured:
            self._configure(fn)
            self._configured.add(fn)
        return fn

    def _dyn_smem(self) -> int:
        return self.probes.get("dyn_smem", 0)

    def _configure(self, fn: int) -> None:
        """The flush kernel opts in to its dynamic shared memory (windows
        longer than 16), or fails naming the limit it exceeds."""
        dyn = self._dyn_smem()
        if not dyn:
            return
        dev = torch.cuda.current_device()
        optin = drv.device_attribute(dev, drv.DEV_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        need = dyn + drv.func_attribute(fn, drv.ATTR_SHARED_SIZE_BYTES)
        c = self.config
        if need > optin:
            raise RuntimeError(
                f"SketchSSM Mamba-2 flush kernel (window {c['SK_W']}, head dim "
                f"{c['SK_P']}, state size {c['SK_N']}, WARPS={c.get('WARPS')}, "
                f"FL_STAGES={c.get('FL_STAGES')}) needs {need} bytes of shared memory "
                f"per block, more than this GPU's limit of {optin} bytes "
                "(cudaDevAttrMaxSharedMemoryPerBlockOptin); lower WARPS or FL_STAGES"
            )
        drv.set_max_dynamic_smem(fn, dyn)

    def resources(self) -> list[int]:
        """Registers, local memory, shared memory (static + dynamic) and the
        maximum threads per block of the kernel."""
        fn = self.function()
        return [
            drv.func_attribute(fn, drv.ATTR_NUM_REGS),
            drv.func_attribute(fn, drv.ATTR_LOCAL_SIZE_BYTES),
            drv.func_attribute(fn, drv.ATTR_SHARED_SIZE_BYTES) + self._dyn_smem(),
            drv.func_attribute(fn, drv.ATTR_MAX_THREADS_PER_BLOCK),
        ]


def mamba2_supported(
    num_heads: int,
    head_dim: int,
    state_size: int,
    n_groups: int,
    window: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
) -> Support:
    """Whether a Mamba-2 layer can use the SketchSSM CUDA kernels, and where
    they come from (``Support.source``)."""
    shape_ok = (
        head_dim % 16 == 0
        and head_dim <= 128
        and state_size in STATE_SIZES
        and num_heads % n_groups == 0
        and window % WINDOW_TILE == 0
        and activation_dtype == torch.bfloat16
        and state_dtype == torch.float32
    )
    if not shape_ok:
        return Support(False, "unsupported layer shape, window or dtype")
    support = rt.cuda_available(80)
    if not support:
        return support
    # A non-flush CTA's heads must share one B/C group.
    hpg = num_heads // n_groups
    nf, flush = tuned_config(head_dim, hpg, state_size, window)
    if hpg % nf["NF_HEADS"]:
        return Support(False, "heads per group not a multiple of NF_HEADS")
    shape = _shape(num_heads, hpg, head_dim, state_size, window)
    specs = [make_spec("mamba2_sketch_flush", dict(flush, **shape))]
    large = large_batch_config(head_dim, hpg, state_size, window)
    if large is not None:
        specs.append(make_spec("mamba2_sketch_flush", dict(large[1], **shape)))
    support = rt.kernel_support(specs)
    if not support:
        return support
    # The non-flush kernel bakes in the cache strides, known only at run time:
    # NVRTC, unless one was precompiled for this shape (build.py --specs).
    if kn.nvrtc_disabled() or kn._nvrtc.unavailable() is not None:
        nf_defines = dict(nf, **shape, NF_FFMA2=rt.ffma2())
        if kn.aot_any("mamba2_nf", nf_defines, rt.capability()):
            return Support(True, support.reason + ", mamba2_nf: aot (for the precompiled "
                           "cache strides only)", support.source)  # fmt: skip
        why = "disabled" if kn.nvrtc_disabled() else kn._nvrtc.unavailable()
        return Support(
            False,
            "the Mamba-2 non-flush kernel is specialized on the cache strides at "
            f"run time and needs NVRTC ({why})",
        )
    return Support(True, support.reason + ", mamba2_nf: nvrtc (cache strides)", "nvrtc")


def _shape(heads: int, hpg: int, head_dim: int, state_size: int, window: int) -> dict:
    return dict(SK_NHEADS=heads, SK_HPG=hpg, SK_P=head_dim, SK_N=state_size, SK_W=window)


def _shape_defines(state: torch.Tensor, n_groups: int, window: int) -> dict:
    _, heads, head_dim, state_size = state.shape
    return _shape(heads, heads // n_groups, head_dim, state_size, window)


def _shape_key(d: dict) -> tuple[int, int, int, int]:
    return d["SK_P"], d["SK_HPG"], d["SK_N"], d["SK_W"]


def _build(name: str, config: dict, fns: list[str]) -> _Built:
    """Load (precompiled or NVRTC) kernel source ``name`` with ``config``."""
    del fns  # kept for the tuners' and tests' hooks
    return _Built(name, config, kn.load(make_spec(name, config)))


def _build_fitting(name: str, kind: int, d: dict, extra: dict, threads, fns,
                   knobs: dict | None = None):
    """Build ``name`` with the tuned knobs of its shape and check that it fits
    this GPU (the flush kernel raises when its shared memory exceeds the
    limit; ``threads(knobs)`` threads must fit its registers). Knobs taken
    from the window-16 file, or the shape defaults when there is no file,
    that do not build or fit give way to the architecture defaults."""

    def build(knobs: dict):
        ext = _build(name, dict(knobs, **extra, **d), fns)
        regs, _, _, max_threads = ext.resources()
        if max_threads < threads(knobs):
            raise RuntimeError(
                f"{name}: {threads(knobs)} threads of {regs} registers per block "
                f"exceed this GPU's register file"
            )
        return ext

    knobs = knobs if knobs is not None else tuned_config(*_shape_key(d))[kind]
    if not window16_fallback(*_shape_key(d)) and _tuned_path(*_shape_key(d)) is not None:
        return build(knobs)
    try:
        return build(knobs)
    except Exception as e:
        rt.logger.warning(
            "SketchSSM CUDA %s: the knobs %s do not build or fit at window %d "
            "(%s); using the architecture defaults.",
            name, knobs, d["SK_W"], str(e).splitlines()[-1] if str(e) else e,
        )  # fmt: skip
        return build(default_configs()[kind])


@functools.cache
def _flush_ext(shape: tuple, large: bool = False) -> _Built:
    knobs = large_batch_config(*_shape_key(dict(shape)))[1] if large else None
    return _build_fitting(
        "mamba2_sketch_flush", 1, dict(shape), {},
        lambda knobs: 32 * knobs["WARPS"], ["flush", "resources"], knobs,
    )  # fmt: skip


@functools.cache
def _nf_ext(shape: tuple, stride_init: str) -> _Built:
    # The non-flush kernel is specialized on its 27 activation/ring strides.
    extra = dict(NF_FFMA2=rt.ffma2(), NF_STRIDE_INIT=stride_init)
    return _build_fitting(
        "mamba2_sketch_nf", 0, dict(shape), extra,
        lambda knobs: (8 if knobs["NF_LANES8"] else 16) * knobs["NF_HEADS"],
        ["m2_step", "resources"],
    )  # fmt: skip


def _nf_strides(x, dt, dt_bias, A, B, C, D, out, x_cache, dt_cache, B_cache,
                bc_pre, query) -> tuple[int, ...]:  # fmt: skip
    return (
        x.stride(0), x.stride(1), dt.stride(0), dt.stride(1), dt_bias.stride(0),
        A.stride(0), D.stride(0), B.stride(0), B.stride(1), C.stride(0),
        C.stride(1), out.stride(0), out.stride(1), x_cache.stride(0),
        x_cache.stride(1), x_cache.stride(2), dt_cache.stride(0),
        dt_cache.stride(1), dt_cache.stride(2), B_cache.stride(0),
        B_cache.stride(1), B_cache.stride(2), bc_pre.stride(0),
        bc_pre.stride(1), bc_pre.stride(2), query.stride(0), query.stride(1),
    )  # fmt: skip


def _stride_init(strides: tuple[int, ...]) -> str:
    return "{" + ", ".join(f"{int(v)}L" for v in strides) + ", 0L, 0L, 0L}"


_NF_SIG = drv.Signature(["Q"] * 15 + ["i"] + ["Q"] * 10 + ["30q", "i", "i"])
_FLUSH_SIG = drv.Signature(["Q"] * 23 + ["34q"] + ["i"] * 6 + ["Q", "i", "Q", "i", "i"])
_BF16 = torch.bfloat16
_F32 = torch.float32


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"SketchSSM Mamba-2: {msg}")


def _al16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _nf_launch(k: _Built, state, x, dt, dt_bias, A, B, C, D, out, x_cache, dt_cache,
               B_cache, bc_pre, write_pos, is_flush, slots, null_block, nfh, mh, off,
               mp, w, ag, query, u, u_off, w_off, strides) -> None:  # fmt: skip
    """The non-flush launch (``m2_step`` of the former C++ launcher)."""
    c = k.config
    batch, nheads, dim, dstate = x.shape[0], state.shape[1], state.shape[2], state.shape[3]
    W = c["SK_W"]
    _check(nheads == c["SK_NHEADS"] and dim == c["SK_P"] and dstate == c["SK_N"], "layer shape")
    _check(x_cache.shape[2] == W and dt_cache.shape[2] == W and B_cache.shape[2] == W
           and bc_pre.shape[2] >= W,
           f"window rings must hold the {W} steps the kernel was built for")  # fmt: skip
    _check(x.dtype == _BF16 and dt.dtype == _BF16 and B.dtype == _BF16 and C.dtype == _F32,
           "bf16 x/dt/B, fp32 query C")  # fmt: skip
    _check(dt_bias.dtype == _BF16 and D.dtype == _BF16 and A.dtype == _F32, "bf16 bias/D, fp32 A")
    _check(x_cache.dtype == _BF16 and B_cache.dtype == _BF16 and dt_cache.dtype == _F32
           and bc_pre.dtype == _F32, "ring dtypes")  # fmt: skip
    _check(out.dtype in (_BF16, _F32), "out dtype")
    _check(query.dtype == _F32 and query.stride(2) == 1
           and query.shape[1] == c["SK_NHEADS"] // c["SK_HPG"]
           and query.shape[2] == c["SK_N"], "query")  # fmt: skip
    _check(x.stride(2) == 1 and out.stride(2) == 1 and x_cache.stride(3) == 1
           and B_cache.stride(3) == 1 and B.stride(2) == 1 and C.stride(2) == 1,
           "contiguous value/key axes")  # fmt: skip
    _check(write_pos.dtype == torch.int32 and slots.dtype == torch.int32
           and is_flush.dtype in (torch.int8, torch.bool), "control dtypes")  # fmt: skip
    _check(ag.is_contiguous() and ag.dim() == 3 and ag.shape[1] == 5, "AG layout")
    _check(w.is_contiguous() and w.dim() == 3 and w.shape[2] == c["SK_N"], "packed maps")
    uw = _BF16 if k.probes["uw_bf16"] else _F32
    _check(w.dtype == uw and u.dtype == uw, "U/W storage dtype must match UW_BF16")
    _check(u.is_contiguous() and u.dim() == 3 and u.shape[2] == c["SK_P"], "packed U")
    _check(all(t.dtype == torch.int32 for t in (u_off, w_off, mh, off, mp, nfh)), "int32 tables")
    _check(dt.stride(2) == 0 or dt.shape[2] == 1, "dt per head")
    _check(x_cache.stride(2) == c["SK_P"] and dt_cache.stride(2) == 1 and bc_pre.stride(2) == 1
           and B_cache.stride(2) == c["SK_N"], "fixed inner ring strides")  # fmt: skip
    _check(all(_al16(t) for t in (x, out, x_cache, B_cache, B, C, query, w, u)),
           "16-byte alignment")  # fmt: skip
    _check(all(s % 8 == 0 for s in (x.stride(0), x.stride(1), x_cache.stride(0),
               x_cache.stride(1), x_cache.stride(2), B_cache.stride(0), B_cache.stride(1),
               B_cache.stride(2), B.stride(0), B.stride(1)))
           and all(s % 4 == 0 for s in (C.stride(0), C.stride(1), query.stride(0),
                                        query.stride(1)))
           and out.stride(0) % 4 == 0 and out.stride(1) % 4 == 0, "vector strides")  # fmt: skip
    st = (*strides, w.shape[1], u.shape[1], ag.shape[2])
    heads = k.probes["heads"]
    if k.probes["lanes8"]:
        grid, block = (c["SK_NHEADS"] // heads, batch, 1), (8 * heads, 1, 1)
    elif k.probes["rows"] > 1:
        rows = k.probes["rows"]
        grid, block = (c["SK_NHEADS"] // heads, (batch + rows - 1) // rows, 1), (16 * heads, 1, 1)
    else:
        grid, block = (c["SK_NHEADS"] // heads, batch, 1), (16 * heads, 1, 1)
    args = (
        x.data_ptr(), dt.data_ptr(), dt_bias.data_ptr(), A.data_ptr(), B.data_ptr(),
        C.data_ptr(), D.data_ptr(), out.data_ptr(), x_cache.data_ptr(),
        dt_cache.data_ptr(), B_cache.data_ptr(), bc_pre.data_ptr(), write_pos.data_ptr(),
        is_flush.data_ptr(), slots.data_ptr(), null_block, nfh.data_ptr(), mh.data_ptr(),
        off.data_ptr(), mp.data_ptr(), w.data_ptr(), ag.data_ptr(), query.data_ptr(),
        u.data_ptr(), u_off.data_ptr(), w_off.data_ptr(), *st, int(out.dtype == _F32), batch,
    )  # fmt: skip
    drv.launch(k.function(), _NF_SIG, grid, block, 0, drv.current_stream(x.device.index), args)


def _flush_launch(k: _Built, S, X, DT, Bias, A, B, C, D, O, XR, DR, BR, WP, FL, Slots,
                  Width, Map, Sketch, SkOff, Tail, AG, Offsets, WOff, strides, null_slot,
                  skrows, sm, wrows, Rows, grid_rows, frames,
                  row_range=(0, 1 << 30)) -> None:  # fmt: skip
    """The flush launch (``flush`` of the former C++ launcher)."""
    c, p = k.config, k.probes
    W = c["SK_W"]
    _check(len(strides) == 34, "stride vector")
    _check(B.dtype == _BF16 and C.dtype == _F32, "bf16 B, fp32 query C")
    n, groups = c["SK_N"], c["SK_NHEADS"] // c["SK_HPG"]
    _check(frames.dtype == _F32 and frames.is_contiguous()
           and tuple(frames.shape) == (groups, n, n), "fp32 frames_t (groups, N, N)")  # fmt: skip
    _check(C.stride(-1) == 1 and C.stride(0) % 4 == 0 and C.stride(1) % 4 == 0 and _al16(C),
           "contiguous 16-byte aligned query rows")  # fmt: skip
    _check(S.shape[1] == c["SK_NHEADS"] and S.shape[2] == c["SK_P"] and S.shape[3] == c["SK_N"],
           "layer shape")  # fmt: skip
    _check(XR.shape[2] == W and DR.shape[2] == W and BR.shape[2] == W,
           f"window rings must hold the {W} steps the kernel was built for")  # fmt: skip
    if p["dense_t"]:
        _check(strides[3] == 1 and strides[2] % 2 == 0 and strides[2] >= 128,
               "dense flush needs keys contiguous ([value][key] state)")  # fmt: skip
    elif p["stages"]:
        _check(strides[3] == c["SK_P"] and strides[2] == 1,
               "staged flush needs key-major state blocks")  # fmt: skip
    if not p["dense"]:
        uw = _BF16 if p["uw_bf16"] else _F32
        _check(Sketch.dtype == uw and Tail.dtype == uw, "U/W storage dtype must match UW_BF16")
    rows = X.shape[0]
    warps = p["warps"]
    heads = c["SK_NHEADS"] // (warps * p["hpw"])
    _check(p["row_list"] or tuple(row_range) == (0, 1 << 30), "a flush row range needs FL_ROW_LIST")
    if p["row_list"]:
        _check(Rows.numel() >= rows and Rows.dtype == torch.int32, "flush row list")
        # A CTA serves hpw heads per warp: proportionally fewer rows each.
        grid = (max(1, min(grid_rows * p["hpw"], rows)), heads, 1)
    elif p["grid_hf"]:
        grid = (heads, rows, 1)
    else:
        grid = (rows, heads, 1)
    fn = k.function()
    args = (
        S.data_ptr(), X.data_ptr(), DT.data_ptr(), Bias.data_ptr(), A.data_ptr(),
        B.data_ptr(), C.data_ptr(), D.data_ptr(), O.data_ptr(), XR.data_ptr(),
        DR.data_ptr(), BR.data_ptr(), WP.data_ptr(), FL.data_ptr(), Slots.data_ptr(),
        Width.data_ptr(), Map.data_ptr(), Sketch.data_ptr(), SkOff.data_ptr(),
        Tail.data_ptr(), AG.data_ptr(), Offsets.data_ptr(), WOff.data_ptr(), *strides,
        null_slot, 1, int(O.dtype == _F32), skrows, sm, wrows, Rows.data_ptr(), rows,
        frames.data_ptr(), *row_range,
    )  # fmt: skip
    drv.launch(fn, _FLUSH_SIG, grid, (32 * warps, 1, 1), k._dyn_smem(),
               drv.current_stream(X.device.index), args)  # fmt: skip


_DUMMY: dict[torch.device, torch.Tensor] = {}


@functools.cache
def _aux_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


def _run_with_flush(flush: Callable[[], None], nonflush: Callable[[], None]):
    flush()
    nonflush()


def mamba2_decode(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    bc_pre: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    flush_rows: torch.Tensor,
    slots: torch.Tensor,
    meta: torch.Tensor,
    out: torch.Tensor,
    sketch: Mamba2Sketch,
    null_block_id: int = 0,
    has_flush_rows: bool = True,
    *,
    frames_t: torch.Tensor,
    flush_programs: int,
    run_with_flush: Callable[[Callable[[], None], Callable[[], None]], None]
    | None = None,
) -> None:
    """One SketchSSM decode step of a Mamba-2 layer, after the caller filled
    ``bc_pre`` (the B·C products of the ring, ``(batch, groups, W)`` FP32,
    from the unrotated B and C).

    The state is kept in the layer's rotated frame R (per group), given as
    ``frames_t`` = R^T ``(groups, N, N)`` FP32. ``B`` is the unrotated BF16
    key, appended to the ring as is; ``C`` is the FP32 query R·C ``(batch,
    groups, N)``. The flush rotates its window keys R·B_t itself, once per
    group and CTA, to FP32 accuracy.

    Non-flush rows read their sketch; flush rows replay the window into the
    full state and rebuild the sketch. ``dt``, ``A``, ``D`` and ``dt_bias``
    are per head, expanded over dim/dstate; ``state`` is the
    ``(slots, H, dim, dstate)`` view of the key-major state, ``meta`` holds
    each row's sketch index, and ``flush_rows`` lists the flush rows followed
    by -1 padding, walked by ``flush_programs`` CTAs. With
    ``has_flush_rows=False`` the flush launch is skipped.
    ``run_with_flush(flush, nonflush)`` runs the two launches (default: one
    after the other on the current stream).
    """
    batch = x.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    if slots.dim() == 2:
        slots = slots[:, 0]
    assert slots.is_contiguous() and slots.dtype == torch.int32
    assert meta.is_contiguous() and meta.dtype == torch.int32
    assert write_pos.dtype == torch.int32 and is_flush.is_contiguous()
    L = x_cache.shape[2]
    n_groups = B.shape[1]
    shape = tuple(sorted(_shape_defines(state, n_groups, L).items()))
    if is_flush.dtype == torch.bool:
        is_flush = is_flush.view(torch.int8)
    query = C
    nf_strides = _nf_strides(x, dt, dt_bias, A, B, C, D, out, x_cache, dt_cache,
                             B_cache, bc_pre, query)  # fmt: skip
    nf = _nf_ext(shape, _stride_init(nf_strides))
    device = x.device
    if device not in _DUMMY:
        _DUMMY[device] = torch.zeros(1, 8, 128, device=device)

    def nonflush() -> None:
        _nf_launch(
            nf, state, x, dt, dt_bias, A, B, C, D, out, x_cache, dt_cache, B_cache,
            bc_pre, write_pos, is_flush, slots, null_block_id, t.dense_rows, t.ranks,
            t.ag_offsets, meta, sketch.w, sketch.ag, query, sketch.u, t.u_offsets,
            t.w_offsets, nf_strides,
        )  # fmt: skip

    if not has_flush_rows:
        nonflush()
        return
    strides = [
        *state.stride(), *x.stride(), dt.stride(0), dt.stride(1),
        dt_bias.stride(0), A.stride(0), *B.stride(), *C.stride(), D.stride(0),
        D.stride(1), *out.stride(), *x_cache.stride(), *dt_cache.stride(),
        *B_cache.stride(), slots.stride(0),
    ]  # fmt: skip

    # With "flush_large" knobs, batches that can reach min_batch flush rows
    # launch both builds; each runs only for its range of flush rows.
    large = large_batch_config(*_shape_key(dict(shape)))
    split = large[0] if large is not None and batch >= large[0] else None
    # The small build then sees fewer than split rows: a grid for those.
    exts = [(_flush_ext(shape), (0, 1 << 30), flush_programs)]
    if split is not None:
        exts = [(exts[0][0], (0, split), min(flush_programs, (split + 3) // 4)),
                (_flush_ext(shape, True), (split, 1 << 30), flush_programs)]

    def launch(ext, row_range, programs) -> None:
        _flush_launch(
            ext, state, x, dt, dt_bias, A, B, C, D, out, x_cache, dt_cache,
            B_cache, write_pos, is_flush, slots, t.ranks, meta, sketch.u,
            t.u_offsets, sketch.w, sketch.ag, t.ag_offsets, t.w_offsets, strides,
            null_block_id, sketch.u.shape[1], sketch.ag.shape[2],
            sketch.w.shape[1], flush_rows, programs, frames_t, row_range,
        )  # fmt: skip

    def flush() -> None:
        if len(exts) == 1:
            launch(*exts[0])
            return
        # The build that does not run for this row count exits at once; on a
        # stream of its own its exits overlap the other build's work.
        main = torch.cuda.current_stream()
        aux = _aux_stream(main.device)
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            launch(*exts[0])
        launch(*exts[1])
        main.wait_stream(aux)

    (run_with_flush or _run_with_flush)(flush, nonflush)


def aot_specs(
    num_heads: int,
    head_dim: int,
    state_size: int,
    n_groups: int,
    window: int,
    target: Target,
) -> list[tuple[kn.Spec, bool]]:
    """Specializations of a layer to precompile for ``target``, each with
    whether it must build: the flush kernel with its resolved knobs (the
    architecture defaults too, optional, when those are borrowed from the
    window-16 file). The non-flush kernel bakes in the cache strides, so it
    is compiled at run time."""
    hpg = num_heads // n_groups
    shape = _shape(num_heads, hpg, head_dim, state_size, window)
    out = [(make_spec("mamba2_sketch_flush",
                      dict(tuned_config(head_dim, hpg, state_size, window, target)[1], **shape)),
            True)]  # fmt: skip
    if window16_fallback(head_dim, hpg, state_size, window, target):
        out.append((make_spec("mamba2_sketch_flush", dict(default_configs(target)[1], **shape)), True))
        out[0] = (out[0][0], False)
    large = large_batch_config(head_dim, hpg, state_size, window, target)
    if large is not None:
        out.append((make_spec("mamba2_sketch_flush", dict(large[1], **shape)), True))
    return out
