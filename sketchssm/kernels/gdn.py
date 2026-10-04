# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""SketchSSM Gated DeltaNet decode with CUDA kernels.

The step and the exact flush + sketch rebuild kernels come from the SketchSSM
paper; they live in ``csrc/gdn/`` and are specialized per layer shape, window
and build knobs (precompiled when available, else compiled with NVRTC).
"""

import functools
import logging

import torch

from . import _driver as drv
from . import _kernel as kn
from . import _runtime as rt
from ._runtime import Support, Target
from .types import GDNSketch

FAMILY = "gdn"
HEAD_DIM = 128
# Default window; the kernels take any multiple of WINDOW_ALIGN.
WINDOW = 16
WINDOW_ALIGN = 16

# Default build knobs on capability >= 10. Packed FP32 FMA (NF_FFMA2) needs
# sm_100; the step has a scalar fallback. The minimum CTAs per SM (NF_MINB,
# W1_MINB) are for 3 value heads per key head and scale down for more, to keep
# the per-thread register budget.
_STEP_CONFIG = dict(
    NF_MINB=9, NF_UROWS=6, NF_AG_FG=32, NF_FS_FG=32, NF_UBATCH=4, NF_FFMA2=1,
    NF_PF_ROWS=0, NF_ABLATE=0, NF_SMEM_PAD=0, SKETCH_BF16=1,
)  # fmt: skip
_FLUSH_CONFIG = dict(
    W1_MINB=4, W1_NOSKETCH=0, W1_ABLATE=0, W1_NOSK=0,
    W1_SMEM_PAD=0, SKETCH_BF16=1,
)  # fmt: skip
# Overrides below capability 10; the same heads-per-group scaling applies.
_SM90_STEP = dict(NF_MINB=6, NF_UROWS=8)
_SM90_FLUSH = dict(W1_MINB=1)
# Flush rows per CTA of the flush row list, and a cap on the CTAs per key head
# (launch knobs, not defines).
_FLUSH_ROWS_PER_PROGRAM = 1
_FLUSH_CTAS = 256

_DT_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
# Value heads per key head (one warp per value head of a key head in a CTA).
SUPPORTED_HEADS_PER_GROUP = (1, 2, 3, 4, 5, 6, 7, 8)
_PAPER_HEADS_PER_GROUP = 3
# Shared memory the driver reserves per CTA.
_SMEM_RESERVED_PER_CTA = 1024


def window_supported(window: int) -> bool:
    return window >= WINDOW_ALIGN and window % WINDOW_ALIGN == 0


def _scaled_minb(minb: int, heads_per_group: int) -> int:
    paper = _PAPER_HEADS_PER_GROUP
    return max(1, minb * paper // max(heads_per_group, paper))


def step_smem_bytes(config: dict, num_k_heads: int, num_v_heads: int,
                    window: int) -> int:  # fmt: skip
    """Shared memory per CTA of the step kernel (dynamic + static)."""
    hpg = num_v_heads // num_k_heads
    sk = 2 if config["SKETCH_BF16"] else 4
    v = k = HEAD_DIM
    staged_rows = 15
    warp = (
        staged_rows * v * 2
        + config["NF_UROWS"] * v * sk
        + (4 * k + 5 * config["NF_AG_FG"]) * sk
        + staged_rows * config["NF_FS_FG"] * sk
    )
    static = 2 * window * 4 + 2 * hpg * window * 8 + hpg * 128 * 4
    return hpg * warp + config["NF_SMEM_PAD"] + static


def flush_smem_bytes(config: dict, num_k_heads: int, num_v_heads: int,
                     window: int) -> int:  # fmt: skip
    """Shared memory per CTA of the flush kernel (``W1_SMEM`` + pad)."""
    hpg = num_v_heads // num_k_heads
    tiles = window // 16
    cta = 2 * window * 136 * 2 + 256 * 4 + 128 * 4
    t_bytes = 16 * 20 * 4 if window == 16 else tiles * (tiles + 1) // 2 * 1024
    warp = 16 * 128 * 4 + 5 * 128 * 4 + t_bytes + window * 64 + 2 * window * 4
    return cta + hpg * warp + config["W1_SMEM_PAD"]


def _target(target: Target | None) -> Target:
    return target if target is not None else rt.current_target()


def _smem_limits(target: Target | None = None) -> tuple[int, int]:
    """(per-CTA opt-in maximum, per-SM capacity) of the shared memory."""
    t = _target(target)
    return t.smem_per_block_optin, t.smem_per_sm


def _ctas_per_sm(smem: int, target: Target | None = None) -> int:
    return _smem_limits(target)[1] // (smem + _SMEM_RESERVED_PER_CTA)


def _fits(step: dict, flush: dict, num_k_heads: int, num_v_heads: int,
          window: int, occupancy: bool = True,
          target: Target | None = None) -> str | None:  # fmt: skip
    """None if the knobs fit this shape and window on this GPU, else why not
    (per-CTA shared memory, or with ``occupancy`` the minimum CTAs per SM)."""
    per_cta = _smem_limits(target)[0]
    for name, cfg, smem, minb in (
        ("step", step, step_smem_bytes, "NF_MINB"),
        ("flush", flush, flush_smem_bytes, "W1_MINB"),
    ):
        need = smem(cfg, num_k_heads, num_v_heads, window)
        if need > per_cta:
            return (
                f"the {name} kernel needs {need} bytes of shared memory per "
                f"CTA, above the GPU's limit of {per_cta} "
                "(sharedMemPerBlockOptin)"
            )
        if occupancy and cfg[minb] > _ctas_per_sm(need, target):
            return (
                f"{minb}={cfg[minb]} {name} CTAs of {need} bytes do not fit "
                "in an SM's shared memory"
            )
    return None


def config_file_name(num_k_heads: int, num_v_heads: int,
                     window: int = WINDOW, target: Target | None = None) -> str:  # fmt: skip
    """Tuned-config file name of a layer shape and window on this GPU."""
    w = "" if window == WINDOW else f"window={window},"
    return (
        f"num_k_heads={num_k_heads},num_v_heads={num_v_heads},{w}"
        f"device_name={_target(target).device_name}.json"
    )


def default_config(
    num_k_heads: int,
    num_v_heads: int,
    window: int = WINDOW,
    target: Target | None = None,
) -> tuple[dict, dict]:
    """Architecture default step and flush knobs.

    The minimum CTAs per SM scale down for more than 3 value heads per key
    head, and for a window above 16 are capped at what shared memory allows.
    """
    hpg = num_v_heads // num_k_heads
    step, flush = dict(_STEP_CONFIG), dict(_FLUSH_CONFIG)
    if _target(target).major < 10:
        step.update(_SM90_STEP)
        flush.update(_SM90_FLUSH)
    step["NF_MINB"] = _scaled_minb(step["NF_MINB"], hpg)
    flush["W1_MINB"] = _scaled_minb(flush["W1_MINB"], hpg)
    if window != WINDOW:
        for cfg, smem, minb in (
            (step, step_smem_bytes, "NF_MINB"),
            (flush, flush_smem_bytes, "W1_MINB"),
        ):
            room = _ctas_per_sm(smem(cfg, num_k_heads, num_v_heads, window), target)
            cfg[minb] = max(1, min(cfg[minb], room))
    flush["ROWS_PER_PROGRAM"] = _FLUSH_ROWS_PER_PROGRAM
    flush["FLUSH_CTAS"] = _FLUSH_CTAS
    return step, flush


@functools.cache
def tuned_config(num_k_heads: int, num_v_heads: int, window: int = WINDOW,
                 target: Target | None = None) -> tuple[dict, dict]:  # fmt: skip
    """Step and flush build knobs for a layer shape and window on this GPU
    (or ``target``).

    A JSON file ``{"step": {...}, "flush": {...}}`` named by
    ``config_file_name`` in a config folder overrides ``default_config``. The
    shape's window-16 file is used for other windows when its knobs fit.
    """
    defaults = default_config(num_k_heads, num_v_heads, window, target)
    names = [config_file_name(num_k_heads, num_v_heads, window, target)]
    if window != WINDOW:
        names.append(config_file_name(num_k_heads, num_v_heads, target=target))
    for name in names:
        path = rt.find_config(FAMILY, name)
        if path is None:
            continue
        raw = rt.read_config(path)
        step = dict(defaults[0], **raw.get("step", {}))
        flush = dict(defaults[1], **raw.get("flush", {}))
        exact = name == names[0]
        reason = None if exact else _fits(step, flush, num_k_heads, num_v_heads,
                                          window, target=target)  # fmt: skip
        if reason is None:
            rt.log_once(
                logging.INFO,
                "Using SketchSSM GDN CUDA config %s for window %d",
                path,
                window,
            )
            return step, flush
        rt.log_once(
            logging.INFO,
            "SketchSSM GDN CUDA config %s does not fit window %d (%s); using "
            "the architecture default build knobs.",
            path,
            window,
            reason,
        )
        return defaults
    rt.log_once(
        logging.INFO,
        "No tuned SketchSSM GDN CUDA config %s; using the architecture default "
        "build knobs.",
        names[0],
    )
    return defaults


def check_resources(num_k_heads: int, num_v_heads: int, window: int) -> None:
    """Raise if no build knobs fit this layer shape and window on this GPU."""
    reason = _fits(*tuned_config(num_k_heads, num_v_heads, window), num_k_heads,
                   num_v_heads, window, occupancy=False)  # fmt: skip
    if reason is not None:
        raise ValueError(
            f"SketchSSM GDN CUDA kernels at window {window} with "
            f"{num_v_heads // num_k_heads} value heads per key head: {reason}"
        )


def gdn_supported(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    window: int,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
) -> Support:
    """Whether a GDN layer can use the SketchSSM CUDA kernels, and where they
    come from (``Support.source``; the step kernel is looked up for gates
    ``a``/``b`` and ``dt_bias`` in the activation dtype and an FP32 ``A_log``)."""
    shape_ok = (
        head_k_dim == HEAD_DIM
        and head_v_dim == HEAD_DIM
        and num_v_heads % num_k_heads == 0
        and num_v_heads // num_k_heads in SUPPORTED_HEADS_PER_GROUP
        and window_supported(window)
        and activation_dtype == torch.bfloat16
        and state_dtype == torch.float32
    )
    if not shape_ok:
        return Support(False, "unsupported layer shape, window or dtype")
    support = rt.cuda_available(80)
    if not support:
        return support
    code = _DT_CODE[activation_dtype]
    specs = layer_specs(num_k_heads, num_v_heads, window, code, 0, code, activation_dtype)
    support = rt.kernel_support(specs)
    if not support and rt.kernel_support(specs[1:]):
        # Gates of other dtypes may have been precompiled.
        step = {k: v for k, v in specs[0].defines if not k.endswith("_CODE")}
        if kn.aot_any("gdn_step", step, rt.capability()):
            return Support(True, "gdn_step: aot (gate dtypes as precompiled), "
                           + rt.kernel_support(specs[1:]).reason, "aot")  # fmt: skip
    return support


# ── specializations ──

_GTS = (8, 16, 32, 48, 64, 80, 128)
_TIO = {torch.float32: "float", torch.bfloat16: "__nv_bfloat16"}
_STEP_PROBES = (("smem", "Smem<128, 128, NF_HV / NF_H>::BYTES"),)
_FLUSH_PROBES = (
    ("smem", "W1_SMEM + W1_SMEM_PAD"),
    ("threads", "W1_THREADS"),
    ("warps", "W1_WARPS"),
)


def _step_name(gt: int, hpg: int, tio: str) -> str:
    return f"gdn_step_kernel<128, 128, {gt}, {hpg}, {tio}>"


def step_spec(config: dict, io_dtype: torch.dtype) -> kn.Spec:
    """The step kernel of ``config`` (with its ``NF_*`` shape defines) for
    inputs and outputs in ``io_dtype``: one kernel per rank bucket."""
    hpg = config["NF_HV"] // config["NF_H"]
    names = [_step_name(gt, hpg, _TIO[io_dtype]) for gt in _GTS]
    return kn.Spec.make("gdn_step", "gdn/gdn_sketch_step.cuh", config, names,
                        ("--use_fast_math",), _STEP_PROBES)  # fmt: skip


def flush_spec(config: dict) -> kn.Spec:
    return kn.Spec.make("gdn_flush", "gdn/gdn_sketch_flush.cuh", config,
                        ["gdn_flush_warp_kernel", "gdn_finish_kernel"], (),
                        _FLUSH_PROBES)  # fmt: skip


def _knobs(h: int, hv: int, window: int, target: Target | None) -> tuple[dict, dict]:
    # Called without ``target`` at run time, so that the tuners can replace
    # ``tuned_config``.
    if target is None:
        return tuned_config(h, hv, window)
    return tuned_config(h, hv, window, target)


def _step_config(h, hv, ab_code, p_code, bias_code, window, target=None) -> dict:
    config = dict(
        _knobs(h, hv, window, target)[0], NF_H=h, NF_HV=hv, NF_AB_CODE=ab_code,
        NF_P_CODE=p_code, NF_BIAS_CODE=bias_code, WMAX=window,
    )  # fmt: skip
    config["NF_FFMA2"] = config["NF_FFMA2"] and _target(target).ffma2
    return config


def _flush_config(h, hv, window, target=None) -> dict:
    config = dict(_knobs(h, hv, window, target)[1], W1_WARPS=hv // h, WMAX=window)
    config.pop("ROWS_PER_PROGRAM")
    config.pop("FLUSH_CTAS", None)
    return config


def layer_specs(h, hv, window, ab_code, p_code, bias_code, io_dtype,
                target: Target | None = None) -> list[kn.Spec]:  # fmt: skip
    return [
        step_spec(_step_config(h, hv, ab_code, p_code, bias_code, window, target), io_dtype),
        flush_spec(_flush_config(h, hv, window, target)),
    ]


class _Kernel:
    def __init__(self, spec: kn.Spec):
        self.spec = spec
        self.kernel = kn.load(spec)
        self.source = self.kernel.source
        self.probes = self.kernel.probes
        self._fns: dict[str, int] = {}

    def function(self, name: str, sig=None) -> int:
        fn = self.kernel.function(name, sig)
        if self._fns.get(name) != fn:
            self._configure(fn)
            self._fns[name] = fn
        return fn

    def _configure(self, fn: int) -> None:
        smem = self.probes["smem"]
        dev = torch.cuda.current_device()
        optin = drv.device_attribute(dev, drv.DEV_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        if smem > optin:
            c = dict(self.spec.defines)
            raise RuntimeError(
                f"SketchSSM GDN {self.spec.kind}: window {c.get('WMAX')} needs {smem} "
                f"bytes of shared memory per CTA, above this GPU's limit of {optin} "
                "(cudaDevAttrMaxSharedMemoryPerBlockOptin)"
            )
        drv.set_max_dynamic_smem(fn, smem)


@functools.cache
def _step_ext(h: int, hv: int, ab_code: int, p_code: int, bias_code: int,
              window: int = WINDOW, io_dtype: torch.dtype = torch.bfloat16):  # fmt: skip
    return _Kernel(step_spec(_step_config(h, hv, ab_code, p_code, bias_code, window), io_dtype))


@functools.cache
def _flush_ext(h: int, hv: int, window: int = WINDOW):
    return _Kernel(flush_spec(_flush_config(h, hv, window)))


def flush_programs(batch: int, num_k_heads: int, num_v_heads: int,
                   window: int = WINDOW) -> int:  # fmt: skip
    """CTAs (per key head) that walk a flush row list of ``batch`` entries: one
    per ``ROWS_PER_PROGRAM`` rows, at most ``FLUSH_CTAS``. When the row list
    carries its flush count (padding ``-2 - n``), the CTAs split the values of
    few flush rows among them."""
    flush = tuned_config(num_k_heads, num_v_heads, window)[1]
    rows, cap = flush["ROWS_PER_PROGRAM"], flush.get("FLUSH_CTAS", _FLUSH_CTAS)
    # at least 4 CTAs, so that a single flush row is split 4 ways
    return max(4, min(-(-batch // rows), cap))


# mixed, a, b, ab_code, A_log, dt_bias, p_code, bias_code, out, state, d/k/g rings,
# slots, pos, scale, u, phi, ranks, fs, meta, beta, rotated q/k, layout,
# 7 int strides, 4 long strides
_STEP_SIG = drv.Signature(
    ["Q"] * 3 + ["i"] + ["Q"] * 2 + ["i", "i"] + ["Q"] * 7 + ["f"] + ["Q"] * 8
    + ["i"] * 7 + ["q"] * 4
)
# state, d ring, k ring, g ring, rows, n_rows, slots, meta, ranks, u, layout,
# finish statistics, beta, 7 strides, H, HV, G, query, output, query stride,
# query_bf16, output_bf16, scale, emit_output
_FLUSH_SIG = drv.Signature(
    ["Q"] * 5 + ["i"] + ["Q"] * 7 + ["q"] * 7 + ["i"] * 3
    + ["Q", "Q", "q", "?", "?", "f", "?"]
)
# finish statistics, rows, n_rows, meta, ranks, layout, phi, phi stride, HV,
# flush CTAs per key head
_FINISH_SIG = drv.Signature(["Q", "Q", "i"] + ["Q"] * 4 + ["q", "i", "i"])
# Floats of the coefficient-finish statistics per flushed head (energy and 4
# Gram rows).
_FINISH_STATS = 5 * HEAD_DIM
# CTAs (per 8 value heads) that walk the flush rows of the finish launch.
_FINISH_PROGRAMS = 64
_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _step_launch(k: _Kernel, mixed, a, b, alog, bias, out, state, writes, keys, gates,
                 index, pos, u, phi, widths, factors, meta, beta_ring, qk, layout,
                 rank_cap, scale) -> None:  # fmt: skip
    """The step launch (``step`` of the former C++ launcher)."""
    c = dict(k.spec.defines)
    nf_h, nf_hv, wmax = int(c["NF_H"]), int(c["NF_HV"]), int(c["WMAX"])
    K = V = HEAD_DIM
    B, H, HV = mixed.shape[0], keys.shape[1], state.shape[1]
    _check(H == nf_h and HV == nf_hv, f"gdn_step: compiled for H={nf_h} HV={nf_hv}")
    _check(_CODE.get(a.dtype) == int(c["NF_AB_CODE"]) and _CODE.get(b.dtype) == int(c["NF_AB_CODE"])
           and _CODE.get(alog.dtype) == int(c["NF_P_CODE"])
           and _CODE.get(bias.dtype) == int(c["NF_BIAS_CODE"]),
           "gdn_step: compiled for other gate dtypes")  # fmt: skip
    _check(state.stride(1) == K * V and state.stride(2) == V and state.stride(3) == 1
           and writes.stride(1) == 2 * wmax * V and writes.stride(2) == V and writes.stride(3) == 1
           and keys.stride(1) == 3 * wmax * K and keys.stride(2) == K and keys.stride(3) == 1
           and gates.stride(1) == wmax and gates.stride(2) == 1,
           "gdn_step: dense per-slot state/ring layout required")  # fmt: skip
    _check(layout.dtype == torch.int32 and layout.is_contiguous() and layout.numel() == 4 * HV,
           "gdn_step: int32 [HV,4] layout required")  # fmt: skip
    _check(writes.shape[2] == 2 * wmax and keys.shape[2] == 3 * wmax and gates.shape[2] == wmax,
           f"gdn_step: compiled for window {wmax} (d ring 2 W rows, k ring 3 W rows)")  # fmt: skip
    _check(qk is None or (qk.dtype == torch.float32 and qk.shape[1:] == (2, H, K)
                          and qk.stride(1) == H * K and qk.stride(2) == K
                          and qk.stride(3) == 1 and qk.stride(0) < 2**31),
           "gdn_step: FP32 (batch, 2, H, K) rotated q/k required")  # fmt: skip
    _check(index.stride(0) == 1 and meta.stride(0) == 1 and meta.dtype == torch.int32
           and u.stride(0) < 2**31 and phi.stride(0) < 2**31 and factors.stride(0) < 2**31,
           "gdn_step: 32-bit strides")  # fmt: skip
    gt = next((g for g in _GTS if rank_cap <= g), None)
    _check(gt is not None, "Unsupported G")
    tio = _TIO.get(mixed.dtype, "__nv_bfloat16")
    fn = k.function(_step_name(gt, nf_hv // nf_h, tio), _STEP_SIG)
    args = (
        mixed.data_ptr(), a.data_ptr(), b.data_ptr(), _CODE[a.dtype], alog.data_ptr(),
        bias.data_ptr(), _CODE[alog.dtype], _CODE[bias.dtype], out.data_ptr(),
        state.data_ptr(), writes.data_ptr(), keys.data_ptr(), gates.data_ptr(),
        index.data_ptr(), pos.data_ptr(), scale, u.data_ptr(), phi.data_ptr(),
        widths.data_ptr(), factors.data_ptr(), meta.data_ptr(),
        beta_ring.data_ptr() if beta_ring is not None else 0,
        qk.data_ptr() if qk is not None else 0, layout.data_ptr(), mixed.stride(0),
        a.stride(0), b.stride(0), u.stride(0), phi.stride(0), factors.stride(0),
        qk.stride(0) if qk is not None else 0, state.stride(0),
        writes.stride(0), keys.stride(0), gates.stride(0),
    )  # fmt: skip
    drv.launch(fn, _STEP_SIG, (B, H, 1), (32 * (nf_hv // nf_h), 1, 1), k.probes["smem"],
               drv.current_stream(mixed.device.index), args)  # fmt: skip


def _flush_launch(k: _Kernel, h0, writes, keys, gates, rows, programs, slots, meta,
                  widths, beta, u, phi, layout, rank_cap, query, output,
                  output_scale, emit_output, finish=True) -> None:  # fmt: skip
    """The flush launch (``flush`` of the former C++ launcher)."""
    c = dict(k.spec.defines)
    warps, wmax = k.probes["warps"], int(c["WMAX"])
    SK = SV = HEAD_DIM
    H, HV, G, n_rows = keys.shape[1], h0.shape[1], rank_cap, rows.numel()
    _check(HV == warps * H, f"flush w1: compiled for HV/H = {warps}")
    _check(4 <= G <= SK and G % 4 == 0, "flush w1: G must be a multiple of 4 in [4,128]")
    _check(h0.stride(0) % SK == 0 and h0.stride(1) == SK * SV and h0.stride(2) == SK
           and h0.stride(3) == 1, "flush w1: dense (HV,128,128) state pages")  # fmt: skip
    _check(layout.dtype == torch.int32 and layout.is_contiguous() and layout.numel() == 4 * HV,
           "flush w1: int32 [HV,4] layout required")  # fmt: skip
    _check(writes.shape[2] == 2 * wmax and keys.shape[2] == 3 * wmax
           and gates.shape[2] == wmax and beta.shape[-1] == wmax,
           f"flush w1: compiled for window {wmax}")  # fmt: skip
    _check(writes.stride(1) == 2 * wmax * SV and writes.stride(2) == SV and writes.stride(3) == 1
           and keys.stride(1) == 3 * wmax * SK and keys.stride(2) == SK and keys.stride(3) == 1,
           "flush w1: dense per-slot d and k rings required")  # fmt: skip
    _check(all(t.dtype == torch.int32 and t.is_contiguous() for t in (rows, slots, meta)),
           "flush w1: contiguous int32 rows, slots and meta")  # fmt: skip
    if n_rows == 0 or programs <= 0:
        return
    fn = k.function("gdn_flush_warp_kernel", _FLUSH_SIG)
    # One statistics record per work unit (flush row and value part): at most
    # max(rows, CTAs per key head) units.
    stats = torch.empty(max(n_rows, programs), HV, _FINISH_STATS, dtype=torch.float32,
                        device=h0.device)  # fmt: skip
    args = (
        h0.data_ptr(), writes.data_ptr(), keys.data_ptr(), gates.data_ptr(),
        rows.data_ptr(), n_rows, slots.data_ptr(), meta.data_ptr(), widths.data_ptr(),
        u.data_ptr(), layout.data_ptr(), stats.data_ptr(), beta.data_ptr(),
        h0.stride(0), h0.stride(1), writes.stride(0), keys.stride(0), gates.stride(0),
        u.stride(0), beta.stride(0),
        H, HV, G, query.data_ptr(), output.data_ptr(), query.stride(0),
        query.dtype == torch.bfloat16, output.dtype == torch.bfloat16, output_scale,
        emit_output,
    )  # fmt: skip
    stream = drv.current_stream(h0.device.index)
    drv.launch(fn, _FLUSH_SIG, (H, programs, 1), (k.probes["threads"], 1, 1),
               k.probes["smem"], stream, args)  # fmt: skip
    if not finish:                                  # no sketch heads of rank 0 < m < K
        return
    # The coefficient finish of the rebuilt sketches: one warp per (row, head).
    fin = k.function("gdn_finish_kernel", _FINISH_SIG)
    args = (stats.data_ptr(), rows.data_ptr(), n_rows, meta.data_ptr(), widths.data_ptr(),
            layout.data_ptr(), phi.data_ptr(), phi.stride(0), HV, programs)  # fmt: skip
    drv.launch(fin, _FINISH_SIG, (min(n_rows, _FINISH_PROGRAMS), -(-HV // 8), 1), (256, 1, 1), 0,
               stream, args)  # fmt: skip


def gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    slots: torch.Tensor,
    write_pos: torch.Tensor,
    meta: torch.Tensor,
    flush_rows: torch.Tensor,
    sketch: GDNSketch,
    scale: float,
    null_block_id: int = 0,
    has_flush_rows: bool = True,
    *,
    qk: torch.Tensor | None = None,
) -> None:
    """One SketchSSM decode step of a GDN layer.

    Every row runs the step (sketch readout, ring append); then the flush
    rows (``write_pos == W - 1``) fold the window into the full state,
    rebuild their sketch and overwrite their output with the exact one.

    The state is kept in the layer's rotated key frame R (per key head). q and
    k arrive unrotated: the key ring holds the BF16 keys as they are and ring
    dots use them unrotated; the sketch and state reads take ``qk``, the FP32
    rotated R q and R k ``(batch, 2, H, K)`` (None without a frame, as for
    ReplaySSM). The flush folds the window as the reference recurrence does in
    FP32: from the BF16 d ring rows, their residuals and the unit keys that
    the steps write (flush-only rows of the rings), it solves the window's WY
    system for the exact updates.

    Args:
        mixed_qkv: ``(batch, 2 H K + HV V)``, q/k unrotated.
        a, b: ``(batch, HV)`` gate activations; ``A_log``, ``dt_bias``: ``(HV,)``.
        out: ``(batch, HV, V)`` contiguous output, dtype of ``mixed_qkv``.
        state: ``(slots, HV, V, K)`` FP32 state in rotated key coordinates.
        d_cache, k_cache, g_cache: window rings ``(slots, HV, 2 W, V)``,
            ``(slots, H, 3 W, K)`` (BF16 storage) and ``(slots, HV, W)``
            (FP32); the window ``W`` is a multiple of 16. d: W BF16 rows d,
            then W fp16 rows of d - bf16(d) in units of its bf16 ulp; k: W - 1
            BF16 rows of the raw keys (row W - 1 unused), then W fp16 rows each
            of hi and lo of 2^10 x the unit keys in the rotated frame. The
            fp16 rows are for the flush only.
        slots: ``(batch,)`` int32 state slots; ``null_block_id`` (0) marks
            padding.
        write_pos: ``(batch,)`` int32 ring positions in ``[0, W - 1]``.
        meta: ``(batch,)`` int32 sketch rows (persistent request indices).
        flush_rows: ``(batch,)`` int32 flush rows, then padding: ``-2 - n``
            with n the number of flush rows lets the flush split the values of
            few flush rows over its CTAs (lower latency); -1 (count unknown)
            folds each row in one CTA.
        has_flush_rows: False skips the flush launch (no row flushes).
    """
    batch = mixed_qkv.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    h, hv, w = k_cache.shape[1], state.shape[1], g_cache.shape[2]
    assert w == sketch.tables.window
    if slots.dim() == 2:
        slots = slots[:, 0]
    # The step treats slots <= 0 as padding.
    assert null_block_id == 0
    assert slots.is_contiguous() and slots.dtype == torch.int32
    assert meta.is_contiguous() and meta.dtype == torch.int32
    assert write_pos.is_contiguous() and write_pos.dtype == torch.int32
    assert out.is_contiguous() and out.dtype == mixed_qkv.dtype
    step_out = out.view(batch, hv, HEAD_DIM)
    io = torch.float32 if mixed_qkv.dtype == torch.float32 else torch.bfloat16
    step = _step_ext(h, hv, _DT_CODE[a.dtype], _DT_CODE[A_log.dtype],
                     _DT_CODE[dt_bias.dtype], w, io)  # fmt: skip
    _step_launch(
        step, mixed_qkv, a, b, A_log, dt_bias, step_out, state, d_cache, k_cache,
        g_cache, slots, write_pos, sketch.u, sketch.phi, t.ranks, sketch.fs,
        meta, sketch.beta, qk, t.layout, t.rank_cap, scale,
    )  # fmt: skip
    if not has_flush_rows:
        return
    assert flush_rows.is_contiguous() and flush_rows.dtype == torch.int32
    query = mixed_qkv if qk is None else qk.view(batch, 2 * h * HEAD_DIM)
    _flush_launch(
        _flush_ext(h, hv, w), state, d_cache, k_cache, g_cache, flush_rows,
        flush_programs(batch, h, hv, w), slots, meta, t.ranks,
        sketch.beta, sketch.u, sketch.phi, t.layout, t.rank_cap, query, step_out,
        scale, True, getattr(t, "has_sketch_heads", True),
    )  # fmt: skip


def aot_specs(num_k_heads: int, num_v_heads: int, window: int, target: Target,
              io_dtype: torch.dtype = torch.bfloat16) -> list[tuple[kn.Spec, bool]]:  # fmt: skip
    """Specializations of a layer to precompile for ``target`` (all must
    build): gates ``a``/``b`` and ``dt_bias`` in ``io_dtype`` and an FP32
    ``A_log``."""
    code = _DT_CODE[io_dtype]
    specs = layer_specs(num_k_heads, num_v_heads, window, code, 0, code, io_dtype, target)
    return [(s, True) for s in specs]
