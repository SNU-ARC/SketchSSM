# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""The CUDA driver API through ``ctypes``: load a cubin into the current
context and launch its kernels on torch's current stream (so launches are
captured by CUDA graphs like any torch kernel)."""

import ctypes
import functools
import struct
import threading

import torch

CUDA_SUCCESS = 0
# CUfunction_attribute
ATTR_MAX_THREADS_PER_BLOCK = 0
ATTR_SHARED_SIZE_BYTES = 1
ATTR_LOCAL_SIZE_BYTES = 3
ATTR_NUM_REGS = 4
ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
# CUdevice_attribute
DEV_MULTIPROCESSOR_COUNT = 16
DEV_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
DEV_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
# CUtensorMap enums
TENSOR_MAP_FLOAT32 = 7
TENSOR_MAP_SWIZZLE_128B = 3
TENSOR_MAP_BYTES = 128


class CUDADriverError(RuntimeError):
    pass


@functools.cache
def lib() -> ctypes.CDLL:
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p, *[ctypes.c_uint] * 6, ctypes.c_uint, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]  # fmt: skip
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p,
    ]  # fmt: skip
    cuda.cuFuncGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p,
    ]  # fmt: skip
    cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    ]  # fmt: skip
    cuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
    ]  # fmt: skip
    cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cuda.cuTensorMapEncodeTiled.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]  # fmt: skip
    if hasattr(cuda, "cuFuncGetParamInfo"):
        cuda.cuFuncGetParamInfo.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ]  # fmt: skip
    return cuda


def check(status: int, what: str) -> None:
    if status != CUDA_SUCCESS:
        msg = ctypes.c_char_p()
        lib().cuGetErrorString(status, ctypes.byref(msg))
        text = msg.value.decode() if msg.value else f"error {status}"
        raise CUDADriverError(f"{what}: {text}")


def current_context() -> int:
    """The current driver context (torch's primary context of the current
    device), initialized by torch if needed."""
    ctx = ctypes.c_void_p()
    check(lib().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    if not ctx.value:
        torch.cuda.init()
        torch.cuda.current_stream()  # makes the primary context current
        torch.empty(0, device="cuda")
        check(lib().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            raise CUDADriverError("no current CUDA context")
    return ctx.value


@functools.cache
def device_attribute(device: int, attr: int) -> int:
    out = ctypes.c_int()
    check(lib().cuDeviceGetAttribute(ctypes.byref(out), attr, device), "cuDeviceGetAttribute")
    return out.value


_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def current_stream(device: int) -> int:
    """torch's current stream on ``device`` (a ``cudaStream_t``)."""
    if _raw_stream is not None:
        return _raw_stream(device)
    return torch.cuda.current_stream(device).cuda_stream


class Module:
    """A cubin loaded into one context."""

    def __init__(self, cubin: bytes, lowered: dict[str, str]):
        handle = ctypes.c_void_p()
        check(lib().cuModuleLoadData(ctypes.byref(handle), cubin), "cuModuleLoadData")
        self.handle = handle
        self.lowered = lowered
        self.functions: dict[str, int] = {}

    def function(self, name: str) -> int:
        fn = self.functions.get(name)
        if fn is None:
            out = ctypes.c_void_p()
            symbol = self.lowered.get(name, name)
            check(
                lib().cuModuleGetFunction(ctypes.byref(out), self.handle, symbol.encode()),
                f"cuModuleGetFunction({name})",
            )
            fn = self.functions[name] = out.value
        return fn


def func_attribute(fn: int, attr: int) -> int:
    out = ctypes.c_int()
    check(lib().cuFuncGetAttribute(ctypes.byref(out), attr, fn), "cuFuncGetAttribute")
    return out.value


def set_max_dynamic_smem(fn: int, nbytes: int) -> None:
    check(
        lib().cuFuncSetAttribute(fn, ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES, nbytes),
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
    )


def occupancy(fn: int, threads: int, smem: int) -> int:
    out = ctypes.c_int()
    check(
        lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(out), fn, threads, smem),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return out.value


def param_sizes(fn: int) -> list[int] | None:
    """The kernel's parameter sizes from the cubin (driver >= 12.4), else None."""
    cuda = lib()
    if not hasattr(cuda, "cuFuncGetParamInfo"):
        return None
    sizes = []
    offset, size = ctypes.c_size_t(), ctypes.c_size_t()
    while cuda.cuFuncGetParamInfo(fn, len(sizes), ctypes.byref(offset), ctypes.byref(size)) == 0:
        sizes.append(size.value)
    return sizes


class Signature:
    """Kernel parameters as ``struct`` codes, packed into one buffer per
    launch; the driver copies each parameter from its offset."""

    def __init__(self, codes: list[str]):
        self.codes = codes
        self.struct = struct.Struct("=" + "".join(codes))
        self.sizes = [struct.calcsize("=" + c) for c in codes]
        self.offsets = [sum(self.sizes[:i]) for i in range(len(codes))]
        self.size = self.struct.size
        self._local = threading.local()

    def buffers(self):
        """This thread's parameter buffer and its pointer array."""
        bufs = getattr(self._local, "bufs", None)
        if bufs is None:
            buf = ctypes.create_string_buffer(self.size)
            base = ctypes.addressof(buf)
            ptrs = (ctypes.c_void_p * len(self.offsets))(*[base + o for o in self.offsets])
            bufs = self._local.bufs = (buf, ptrs)
        return bufs


def launch(fn: int, sig: Signature, grid, block, smem: int, stream: int, args) -> None:
    # cuLaunchKernel copies the parameters before it returns, so the buffer
    # can be reused by the next launch.
    buf, params = sig.buffers()
    sig.struct.pack_into(buf, 0, *args)
    status = _launch_kernel(
        fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, stream,
        params, None,
    )  # fmt: skip
    if status != CUDA_SUCCESS:
        check(status, "cuLaunchKernel")


def _launch_kernel(*args) -> int:
    global _launch_kernel
    _launch_kernel = lib().cuLaunchKernel
    return _launch_kernel(*args)


def encode_tensor_map_tiled(
    ptr: int,
    dims: list[int],
    strides: list[int],
    box: list[int],
    swizzle: int = TENSOR_MAP_SWIZZLE_128B,
) -> bytes:
    """``cuTensorMapEncodeTiled`` of an FP32 tensor (no interleave, no L2
    promotion, no OOB fill), as the 128 bytes of a ``CUtensorMap``."""
    rank = len(dims)
    # The descriptor must be 64-byte aligned (128 in CUDA 13's cuda.h).
    raw = ctypes.create_string_buffer(TENSOR_MAP_BYTES + 128)
    addr = (ctypes.addressof(raw) + 127) // 128 * 128
    c_dims = (ctypes.c_uint64 * rank)(*dims)
    c_strides = (ctypes.c_uint64 * max(1, rank - 1))(*strides)
    c_box = (ctypes.c_uint32 * rank)(*box)
    c_step = (ctypes.c_uint32 * rank)(*[1] * rank)
    status = lib().cuTensorMapEncodeTiled(
        addr, TENSOR_MAP_FLOAT32, rank, ptr, c_dims, c_strides, c_box, c_step,
        0, swizzle, 0, 0,
    )  # fmt: skip
    check(status, "cuTensorMapEncodeTiled")
    return ctypes.string_at(addr, TENSOR_MAP_BYTES)
