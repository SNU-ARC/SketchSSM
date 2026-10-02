# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""NVRTC through ``ctypes``: compiles a kernel source to a cubin at run time.

The library comes from the CUDA pip wheels that torch already depends on
(``nvidia-cuda-nvrtc``; the headers from ``nvidia-cuda-runtime``), or from a
CUDA toolkit. Neither nvcc nor a host compiler is needed.
"""

import ctypes
import functools
import glob
import os
import sys
from pathlib import Path

# Shim headers (ours): empty stand-ins for the host-only includes of the
# kernel sources (torch, ATen, c10, the C++ library) and the two C headers
# that cuda.h pulls in.
SHIM = Path(__file__).parent / "csrc" / "include"


class NVRTCError(RuntimeError):
    pass


def _wheel_roots() -> list[Path]:
    roots = []
    for entry in sys.path:
        p = Path(entry) / "nvidia"
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def _torch_cuda_major() -> int | None:
    torch = sys.modules.get("torch")
    version = getattr(getattr(torch, "version", None), "cuda", None)
    return int(version.split(".")[0]) if version else None


def _candidates() -> list[Path]:
    """libnvrtc candidates, best first."""
    env = os.environ.get("SKETCHSSM_KERNELS_NVRTC")
    if env:
        return [Path(env)]
    found: list[tuple[int, Path]] = []
    major = _torch_cuda_major()
    for root in _wheel_roots():
        # CUDA 13 wheels share nvidia/cu13/; CUDA 12 wheels have one folder each.
        for lib in sorted(root.glob("cu*/lib/libnvrtc.so.*")) + sorted(
            root.glob("cuda_nvrtc/lib/libnvrtc.so.*")
        ):
            if "alt" in lib.name or "builtins" in lib.name:
                continue
            ver = lib.name.split(".so.")[1].split(".")[0]
            found.append((0 if major is not None and ver == str(major) else 1, lib))
    found.sort(key=lambda t: t[0])
    out = [p for _, p in found]
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            out += sorted(Path(os.environ[var]).glob("lib64/libnvrtc.so*"))[:1]
    out += sorted(Path("/usr/local/cuda").glob("lib64/libnvrtc.so*"))[:1]
    return out


def _include_for(lib: Path) -> Path | None:
    """The CUDA headers matching ``lib``: the wheel's or toolkit's include/."""
    env = os.environ.get("SKETCHSSM_KERNELS_CUDA_INCLUDE")
    if env:
        return Path(env)
    for cand in (
        lib.parent.parent / "include",  # nvidia/cu13/include, $CUDA_HOME/include
        lib.parent.parent.parent / "cuda_runtime" / "include",  # CUDA 12 wheels
    ):
        if (cand / "cuda_bf16.h").exists():
            return cand
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var) and (Path(os.environ[var]) / "include/cuda_bf16.h").exists():
            return Path(os.environ[var]) / "include"
    return None


class NVRTC:
    def __init__(self, path: Path):
        # libnvrtc dlopens its builtins library by soname; load it first from
        # the same folder so that a wheel install needs no LD_LIBRARY_PATH.
        for b in sorted(glob.glob(str(path.parent / "libnvrtc-builtins.so.*"))):
            if ".alt." not in b:
                try:
                    ctypes.CDLL(b, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
        self.version = (major.value, minor.value)
        self.include = _include_for(path)

    def _check(self, status: int, what: str = "") -> None:
        if status != 0:
            msg = self.lib.nvrtcGetErrorString(status).decode()
            raise NVRTCError(f"NVRTC {what}: {msg}")

    def supported_archs(self) -> list[int]:
        n = ctypes.c_int()
        self._check(self.lib.nvrtcGetNumSupportedArchs(ctypes.byref(n)))
        arr = (ctypes.c_int * n.value)()
        self._check(self.lib.nvrtcGetSupportedArchs(arr))
        return list(arr)

    def compile(
        self,
        source: str,
        name: str,
        options: list[str],
        name_expressions: list[str],
    ) -> tuple[bytes, dict[str, str], str]:
        """Compile ``source`` to a cubin. Returns the cubin, the lowered
        (mangled) name of each name expression and the compile log."""
        if self.include is None:
            raise NVRTCError(
                "no CUDA headers (cuda_bf16.h) next to "
                f"{self.path}; install nvidia-cuda-runtime or set "
                "SKETCHSSM_KERNELS_CUDA_INCLUDE"
            )
        lib = self.lib
        prog = ctypes.c_void_p()
        self._check(
            lib.nvrtcCreateProgram(
                ctypes.byref(prog), source.encode(), name.encode(), 0, None, None
            ),
            "create",
        )
        try:
            for expr in name_expressions:
                self._check(lib.nvrtcAddNameExpression(prog, expr.encode()), expr)
            opts = [*options, f"-I{self.include}", f"-I{SHIM}"]
            arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
            status = lib.nvrtcCompileProgram(prog, len(opts), arr)
            size = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value)
            lib.nvrtcGetProgramLog(prog, buf)
            log = buf.value.decode(errors="replace")
            if status != 0:
                raise NVRTCError(f"compiling {name} failed:\n{log}")
            self._check(lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "cubin")
            cubin = ctypes.create_string_buffer(size.value)
            self._check(lib.nvrtcGetCUBIN(prog, cubin), "cubin")
            lowered = {}
            for expr in name_expressions:
                out = ctypes.c_char_p()
                self._check(
                    lib.nvrtcGetLoweredName(prog, expr.encode(), ctypes.byref(out)),
                    expr,
                )
                lowered[expr] = out.value.decode()
            return cubin.raw, lowered, log
        finally:
            lib.nvrtcDestroyProgram(ctypes.byref(prog))


@functools.cache
def _load() -> NVRTC | str:
    errors = []
    for cand in _candidates():
        try:
            return NVRTC(cand)
        except OSError as e:
            errors.append(f"{cand}: {e}")
    return "libnvrtc not found" + (f" ({'; '.join(errors)})" if errors else "")


def nvrtc() -> NVRTC:
    lib = _load()
    if isinstance(lib, str):
        raise NVRTCError(lib)
    return lib


def unavailable() -> str | None:
    """Why NVRTC cannot compile here, or None."""
    lib = _load()
    if isinstance(lib, str):
        return lib
    if lib.include is None:
        return f"no CUDA headers for {lib.path}"
    return None
