# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Kernel specializations: their key, the precompiled (AOT) lookup, the NVRTC
build with its disk cache, and the loaded modules.

A specialization (``Spec``) is one kernel source with its ``#define``s
(shape, window, build knobs), the kernels to instantiate and the extra
compiler flags. It is compiled exactly the same way ahead of time
(``python -m sketchssm.kernels.build``) and at run time.
"""

import fcntl
import functools
import hashlib
import json
import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import _driver, _nvrtc
from . import _runtime as rt

# Flags of every build. They are the device-side flags of the former nvcc
# build (torch.utils.cpp_extension: the __CUDA_NO_* defines, -O3 and
# -std=c++17); nvcc's --expt-relaxed-constexpr has no NVRTC counterpart and is
# not needed, and -default-device lets the host-only declarations the
# sources include parse in NVRTC's device-only mode.
BASE_FLAGS = (
    "-std=c++17",
    "-default-device",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
)
PROBE_PREFIX = "sk_probe_"
MANIFEST = "manifest.json"
MANIFEST_FORMAT = "sketchssm-kernels-aot"
MANIFEST_VERSION = 1


@functools.cache
def source_digest() -> str:
    """sha256 of every file the kernels are compiled from (``csrc/``)."""
    h = hashlib.sha256()
    for path in sorted(p for p in rt.CSRC.rglob("*") if p.is_file()):
        rel = path.relative_to(rt.CSRC).as_posix()
        h.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class Spec:
    """One kernel specialization."""

    kind: str  # mamba2_nf, mamba2_flush, gdn_step, gdn_flush, kda_step, kda_flush
    source: str  # csrc-relative .cuh
    defines: tuple[tuple[str, str], ...]
    names: tuple[str, ...]  # kernels (NVRTC name expressions)
    flags: tuple[str, ...] = ()
    # Host-side constants read from the compiled cubin: (name, C expression).
    probes: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def make(kind, source, defines: dict, names, flags=(), probes=()) -> "Spec":
        return Spec(
            kind, source, tuple(sorted((k, str(v)) for k, v in defines.items())),
            tuple(names), tuple(flags), tuple(probes),
        )  # fmt: skip

    @functools.cached_property
    def key(self) -> str:
        text = json.dumps(
            [self.kind, self.source, self.defines, self.names, self.flags, self.probes]
        )
        return hashlib.sha256(text.encode()).hexdigest()[:20]

    def text(self) -> str:
        """The translation unit: the defines, the kernel source, and one
        ``__device__`` constant per probe (no kernel reads them)."""
        lines = [f"#define {k} {v}" for k, v in self.defines]
        lines.append(f'#include "{self.source}"')
        for name, expr in self.probes:
            lines.append(
                f'extern "C" __device__ long long {PROBE_PREFIX}{name} = '
                f"(long long)({expr});"
            )
        return "\n".join(lines) + "\n"

    def options(self, arch: str) -> list[str]:
        return [*BASE_FLAGS, *self.flags, f"-arch={arch}", f"-I{rt.CSRC}"]

    def to_json(self) -> dict:
        return dict(
            kind=self.kind, source=self.source, defines=dict(self.defines),
            names=list(self.names), flags=list(self.flags),
            probes=[list(p) for p in self.probes],
        )  # fmt: skip

    @staticmethod
    def from_json(d: dict) -> "Spec":
        return Spec(
            d["kind"], d["source"], tuple(sorted(d["defines"].items())),
            tuple(d["names"]), tuple(d["flags"]), tuple(tuple(p) for p in d["probes"]),
        )  # fmt: skip


@dataclass
class Artifact:
    cubin: bytes
    lowered: dict[str, str]
    probes: dict[str, int]
    origin: str  # "aot" or "nvrtc"
    arch: str
    path: str = ""


# ── architectures ──


def normalize_arch(arch: str) -> str:
    """``sm_90a`` from ``9.0a``, ``90a``, ``sm_90a`` or ``compute_90a`` (the
    CMake ``CUDA_ARCHS`` spellings are accepted)."""
    a = arch.strip().lower()
    for prefix in ("sm_", "compute_"):
        if a.startswith(prefix):
            a = a[len(prefix):]
    suffix = a[-1] if a and a[-1] in "af" else ""
    digits = a[: len(a) - len(suffix)]
    if "." in digits:
        major, minor = digits.split(".")
        digits = f"{int(major)}{int(minor)}"
    if not digits.isdigit() or len(digits) < 2:
        raise ValueError(f"not a CUDA architecture: {arch!r}")
    return f"sm_{digits}{suffix}"


def arch_capability(arch: str) -> tuple[int, int]:
    digits = normalize_arch(arch)[3:].rstrip("af")
    return int(digits[:-1]), int(digits[-1])


def runtime_arch(capability: tuple[int, int]) -> str:
    """The architecture NVRTC compiles for on this GPU (arch-specific ``a``
    from sm_90 on, as the former JIT build)."""
    major, minor = capability
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def compatible_archs(capability: tuple[int, int]) -> list[str]:
    """Cubin architectures that run on ``capability``, best first: the exact
    arch-specific build, then family builds (``f``) of this or a lower minor,
    then plain builds."""
    major, minor = capability
    out = [f"sm_{major}{minor}a", f"sm_{major}{minor}f"]
    out += [f"sm_{major}{m}f" for m in range(minor - 1, -1, -1)]
    out += [f"sm_{major}{m}" for m in range(minor, -1, -1)]
    return out


# ── cubin probes ──


def read_probes(cubin: bytes) -> dict[str, int]:
    """Values of the ``sk_probe_*`` 8-byte globals of an ELF cubin."""
    if cubin[:4] != b"\x7fELF" or cubin[4] != 2:
        raise ValueError("not an ELF64 cubin")
    (shoff,) = struct.unpack_from("<Q", cubin, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)
    sections = []
    for i in range(shnum):
        name, typ, _, _, off, size, link, _, _, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", cubin, shoff + i * shentsize
        )
        sections.append((name, typ, off, size, link, entsize))
    out = {}
    for _, typ, off, size, link, entsize in sections:
        if typ != 2:  # SHT_SYMTAB
            continue
        stroff = sections[link][2]
        for j in range(size // entsize):
            st_name, _, _, shndx, value, st_size = struct.unpack_from(
                "<IBBHQQ", cubin, off + j * entsize
            )
            end = cubin.index(b"\0", stroff + st_name)
            sym = cubin[stroff + st_name : end].decode()
            if sym.startswith(PROBE_PREFIX) and st_size == 8 and shndx < len(sections):
                data = sections[shndx][2] + value
                out[sym[len(PROBE_PREFIX) :]] = struct.unpack_from("<q", cubin, data)[0]
    return out


# ── NVRTC build and cache ──


def compile_spec(spec: Spec, arch: str) -> Artifact:
    """Compile ``spec`` for ``arch`` with NVRTC (no cache)."""
    arch = normalize_arch(arch)
    nv = _nvrtc.nvrtc()
    cubin, lowered, _ = nv.compile(
        spec.text(), f"{spec.kind}_{spec.key}.cu", spec.options(arch), list(spec.names)
    )
    probes = read_probes(cubin)
    missing = {n for n, _ in spec.probes} - set(probes)
    if missing:
        raise RuntimeError(f"{spec.kind}: probes {sorted(missing)} not in the cubin")
    return Artifact(cubin, lowered, probes, "nvrtc", arch)


def _cache_path(spec: Spec, arch: str) -> Path:
    nv = _nvrtc.nvrtc()
    tag = f"nvrtc{nv.version[0]}.{nv.version[1]}"
    return (
        rt.cache_dir() / "nvrtc" / arch
        / f"{spec.kind}_{spec.key}_{source_digest()[:12]}_{tag}.cubin"
    )  # fmt: skip


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def nvrtc_build(spec: Spec, arch: str) -> Artifact:
    """``spec`` for ``arch`` from the disk cache, else compiled and cached.
    Processes building the same kernel wait for each other."""
    path = _cache_path(spec, arch)
    meta = path.with_suffix(".json")

    def cached() -> Artifact | None:
        if path.exists() and meta.exists():
            m = json.loads(meta.read_text())
            return Artifact(
                path.read_bytes(), m["lowered"], m["probes"], "nvrtc", arch, str(path)
            )
        return None

    art = cached()
    if art is not None:
        return art
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        art = cached()
        if art is not None:
            return art
        rt.log_once(logging.INFO, "Compiling SketchSSM CUDA kernel %s_%s for %s with NVRTC",
                    spec.kind, spec.key, arch)  # fmt: skip
        art = compile_spec(spec, arch)
        _write_atomic(path, art.cubin)
        _write_atomic(
            meta,
            json.dumps(
                dict(spec=spec.to_json(), arch=arch, lowered=art.lowered,
                     probes=art.probes, source_digest=source_digest()),
                indent=1,
            ).encode(),
        )  # fmt: skip
        art.path = str(path)
        return art


# ── AOT lookup ──


@dataclass
class _AotIndex:
    entries: dict[tuple[str, str], dict] = field(default_factory=dict)  # (key, arch)
    roots: dict[tuple[str, str], Path] = field(default_factory=dict)


_aot_index: _AotIndex | None = None


def reset_aot_index() -> None:
    global _aot_index
    _aot_index = None


def _load_aot_index() -> _AotIndex:
    global _aot_index
    if _aot_index is not None:
        return _aot_index
    index = _AotIndex()
    if os.environ.get("SKETCHSSM_KERNELS_DISABLE_AOT") != "1":
        for root in rt.aot_dirs():
            path = root / MANIFEST
            if not path.exists():
                continue
            try:
                m = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                rt.log_once(logging.WARNING, "Ignoring SketchSSM AOT manifest %s: %s", path, e)
                continue
            if m.get("format") != MANIFEST_FORMAT or m.get("version") != MANIFEST_VERSION:
                rt.log_once(logging.WARNING, "Ignoring SketchSSM AOT manifest %s: unknown format", path)
                continue
            if m.get("source_digest") != source_digest():
                rt.log_once(
                    logging.WARNING,
                    "Ignoring SketchSSM AOT kernels in %s: built from other kernel "
                    "sources (digest %s, this package %s)",
                    root, str(m.get("source_digest"))[:12], source_digest()[:12],
                )  # fmt: skip
                continue
            for e in m["kernels"]:
                k = (e["key"], e["arch"])
                if k not in index.entries:  # earlier folders win
                    index.entries[k] = e
                    index.roots[k] = root
    _aot_index = index
    return index


def aot_lookup(spec: Spec, capability: tuple[int, int]) -> tuple[Path, dict] | None:
    index = _load_aot_index()
    for arch in compatible_archs(capability):
        e = index.entries.get((spec.key, arch))
        if e is not None:
            return index.roots[(spec.key, arch)], e
    return None


def aot_any(kind: str, defines: dict, capability: tuple[int, int]) -> bool:
    """Whether some precompiled ``kind`` kernel for this GPU has all of
    ``defines`` (e.g. a stride-specialized Mamba-2 non-flush kernel)."""
    want = {k: str(v) for k, v in defines.items()}
    archs = set(compatible_archs(capability))
    for (_, arch), e in _load_aot_index().entries.items():
        if arch in archs and e["kind"] == kind:
            have = e["spec"]["defines"]
            if all(have.get(k) == v for k, v in want.items()):
                return True
    return False


def aot_load(spec: Spec, capability: tuple[int, int]) -> Artifact | None:
    hit = aot_lookup(spec, capability)
    if hit is None:
        return None
    root, e = hit
    path = root / e["file"]
    return Artifact(path.read_bytes(), e["lowered"], e["probes"], "aot", e["arch"], str(path))


def nvrtc_disabled() -> bool:
    return os.environ.get("SKETCHSSM_KERNELS_DISABLE_NVRTC") == "1"


def source_of(spec: Spec, capability: tuple[int, int]) -> str | None:
    """Where ``spec`` would come from: "aot", "nvrtc" or None."""
    if aot_lookup(spec, capability) is not None:
        return "aot"
    if not nvrtc_disabled() and _nvrtc.unavailable() is None:
        return "nvrtc"
    return None


def unavailable_reason(spec: Spec, capability: tuple[int, int]) -> str:
    if nvrtc_disabled():
        return f"no precompiled {spec.kind} kernel and NVRTC disabled"
    return f"no precompiled {spec.kind} kernel and NVRTC unavailable ({_nvrtc.unavailable()})"


# ── loaded kernels ──


class Kernel:
    """A compiled specialization; its module is loaded once per context."""

    def __init__(self, spec: Spec, artifact: Artifact):
        self.spec = spec
        self.artifact = artifact
        self.source = artifact.origin
        self.probes = artifact.probes
        self._modules: dict[int, object] = {}
        self._verified: set[int] = set()

    def module(self):
        ctx = _driver.current_context()
        mod = self._modules.get(ctx)
        if mod is None:
            mod = self._modules[ctx] = _driver.Module(self.artifact.cubin, self.artifact.lowered)
        return mod

    def function(self, name: str, sig=None) -> int:
        """The kernel ``name`` in the current context. With ``sig`` (a
        ``_driver.Signature``), the first lookup checks the parameter sizes
        against the cubin's."""
        fn = self.module().function(name)
        if sig is not None and fn not in self._verified:
            sizes = _driver.param_sizes(fn)
            if sizes is not None and sizes != sig.sizes:
                raise RuntimeError(
                    f"SketchSSM {self.spec.kind} {name}: launch parameters {sig.sizes} "
                    f"do not match the kernel's {sizes}"
                )
            self._verified.add(fn)
        return fn


def load(spec: Spec, capability: tuple[int, int] | None = None) -> Kernel:
    """The kernel of ``spec`` for this GPU: precompiled if available, else
    built with NVRTC."""
    if capability is None:
        capability = rt.capability()
    art = None if os.environ.get("SKETCHSSM_KERNELS_DISABLE_AOT") == "1" else aot_load(spec, capability)
    if art is None:
        if nvrtc_disabled():
            raise RuntimeError(unavailable_reason(spec, capability))
        art = nvrtc_build(spec, runtime_arch(capability))
    require = os.environ.get("SKETCHSSM_KERNELS_REQUIRE")
    if require and require != art.origin:
        raise RuntimeError(
            f"SketchSSM {spec.kind} kernel {spec.key}: from {art.origin}, but "
            f"SKETCHSSM_KERNELS_REQUIRE={require}"
        )
    rt.log_once(logging.DEBUG, "SketchSSM %s %s: %s (%s)", spec.kind, spec.key,
                art.origin, art.path)  # fmt: skip
    return Kernel(spec, art)
