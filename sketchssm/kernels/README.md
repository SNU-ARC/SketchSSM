# SketchSSM CUDA kernels (`sketchssm.kernels`)

The CUDA decode kernels of SketchSSM for three layer families: Mamba-2, Gated DeltaNet (GDN) and Kimi Delta Attention (KDA). Each family has a step kernel, which reads the per-request sketch on non-flush steps, and a flush kernel. The flush folds the window into the exact state and rebuilds the sketch. These are the kernels of the paper, adapted to vLLM's paged state cache and continuous batching (a sketch per request, flush row lists) and generalized in layer shape and window length.

vLLM uses these kernels when they are built into vLLM or this package is installed, and falls back to its own Triton kernels otherwise.

This directory is self-contained: Python, `csrc/` and `configs/`. It imports only `torch`, never vLLM.

## Install

```bash
pip install sketchssm                     # from PyPI
pip install -e /path/to/SketchSSM         # development checkout
```

No CUDA toolkit is needed with a pip-installed PyTorch: kernels that are not
precompiled are compiled at run time with NVRTC and the CUDA headers from the
CUDA pip packages PyTorch already installs (otherwise from `$CUDA_HOME`). The GPU must be sm_80 or newer for Mamba-2 and GDN, and sm_90 or newer
for KDA (TMA). Otherwise `*_supported` returns false and vLLM uses Triton.

## How the kernels are built

Each kernel is specialized at compile time for its layer shape, window, build
knobs and GPU architecture. A call uses, in order:

1. **Precompiled (AOT) kernels**, from `$SKETCHSSM_KERNELS_AOT_DIR`, the folders
   given to `set_aot_dirs`, then the package's `aot/` folder.
2. **NVRTC**: the exact specialization is compiled at run time on first use
   (about a second) and cached on disk.

The output is the same either way, and `Support.source` says which one a layer
uses. To precompile the default set (the published models' shapes for TP 1-8
and W 16/32/64) for H100 and B200/B300:

```bash
python sketchssm/kernels/build.py --arch "9.0a;10.0f" --out sketchssm/kernels/aot --clean
```

`build.py` runs from a source checkout without installing the package and needs
no GPU. Other options:

- `--layers my.json`: more layer shapes; `--no-default` skips the default set.
- `--configs DIR --device-name NAME`: build with tuned-config files for a GPU.
- `--tp`, `--windows`: tensor-parallel degrees and windows.
- `--specs <cache>/nvrtc`: precompile exactly what NVRTC compiled on a machine.
  The Mamba-2 non-flush kernel bakes in the cache page strides, so it is
  precompiled only this way; otherwise NVRTC compiles it.
- `--list` (dry run), `--jobs`, `--nvrtc`, `--cuda-include`.

A precompiled set is used only if it was built from the same `csrc/`.

**Built into vLLM.** vLLM's build fetches this repository at a pinned commit
(`cmake/external_projects/sketchssm.cmake`, the same way it vendors DeepGEMM and
FlashKDA). It then copies this package to `vllm/third_party/sketchssm_kernels/`
and precompiles the default set for the target architectures. vLLM uses that
copy first, then an installed `sketchssm` package, then its Triton kernels.
A precompiled vLLM wheel (`VLLM_USE_PRECOMPILED=1`) contains only what the wheel
was built with, so install `sketchssm` alongside it.

## API

```python
from sketchssm import kernels as sk

sk.API_VERSION                     # 5; bumped on a breaking change
sk.set_config_dirs([folder, ...])  # extra tuned-config folders
sk.set_aot_dirs([folder, ...])     # extra precompiled-kernel folders
sk.set_cache_dir(path)             # where the NVRTC builds go

# Mamba-2
sk.mamba2_supported(num_heads, head_dim, state_size, n_groups, window,
                    activation_dtype, state_dtype) -> Support
sk.mamba2_decode(state, x, dt, A, B, C, D, dt_bias, x_cache, dt_cache, B_cache,
                 bc_pre, write_pos, is_flush, flush_rows, slots, meta, out, sketch,
                 null_block_id=0, has_flush_rows=True, *, frames_t,
                 flush_programs, run_with_flush=None)
# GDN
sk.gdn_supported(num_k_heads, num_v_heads, head_k_dim, head_v_dim, window,
                 activation_dtype, state_dtype) -> Support
sk.gdn.check_resources(num_k_heads, num_v_heads, window)  # raises ValueError
sk.gdn_decode(mixed_qkv, a, b, A_log, dt_bias, out, state, d_cache, k_cache,
              g_cache, slots, write_pos, meta, flush_rows, sketch, scale,
              null_block_id=0, has_flush_rows=True)
# KDA
sk.kda_supported(num_heads, head_k_dim, head_v_dim, window, activation_dtype,
                 state_dtype, lower_bound=-5.0) -> Support
sk.kda_decode(q, k, v, g, beta, A_log, dt_bias, out, state, rings, slots, meta,
              pos, flush_rows, sketch, scratch, scale=128**-0.5,
              null_block_id=0, has_flush_rows=True)
sk.kda_cold_build(state, rings, slots, meta, rows, sketch, scratch,
                  null_block_id=0)
```

`Support` is truthy when the layer can use the kernels; `source` is `aot` or `nvrtc`. Otherwise its `reason` says why not.

**Containers.** The package does not allocate the sketch, its tables or the KDA window rings. The caller owns them in its storage layout. `sketchssm.kernels.types` lists, as `typing.Protocol`s, exactly the attributes the kernels read (`Mamba2Sketch`, `GDNSketch`, `KDASketch`, `KDARings`). vLLM's `SketchArgs`, `GDNSketchArgs`, `KDASketchArgs` and `KDASketchRings` satisfy them.

**Metadata.**
- `slots`, `meta`, `write_pos`/`pos` and `flush_rows` are contiguous int32.
- `flush_rows` lists the flush rows, padded with -1.
- `null_block_id` (slot 0) marks padding rows.
- `has_flush_rows=False` skips the flush launch.

**The window.** `W` is read from the ring shapes. It can be any multiple of 16.

**Mamba-2 frames.** The Mamba-2 state is kept in a rotated frame R per group (`frames_t` = Rᵀ, `(groups, N, N)` FP32), but B and C stay unrotated where precision matters: the B ring holds the BF16 B (as ReplaySSM does) and `bc_pre`, the ring's B·C products, comes from the unrotated B and C. The caller fills `bc_pre` and passes the FP32 query R·C as `C` (in vLLM, the Triton `sketch_bc_pre` and `sketch_query`). The flush kernel rotates the window keys R·B_t itself to FP32 accuracy (BF16 keys times the frame split into three BF16 terms on tensor cores), once per CTA for the `WARPS` heads of a group it serves. `run_with_flush(flush, nonflush)` lets the caller overlap the two launches. vLLM runs the flush on a side stream.

## Supported shapes

| Family | Shapes | Window W | dtypes | GPU |
| --- | --- | --- | --- | --- |
| Mamba-2 | head dim a multiple of 16, ≤ 128; state size 64, 128 or 256; heads per group a multiple of the `NF_HEADS` knob | any multiple of 16 | BF16 activations, FP32 state | sm_80+ |
| GDN | head dims 128; 1 to 8 value heads per key head | any multiple of 16 that fits shared memory (`gdn.check_resources`) | BF16 activations, FP32 state | sm_80+ |
| KDA | head dim 128; 1 to 128 heads; gate lower bound -5 | any multiple of 16 | BF16 activations, FP32 state | sm_90+ |

## Compile cache

NVRTC builds are cached under `nvrtc/<arch>/` in:

- the folder given to `set_cache_dir` (vLLM passes `$VLLM_CACHE_ROOT/sketchssm_cuda`), else
- `$SKETCHSSM_KERNELS_CACHE_DIR`, else
- `~/.cache/sketchssm/kernels`.

**Environment variables:**

| Variable | Effect |
| --- | --- |
| `SKETCHSSM_KERNELS_AOT_DIR` | extra precompiled-kernel folders (path list) |
| `SKETCHSSM_KERNELS_DISABLE_AOT`, `SKETCHSSM_KERNELS_DISABLE_NVRTC` | skip that source |
| `SKETCHSSM_KERNELS_REQUIRE=aot\|nvrtc` | fail instead of falling back |
| `SKETCHSSM_KERNELS_NVRTC`, `SKETCHSSM_KERNELS_CUDA_INCLUDE` | the NVRTC library and CUDA headers to use |

## Tuned build knobs

The build knobs change scheduling only (occupancy, staging, prefetch), so outputs are bitwise identical across knobs. Each family has architecture defaults: the paper's B300 knobs on sm_100 and newer, and H100-derived overrides below sm_100. Per-GPU JSON files override them. The package ships files for the published model shapes on H100 (tuned) and B300 (the paper's knobs, written out):

| Family | File name | Keys |
| --- | --- | --- |
| Mamba-2 | `head_dim=P,heads_per_group=R[,state_size=N][,window=W],device_name=<GPU>.json` | `nf`, `flush` |
| GDN | `num_k_heads=H,num_v_heads=HV[,window=W],device_name=<GPU>.json` | `step`, `flush` |
| KDA | `kda,num_heads=H[,window=W],device_name=<GPU>.json` | `step`, `flush` |

`<GPU>` is `torch.cuda.get_device_name()` with spaces replaced by `_`.

**Search order for files:**
1. `$SKETCHSSM_KERNELS_CONFIG_DIR`;
2. the folders given to `set_config_dirs` (vLLM passes `VLLM_TUNED_CONFIG_FOLDER`);
3. the shipped `configs/{mamba2,gdn,kda}/`.

**Choice of file for a window:** the shape's file for that window; else its window-16 file, used only if those knobs fit the window; else the architecture defaults.

**Tuning on your GPU:**

```bash
python -m sketchssm.kernels.tools.tune_mamba2 --calibration nano_frames.pt \
    --head-dim 80 --save-configs --out-dir my_configs
python -m sketchssm.kernels.tools.tune_gdn --calibration qwen_frames.pt --num-v-heads 32 --window 32
python -m sketchssm.kernels.tools.tune_kda --calibration glm_frames.pt
```

Each tuner runs a coordinate descent over one family's knobs, with every candidate compiled with NVRTC. A candidate is timed over every layer of the calibration with its own ranks (`--layer` for one) and over a whole window, W - 1 non-flush steps plus one flush step (`--objective mixed` for steps with the rows spread over the window), at `--batch` 256; the result is also compared with the defaults at `--check-batch` 128. A file is written only when the tuned knobs beat the knobs in effect (the shape's file for the window, else its window-16 file). Put the resulting files in a config folder, and precompile them with `build.py --configs` if needed.

**Benchmarks:** `python -m sketchssm.kernels.tools.benchmark_{mamba2,gdn,kda}` times one decode step of one layer and compares it with the standard decode. The tuners and benchmarks import vLLM for the sketch tables, the Triton helpers and the baseline decoders.

## Tests

The kernel tests live in the repository's `tests/kernels/`. They compare against FP64 oracles and import vLLM for the sketch tables and cold builds.

```bash
pytest tests/kernels -m "not slow"   # quick: W = 16 and one long window
pytest tests/kernels                 # full window sweeps (W = 16 ... 64)
```

## License

Apache-2.0 (the repository's `LICENSE`).
