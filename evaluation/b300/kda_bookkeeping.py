# SPDX-License-Identifier: Apache-2.0
"""Slot bookkeeping for the KDA window runtime (v3): resolve also assigns the fresh rows' slots (Owners = page, Pos = 0,
the fork's `_acquire_row` h == 0 duty; the eviction write-back moved into the flush main kernel).

v2: `resolve` reproduces the fork's Triton `_resolve_work` semantics
(existing / fresh rows, free-slot assignment: empty slots first then occupied ones in slot order, work lists) with
O(1) page -> slot lookups through a versioned map (rebuilt from the owners each call, no clearing) and three block
scans in one 1024-thread CTA."""
import os
from functools import lru_cache
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#define NT 1024

__device__ __forceinline__ int block_scan_inclusive(int v, int* sw) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) sw[warp] = x;
    __syncthreads();
    int base = 0;
    for (int w = 0; w < warp; ++w) base += sw[w];
    __syncthreads();
    return x + base;
}

__global__ void __launch_bounds__(NT)
resolve_kernel(const int* __restrict__ ids, int* __restrict__ owners, int* __restrict__ slots_out, int* __restrict__ old_out,
               bool* __restrict__ fresh_out, int* __restrict__ old_pos, int* __restrict__ pos, long long* __restrict__ counts,
               int* __restrict__ work_rows, int* __restrict__ work_counts, int* __restrict__ map_slot, int* __restrict__ map_gen,
               int* __restrict__ gen_counter, int* __restrict__ rowinfo, int B, int P, int num_pages)   // pos: read and written
{
    __shared__ int s_own[NT], s_free[NT], sw[32], s_nempty, s_gen;
    __shared__ unsigned char s_matched[NT], s_avail[NT];
    const int t = threadIdx.x;
    if (t == 0) s_gen = ++gen_counter[0];
    const int own = (t < P) ? owners[t] : -2;
    s_own[t] = own; s_matched[t] = 0;
    __syncthreads();
    const int gen = s_gen;
    if (t < P && own > 0 && own < num_pages) { map_slot[own] = t; map_gen[own] = gen; }   // owners are unique
    __syncthreads();
    // rows: existing slot through the map
    const int id = (t < B) ? ids[t] : -1;
    int existing = 0, found = 0;
    if (t < B && id > 0 && id < num_pages && map_gen[id] == gen) { existing = map_slot[id]; found = 1; }
    if (found) s_matched[existing] = 1;
    __syncthreads();
    const int avail = (t < P) && !s_matched[t];
    s_avail[t] = avail;
    const int fresh = (t < B) && !found && (id > 0);
    const int ordinal = block_scan_inclusive(fresh, sw);
    const int empty = avail && own < 0, occupied = avail && own >= 0;
    const int cs_empty = block_scan_inclusive(empty, sw);
    const int cs_occ = block_scan_inclusive(occupied, sw);
    if (t == NT - 1) s_nempty = cs_empty;
    __syncthreads();
    s_free[t] = empty ? cs_empty : (s_nempty + cs_occ);
    __syncthreads();
    int chosen = 0;
    if (fresh) for (int p = 0; p < P; ++p) if (s_avail[p] && s_free[p] == ordinal) chosen = p;
    int slot = -1;
    if (t < B && id > 0) slot = fresh ? chosen : existing;
    int old = -1;
    if (slot >= 0 && fresh) old = s_own[slot];
    if (t < B) {
        slots_out[t] = slot; old_out[t] = old; fresh_out[t] = fresh;
        old_pos[t] = (slot >= 0 && fresh && old > 0) ? pos[slot] : 0;
    }
    __syncthreads();                                              // every old owner / pending count is read before the fresh rows take their slots
    if (fresh) { owners[slot] = id; pos[slot] = 0; }
    __syncthreads();
    const int cnt = block_scan_inclusive((t < B) && slot >= 0, sw);
    if (t == NT - 1) counts[0] += cnt;
    if (fresh) work_rows[ordinal - 1] = t;
    const int cur = (slot >= 0) ? pos[slot] : 0;
    if (t < B) rowinfo[t] = (slot >= 0) ? (slot | (cur << 16)) : -1;   // the step kernel's one-load slot chain
    const int flush = (t < B) && slot >= 0 && !fresh && cur == 15;
    const int fo = block_scan_inclusive(flush, sw);
    if (flush) work_rows[P + fo - 1] = t;
    if (t == NT - 1) { work_counts[0] = ordinal; work_counts[1] = fo; }
}

__global__ void bump_kernel(const int* __restrict__ slots, int* __restrict__ pos, int B) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < B) { const int s = slots[t]; if (s >= 0) pos[s] = (pos[s] + 1) & 15; }
}

void resolve(torch::Tensor ids, torch::Tensor owners, torch::Tensor slots, torch::Tensor old, torch::Tensor fresh, torch::Tensor old_pos,
             torch::Tensor pos, torch::Tensor counts, torch::Tensor work_rows, torch::Tensor work_counts, torch::Tensor map_slot,
             torch::Tensor map_gen, torch::Tensor gen_counter, torch::Tensor rowinfo, int B, int P)
{
    TORCH_CHECK(B <= NT && P <= NT, "resolve: batch and capacity must be <= 1024");
    resolve_kernel<<<1, NT, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int>(), owners.data_ptr<int>(), slots.data_ptr<int>(), old.data_ptr<int>(), fresh.data_ptr<bool>(), old_pos.data_ptr<int>(),
        pos.data_ptr<int>(), (long long*)counts.data_ptr<int64_t>(), work_rows.data_ptr<int>(), work_counts.data_ptr<int>(),
        map_slot.data_ptr<int>(), map_gen.data_ptr<int>(), gen_counter.data_ptr<int>(), rowinfo.data_ptr<int>(), B, P, (int)map_slot.numel());
}
void bump(torch::Tensor slots, torch::Tensor pos, int B) {
    bump_kernel<<<(B + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(slots.data_ptr<int>(), pos.data_ptr<int>(), B);
}
"""
_CPP = r"""
#include <torch/extension.h>
void resolve(torch::Tensor ids, torch::Tensor owners, torch::Tensor slots, torch::Tensor old, torch::Tensor fresh, torch::Tensor old_pos,
             torch::Tensor pos, torch::Tensor counts, torch::Tensor work_rows, torch::Tensor work_counts, torch::Tensor map_slot,
             torch::Tensor map_gen, torch::Tensor gen_counter, torch::Tensor rowinfo, int B, int P);
void bump(torch::Tensor slots, torch::Tensor pos, int B);
"""


@lru_cache(maxsize=None)
def module():
    build = Path(os.environ.get("KDA_BOOKKEEPING_BUILD_DIR", Path.home() / ".cache" / "sketchssm" / "kda_bookkeeping"))
    build.mkdir(parents=True, exist_ok=True)
    import hashlib
    return load_inline(name="kda_control3_" + hashlib.sha256(_SRC.encode()).hexdigest()[:10], cpp_sources=_CPP, cuda_sources=_SRC,
                       functions=["resolve", "bump"], build_directory=str(build),
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_100f,code=sm_100f", "-std=c++20"], extra_cflags=["-std=c++20"], verbose=False)
