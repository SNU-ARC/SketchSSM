// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
// NVRTC shim: the kernel sources include torch for their host helpers only
// (kda_sketch_flush.cuh: host_head_table). These declarations let that helper
// parse; it is never called in device code, so nothing here is defined.
#pragma once
namespace torch {
enum ScalarType { kInt32 };
struct Device {
  bool is_cpu() const;
};
struct Tensor {
  Device device() const;
  ScalarType scalar_type() const;
  bool is_contiguous() const;
  long numel() const;
  template <typename T>
  T* data_ptr() const;
};
}  // namespace torch
#define TORCH_CHECK(cond, ...) ((void)(cond))
