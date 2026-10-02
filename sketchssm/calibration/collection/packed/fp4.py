# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"Unpack individual NVFP4 tensors for the frozen-weight gradient estimator."

import torch

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_LUT = torch.cat([_E2M1, -_E2M1])


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """U8 [O, I/2] to float32 [O, I], with the low nibble first."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    lut = FP4_LUT.to(packed.device)
    out = torch.stack([lut[lo.long()], lut[hi.long()]], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
