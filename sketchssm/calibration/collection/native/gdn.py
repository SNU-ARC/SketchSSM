# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Native covariance binding for gdn; unsupported engine ABIs fail explicitly."""

import torch

from ..observers.gdn import Accumulator, factors
from .common import CovarianceWorker


class Worker(CovarianceWorker):
    def install_covariance(self, config, resume=None):
        self.configure(config)
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as module

        self._layer_ids = []
        self._cov = []
        self._request_indices = {}
        active = []
        rows = []
        model = self.model_runner.get_model()
        for name, m in model.named_modules():
            if not (
                hasattr(m, "gdn_decode_kernel")
                and hasattr(m, "A_log")
                and hasattr(m, "_forward_core")
            ):
                continue
            assert not getattr(m, "use_replayssm", False)
            assert getattr(m, "sketchssm", None) is None
            assert len(m.kv_cache) == 2
            self.register_layer(name, m)
            state = m.kv_cache[1]
            assert state.dtype == torch.float32 and tuple(state.shape[2:]) == (
                self.V,
                self.K,
            ), (name, state.shape, state.dtype)
            acc = Accumulator(
                self.slots,
                state.shape[1],
                self.K,
                state.device,
                window=self.W,
                tokens=self.tokens,
            )
            self._cov.append(acc)
            original = m._forward_core

            def core(*args, _original=original, _acc=acc, **kw):
                active.append(_acc)
                try:
                    return _original(*args, **kw)
                finally:
                    active.pop()

            m._forward_core = core
            rows.append(
                dict(
                    name=name,
                    state_shape=list(state.shape),
                    state_dtype=str(state.dtype),
                    decode=m.gdn_decode_kernel,
                )
            )
        assert rows, "No supported GDN layers found"
        if resume:
            d = torch.load(resume, map_location="cpu", weights_only=True)
            if d.get("layer_ids") != self._layer_ids:
                raise ValueError("Native recurrent layer order changed")
            for i, a in enumerate(self._cov):
                a.E.copy_(d["head_scov"][i])
                a.C.copy_(d["head_qcov"][i])
                a.windows = int(d["head_cov_windows"][i])
                a.queries = int(d["head_cov_queries"][i])

        def observe(kw, q, k, rounded_beta):
            assert len(active) == 1
            B = q.shape[0]
            a = kw["a"].reshape(B, -1)
            b = kw["b"].reshape(B, -1)
            physical = kw["ssm_state_indices"].reshape(-1)[:B].long()
            assert kw.get("use_qk_l2norm_in_kernel") is True
            logical = []
            for req in self.model_runner.input_batch.req_ids[:B]:
                assert req is not None
                if req not in self._request_indices:
                    self._request_indices[req] = len(self._request_indices)
                logical.append(self._request_indices[req])
            logical = torch.tensor(logical, device=physical.device, dtype=torch.long)
            q, k, alpha, beta = factors(
                q,
                k,
                a,
                b,
                kw["A_log"],
                kw["dt_bias"],
                kw.get("scale") or self.K**-0.5,
                rounded_beta,
            )
            positions = self.model_runner.positions[:B].long() - self.prompt_tokens
            active[0].observe(
                kw["initial_state"], q, k, alpha, beta, physical, logical, positions
            )

        packed = module.fused_recurrent_gated_delta_rule_packed_decode

        def packed_observer(*args, **kw):
            assert not args
            x = kw["mixed_qkv"]
            B = x.shape[0]
            assert (
                x.shape[-1] == 2 * self.groups * self.K + active[0].E.shape[0] * self.V
            )
            observe(
                kw,
                x[:, : self.groups * self.K].reshape(B, self.groups, self.K),
                x[:, self.groups * self.K : 2 * self.groups * self.K].reshape(
                    B, self.groups, self.K
                ),
                True,
            )
            return packed(**kw)

        generic = module.fused_sigmoid_gating_delta_rule_update

        def generic_observer(*args, **kw):
            assert not args
            q = kw["q"]
            k = kw["k"]
            B = q.numel() // (self.groups * self.K)
            assert kw["inplace_final_state"] is True
            cu = kw.get("cu_seqlens")
            assert (
                cu is not None
                and cu.numel() == B + 1
                and bool(((cu[1:] - cu[:-1]) == 1).all())
            ), "Only one-token dense decode is observable"
            observe(
                kw,
                q.reshape(B, self.groups, self.K),
                k.reshape(B, self.groups, self.K),
                False,
            )
            return generic(**kw)

        module.fused_recurrent_gated_delta_rule_packed_decode = packed_observer
        module.fused_sigmoid_gating_delta_rule_update = generic_observer
        quant = {}
        packed_bytes = 0
        for m in model.modules():
            method = getattr(m, "quant_method", None)
            if method is not None:
                quant[type(method).__name__] = quant.get(type(method).__name__, 0) + 1
            for p in m.parameters(recurse=False):
                if p.dtype == torch.uint8:
                    packed_bytes += p.numel()
        self.check_native_weights(packed_bytes, quant)
        return dict(
            mixers=rows,
            quant_methods=quant,
            packed_uint8_bytes=packed_bytes,
            native_forward=True,
            replay=False,
            sketch=False,
            covariance="native state and full ordered GDN erase/query transition",
        )
