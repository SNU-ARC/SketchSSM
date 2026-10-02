# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Native covariance binding for kda; unsupported engine ABIs fail explicitly."""

import torch

from ..observers.kda import Accumulator, factors
from .common import CovarianceWorker


class Worker(CovarianceWorker):
    def install_covariance(self, config, resume=None):
        self.configure(config)
        import vllm.models.glm5next.nvidia.kda as module

        self._cov = []
        self._layer_ids = []
        active = []
        for name, m in self.get_model().named_modules():
            if type(m).__name__ != "Glm5NextLinearAttention":
                continue
            state = m.kv_cache[1]
            assert state.dtype == torch.float32 and tuple(state.shape[2:]) == (
                self.V,
                self.K,
            )
            assert not getattr(m, "use_replayssm", False)
            assert getattr(m, "sketchssm", None) is None
            acc = Accumulator(
                self.slots,
                state.shape[1],
                self.K,
                state.device,
                window=self.W,
                tokens=self.tokens,
                physical_slots=state.shape[0],
            )
            self._cov.append(acc)
            self._layer_ids.append(m.layer_idx)
            original = m._forward

            def core(*args, _original=original, _acc=acc, _module=m, **kw):
                active.append((_acc, _module))
                try:
                    return _original(*args, **kw)
                finally:
                    active.pop()

            m._forward = core
        assert self._cov, "No supported KDA layers found"
        if resume:
            d = torch.load(resume, map_location="cpu", weights_only=True)
            assert d["layer_ids"] == self._layer_ids
            for i, a in enumerate(self._cov):
                a.E.copy_(d["head_scov"][i])
                a.C.copy_(d["head_qcov"][i])
                a.windows = int(d["head_cov_windows"][i])
                a.queries = int(d["head_cov_queries"][i])
        original = module.fused_recurrent_kda

        def observe(*args, **kw):
            assert not args and len(active) == 1
            assert (
                kw.get("compute_gate")
                and kw.get("sigmoid_beta")
                and kw.get("use_qk_l2norm_in_kernel")
            )
            assert kw.get("lower_bound") == -5 and kw.get("num_accepted_tokens") is None
            q = kw["q"]
            k = kw["k"]
            B = q.shape[1]
            H = active[0][0].E.shape[0]
            assert q.shape == (1, B, H, self.K)
            cu = kw["cu_seqlens"]
            assert cu.numel() == B + 1 and bool(((cu[1:] - cu[:-1]) == 1).all()), (
                "Only pure single-token decode is observable"
            )
            physical = kw["ssm_state_indices"].reshape(-1)[:B].long()
            q, k, alpha, beta = factors(
                q[0],
                k[0],
                kw["g"].reshape(B, H, self.K),
                kw["beta"].reshape(B, H),
                kw["a_log"],
                kw["g_bias"],
            )
            active[0][0].observe(kw["initial_state"], q, k, alpha, beta, physical)
            return original(**kw)

        module.fused_recurrent_kda = observe
        original_chunk = module.chunk_kda_with_fused_gate

        def observe_chunk(*args, **kw):
            assert not args and len(active) == 1
            acc, m = active[0]
            H = acc.E.shape[0]
            metadata = module.get_forward_context().attn_metadata[m.prefix]
            B = metadata.num_decodes
            if B:
                assert metadata.num_spec_decodes == 0
                cu = kw["cu_seqlens"]
                assert bool(((cu[1 : B + 1] - cu[:B]) == 1).all())
                assert (
                    kw.get("safe_gate")
                    and kw.get("lower_bound") == -5
                    and kw.get("use_qk_l2norm_in_kernel")
                )
                q = kw["q"][0, :B]
                k = kw["k"][0, :B]
                q, k, alpha, _ = factors(
                    q,
                    k,
                    kw["raw_g"].reshape(-1, H, self.K)[:B],
                    torch.zeros(B, H, device=q.device),
                    kw["A_log"],
                    kw["g_bias"],
                )
                beta = kw["beta"].reshape(-1, H)[:B].float()
                physical = metadata.non_spec_state_indices_tensor.reshape(-1)[:B].long()
                acc.observe(m.kv_cache[1], q, k, alpha, beta, physical)
            return original_chunk(**kw)

        module.chunk_kda_with_fused_gate = observe_chunk
        self.audit_model(self.get_model())
        return dict(
            native_forward=True,
            covariance="native state; ordered D_t(I-beta k kT); no rotated-state assumption",
            layer_ids=self._layer_ids,
        )
