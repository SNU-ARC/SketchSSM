# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Native covariance binding for mamba2; unsupported engine ABIs fail explicitly."""

import torch

from ..observers.mamba2 import Accumulator
from .common import CovarianceWorker


class Worker(CovarianceWorker):
    def install_covariance(self, config, resume=None):
        self.configure(config)
        import vllm.model_executor.layers.mamba.mamba_mixer2 as module

        model = self.model_runner.get_model()
        self._layer_ids = []
        self._cov = []
        rows = []
        active = []
        self._request_indices = {}
        for name, m in model.named_modules():
            if type(m).__name__ != "MambaMixer2":
                continue
            assert not getattr(m, "use_replayssm", False), name
            assert getattr(m, "sketchssm", None) is None, name
            assert len(m.kv_cache) == 2, name
            self.register_layer(name, m)
            state = m.kv_cache[1]
            assert state.dtype == torch.float32 and tuple(state.shape[2:]) == (
                self.V,
                self.K,
            ), (name, state.shape, state.dtype)
            a = Accumulator(
                self.slots,
                state.shape[1],
                self.K,
                state.device,
                window=self.W,
                tokens=self.tokens,
            )
            self._cov.append(a)
            # vLLM may share the page storage between layer views; a storage
            # pointer does not identify a layer. Bind to the actual mixer call.
            original_conv = m.conv_ssm_forward

            def conv(*args, _original=original_conv, _acc=a, **kwargs):
                active.append(_acc)
                try:
                    return _original(*args, **kwargs)
                finally:
                    active.pop()

            m.conv_ssm_forward = conv
            rows.append(
                dict(
                    name=name,
                    state_shape=list(state.shape),
                    state_dtype=str(state.dtype),
                    state_ptr=state.data_ptr(),
                )
            )
        assert rows, "No supported Mamba2 layers found"
        if resume:
            d = torch.load(resume, map_location="cpu", weights_only=True)
            if d.get("layer_ids") != self._layer_ids:
                raise ValueError("Native recurrent layer order changed")
            for i, a in enumerate(self._cov):
                a.E.copy_(d["head_scov"][i])
                a.C.copy_(d["head_qcov"][i])
                a.windows = int(d["head_cov_windows"][i])
                a.queries = int(d["head_cov_queries"][i])
        original = module.selective_state_update

        def observe(state, x, dt, A, B, C, D, dt_bias, *args, **kw):
            assert C.shape[1] == self.groups, "Native group count differs from config"
            assert not args and kw.get("dt_softplus") is True
            slots = kw["state_batch_indices"].reshape(-1).long()
            dst = kw["dst_state_batch_indices"].reshape(-1).long()
            assert slots.numel() == x.shape[0] and torch.equal(slots, dst)
            assert bool((slots >= 0).all()) and slots.unique().numel() == slots.numel()
            assert kw.get("num_accepted_tokens") is None
            assert len(active) == 1
            # Physical pages can be recycled within one generate() batch.
            # Track decay/window phase by persistent request identity, and
            # check it against the engine's actual input-token positions.
            request_ids = self.model_runner.input_batch.req_ids[: x.shape[0]]
            indices = []
            for request_id in request_ids:
                assert request_id is not None
                if request_id not in self._request_indices:
                    self._request_indices[request_id] = len(self._request_indices)
                indices.append(self._request_indices[request_id])
            logical = torch.tensor(indices, device=slots.device, dtype=torch.long)
            positions = (
                self.model_runner.positions[: x.shape[0]].long() - self.prompt_tokens
            )
            active[0].observe(
                state,
                dt[:, :, 0],
                A[:, 0, 0],
                C,
                dt_bias[:, 0],
                slots,
                logical,
                positions,
            )
            return original(state, x, dt, A, B, C, D, dt_bias, **kw)

        module.selective_state_update = observe
        self._cov_original = original
        quant = {}
        packed_bytes = 0
        for name, m in model.named_modules():
            method = getattr(m, "quant_method", None)
            if method is not None:
                key = type(method).__name__
                quant[key] = quant.get(key, 0) + 1
            for p in m.parameters(recurse=False):
                if p.dtype == torch.uint8:
                    packed_bytes += p.numel()
        self.check_native_weights(packed_bytes, quant)
        return dict(
            mixers=rows,
            quant_methods=quant,
            packed_uint8_bytes=packed_bytes,
            weights="original native checkpoint",
            replay=False,
            sketch=False,
        )
