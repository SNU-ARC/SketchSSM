# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from sketchssm.calibration.collection.config import read_config
from sketchssm.calibration.collection.observers import gdn, kda, mamba2
from sketchssm.calibration.collection.pipeline import run
from sketchssm.calibration.collection.storage import atomic_torch

ROOT = Path(__file__).resolve().parents[1]


class CollectionTests(unittest.TestCase):
    def test_readout_hook_accepts_positional_and_keyword_hidden_states(self):
        from sketchssm.calibration.collection.hooks.mamba2 import JointCollector

        class Mixer(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states + 1

        collector = JointCollector.__new__(JointCollector)
        collector.pending_inputs = [None]
        mixer = Mixer()
        handle = mixer.register_forward_pre_hook(
            collector._mixer_pre_hook(0), with_kwargs=True
        )
        x = torch.randn(1, 8, 4, requires_grad=True)
        try:
            for keyword in (False, True):
                collector.pending_inputs[0] = None
                output = mixer(hidden_states=x) if keyword else mixer(x)
                torch.testing.assert_close(collector.pending_inputs[0], x)
                self.assertFalse(collector.pending_inputs[0].requires_grad)
                torch.testing.assert_close(output, x + 1)
        finally:
            handle.remove()

    def test_native_observers_against_direct_boundary_recurrence(self):
        torch.manual_seed(10)
        H, V, K, T, W = 4, 3, 4, 8, 4
        initial = torch.randn(H, V, K)
        for family in (mamba2, gdn, kda):
            acc = family.Accumulator(1, H, K, "cpu", window=W, tokens=T)
            state = torch.zeros(3, H, V, K)
            state[1] = initial
            expected_e = torch.zeros(H, K, K, dtype=torch.double)
            expected_c = expected_e.clone()
            transition = torch.eye(K).expand(H, K, K).clone()
            for t in range(T):
                physical = 1 if t < 6 else 2
                if t == 6:
                    state[2] = state[1]  # A request moves to another physical page.
                q, key = torch.randn(H, K), torch.randn(H, K)
                key = key / key.norm(dim=-1, keepdim=True)
                beta = torch.rand(H)
                decay = (
                    torch.rand(H) * 0.1 + 0.8
                    if family != kda
                    else torch.rand(H, K) * 0.1 + 0.8
                )
                if t % W == 0:
                    boundary = state[physical].clone()
                    transition = torch.eye(K).expand(H, K, K).clone()
                    if t >= W:
                        expected_e += torch.einsum(
                            "hvk,hvn->hkn", boundary, boundary
                        ).double()
                if family == mamba2:
                    # softplus(0) * A = log(decay)
                    A = decay.log() / torch.nn.functional.softplus(torch.tensor(0.0))
                    acc.observe(
                        state,
                        torch.zeros(1, H),
                        A,
                        q[None],
                        torch.zeros(H),
                        torch.tensor([physical]),
                        torch.tensor([0]),
                        torch.tensor([t]),
                    )
                    step = torch.diag_embed(decay[:, None].expand(-1, K))
                else:
                    if family == kda:
                        # This observer uses a fixed physical mapping for one homogeneous batch.
                        # Physical relocation is validated by the logical-ID GDN/Mamba observers.
                        acc.observe(
                            state,
                            q[None],
                            key[None],
                            decay[None],
                            beta[None],
                            torch.tensor([0]),
                        )
                    else:
                        acc.observe(
                            state,
                            q[None],
                            key[None],
                            decay[None],
                            beta[None],
                            torch.tensor([physical]),
                            torch.tensor([0]),
                            torch.tensor([t]),
                        )
                    erase = (
                        torch.eye(K)
                        - beta[:, None, None] * key[:, :, None] * key[:, None, :]
                    )
                    step = (
                        torch.diag_embed(decay) @ erase
                        if family == kda
                        else decay[:, None, None] * erase
                    )
                transition = transition @ step
                effective = (transition @ q[..., None]).squeeze(-1)
                if t >= W:
                    expected_c += torch.einsum(
                        "hk,hn->hkn", effective, effective
                    ).double()
                state[physical] = state[physical] @ step + torch.randn(H, V, K) * 0.03
                state[0] = state[physical]  # KDA single stable logical slot.
            torch.testing.assert_close(acc.C, expected_c, rtol=3e-6, atol=3e-6)
            torch.testing.assert_close(acc.E, expected_e, rtol=3e-6, atol=3e-6)
            self.assertEqual(acc.windows, 1)
            self.assertEqual(acc.queries, 4)

    def test_packed_linear_input_gradient_and_storage(self):
        from sketchssm.calibration.collection.packed.qwen import QLinearFn, dequant
        from sketchssm.calibration.collection.packed.super import (
            PackedLinearFn,
            weight_view,
        )

        torch.manual_seed(9)
        w = torch.randint(0, 256, (5, 16), dtype=torch.uint8)
        scale, global_scale = (
            torch.rand(5, 2).to(torch.float8_e4m3fn),
            torch.tensor(0.8),
        )
        for fn, view in ((PackedLinearFn, weight_view), (QLinearFn, dequant)):
            x = torch.randn(3, 32).bfloat16().requires_grad_()
            upstream = torch.randn(3, 5).bfloat16()
            output = fn.apply(x, w, scale, global_scale)
            self.assertEqual(
                [t.dtype for t in output.grad_fn.saved_tensors],
                [w.dtype, scale.dtype, global_scale.dtype],
            )
            output.backward(upstream)
            reference = view(w, scale, global_scale)
            torch.testing.assert_close(
                output,
                torch.nn.functional.linear(x.detach(), reference),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(x.grad, upstream @ reference, rtol=0, atol=0)

    def test_partial_stage_resume_rejects_modified_inputs(self):
        cfg = yaml.safe_load(
            (
                ROOT / "sketchssm/calibration/example/nemotron_super/config.yaml"
            ).read_text()
        )
        cfg["model"]["checkpoint"] = "test/original-checkpoint"
        cfg["runtime"] = {"batch_size": 2}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, out = root / "input.yaml", root / "output"
            path.write_text(yaml.safe_dump(cfg))
            calls = []

            def fake(cmd, **kwargs):
                calls.append(cmd)
                stage = cmd[cmd.index("--stage") + 1]
                if stage == "generate":
                    for name in ("generation_tokens", "allocation_tokens"):
                        atomic_torch({"ids": torch.arange(5)}, out / f"data/{name}.pt")
                else:
                    atomic_torch(
                        {"cov": torch.eye(3)}, out / "statistics/covariance.pt"
                    )

            with patch(
                "sketchssm.calibration.collection.pipeline.subprocess.run",
                side_effect=fake,
            ):
                run(path, out, stage="generate")
                run(path, out, stage="generate", resume=True)
                self.assertEqual(len(calls), 1)
                run(path, out, stage="covariance", resume=True)
                atomic_torch({"changed": True}, out / "data/generation_tokens.pt")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    run(path, out, stage="covariance", resume=True)

    def test_covariance_trace_binding_and_sentinel(self):
        import sys
        from types import SimpleNamespace

        from sketchssm.calibration.collection.engine import covariance

        cfg = yaml.safe_load(
            (ROOT / "sketchssm/calibration/example/glm_flash/config.yaml").read_text()
        )
        cfg["runtime"] = {"batch_size": 2, "teacher_force": "trace_decode"}
        cfg["generation"]["new_tokens"] = 4
        seen = []
        closed = []

        class Engine:
            llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=lambda: closed.append(True))
            )

            def collective_rpc(self, name, args=()):
                seen.append((name, args))
                return [{"status": "ok"}]

            def generate(self, inputs, params, **kwargs):
                result = []
                for inp, param in zip(inputs, params):
                    self_test.assertEqual(param.trace_decode_token_ids[-1], 0)
                    self_test.assertEqual(
                        len(param.trace_decode_token_ids), param.max_tokens
                    )
                    result.append(
                        SimpleNamespace(
                            prompt_token_ids=inp["prompt_token_ids"],
                            outputs=[
                                SimpleNamespace(token_ids=param.trace_decode_token_ids)
                            ],
                        )
                    )
                return result

        self_test = self
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_torch(
                {
                    "prompt_token_ids": torch.tensor([[5, 6], [7, 8]]),
                    "generated_token_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
                },
                root / "data/generation_tokens.pt",
            )
            with (
                patch.dict(
                    sys.modules,
                    {
                        "vllm": SimpleNamespace(
                            SamplingParams=lambda **kw: SimpleNamespace(**kw)
                        )
                    },
                ),
                patch(
                    "sketchssm.calibration.collection.engine.create_engine",
                    return_value=Engine(),
                ),
            ):
                covariance(cfg, root, "identity")
        self.assertEqual(
            [x[0] for x in seen],
            ["install_covariance", "reset_covariance_slots", "save_covariance"],
        )
        self.assertEqual(closed, [True])

    def test_invalid_protocol_rejected_before_engine(self):
        cfg = yaml.safe_load(
            (
                ROOT / "sketchssm/calibration/example/nemotron_super/config.yaml"
            ).read_text()
        )
        cfg["model"]["checkpoint"] = "test/model"
        cfg["paired"]["objective"] = "p4"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "Full-Gram"):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
