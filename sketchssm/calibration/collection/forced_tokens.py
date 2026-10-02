# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""Teacher-force the immutable, previously generated native NVFP4 corpus."""

from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor,
    MoveDirectionality,
)


class SavedTokens(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.entries = {}

    def is_argmax_invariant(self):
        return False

    @classmethod
    def validate_params(cls, params):
        tokens = (params.extra_args or {}).get("saved_tokens")
        if tokens is not None and (
            not tokens or not all(isinstance(t, int) and t >= 0 for t in tokens)
        ):
            raise ValueError("Expected a nonempty list of saved generated token IDs")

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for i in batch_update.removed:
            self.entries.pop(i, None)
        for i, params, prompt, output in batch_update.added:
            tokens = (params.extra_args or {}).get("saved_tokens")
            if tokens is None:
                self.entries.pop(i, None)
            else:
                self.entries[i] = (tokens, output)
        for a, b, direction in batch_update.moved:
            av = self.entries.pop(a, None)
            bv = self.entries.pop(b, None)
            if av is not None:
                self.entries[b] = av
            if bv is not None and direction == MoveDirectionality.SWAP:
                self.entries[a] = bv

    def apply(self, logits):
        # One extra output makes the last saved token enter the recurrent update.
        # This sentinel is not consumed or included in covariance.
        for row, (tokens, output) in self.entries.items():
            n = len(output)
            if n < len(tokens):
                logits[row].fill_(-float("inf"))
                logits[row, tokens[n]] = 0.0
        return logits
