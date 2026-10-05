"""Decode barrier + per-step timing for in-process vLLM (VLLM_ENABLE_V1_MULTIPROCESSING=0).

Profiling harness only; no change to vLLM or SketchSSM code paths.

* BarrierScheduler: normal chunked prefill; a request that has its first output token is held out of decode
  (next_decode_eligible_step = inf) until the armed wave's TARGET requests are all ready, so the whole cohort
  enters decode on the same step (rings aligned: W-1 non-flush steps, then one flush step).
* Runner wrapper (V1 and V2 GPUModelRunner): a pure-decode step with exactly TARGET requests is bracketed by
  the NVTX range ``decode/batch_size_<B>`` and by CUDA events (model-forward GPU time per step).
* V1 runner + ReplaySSM: a held request leaves and re-enters the persistent batch; re-anchor its ring origin
  at the prompt length (its state and KV stay resident while held).
"""
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

TARGET = None
ARMED = False
RELEASES = []
RECORDS = []
_PENDING = []
_HOLD = 1 << 60


def arm(batch):
    global TARGET, ARMED
    TARGET, ARMED = batch, True


def _ready(r):
    return (r.status == RequestStatus.RUNNING and r.num_in_flight_tokens == 0
            and r.num_computed_tokens >= r.num_prompt_tokens and 1 <= len(r.output_token_ids) < r.max_tokens)


class BarrierScheduler(Scheduler):
    def schedule(self, throttle_prefills: bool = False):
        global ARMED
        if ARMED:
            ready = [r for r in self.running if _ready(r)]
            if len(ready) >= TARGET:
                for r in self.running:
                    if r.next_decode_eligible_step == _HOLD:
                        r.next_decode_eligible_step = 0
                ARMED = False
                RELEASES.append(dict(reason='full_cohort', ready=len(ready), target=TARGET, t=time.time()))
            else:
                for r in ready:
                    r.next_decode_eligible_step = _HOLD
        out = super().schedule(throttle_prefills=throttle_prefills)
        if ARMED and out.total_num_scheduled_tokens == 0 and (self.waiting or self.skipped_waiting):
            ready = [r for r in self.running if _ready(r)]
            if ready and len(ready) == len(self.running):
                raise RuntimeError(f'capacity stall: {len(ready)} ready of target {TARGET}, '
                                   f'{len(self.waiting) + len(self.skipped_waiting)} waiting')
        return out


def _wrap(cls):
    import torch
    if getattr(cls, '_bench_wrapped', False):
        return
    orig = cls.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        counts = getattr(scheduler_output, 'num_scheduled_tokens', None) or {}
        if counts and TARGET is not None and len(counts) == TARGET and all(n == 1 for n in counts.values()):
            b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f'decode/batch_size_{len(counts)}')
            b.record(); t0 = time.perf_counter()
            try:
                return orig(self, scheduler_output, *args, **kwargs)
            finally:
                e.record(); t1 = time.perf_counter()
                torch.cuda.nvtx.range_pop()
                _PENDING.append((len(counts), t0, t1, b, e, type(self).__module__))
        return orig(self, scheduler_output, *args, **kwargs)

    cls.execute_model = execute_model
    cls._bench_wrapped = True


def _install():
    from vllm.v1.worker import gpu_model_runner as v1
    from vllm.v1.worker.gpu import model_runner as v2
    _wrap(v1.GPUModelRunner)
    _wrap(v2.GPUModelRunner)
    from vllm.v1.worker import gpu_input_batch
    ib = gpu_input_batch.InputBatch
    if not getattr(ib, '_bench_anchor', False):
        orig = ib.add_request

        def add_request(self, request, *a, **k):
            out = orig(self, request, *a, **k)
            if getattr(self, 'use_replayssm', False) and request.num_computed_tokens > 0:
                self.replayssm_decode_base[self.req_id_to_index[request.req_id]] = request.num_prompt_tokens
            return out

        ib.add_request = add_request
        ib._bench_anchor = True


_install()


def drain():
    import torch
    torch.cuda.synchronize()
    for batch, t0, t1, b, e, mod in _PENDING:
        RECORDS.append(dict(batch=batch, wall_start=t0, wall_end=t1, gpu_ms=b.elapsed_time(e), runner=mod))
    _PENDING.clear()
    out = list(RECORDS)
    RECORDS.clear()
    return out
