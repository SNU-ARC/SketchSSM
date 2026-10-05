"""Actual-model decode cohorts for Figure 8 (Nemotron 3 Super on one B300), the vLLM fork with SketchSSM.

One synchronized cohort of B random-token prompts (C tokens each, one distinct sequence per row, fixed seed),
prefilled normally, released into decode together (barrier.py), then --tokens output tokens (127 full-batch
decode steps after the first sampled token); per step = model-forward GPU time (CUDA events) averaged over
steps 16-111. One 32-token warm wave per engine precedes the measured waves. --endpoint adds the arm's capacity
endpoint: (KV blocks - 1) // blocks per request, // 8 * 8, at most --max-seqs. --profile brackets each measured
wave with cudaProfilerStart/Stop for `nsys profile --capture-range=cudaProfilerApi` (then trace.py).

Arms: standard | replayssm (use_replayssm, W=16) | sketch (sketchssm=<calibration>, sketchssm_mean_rank=G).
All arms run on Model Runner V2.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    'super': dict(path='/disk2/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4', calib='SketchSSM/Nemotron-3-Super-NVFP4', layers=40),
    'qwen': dict(path='/disk2/models/Qwen3.8-Flash-Next-NVFP4', calib='SketchSSM/Qwen3.8-Flash-Next-NVFP4', layers=36),
    'glm': dict(path='/disk2/models/GLM-5.3-Flash-NVFP4', calib='SketchSSM/GLM-5.3-Flash-NVFP4', layers=34),
}
W = 16


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=1)); tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--arm', choices=('standard', 'replayssm', 'sketch'), required=True)
    p.add_argument('--g', type=float, default=8)
    p.add_argument('--batches', default='128,256,512')
    p.add_argument('--max-seqs', type=int, default=0, help='max_num_seqs (default: max batch)')
    p.add_argument('--endpoint', action='store_true', help='add the capacity endpoint (block-pool bound, //8*8) and drop batches above it')
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--util', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--capacity-only', action='store_true')
    p.add_argument('--model-path', help='local checkpoint (default: the path in MODELS)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    # every arm on Model Runner V2
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    sys.path.insert(0, str(HERE))
    import torch
    import barrier
    from vllm import LLM, SamplingParams
    m = dict(MODELS[a.model], **({'path': a.model_path} if a.model_path else {}))
    batches = [int(x) for x in a.batches.split(',')]
    max_seqs = a.max_seqs or max(batches)
    sizes = sorted(set(batches) | ({*range(8, max_seqs + 1, 8)} if a.endpoint else set()) | {max_seqs})
    # max_model_len has no slack: at C=8K a +16 margin needs a second attention block per request in Standard
    kw = dict(model=m['path'], trust_remote_code=True, mamba_ssm_cache_dtype='float32', gpu_memory_utilization=a.util,
              enable_prefix_caching=False, max_model_len=a.context + a.tokens, max_num_batched_tokens=8192,
              max_num_seqs=max_seqs, compilation_config={'cudagraph_capture_sizes': sizes, 'max_cudagraph_capture_size': max_seqs},
              scheduler_cls='barrier.BarrierScheduler', disable_log_stats=True)
    if a.model != 'super':
        kw.update(language_model_only=True, limit_mm_per_prompt={'image': 0, 'video': 0})
    if a.arm == 'replayssm':
        kw.update(use_replayssm=True, replayssm_buffer_len=W)
    if a.arm == 'sketch':
        kw.update(sketchssm=m['calib'], sketchssm_mean_rank=a.g, replayssm_buffer_len=W)
    t0 = time.time()
    llm = LLM(**kw)
    init_s = time.time() - t0
    core = llm.llm_engine.engine_core.engine_core
    sched = core.scheduler
    runner = core.model_executor.driver_worker.worker.model_runner
    kvc = sched.kv_cache_config
    cfg = llm.llm_engine.vllm_config
    # count blocks for the actual request length (context + tokens), not max_model_len
    per_req = 0
    groups = []
    for g in kvc.kv_cache_groups:
        n = g.kv_cache_spec.max_num_blocks_per_req(cfg, a.context + a.tokens) if hasattr(g.kv_cache_spec, 'max_num_blocks_per_req') else None
        groups.append(dict(kind=type(g.kv_cache_spec).__name__, layers=len(g.layer_names), block_size=g.kv_cache_spec.block_size,
                           page_size_bytes=g.kv_cache_spec.page_size_bytes, blocks_per_request=n))
        per_req += n or 0
    cap = (kvc.num_blocks - 1) // per_req if per_req else None   # block 0 is the null block
    capacity = dict(num_blocks=kvc.num_blocks, blocks_per_request=per_req, block_pool_upper_bound=cap,
                    max_concurrency_reported=None, groups=groups, runner=type(runner).__module__,
                    free_gib=torch.cuda.mem_get_info()[0] / 2**30)
    print('CAPACITY', json.dumps(capacity), flush=True)
    meta = dict(model=a.model, model_path=m['path'], arm=a.arm, g=a.g if a.arm in ('sketch', 'control') else None,
                calibration=m['calib'] if a.arm in ('sketch', 'control') else None, context=a.context, tokens=a.tokens,
                util=a.util, seed=a.seed, init_s=init_s, capacity=capacity, llm_kwargs={k: v for k, v in kw.items() if k != 'compilation_config'},
                capture_sizes=sizes, env={k: v for k, v in os.environ.items() if k.startswith(('SKETCHSSM_', 'VLLM_'))})
    if a.capacity_only:
        save(a.out, dict(status='CAPACITY_ONLY', **meta)); return
    if a.endpoint:
        end = min(cap // 8 * 8, max_seqs)
        batches = sorted({b for b in batches if b <= end} | {end})
        print('ENDPOINT', end, batches, flush=True)
    vocab = cfg.model_config.get_vocab_size()
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(100, min(vocab, 150000) - 100, (max(batches), a.context), generator=g).tolist()

    def wave(batch, tokens):
        barrier.arm(batch)
        sp = SamplingParams(temperature=0, max_tokens=tokens, min_tokens=tokens, ignore_eos=True, seed=0)
        t = time.time()
        outs = llm.generate([dict(prompt_token_ids=r) for r in ids[:batch]], sp, use_tqdm=False)
        assert len(outs) == batch and all(len(o.outputs[0].token_ids) == tokens for o in outs)
        assert barrier.RELEASES and barrier.RELEASES[-1]['ready'] == batch, barrier.RELEASES[-1:]
        return outs, time.time() - t

    records = []
    _, wt = wave(batches[0], 32)
    print(f'WARM b{batches[0]} {wt:.1f}s', flush=True)
    barrier.drain()
    for batch in batches:
        for r in range(a.repeats):
            if a.profile:
                torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
            outs, wt = wave(batch, a.tokens)
            if a.profile:
                torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
            steps = barrier.drain()
            assert len(steps) == a.tokens - 1 and all(s['batch'] == batch for s in steps), (len(steps), batch)
            steady = steps[16:112]
            gpu_ms = sum(s['gpu_ms'] for s in steady) / len(steady)
            span = steady[-1]['wall_end'] - steady[0]['wall_start']
            summary = dict(batch=batch, repeat=r, steps=len(steps), steady_steps=len(steady), model_forward_ms_per_step=gpu_ms,
                           wall_ms_per_step=1000 * span / len(steady), tokens_per_s_forward=batch * 1000 / gpu_ms,
                           tokens_per_s_wall=batch * len(steady) / span, wave_s=wt)
            print('STEP', json.dumps(summary), flush=True)
            records.append(dict(summary=summary, first_tokens=[o.outputs[0].token_ids[:4] for o in outs[:4]], steps=steps))
            save(a.out, dict(status='RUNNING', **meta, records=records))
    save(a.out, dict(status='DONE', **meta, batches=batches, records=records,
                     prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
