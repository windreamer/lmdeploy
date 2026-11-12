#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Op-level benchmark for the gated-delta-rule family: existing FLA/TileLang
stack vs the opt-in TurboMind native stack, on the prefill (chunked) and
decode (recurrent) engine paths.

Usage (A100 / pre_sm90 bucket, GPU 4):
  CUDA_VISIBLE_DEVICES=4 python benchmarks/ops/bench_gated_delta_rule_tm.py \
      --output benchmarks/ops/gated_delta_rule_pre_sm90.json

Selection behavior mirrors production builds: impl is chosen through
LMDEPLOY_TURBOMIND_OPS via the same _build_gated_delta_rule seam the engine
uses (force=gated_delta -> TM, auto -> FLA/TileLang).
"""
import argparse
import json
import os
import subprocess
import time

import torch

from lmdeploy.pytorch.model_inputs import StepContextManager, set_step_ctx_manager

set_step_ctx_manager(StepContextManager())

SHAPES = [
    # (name, lengths, kv_lengths, state_ids)
    ('prefill_short', (5, 9, 1), (0, 0, 0), [2, 5, 7]),
    ('prefill_long', (1, 256, 1024), (3, 65, 100), [1, 3, 8]),
    ('decode_b8', (8, ), None, [2, 5, 7, 1, 3, 8, 4, 6]),
    ('decode_b64', (64, ), None, list(range(4, 68))),
]

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 2, 4, 128
WARMUP, ITERS = 5, 20


def build_inputs(lengths, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    total = sum(lengths)
    t = (lambda *s: ((torch.rand(*s, generator=gen, device=device) - 0.5) * 2).to(torch.bfloat16))
    return (t(1, total, NUM_K_HEADS, HEAD_DIM), t(1, total, NUM_K_HEADS, HEAD_DIM),
            t(1, total, NUM_V_HEADS, HEAD_DIM),
            torch.rand(1, total, NUM_K_HEADS, NUM_V_HEADS // NUM_K_HEADS, generator=gen, device=device).to(torch.bfloat16),
            torch.rand(1, total, NUM_K_HEADS, NUM_V_HEADS // NUM_K_HEADS, generator=gen, device=device).to(torch.bfloat16),
            torch.rand(NUM_V_HEADS, generator=gen, device=device) * 0.2,
            -torch.rand(NUM_V_HEADS, generator=gen, device=device) * 2)


def build_rule(tm: bool):
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import _build_gated_delta_rule
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaRuleBuildSpec
    old = os.environ.get('LMDEPLOY_TURBOMIND_OPS')
    os.environ['LMDEPLOY_TURBOMIND_OPS'] = 'force=gated_delta' if tm else 'auto'
    try:
        return _build_gated_delta_rule(GatedDeltaRuleBuildSpec(indexed_state=False))
    finally:
        if old is None:
            os.environ.pop('LMDEPLOY_TURBOMIND_OPS', None)
        else:
            os.environ['LMDEPLOY_TURBOMIND_OPS'] = old


def make_meta(lengths, kv_lengths, state_ids, device):
    from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaMeta
    total = sum(lengths)
    decoding = kv_lengths is None
    attn = type('Attn', (), {})()
    attn.is_decoding = decoding
    batch = len(state_ids)
    if decoding:
        cu = torch.arange(batch + 1, device=device, dtype=torch.int32)
        attn.q_seqlens = torch.ones(batch, device=device, dtype=torch.int32)
        attn.kv_seqlens = torch.full((batch, ), 3, device=device, dtype=torch.int32)
    else:
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)),
                          device=device,
                          dtype=torch.int32)
        attn.q_seqlens = torch.tensor(lengths, device=device, dtype=torch.int32)
        attn.kv_seqlens = torch.tensor(kv_lengths, device=device, dtype=torch.int32)
    attn.cu_seqlens_q = cu
    return GatedDeltaMeta(total, 4, torch.tensor(state_ids, device=device), attn)


def bench_case(rule, lengths, kv_lengths, state_ids, device):
    q, k, v, b, a, dt_bias, a_log_exp = build_inputs(lengths, device)
    meta = make_meta(lengths, kv_lengths, state_ids, device)
    pool = torch.zeros(80, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device)

    def run():
        rule.forward(q, k, v, b, a, dt_bias, a_log_exp, pool, meta, 2, True)

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(ITERS):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) / 1000.0)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--iters', type=int, default=ITERS)
    args = parser.parse_args()
    device = 'cuda'
    capability = list(torch.cuda.get_device_capability())
    props = torch.cuda.get_device_properties(0)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

    impl_names = {'fla_tilelang': build_rule(False), 'turbomind': build_rule(True)}
    timings = {}
    errors = {}
    for name, impl in impl_names.items():
        for case, lengths, kv_lengths, state_ids in SHAPES:
            try:
                timings[f'{name}:{case}'] = bench_case(impl, lengths, kv_lengths, state_ids,
                                                       device)
            except Exception as exc:  # noqa: BLE001 - record and continue
                errors[f'{name}:{case}'] = f'{type(exc).__name__}: {exc}'

    record = {
        'device': props.name,
        'capability': capability,
        'shapes': [{
            'case': case,
            'lengths': list(lengths),
            'kv_lengths': list(kv_lengths) if kv_lengths else None,
            'num_k_heads': NUM_K_HEADS,
            'num_v_heads': NUM_V_HEADS,
            'head_dim': HEAD_DIM,
        } for case, lengths, kv_lengths, state_ids in SHAPES],
        'impl': list(impl_names),
        'timings': timings,
        'iters': args.iters,
        'warmup': WARMUP,
        'git_sha': sha,
        'gate_bucket': ('pre_sm90' if capability[0] < 9 else
                        'sm90' if capability[0] == 9 else 'sm120'),
        'errors': errors,
        'notes': ('Selection via LMDEPLOY_TURBOMIND_OPS through the engine build seam; '
                  'schema self-built by ticket 03 pending ticket-02 harness merge. '
                  'fla_tilelang decode uses the TileLang recurrent kernel, which fails '
                  'to JIT-compile under nvcc 13.0 in this environment (pre-existing); '
                  'those entries land in errors.'),
    }
    with open(args.output, 'w') as f:
        json.dump([record], f, indent=2)
    for key, values in timings.items():
        print(f'{key}: median={sorted(values)[len(values) // 2] * 1e3:.3f} ms')


if __name__ == '__main__':
    main()
