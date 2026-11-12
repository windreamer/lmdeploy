# Copyright (c) OpenMMLab. All rights reserved.
"""Op-level benchmark for the blocked-FP8 linear family (SM80).

Collects evidence records for the TurboMind family 15 implementation and
the incumbent Triton implementation so a future factory-table row can be
decided by data. Usage:

    CUDA_VISIBLE_DEVICES=4 python benchmarks/ops/bench_blocked_fp8_linear.py \
        [--out benchmarks/ops/blocked_fp8_sm80.json]

Record schema (shared across tickets):
    family, impl, device, capability, shapes[{M,K,N,scale_layout}],
    timings_us{shape_key: us}, git_sha, timestamp, torch_version, triton_version
"""

import argparse
import json
import subprocess
import time

import torch

from lmdeploy.pytorch.backends.blockedf8_modules import LinearBlockedF8BuildSpec

SHAPES = [
    # (M, K, N)
    (1, 4096, 4096),
    (128, 4096, 11008),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
]
SCALE_LAYOUTS = [1, 128]
WARMUP = 10
ITERS = 50


def _git_sha() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _make_inputs(m, k, n, block_out, device):
    torch.manual_seed(0)
    weight = (torch.randn(n, k, device=device) * 0.05).to(torch.float8_e4m3fn).contiguous()
    k_groups = k // 128
    rows = n if block_out == 1 else n // 128
    scale = (torch.rand(rows, k_groups, device=device) * 0.01 + 0.005).contiguous()
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.5
    return x, weight, scale


def _time(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # us


def _build_tm(spec):
    from lmdeploy.pytorch.backends.cuda.blockedf8_modules import _build_linear_blocked_f8

    return _build_linear_blocked_f8(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='benchmarks/ops/blocked_fp8_sm80.json')
    args = parser.parse_args()

    # reach the TM implementation (opt-in): the bench records it explicitly
    import os

    os.environ['LMDEPLOY_TURBOMIND_OPS'] = 'force=blocked_fp8'

    from lmdeploy.pytorch.backends.cuda.blockedf8_modules import TritonLinearBlockedF8Impl

    capability = torch.cuda.get_device_capability()
    device = torch.cuda.get_device_name()
    records = []

    for block_out in SCALE_LAYOUTS:
        timings = {}
        for (m, k, n) in SHAPES:
            spec = LinearBlockedF8BuildSpec(in_features=k,
                                            out_features=n,
                                            block_size=128,
                                            bias=False,
                                            output_dtype=torch.bfloat16,
                                            fp8_dtype=torch.float8_e4m3fn,
                                            scale_fmt=None)
            x, weight, scale = _make_inputs(m, k, n, block_out, 'cuda')

            tm = _build_tm(spec)
            assert type(tm).__name__ == 'TurbomindLinearBlockedF8Impl'
            tm.update_weights(weight, scale)
            timings[f'M{m}_K{k}_N{n}_bo{block_out}_turbomind'] = _time(lambda: tm.forward(x, weight, scale))

            try:
                triton_impl = TritonLinearBlockedF8Impl(k, n, 128)
                timings[f'M{m}_K{k}_N{n}_bo{block_out}_triton'] = _time(
                    lambda: triton_impl.forward(x, weight, scale))
            except Exception as e:
                # triton fp8 requires cc >= 9 (fp8e4nv); record the gap
                timings[f'M{m}_K{k}_N{n}_bo{block_out}_triton'] = f'unavailable: {e}'

        records.append({
            'family': 'blocked_fp8',
            'impl': 'turbomind-family15-vs-triton',
            'device': device,
            'capability': list(capability),
            'shapes': [{
                'M': m,
                'K': k,
                'N': n,
                'scale_layout': f'128x{block_out}',
            } for (m, k, n) in SHAPES],
            'timings_us': timings,
            'git_sha': _git_sha(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'torch_version': torch.__version__,
            'triton_version': __import__('triton').__version__,
        })

    with open(args.out, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
