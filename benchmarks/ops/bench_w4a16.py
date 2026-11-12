# Copyright (c) OpenMMLab. All rights reserved.
"""Minimal op-level benchmark driver for the W4A16 AWQ linear family.

Compares the triton and TurboMind implementations on gate-passing shapes
and writes a benchmark record JSON under benchmarks/ops/.

Usage (GPU 4 by house rule):
    CUDA_VISIBLE_DEVICES=4 python benchmarks/ops/bench_w4a16.py \
        --output benchmarks/ops/w4a16_sm80.json
"""
import argparse
import json
import subprocess
import time

import torch


AWQ_GEMM_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 into the AWQ-gemm int32 layout (see the numeric
    tests for the interleave contract shared by triton and TurboMind)."""
    codes = (values.to(torch.int32) + 8) % 16
    grouped = codes.unflatten(-1, (-1, 8))
    packed = torch.empty_like(grouped)
    for nibble in range(8):
        packed[..., nibble] = grouped[..., AWQ_GEMM_ORDER.index(nibble)]
    out = torch.zeros(packed.shape[:-1], dtype=torch.int32)
    for nibble in range(8):
        out |= packed[..., nibble] << (4 * nibble)
    return out


def _make_weights(in_features, out_features, group_size=128, seed=0):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    scales = (torch.randn((in_features // group_size, out_features), generator=generator) * 0.05 + 1.0).half()
    q4 = torch.randint(-8, 8, (in_features, out_features), generator=generator)
    z4 = torch.randint(-8, 8, (in_features // group_size, out_features), generator=generator)
    qweight = _pack_int4(q4).cuda()
    qzeros = _pack_int4(z4).cuda()
    scales = scales.cuda()
    return qweight, scales, qzeros


def _run_triton(x, qweight, scales, qzeros):
    from lmdeploy.pytorch.backends.cuda.awq_modules import wq_gemm_forward

    return wq_gemm_forward(x, qweight, qzeros, scales, 4, 128, None, scales.size(1))


def _make_tm_impl(qweight, scales, qzeros):
    from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl

    impl = TurbomindAwqLinearW4A16Impl(scales.shape[0] * 128, scales.size(1), 4, 128)
    impl.update_weights(qweight, scales, qzeros)
    return impl


def _time(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3  # ms


SHAPES = [
    # (batch, in_features, out_features)
    (1, 4096, 4096),
    (8, 4096, 4096),
    (64, 4096, 4096),
    (1, 4096, 11008),
    (8, 4096, 11008),
    (1, 512, 1024),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--capability-note', default=None)
    args = parser.parse_args()

    device = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name(0)
    git_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    timestamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')

    timings = {'triton': {}, 'turbomind': {}}
    shapes = []
    for batch, k, n in SHAPES:
        qweight, scales, qzeros = _make_weights(k, n)
        x = torch.randn((batch, k), dtype=torch.float16, device='cuda')
        shapes.append({'M': batch, 'K': k, 'N': n})

        impl = _make_tm_impl(qweight, scales, qzeros)
        key = f'M{batch}_K{k}_N{n}'
        timings['turbomind'][key] = _time(lambda: impl.forward(x, qweight, scales, qzeros)) * 1000
        impl._release()
        timings['triton'][key] = _time(lambda: _run_triton(x, qweight, scales, qzeros)) * 1000
        print(key, timings['turbomind'][key], timings['triton'][key])

    records = [
        {
            'family': 'w4a16',
            'impl': impl_name,
            'device': device_name,
            'capability': list(device),
            'shapes': shapes,
            'timings_us': timings[impl_name],
            'git_sha': git_sha,
            'timestamp': timestamp,
        }
        for impl_name in ('triton', 'turbomind')
    ]
    with open(args.output, 'w') as handle:
        json.dump(records, handle, indent=2)
        handle.write('\n')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
