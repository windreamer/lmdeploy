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


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    values = (values.to(torch.int32) + 8) % 16
    unpacked = values.unflatten(-1, (-1, 8))
    return sum(unpacked[..., i] << (4 * i) for i in range(8)).to(torch.int32)


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
    from lmdeploy.pytorch.kernels.cuda.awq_modules import wq_gemm_forward

    return wq_gemm_forward(x, qweight, qzeros, scales, 4, 128, None, scales.size(1))


def _run_tm(x, qweight, scales, qzeros):
    from lmdeploy.pytorch.backends.cuda.turbomind_awq_modules import TurbomindAwqLinearW4A16Impl

    impl = TurbomindAwqLinearW4A16Impl(x.size(-1), scales.size(1), 4, 128)
    impl.update_weights(qweight, scales, qzeros)
    try:
        return impl.forward(x, qweight, scales, qzeros)
    finally:
        impl._release()


def _time(fn, iters=50, warmup=10):
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
    git_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    props = torch.cuda.get_device_properties(0)

    shapes = []
    for batch, k, n in SHAPES:
        qweight, scales, qzeros = _make_weights(k, n)
        x = torch.randn((batch, k), dtype=torch.float16, device='cuda')
        entry = {'batch': batch, 'in_features': k, 'out_features': n, 'impls': {}}
        entry['impls']['triton_ms'] = _time(lambda: _run_triton(x, qweight, scales, qzeros))
        try:
            entry['impls']['turbomind_ms'] = _time(lambda: _run_tm(x, qweight, scales, qzeros))
        except Exception as exc:  # gate-blocked or impl error
            entry['impls']['turbomind_error'] = str(exc)
        print(entry)
        shapes.append(entry)

    record = {
        'family': 'w4a16',
        'device': {
            'name': props.name,
            'capability': list(device),
            'multi_processor_count': props.multi_processor_count,
            'total_memory_gb': round(props.total_memory / 2**30, 1),
        },
        'git_sha': git_sha,
        'dtype': 'float16',
        'group_size': 128,
        'shapes': shapes,
    }
    if args.capability_note:
        record['notes'] = args.capability_note
    with open(args.output, 'w') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
