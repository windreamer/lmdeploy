# Copyright (c) OpenMMLab. All rights reserved.
"""Op-level benchmark for the TurboMind RMSNorm / SiluAndMul kernels vs
the Triton implementations (ticket 04 evidence).

Writes evidence JSON records under ``benchmarks/ops/``:

    {
      "family": ..., "device": ..., "capability": ...,
      "impl": ..., "shapes": [...], "dtype": ...,
      "timings_us": {...}, "git_sha": ..., "timestamp": ...
    }

Usage (GPU 4 per tm-kernel-integration notes):
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python \\
        benchmarks/ops/bench_rmsnorm_siluandmul_tm.py
"""
import argparse
import datetime
import json
import os
import platform
import subprocess

import torch


def _git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def _bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def _bench_rms_norm(device, cap):
    from lmdeploy.pytorch.backends.cuda.norm import build_rms_norm
    from lmdeploy.pytorch.kernels.cuda import rms_norm as triton_rms_norm

    records = []
    for dtype in (torch.bfloat16, torch.float16):
        for (tokens, hidden) in ((8, 512), (128, 4096), (4096, 4096), (2048, 8192)):
            x = torch.randn(tokens, hidden, dtype=dtype, device=device)
            w = torch.randn(hidden, dtype=dtype, device=device)
            residual = torch.randn(tokens, hidden, dtype=dtype, device=device)

            triton_impl = build_rms_norm(type('S', (), {'hidden_size': hidden, 'eps': 1e-6})())
            os.environ['LMDEPLOY_TURBOMIND_OPS'] = 'force=rms_norm'
            tm_impl = build_rms_norm(type('S', (), {'hidden_size': hidden, 'eps': 1e-6})())
            del os.environ['LMDEPLOY_TURBOMIND_OPS']

            timings = {
                'triton': _bench(lambda: triton_impl.forward(x, w)),
                'triton_residual': _bench(lambda: triton_impl.forward(x, w, residual)),
                'turbomind': _bench(lambda: tm_impl.forward(x, w)),
                'turbomind_residual': _bench(lambda: tm_impl.forward(x, w, residual)),
            }
            records.append(dict(dtype=str(dtype).replace('torch.', ''),
                                shapes=[[tokens, hidden]],
                                timings_us=timings))
    return records


def _bench_silu_and_mul(device, cap):
    from lmdeploy.pytorch.backends.cuda.activation import build_silu_and_mul

    records = []
    for dtype in (torch.bfloat16, torch.float16):
        for (tokens, dim) in ((8, 256), (128, 4096), (4096, 4096), (2048, 8192)):
            x = torch.randn(tokens, 2 * dim, dtype=dtype, device=device)

            os.environ['LMDEPLOY_TURBOMIND_OPS'] = 'auto'
            triton_impl = build_silu_and_mul(type('S', (), {'inplace': False})())
            os.environ['LMDEPLOY_TURBOMIND_OPS'] = 'force=silu_and_mul'
            tm_impl = build_silu_and_mul(type('S', (), {'inplace': False})())
            del os.environ['LMDEPLOY_TURBOMIND_OPS']

            timings = {
                'triton': _bench(lambda: triton_impl.forward(x)),
                'turbomind': _bench(lambda: tm_impl.forward(x)),
            }
            records.append(dict(dtype=str(dtype).replace('torch.', ''),
                                shapes=[[tokens, 2 * dim]],
                                timings_us=timings))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    device = 'cuda'
    cap = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)

    rms_cases = _bench_rms_norm(device, cap)
    silu_cases = _bench_silu_and_mul(device, cap)
    cases = [('rms_norm', c) for c in rms_cases] + [('silu_and_mul', c) for c in silu_cases]

    timings = {}
    for family, case in cases:
        prefix = f"{family}/{case['dtype']}/{'x'.join(map(str, case['shapes'][0]))}"
        for name, us in case['timings_us'].items():
            timings[f'{prefix}/{name}'] = [us]

    record = {
        'ticket': '04-rmsnorm-siluandmul',
        'families': ['rms_norm', 'silu_and_mul'],
        'device': props.name,
        'capability': list(cap),
        'torch': torch.__version__,
        'git_sha': _git_sha(),
        'timestamp': datetime.datetime.now().isoformat(),
        'host': platform.node(),
        # Shared-schema fields (see validate_tm_ops_json.py):
        'shapes': sorted({tuple(s) for _, c in cases for s in c['shapes']}),
        'impl': ['triton', 'turbomind'],
        'timings': timings,
        # Per-case detail (dtype, shape, all impl variants together):
        'cases': {family: group for family, group in (('rms_norm', rms_cases), ('silu_and_mul', silu_cases))},
    }

    out = args.out or os.path.join(os.path.dirname(__file__), 'rmsnorm_siluandmul_tm.json')
    with open(out, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
