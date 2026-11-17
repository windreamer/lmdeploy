#!/usr/bin/env python
"""T2 (minimal case 2): tl.program_id(0/1/2) correctness.

Claim under test: "tl.program_id(1/2) 返回垃圾" on triton 3.7.1 / SM86.

Design: 3D grid; every program writes its own (pid0, pid1, pid2) triple at its
linear slot. Golden is torch.meshgrid (pure python index math). Exact
torch.equal on int32. Includes the claimed probe shape (1D grid of 128) and a
large-x stress grid (65535, 2, 1).
"""
import json
import sys

import torch
import triton
import triton.language as tl

DEV = 'cuda'
results = []


def check(name, ok, detail):
    results.append(dict(name=name, ok=bool(ok), detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] t2/{name}: {detail}")


@triton.jit
def t2_3d(out_ptr, DX: tl.constexpr, DY: tl.constexpr):
    p0 = tl.program_id(0)
    p1 = tl.program_id(1)
    p2 = tl.program_id(2)
    lin = p0 + DX * (p1 + DY * p2)
    tl.store(out_ptr + 3 * lin + 0, p0)
    tl.store(out_ptr + 3 * lin + 1, p1)
    tl.store(out_ptr + 3 * lin + 2, p2)


def run_3d(dx, dy, dz, name):
    n = dx * dy * dz
    out = torch.full((n, 3), -1, dtype=torch.int32, device=DEV)
    t2_3d[(dx, dy, dz)](out, DX=dx, DY=dy)
    torch.cuda.synchronize()
    p0 = torch.arange(dx, dtype=torch.int64, device=DEV)
    p1 = torch.arange(dy, dtype=torch.int64, device=DEV)
    p2 = torch.arange(dz, dtype=torch.int64, device=DEV)
    # golden linearization must place each triple at the slot indexed by its
    # own coordinates: lin = x + dx*(y + dy*z)  (x fastest, as in CUDA blockIdx
    # linearization); pure-python index math, no kernel formula reuse.
    lin = (p0[:, None, None] + dx * (p1[None, :, None] + dy * p2[None, None, :])).reshape(-1)
    g0f = p0[:, None, None].expand(dx, dy, dz).reshape(-1)
    g1f = p1[None, :, None].expand(dx, dy, dz).reshape(-1)
    g2f = p2[None, None, :].expand(dx, dy, dz).reshape(-1)
    exp = torch.zeros((lin.numel(), 3), dtype=torch.int64, device=DEV)
    exp[lin, 0] = g0f
    exp[lin, 1] = g1f
    exp[lin, 2] = g2f
    exp = exp.to(torch.int32)
    ok = torch.equal(out, exp)
    bad = int((out != exp).any(dim=1).sum())
    check(name, ok, f'grid=({dx},{dy},{dz}) programs={n} garbage_slots={bad}/{n}')
    return ok


@triton.jit
def t2_1d(out_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, pid + 1)


def run_1d(n=128):
    out = torch.zeros(n, dtype=torch.int32, device=DEV)
    t2_1d[(n,)](out)
    torch.cuda.synchronize()
    exp = torch.arange(1, n + 1, dtype=torch.int32, device=DEV)
    visible = int((out != 0).sum())
    ok = torch.equal(out, exp)
    check('1d_visibility_128', ok,
          f'grid=({n},) visible_stores={visible}/{n} (claim: only ~16-32 of 128)')
    return ok


def main():
    torch.cuda.init()
    print(f'triton={triton.__version__} torch={torch.__version__} dev={torch.cuda.get_device_name(0)}')
    run_1d(128)
    run_3d(256, 16, 4, '3d_grid_256x16x4')
    run_3d(4, 5, 8, '3d_grid_4x5x8_production_shape')
    run_3d(65535, 2, 1, '3d_grid_65535x2x1_large_x')
    with open('benchmark/b301_reverify/t2_results.json', 'w') as f:
        json.dump(dict(triton=triton.__version__, torch=torch.__version__,
                       results=results), f, indent=1)
    ok = all(r['ok'] for r in results)
    print(f"T2 VERDICT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
