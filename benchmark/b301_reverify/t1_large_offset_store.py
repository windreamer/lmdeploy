#!/usr/bin/env python
"""T1 (minimal case 1): large-offset vector store.

Claim under test: "自写 kernel 大偏移 store 大面积丢失" (large-offset vector
stores大面积丢失 on triton 3.7.1 / SM86; probe saw 128 programs with only
~16-32 visible stores).

Independent design (no reuse of any prior kernel):
- Sentinel-prefill a uint8 buffer with 0xEE, each program writes a vector of
  bytes equal to (pid+1) at a known offset; golden is pure python bytes.
- A: 1D grid, per-program block of BLK bytes (TM int8 block geometry,
  135168 B = 2*8*64*128 codes + 2*8*64*4 params), int64 in-kernel arithmetic,
  unmasked full-block vector stores (VEC=1024), NB=128 programs.
- B: launcher-precomputed int64 offsets tensor; kernel loads offset, stores
  VEC bytes at base+offset.
- C: offsets straddling the int32 boundary (2**31): 288 programs x 8 MiB
  stride, int64 arithmetic.
- D: int32-arithmetic wrap documentation: product computed in int32 in-kernel,
  stored to a small int32 buffer (no OOB write); golden is modular-2^32 wrap.
  This documents wrap-around semantics, NOT store loss.
- E: masked vector store (mask all-true) at large offsets.

Every check is torch.equal / exact byte compare on uint8.
"""
import json
import sys

import torch
import triton
import triton.language as tl

DEV = 'cuda'
SENT = 0xEE
BLK_BYTES = 135168  # TM int8 physical block geometry (per bench_b301_bridge pool_geometry)
VEC = 1024

results = []


def check(name, ok, detail):
    results.append(dict(name=name, ok=bool(ok), detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] t1/{name}: {detail}")


@triton.jit
def t1a_store_block(out_ptr, BLK: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * BLK
    val = tl.zeros([VEC], dtype=tl.uint8) + (pid + 1).to(tl.uint8)
    for i in tl.static_range(BLK // VEC):
        offs = base + i * VEC + tl.arange(0, VEC).to(tl.int64)
        tl.store(out_ptr + offs, val)


def run_a(nb=128):
    buf = torch.full((nb, BLK_BYTES), SENT, dtype=torch.uint8, device=DEV)
    t1a_store_block[(nb,)](buf, BLK=BLK_BYTES, VEC=VEC)
    torch.cuda.synchronize()
    exp = torch.full_like(buf, SENT)
    for p in range(nb):
        exp[p, :] = (p + 1) % 256
    ok = torch.equal(buf, exp)
    nbad = int((buf.view(nb, -1) != exp.view(nb, -1)).any(dim=1).sum())
    mismatch_bytes = int((buf != exp).sum())
    check('A_int64_full_blocks', ok,
          f'nb={nb} blk_bytes={BLK_BYTES} visible_blocks={nb - nbad}/{nb} '
          f'mismatched_bytes={mismatch_bytes}')
    return ok


@triton.jit
def t1b_store_precomp(out_ptr, off_ptr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.load(off_ptr + pid)
    val = tl.zeros([VEC], dtype=tl.uint8) + (pid + 1).to(tl.uint8)
    tl.store(out_ptr + off + tl.arange(0, VEC).to(tl.int64), val)


def run_b(nb=128):
    buf = torch.full((nb * BLK_BYTES,), SENT, dtype=torch.uint8, device=DEV)
    offs = (torch.arange(nb, dtype=torch.int64, device=DEV) * BLK_BYTES)
    t1b_store_precomp[(nb,)](buf, offs, VEC=VEC)
    torch.cuda.synchronize()
    exp = torch.full_like(buf, SENT)
    for p in range(nb):
        exp[p * BLK_BYTES:p * BLK_BYTES + VEC] = (p + 1) % 256
    ok = torch.equal(buf, exp)
    check('B_launcher_precomputed_i64_offsets', ok,
          f'nb={nb} vec={VEC} mismatched_bytes={int((buf != exp).sum())}')
    return ok


@triton.jit
def t1c_store_i32_boundary(out_ptr, STRIDE: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * STRIDE
    val = tl.zeros([VEC], dtype=tl.uint8) + (pid + 1).to(tl.uint8)
    offs = base + tl.arange(0, VEC).to(tl.int64)
    tl.store(out_ptr + offs, val)


def run_c():
    nprog, stride = 288, 8 * 1024 * 1024  # total 2.25 GiB > 2**31
    buf = torch.empty(nprog * stride, dtype=torch.uint8, device=DEV)
    buf.fill_(SENT)
    t1c_store_i32_boundary[(nprog,)](buf, STRIDE=stride, VEC=VEC)
    torch.cuda.synchronize()
    exp = torch.full_like(buf, SENT)
    for p in range(nprog):
        exp[p * stride:p * stride + VEC] = (p + 1) % 256
    ok = torch.equal(buf, exp)
    b31 = 2**31
    pids_at_boundary = sorted({(b31 - 1) // stride, b31 // stride, (b31 + VEC) // stride})
    # count mismatches in chunks (a bool .sum() over 2.25G elems materializes 18 GiB int64)
    mism = 0
    for i in range(0, buf.numel(), 1 << 28):
        mism += int((buf[i:i + (1 << 28)] != exp[i:i + (1 << 28)]).sum())
    check('C_i32_boundary_int64_math', ok,
          f'total={buf.numel() / 2**30:.3f}GiB stride={stride // 2**20}MiB '
          f'pids_straddling_2^31={pids_at_boundary} mismatched_bytes={mism}')
    return ok


@triton.jit
def t1d_i32_wrap(out_ptr, STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    prod = pid * STRIDE  # int32 multiply; wraps for pid*STRIDE >= 2**31
    tl.store(out_ptr + pid, prod)


def run_d():
    nprog, stride = 300, 2**23
    out = torch.zeros(nprog, dtype=torch.int32, device=DEV)
    t1d_i32_wrap[(nprog,)](out, STRIDE=stride)
    torch.cuda.synchronize()
    exp = torch.tensor([((p * stride + 2**31) % 2**32) - 2**31 for p in range(nprog)],
                       dtype=torch.int32, device=DEV)
    ok = torch.equal(out, exp)
    wrapped = int((exp < 0).sum())
    check('D_i32_wrap_semantics_documented', ok,
          f'nprog={nprog} stride=2^23 wrapped_products={wrapped} '
          f'(expected C wrap-around, not store loss)')
    return ok


@triton.jit
def t1e_store_masked(out_ptr, BLK: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * BLK
    val = tl.zeros([VEC], dtype=tl.uint8) + (pid + 1).to(tl.uint8)
    offs = base + tl.arange(0, VEC).to(tl.int64)
    mask = tl.arange(0, VEC) < VEC  # all-true explicit mask
    tl.store(out_ptr + offs, val, mask=mask)


def run_e(nb=128):
    buf = torch.full((nb, BLK_BYTES), SENT, dtype=torch.uint8, device=DEV)
    t1e_store_masked[(nb,)](buf, BLK=BLK_BYTES, VEC=VEC)
    torch.cuda.synchronize()
    exp = torch.full_like(buf, SENT)
    for p in range(nb):
        exp[p, :VEC] = (p + 1) % 256
    ok = torch.equal(buf, exp)
    check('E_masked_store_large_offset', ok,
          f'nb={nb} vec={VEC} mismatched_bytes={int((buf != exp).sum())}')
    return ok


def main():
    torch.cuda.init()
    print(f'triton={triton.__version__} torch={torch.__version__} dev={torch.cuda.get_device_name(0)}')
    run_a()
    run_b()
    run_c()
    run_d()
    run_e()
    with open('benchmark/b301_reverify/t1_results.json', 'w') as f:
        json.dump(dict(triton=triton.__version__, torch=torch.__version__,
                       results=results), f, indent=1)
    ok = all(r['ok'] for r in results)
    print(f"T1 VERDICT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
