#!/usr/bin/env python
"""T3 (minimal case 3): in-kernel integer // and % correctness.

Claim under test: "kernel 内整数 div/mod 链结果错误" on triton 3.7.1 / SM86.

Design: kernels compute q = a // b, r = a % b and the identity idx = q*b + r;
golden is pure torch on CPU int64 in two documented semantics:
- C-truncation (expected Triton semantics): torch.div(rounding_mode='trunc'),
  torch.fmod
- Python floor: torch.div(rounding_mode='floor'), torch.remainder
Non-negative operands are semantics-independent and must match exactly.
Also replicates the production index chain `kv_block_id = history // BLOCK +
block_id` with the `kv_block_id*BLOCK >= kv_seqlen` guard, pure-python golden.
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
    print(f"[{'PASS' if ok else 'FAIL'}] t3/{name}: {detail}")


@triton.jit
def t3_vec(a_ptr, b_ptr, q_ptr, r_ptr, idx_ptr, N, VEC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * VEC + tl.arange(0, VEC)
    m = offs < N
    a = tl.load(a_ptr + offs, mask=m, other=0)
    b = tl.load(b_ptr + offs, mask=m, other=1)
    q = a // b
    r = a % b
    tl.store(q_ptr + offs, q, mask=m)
    tl.store(r_ptr + offs, r, mask=m)
    tl.store(idx_ptr + offs, q * b + r, mask=m)


def run_vec():
    # deterministic mixed-sign operands incl large magnitudes, b != 0
    g = torch.Generator().manual_seed(7)
    a = torch.randint(-2**31 + 1, 2**31 - 1, (4099,), generator=g, dtype=torch.int64)
    b = torch.randint(-100, 101, (4099,), generator=g, dtype=torch.int64)
    b[b == 0] = 7
    a_t, b_t = a.to(torch.int32).to(DEV), b.to(torch.int32).to(DEV)
    q = torch.full_like(a_t, -77)
    r = torch.full_like(a_t, -77)
    idx = torch.full_like(a_t, -77)
    t3_vec[(triton.cdiv(a_t.numel(), 1024),)](a_t, b_t, q, r, idx, a_t.numel(), VEC=1024)
    torch.cuda.synchronize()
    a64, b64 = a_t.cpu().to(torch.int64), b_t.cpu().to(torch.int64)
    gold = {
        'trunc': (torch.div(a64, b64, rounding_mode='trunc'), torch.fmod(a64, b64)),
        'floor': (torch.div(a64, b64, rounding_mode='floor'), torch.remainder(a64, b64)),
    }
    out = dict(q=q.cpu().to(torch.int64), r=r.cpu().to(torch.int64),
               idx=idx.cpu().to(torch.int64))
    row = {}
    for name, (gq, gr) in gold.items():
        row[name] = dict(q=bool(torch.equal(out['q'], gq)), r=bool(torch.equal(out['r'], gr)),
                         idx=bool(torch.equal(out['idx'], gq * b64 + gr)))
    neg = (a64 < 0) | (b64 < 0)
    q_t, r_t = gold['trunc']
    exact_nonneg = bool(torch.equal(out['q'][~neg], q_t[~neg])) and \
        bool(torch.equal(out['r'][~neg], r_t[~neg]))
    semantics = next((n for n, v in row.items() if all(v.values())), None)
    ok = exact_nonneg and semantics is not None
    check('vector_mixed_sign', ok,
          f'N={a_t.numel()} neg_operands={int(neg.sum())} matches={semantics or "NEITHER"} '
          f'trunc={row["trunc"]} floor={row["floor"]}')
    return ok


@triton.jit
def t3_scalar(a_ptr, b_ptr, q_ptr, r_ptr):
    i = tl.program_id(0)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(q_ptr + i, a // b)
    tl.store(r_ptr + i, a % b)


def run_scalar():
    pairs = [(-7, 2), (7, -2), (-7, -2), (7, 2), (-8, 2), (8, -2), (-1, 3), (0, 5),
             (1, -1), (-2147483648, 3), (2147483647, -3), (2147483647, 3),
             (-2147483648, 2), (6, 7), (-6, 7)]
    a = torch.tensor([p[0] for p in pairs], dtype=torch.int32, device=DEV)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.int32, device=DEV)
    q = torch.zeros(len(pairs), dtype=torch.int32, device=DEV)
    r = torch.zeros(len(pairs), dtype=torch.int32, device=DEV)
    t3_scalar[(len(pairs),)](a, b, q, r)
    torch.cuda.synchronize()
    a64, b64 = a.cpu().to(torch.int64), b.cpu().to(torch.int64)
    tq, tr = torch.div(a64, b64, rounding_mode='trunc'), torch.fmod(a64, b64)
    fq, fr = torch.div(a64, b64, rounding_mode='floor'), torch.remainder(a64, b64)
    q64, r64 = q.cpu().to(torch.int64), r.cpu().to(torch.int64)
    trunc_ok = bool(torch.equal(q64, tq)) and bool(torch.equal(r64, tr))
    floor_ok = bool(torch.equal(q64, fq)) and bool(torch.equal(r64, fr))
    ok = trunc_ok or floor_ok
    check('scalar_pair_table', ok, f'pairs={len(pairs)} trunc={trunc_ok} floor={floor_ok}')
    return ok


@triton.jit
def t3_chain(hist_ptr, bid_ptr, kvlen_ptr, out_ptr, N, BS: tl.constexpr):
    i = tl.program_id(0)
    if i < N:
        h = tl.load(hist_ptr + i)
        bid = tl.load(bid_ptr + i)
        kl = tl.load(kvlen_ptr + i)
        kvb = h // BS + bid
        if kvb * BS < kl:
            tl.store(out_ptr + i, kvb)
        else:
            tl.store(out_ptr + i, -1)


def run_chain():
    # production-style: history in [0,512), block_id in [0,8), kv_seqlen in [1,512]
    g = torch.Generator().manual_seed(11)
    hist = torch.randint(0, 512, (5000,), generator=g, dtype=torch.int32).to(DEV)
    bid = torch.randint(0, 8, (5000,), generator=g, dtype=torch.int32).to(DEV)
    kvlen = torch.randint(1, 512, (5000,), generator=g, dtype=torch.int32).to(DEV)
    out = torch.full_like(hist, -99)
    t3_chain[(5000,)](hist, bid, kvlen, out, 5000, BS=64)
    torch.cuda.synchronize()
    h, bi, kl = hist.cpu(), bid.cpu(), kvlen.cpu()
    kvb = torch.div(h, 64, rounding_mode='floor') + bi  # all operands non-negative
    exp = torch.where(kvb * 64 < kl, kvb, torch.full_like(kvb, -1))
    ok = torch.equal(out.cpu(), exp)
    check('production_style_chain', ok,
          f'N=5000 written={int((out.cpu() >= 0).sum())} guarded_out={int((out.cpu() == -1).sum())}')
    return ok


def main():
    torch.cuda.init()
    print(f'triton={triton.__version__} torch={torch.__version__} dev={torch.cuda.get_device_name(0)}')
    run_vec()
    run_scalar()
    run_chain()
    with open('benchmark/b301_reverify/t3_results.json', 'w') as f:
        json.dump(dict(triton=triton.__version__, torch=torch.__version__,
                       results=results), f, indent=1)
    ok = all(r['ok'] for r in results)
    print(f"T3 VERDICT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
