# Copyright (c) OpenMMLab. All rights reserved.
"""[batch4 TmWriteBench] TM fused KV write vs PT triton INT8 write.

Correctness + performance of the two KV-cache write paths on one pool
geometry (TM `block::Layout`, int8, per-token-per-head quant):

- `tm`: `_tm.tm_process_kv` (`invokeProcessKV_v2`, null rope, quantizing
  append) into a TM block pool.
- `pt`: triton `fill_kv_cache` with `QuantPolicy.INT8` into PT-format caches
  plus the zero -> min transcription (the B3-01/B3-04 seam) into the same
  pool layout.
- `host`: fp32 host-op reference under the TM (scale, min) contract.

Quant contract: PT stores (scale, zero), TM stores (scale, min =
-zero*scale); dequant = code * scale + min under both conventions.

Cumulative-length arrays MUST be int32: `torch.cumsum` on int32 promotes to
int64, and `ProcessKV_v2` reads `const int*` -- an int64 array interleaved
as int32 yields garbage q_len/k_len and out-of-table block indices (the
"q_len=1 illegal access", see the report). `make_cu` below builds int32
arrays on device.

Modes (a CUDA illegal access poisons the context; risky probes run in
subprocesses and last):

- `correctness`  prefill T in {128, 512, 2048} + decode appends m in
                 {1, 8, 32} at kv growth stages + non-aligned 129+1:
                 TM vs PT vs host readback + collateral-sentinel check
- `perf`         median-of-iters timing, TM vs PT on the same shapes
- `int64bug`     reproduces the int64-cu defect per shape (subprocess each)
- `all`          correctness, perf, then int64bug (default)

Results: `benchmark/bench_tm_write_results.json`.
"""

import argparse
import json
import os
import subprocess
import sys

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache
from lmdeploy.turbomind import _tm

BLOCK_LEN = 64
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
BATCH = 4
AMP = 0.5
MAX_KV = 1024  # window cap: 16 blocks per sequence
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(HERE, 'bench_tm_write_results.json')
SENTINEL = 0xCC
RESULTS = {}


def record(key, **kw):
    RESULTS.setdefault(key, []).append(kw)
    print(f'[record] {key}: {kw}', flush=True)


# ------------------------------------------------------------- geometry

def block_bytes():
    """One TM int8 block: per-head [K|V] codes + per-head [Kp|Vp] params."""
    return 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM + 2 * NUM_KV_HEADS * BLOCK_LEN * 4


def alloc_pool(nb):
    return torch.full((nb, block_bytes()), SENTINEL, dtype=torch.uint8, device='cuda')


def identity_table(bsz, cols, pool):
    """Sequence-major `char*` table: entry = pool base + block_id *
    block_bytes."""
    boff = (torch.arange(bsz, device='cuda')[:, None] * cols +
            torch.arange(cols, device='cuda')[None, :]).int()
    ptrs = boff.reshape(-1).to(torch.int64) * block_bytes() + pool.data_ptr()
    base = (torch.arange(bsz + 1, dtype=torch.int32, device='cuda') * cols)
    return ptrs, base, boff


def make_cu(vals):
    """Int32 cumulative array [0, v0, v0+v1, ...] on device.

    torch.cumsum promotes int32 to int64; ProcessKV_v2 takes `const int*`. Keep the whole chain int32 (device fill, no
    cumsum needed).
    """
    out = torch.zeros(len(vals) + 1, dtype=torch.int32)
    acc = 0
    for i, v in enumerate(vals):
        acc += v
        out[i + 1] = acc
    return out.cuda()


def make_kv(seed, bsz, kv_len):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    k = (torch.randn(bsz, kv_len, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    v = (torch.randn(bsz, kv_len, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    return k, v


def quant_scale_min(x):
    """Fp32 per-token-per-head (scale, min), round-half-even codes."""
    xf = x.float()
    lo = xf.amin(-1, keepdim=True)
    scale = ((xf.amax(-1, keepdim=True) - lo) / 255.0).clamp(min=1e-6)
    codes = ((xf - lo) / scale).round().clamp(0, 255).to(torch.uint8)
    return codes, scale, lo


# --------------------------------------------------------------- writers

def tm_write(k_rows, v_rows, cu_q, cu_k, base, max_q_len, ptrs):
    """TM fused quantizing append; rows packed [B*max_q_len, Hkv, D]."""
    _tm.tm_process_kv(
        _tm.from_dlpack(k_rows.contiguous()),
        _tm.from_dlpack(v_rows.contiguous()),
        ptrs.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(), base.data_ptr(),
        stride_c=NUM_KV_HEADS, stride_s=NUM_KV_HEADS, stride_h=1,
        max_q_len=max_q_len, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        batch_size=cu_q.numel() - 1, block_len=BLOCK_LEN, kv_quant=8,
        stream_ptr=torch.cuda.current_stream().cuda_stream)


def host_write(pool, base, k, v, ti0, ti1):
    """Fp32 reference straight into the pool under the (scale, min)
    contract."""
    bb = block_bytes()
    data = 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM
    for b in range(k.shape[0]):
        for ti in range(ti0, ti1):
            e = int(base[b]) + ti // BLOCK_LEN
            s = ti % BLOCK_LEN
            for h in range(NUM_KV_HEADS):
                for name, x in (('k', k), ('v', v)):
                    src = x[b, ti, h].float()
                    codes, scale, lo = quant_scale_min(src)
                    off = h * 2 * BLOCK_LEN * HEAD_DIM + \
                        (0 if name == 'k' else BLOCK_LEN * HEAD_DIM) + s * HEAD_DIM
                    pool[e, off:off + HEAD_DIM] = codes
                    poff = data + h * 2 * BLOCK_LEN * 4 + \
                        (0 if name == 'k' else BLOCK_LEN * 4) + s * 4
                    pool[e, poff:poff + 4] = torch.stack([scale.flatten(), lo.flatten()]).reshape(2).bfloat16().view(torch.uint8)


class PtWriter:
    """Triton INT8 writer into PT caches + zero->min transcription to TM
    pool."""

    def __init__(self, nb):
        self.kc = torch.zeros(nb, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.uint8, device='cuda')
        self.vc = torch.zeros_like(self.kc)
        self.ksz = torch.zeros(nb, BLOCK_LEN, NUM_KV_HEADS, 2, dtype=torch.float32, device='cuda')
        self.vsz = torch.zeros_like(self.ksz)

    def __call__(self, k, v, kv_lens, q_lens, boff):
        bsz = k.shape[0]
        qlen = max(q_lens)
        q_loc = torch.tensor([sum(q_lens[:i]) for i in range(bsz)], dtype=torch.int64, device='cuda')
        q_len = torch.tensor(q_lens, dtype=torch.int64, device='cuda')
        kv_len = torch.tensor(kv_lens, dtype=torch.int64, device='cuda')
        fill_kv_cache(
            k[:, -qlen:].contiguous(), v[:, -qlen:].contiguous(),
            self.kc, self.vc, q_loc, q_len, kv_len, qlen, boff,
            k_scales_zeros=self.ksz, v_scales_zeros=self.vsz,
            quant_policy=QuantPolicy.INT8, kv_layout='bshd')

    def transcribe(self, pool, base, bsz, ti0, ti1):
        bb = block_bytes()
        data = 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM
        for b in range(bsz):
            for ti in range(ti0, ti1):
                e = int(base[b]) + ti // BLOCK_LEN
                s = ti % BLOCK_LEN
                for h in range(NUM_KV_HEADS):
                    off = h * 2 * BLOCK_LEN * HEAD_DIM + \
                        (0 if True else 0) + s * HEAD_DIM
                    pool[e, off:off + HEAD_DIM] = self.kc[e, s, h]
                    off = h * 2 * BLOCK_LEN * HEAD_DIM + BLOCK_LEN * HEAD_DIM + s * HEAD_DIM
                    pool[e, off:off + HEAD_DIM] = self.vc[e, s, h]
                    scale, zero = self.ksz[e, s, h, 0], self.ksz[e, s, h, 1]
                    poff = data + h * 2 * BLOCK_LEN * 4 + s * 4
                    pool[e, poff:poff + 4] = torch.stack([scale.flatten(), (-scale * zero).flatten()]).reshape(2).bfloat16().view(torch.uint8)
                    scale, zero = self.vsz[e, s, h, 0], self.vsz[e, s, h, 1]
                    poff = data + h * 2 * BLOCK_LEN * 4 + BLOCK_LEN * 4 + s * 4
                    pool[e, poff:poff + 4] = torch.stack([scale.flatten(), (-scale * zero).flatten()]).reshape(2).bfloat16().view(torch.uint8)


# ---------------------------------------------------------- verification

def verify(pool, base, k, v, ti0, ti1, seeded=None, tol=0.02):
    """Dequant error vs fp32 source + sentinel collateral outside expected."""
    bsz = k.shape[0]
    bb = block_bytes()
    data = 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM
    hit = torch.zeros(pool.shape[0] * bb, dtype=torch.bool, device='cuda')
    errs = []
    for b in range(bsz):
        for span in ([seeded] if seeded else []) + [(ti0, ti1)]:
            for ti in range(span[0], span[1]):
                e = int(base[b]) + ti // BLOCK_LEN
                hit[e * bb:(e + 1) * bb] = True
    for b in range(bsz):
        for ti in range(ti0, ti1):
            e = int(base[b]) + ti // BLOCK_LEN
            s = ti % BLOCK_LEN
            for h in range(NUM_KV_HEADS):
                src = k[b, ti, h].float()
                off = h * 2 * BLOCK_LEN * HEAD_DIM + s * HEAD_DIM
                codes = pool[e, off:off + HEAD_DIM]
                poff = data + h * 2 * BLOCK_LEN * 4 + s * 4
                prm = pool[e, poff:poff + 4].view(torch.bfloat16).float()
                deq = codes.float() * prm[0] + prm[1]
                errs.append((deq - src).abs().max().item())
    col = int(((pool.reshape(-1) != SENTINEL) & ~hit).sum())
    return max(errs), col


# ----------------------------------------------------------- correctness

def run_correctness():
    """Prefill + decode-append shapes: tm / pt / host readback comparison."""
    # prefill: T in {128, 512, 2048}
    for T in (128, 512, 2048):
        cols = (T + BLOCK_LEN - 1) // BLOCK_LEN
        nb = BATCH * cols
        k, v = make_kv(11, BATCH, T)
        for kind in ('host', 'tm', 'pt'):
            pool = alloc_pool(nb)
            ptrs, base, boff = identity_table(BATCH, cols, pool)
            torch.cuda.synchronize()
            if kind == 'tm':
                tm_write(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                         v.reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                         make_cu([T] * BATCH), make_cu([T] * BATCH), base, T, ptrs)
            elif kind == 'pt':
                w = PtWriter(nb)
                w(k, v, [T] * BATCH, [T] * BATCH, boff)
                w.transcribe(pool, base, BATCH, 0, T)
            else:
                host_write(pool, base, k, v, 0, T)
            torch.cuda.synchronize()
            err, col = verify(pool, base, k, v, 0, T)
            ok = err <= 0.03 and col == 0
            record('correctness', shape=f'prefill_T{T}', writer=kind,
                   dequant_max_abs=round(err, 5), collateral_bytes=col, ok=ok)

    # decode append: m in {1, 8, 32} at kv growth stages, plus 129+1
    stages = [(1, 128), (8, 128), (32, 128), (1, 192), (8, 448), (32, 704), (1, 129)]
    for m, kv0 in stages:
        cols = (MAX_KV // BLOCK_LEN)
        nb = BATCH * cols
        k, v = make_kv(17 + kv0, BATCH, kv0 + m)
        pool = alloc_pool(nb)
        ptrs, base, boff = identity_table(BATCH, cols, pool)
        torch.cuda.synchronize()
        # seed the history region with the host writer so append deltas are meaningful
        host_write(pool, base, k, v, 0, kv0)
        torch.cuda.synchronize()
        tm_write(k[:, kv0:].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 v[:, kv0:].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 make_cu([m] * BATCH), make_cu([kv0 + m] * BATCH), base, m, ptrs)
        torch.cuda.synchronize()
        err, col = verify(pool, base, k, v, kv0, kv0 + m, seeded=(0, kv0))
        ok = err <= 0.03 and col == 0
        record('correctness', shape=f'decode_m{m}_kv{kv0}', writer='tm',
               dequant_max_abs=round(err, 5), collateral_bytes=col, ok=ok)

    # PT decode-append correctness on the same geometry
    for m, kv0 in ((1, 192), (8, 448), (1, 129)):
        cols = MAX_KV // BLOCK_LEN
        nb = BATCH * cols
        k, v = make_kv(17 + kv0, BATCH, kv0 + m)
        pool = alloc_pool(nb)
        ptrs, base, boff = identity_table(BATCH, cols, pool)
        torch.cuda.synchronize()
        w = PtWriter(nb)
        host_write(pool, base, k, v, 0, kv0)
        torch.cuda.synchronize()
        w(k, v, [kv0 + m] * BATCH, [m] * BATCH, boff)
        torch.cuda.synchronize()
        w.transcribe(pool, base, BATCH, kv0, kv0 + m)
        err, col = verify(pool, base, k, v, kv0, kv0 + m, seeded=(0, kv0))
        record('correctness', shape=f'decode_m{m}_kv{kv0}', writer='pt',
               dequant_max_abs=round(err, 5), collateral_bytes=col, ok=err <= 0.03 and col == 0)


# ----------------------------------------------------------------- perf

def time_fn(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort()
    return ts[len(ts) // 2]


def run_perf():
    """Median timing TM vs PT; prefill shapes + decode appends (PT) + TM."""
    for T in (128, 512, 2048):
        cols = (T + BLOCK_LEN - 1) // BLOCK_LEN
        nb = BATCH * cols
        k, v = make_kv(11, BATCH, T)
        rows = k.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        rowsv = v.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        cu_q, cu_k = make_cu([T] * BATCH), make_cu([T] * BATCH)
        pool = alloc_pool(nb)
        ptrs, base, boff = identity_table(BATCH, cols, pool)
        ms_tm = time_fn(lambda: tm_write(rows, rowsv, cu_q, cu_k, base, T, ptrs))
        w = PtWriter(nb)
        ms_pt = time_fn(lambda: w(k, v, [T] * BATCH, [T] * BATCH, boff))
        bytes_w = BATCH * T * NUM_KV_HEADS * HEAD_DIM * 3  # 2 code bytes + 1 param byte per elem
        record('perf', shape=f'prefill_T{T}', tm_ms=round(ms_tm, 4), pt_ms=round(ms_pt, 4),
               tm_gbps=round(bytes_w / ms_tm / 1e6, 1), pt_gbps=round(bytes_w / ms_pt / 1e6, 1))

    cols = MAX_KV // BLOCK_LEN
    nb = BATCH * cols
    for m, kv0 in ((1, 192), (8, 448), (32, 704)):
        k, v = make_kv(17 + kv0, BATCH, kv0 + m)
        pool = alloc_pool(nb)
        ptrs, base, boff = identity_table(BATCH, cols, pool)
        rows = k[:, kv0:].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        rowsv = v[:, kv0:].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        cu_q, cu_k = make_cu([m] * BATCH), make_cu([kv0 + m] * BATCH)
        ms_tm = time_fn(lambda: tm_write(rows, rowsv, cu_q, cu_k, base, m, ptrs))
        w = PtWriter(nb)
        ms_pt = time_fn(lambda: w(k, v, [kv0 + m] * BATCH, [m] * BATCH, boff))
        bytes_w = BATCH * m * NUM_KV_HEADS * HEAD_DIM * 3
        record('perf', shape=f'decode_m{m}_kv{kv0}', tm_ms=round(ms_tm, 4), pt_ms=round(ms_pt, 4),
               tm_gbps=round(bytes_w / ms_tm / 1e6, 1), pt_gbps=round(bytes_w / ms_pt / 1e6, 1))


# --------------------------------------------------------- int64 defect

INT64_SHAPES = [(1, 192), (1, 128), (8, 128), (32, 128), (1, 448), (256, 0)]


def int64_probe(m, kv0):
    """Subprocess body: dispatch with an int64 (cumsum-promoted) cu array."""
    cols = max((kv0 + m + BLOCK_LEN - 1) // BLOCK_LEN + 1, 2)
    nb = BATCH * cols
    k, v = make_kv(17 + kv0, BATCH, kv0 + m)
    pool = alloc_pool(nb)
    ptrs, base, boff = identity_table(BATCH, cols, pool)
    torch.cuda.synchronize()
    # the harness pattern: pad(cumsum(int32)) -- promoted to int64
    cu_q = torch.nn.functional.pad(torch.cumsum(torch.tensor([m] * BATCH, dtype=torch.int32), 0), (1, 0)).cuda()
    cu_k = torch.nn.functional.pad(torch.cumsum(torch.tensor([kv0 + m] * BATCH, dtype=torch.int32), 0), (1, 0)).cuda()
    assert cu_q.dtype == torch.int64
    try:
        tm_write(k[:, kv0:].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 v[:, kv0:].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 cu_q, cu_k, base, m, ptrs)
        torch.cuda.synchronize()
    except RuntimeError as e:
        return {'shape': f'm{m}_kv{kv0}', 'status': 'CRASH', 'error': str(e)[:160]}
    err, col = verify(pool, base, k, v, kv0, kv0 + m)
    status = 'OK' if err <= 0.03 else 'WRONG-CONTENT'
    return {'shape': f'm{m}_kv{kv0}', 'status': status,
            'dequant_max_abs': round(err, 5), 'collateral_bytes': col}


def run_int64bug():
    self_exe = [sys.executable, os.path.abspath(__file__), '--mode', 'int64-one']
    for m, kv0 in INT64_SHAPES:
        out = subprocess.run(self_exe + ['--m', str(m), '--kv0', str(kv0)],
                             capture_output=True, text=True, timeout=180)
        res = None
        for line in out.stdout.splitlines():
            if line.startswith('[bug-json] '):
                res = json.loads(line[len('[bug-json] '):])
        if res is None:
            res = {'shape': f'm{m}_kv{kv0}', 'status': 'HARD-CRASH',
                   'error': (out.stderr or '').strip().splitlines()[-1:]}
        record('int64bug', **res)


def run_int64_one(m, kv0):
    print('[bug-json] ' + json.dumps(int64_probe(m, kv0)), flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='all',
                    choices=['all', 'correctness', 'perf', 'int64bug', 'int64-one'])
    ap.add_argument('--m', type=int, default=1)
    ap.add_argument('--kv0', type=int, default=128)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.init()

    if args.mode == 'int64-one':
        run_int64_one(args.m, args.kv0)
        return
    if args.mode in ('all', 'correctness'):
        run_correctness()
    if args.mode in ('all', 'perf'):
        run_perf()
    if args.mode in ('all', 'int64bug'):
        run_int64bug()

    with open(RESULT_PATH, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f'[results] -> {RESULT_PATH}', flush=True)


if __name__ == '__main__':
    main()
