# Copyright (c) OpenMMLab. All rights reserved.
"""Batch-2 benchmark 3/4: TM decode-attention kernel vs PT triton paged
attention.

- bf16 KV (benchmark 3, low expectation per roadmap principle 2)
- int8 KV (benchmark 4, TM packed-KV read pipeline vs PT int8 read path)

Both sides consume the same q/k/v source data and the same geometry:
fixed GQA decode, [num_blocks, block, H_kv, D] PT-standard pools. The TM side
receives a caller-built TM-layout block pool plus a host-filled char** pointer
array (no state pool machinery), exactly the seam a grafted attention impl
would own. Kernel time only; block layout conversion happens once outside the
timed region.
"""

import argparse
import json
import statistics
import sys

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.kernels.cuda.pagedattention import flash_attn_with_kvcache
from lmdeploy.turbomind import _tm

BLOCK_LEN = 64  # GLM-5.3 kernel_block_size
HEAD_DIM = 128
NUM_HEADS = 64
NUM_KV_HEADS = 8


def median_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        beg.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(beg.elapsed_time(end))
    return statistics.median(times)


def build_kv(bsz: int, klen: int, device):
    """Shared source data: q [B,H,D], k/v pools [nb, S, Hkv, D] bf16."""
    gen = torch.Generator(device='cpu').manual_seed(7)
    q = (torch.randn(bsz, NUM_HEADS, HEAD_DIM, generator=gen) * 0.5).to(device=device, dtype=torch.bfloat16)
    nb = bsz * ((klen + BLOCK_LEN - 1) // BLOCK_LEN)
    k = (torch.randn(nb, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM, generator=gen) * 0.5).to(
        device=device, dtype=torch.bfloat16)
    v = (torch.randn(nb, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM, generator=gen) * 0.5).to(
        device=device, dtype=torch.bfloat16)
    return q, k, v, nb


def quant_kv(kv: torch.Tensor):
    """Per-token-per-head uint8 min-max quant (TM semantics).

    Codes x = round((v - min)/scale) in [0,255]; dequant v = x*scale + min.
    Returns (codes u8, scale, min). PT consumes (scale, zero=-min/scale) with
    the same codes, so both kernels read byte-identical caches.
    """
    vmin = kv.amin(dim=-1, keepdim=True)
    vmax = kv.amax(dim=-1, keepdim=True)
    scale = ((vmax - vmin) / 255.0).clamp(min=1e-6)
    codes = ((kv - vmin) / scale).round().clamp(0, 255).to(torch.uint8)
    return codes, scale, vmin


def build_tm_pool_bf16(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """TM per-block layout: [K_h0][V_h0][K_h1][V_h1]..., each [S, D]."""
    nb, s, h, d = k.shape
    kt = k.permute(0, 2, 1, 3).contiguous()  # [nb, H, S, D]
    vt = v.permute(0, 2, 1, 3).contiguous()
    pool = torch.empty(nb, h, 2, s, d, dtype=torch.bfloat16, device=k.device)
    pool[:, :, 0] = kt
    pool[:, :, 1] = vt
    return pool.view(torch.uint8).view(nb, -1)


def build_tm_pool_int8(kc, ks, km, vc, vs, vm) -> torch.Tensor:
    """TM int8 per-block layout (`block::Layout`, matches
    `lmdeploy/pytorch/backends/tm_kv.py::TmInt8BlockCacheLayout`): per-head
    [K data][V data] code pairs, then per-head [K params][V params] pairs.
    Param slots are 2 x bf16 (scale, zero=min) per token per head; codes are
    uint8 and dequant is code*scale + min.
    """
    nb, s, h, d = kc.shape
    data_bytes = 2 * h * s * d
    param_bytes = 2 * h * s * 4
    pool = torch.empty(nb, data_bytes + param_bytes, dtype=torch.uint8, device=kc.device)
    codes = pool[:, :data_bytes].view(nb, h, 2, s, d)
    params = pool[:, data_bytes:].view(torch.bfloat16).view(nb, h, 2, s, 2)
    codes[:, :, 0] = kc.permute(0, 2, 1, 3)
    codes[:, :, 1] = vc.permute(0, 2, 1, 3)
    params[:, :, 0, :, 0] = ks.squeeze(-1).permute(0, 2, 1)
    params[:, :, 0, :, 1] = km.squeeze(-1).permute(0, 2, 1)
    params[:, :, 1, :, 0] = vs.squeeze(-1).permute(0, 2, 1)
    params[:, :, 1, :, 1] = vm.squeeze(-1).permute(0, 2, 1)
    return pool


def last_token_kv(k, v, bsz, klen):
    """K/V rows of each sequence's newest token, flattened [B, Hkv*D].

    `k`/`v` are the pre-quant source pools [nb, S, Hkv, D] in sequence-major
    block order. The TM decode kernel re-stores exactly these rows (rope'd,
    quantized) into the pool at ti = klen - 1 every dispatch
    (`attention_universal.h` kProcessKV), so the binding must be given the
    real values — garbage gets quantized into the pool's last valid position.
    """
    off = klen - 1
    blk = torch.arange(bsz, device=k.device) * ((klen + BLOCK_LEN - 1) // BLOCK_LEN) + off // BLOCK_LEN
    kl = k[blk, off % BLOCK_LEN]
    vl = v[blk, off % BLOCK_LEN]
    return kl.reshape(bsz, -1).contiguous(), vl.reshape(bsz, -1).contiguous()


def run_tm(q, k, v, bsz, klen, kv_quant, iters, warmup, stream):
    nb = k.shape[0]
    if kv_quant == 0:
        pool = build_tm_pool_bf16(k, v)
    else:
        kc, ks, km = quant_kv(k)
        vc, vs, vm = quant_kv(v)
        pool = build_tm_pool_int8(kc, ks, km, vc, vs, vm)
    block_bytes = pool.shape[1]
    ptrs = torch.tensor([pool.data_ptr() + i * block_bytes for i in range(nb)], dtype=torch.int64, device='cuda')
    blocks_per_seq = (klen + BLOCK_LEN - 1) // BLOCK_LEN
    cu_block_nums = torch.arange(0, (bsz + 1) * blocks_per_seq, blocks_per_seq, dtype=torch.int32, device='cuda')
    out = torch.empty(bsz, NUM_HEADS * HEAD_DIM, dtype=q.dtype, device='cuda')
    k_last, v_last = last_token_kv(k, v, bsz, klen)
    ms = _tm.attn_decode_bench(
        _tm.from_dlpack(q),
        _tm.from_dlpack(k_last),
        _tm.from_dlpack(v_last),
        _tm.from_dlpack(out),
        ptrs.data_ptr(),
        cu_block_nums.data_ptr(),
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        batch_size=bsz,
        block_len=BLOCK_LEN,
        max_k_len=klen,
        kv_quant=kv_quant,
        iters=iters,
        warmup=warmup,
        stream_ptr=stream,
    )
    return statistics.median(ms), out


def run_pt(q, k, v, bsz, klen, kv_quant, iters, warmup):
    if kv_quant == 0:
        k_cache, v_cache = k, v
        ksz = vsz = None
        policy = QuantPolicy.NONE
    else:
        kc, ks, km = quant_kv(k)
        vc, vs, vm = quant_kv(v)
        k_cache, v_cache = kc, vc
        # PT dequant is (code - zero) * scale with zero = -min/scale.
        ksz = torch.stack([ks, -km / ks], dim=-1).bfloat16()  # [nb, S, Hkv, 2]
        vsz = torch.stack([vs, -vm / vs], dim=-1).bfloat16()
        policy = QuantPolicy.INT8
    blocks_per_seq = (klen + BLOCK_LEN - 1) // BLOCK_LEN
    cache_seqlens = torch.full((bsz,), klen, dtype=torch.int32, device='cuda')
    page_table = (torch.arange(bsz, device='cuda')[:, None] * blocks_per_seq +
                  torch.arange(blocks_per_seq, device='cuda')[None, :]).to(torch.int32)
    ms = median_ms(
        lambda: flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens, page_table,
            k_scales_zeros=ksz, v_scales_zeros=vsz, quant_policy=policy, kv_layout='bshd'),
        iters, warmup)
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens, page_table,
        k_scales_zeros=ksz, v_scales_zeros=vsz, quant_policy=policy, kv_layout='bshd')
    return ms, out


def reference(q, k, v, bsz, klen, kv_quant):
    """Naive decode reference on dequantized KV."""
    if kv_quant == 8:
        for name in ('k', 'v'):
            t = locals()[name]
            codes, scale, vmin = quant_kv(t)
            t = codes.float() * scale + vmin
            if name == 'k':
                k = t
            else:
                v = t
    kvh = k.shape[2]
    k = k.view(bsz, -1, kvh, HEAD_DIM)[:, :klen].float()
    v = v.view(bsz, -1, kvh, HEAD_DIM)[:, :klen].float()
    group = NUM_HEADS // kvh
    k = k.repeat_interleave(group, dim=2)  # [B, klen, H, D]
    v = v.repeat_interleave(group, dim=2)
    scores = torch.einsum('bhd,bkhd->bhk', q.float(), k) / HEAD_DIM**0.5
    p = scores.softmax(-1)
    return torch.einsum('bhk,bkhd->bhd', p, v).to(q.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--kv-quants', type=int, nargs='+', default=[0, 8])
    ap.add_argument('--bsz', type=int, nargs='+', default=[1, 8, 32, 128])
    ap.add_argument('--klen', type=int, nargs='+', default=[256, 1024, 4096])
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    torch.cuda.init()
    stream = torch.cuda.current_stream().cuda_stream
    ctx = _tm.create_device_context(stream)
    rows = []

    with ctx:

        # correctness gates (one small shape, per quant mode)
        for kv_quant in args.kv_quants:
            q, k, v, _ = build_kv(4, 256, 'cuda')
            ref = reference(q, k, v, 4, 256, kv_quant)
            _, tm_out = run_tm(q, k, v, 4, 256, kv_quant, 3, 2, stream)
            tm_out = tm_out.view(4, NUM_HEADS, HEAD_DIM)
            _, pt_out = run_pt(q, k, v, 4, 256, kv_quant, 3, 2)
            tol = 0.25 if kv_quant == 8 else 2e-2
            tm_err = (tm_out.float() - ref.float()).abs().max().item()
            pt_err = (pt_out.float() - ref.float()).abs().max().item()
            tag = 'ok' if tm_err < tol and pt_err < tol else 'MISMATCH'
            print(f'[check] kv_quant={kv_quant} tm_max_abs={tm_err:.4f} pt_max_abs={pt_err:.4f} -> {tag}')
            if tag != 'ok':
                sys.exit(f'correctness gate failed at kv_quant={kv_quant}')

        for kv_quant in args.kv_quants:
            for bsz in args.bsz:
                for klen in args.klen:
                    if bsz * klen > 512 * 1024:  # cap pool size for 24 GB
                        continue
                    q, k, v, _ = build_kv(bsz, klen, 'cuda')
                    tm_ms, _ = run_tm(q, k, v, bsz, klen, kv_quant, args.iters, args.warmup, stream)
                    pt_ms, _ = run_pt(q, k, v, bsz, klen, kv_quant, args.iters, args.warmup)
                    speedup = pt_ms / tm_ms
                    kv_gb = 2 * bsz * klen * NUM_KV_HEADS * HEAD_DIM * (1 if kv_quant else 2) / 1e9
                    tm_bw = kv_gb / (tm_ms / 1e3)
                    pt_bw = kv_gb / (pt_ms / 1e3)
                    row = dict(
                        kv_quant=kv_quant, bsz=bsz, klen=klen,
                        tm_ms=round(tm_ms, 4), pt_ms=round(pt_ms, 4),
                        tm_gbps=round(tm_bw, 1), pt_gbps=round(pt_bw, 1),
                        speedup_pt_over_tm=round(speedup, 3))
                    rows.append(row)
                    print('[{kv_quant}] B={bsz:>4} klen={klen:>5}  TM {tm_ms:>8.4f} ms ({tm_gbps:>7.1f} GB/s)'
                          '  PT {pt_ms:>8.4f} ms ({pt_gbps:>7.1f} GB/s)  PT/TM x{speedup_pt_over_tm}'.format(**row))
                    del q, k, v
                    torch.cuda.empty_cache()

        if args.json_out:
            with open(args.json_out, 'w') as f:
                json.dump(rows, f, indent=1)
            print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
