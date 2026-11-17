# Copyright (c) OpenMMLab. All rights reserved.
"""[B3-04 verify] Blind-runnable verification for the bench-binding fix.

Contract under test (production decode convention, `unified_attention_layer.cc`
CreateParams + `Scheduler::PlanContinue`): the decode kernel
(`attention_universal.h`, `kProcessKV`) ALWAYS re-stores the caller's K/V rows
(quantized) into the block pool at `ti = history_len`. A conformant caller
passes the real K/V of that position plus a valid `readonly_block_num` buffer.

Cases:
  1. bf16 pool + real k/v  -> pool byte-identical (idempotent store),
     decode output matches fp32 reference.
  2. int8 pool + real k/v  -> last-slot codes within +-2 of the pre-quant
     round-trip, params ~equal, decode output within int8 tolerance.
  3. int8 pool + scaled k/v -> last slot rewritten to quant(2*k) (proves the
     store is live and driven by caller k/v; the pre-fix bug was garbage k/v,
     not the kernel).
  4. int8 pool + zero k/v  -> last slot zeroed (the reported "zero-write"
     reproduced ONLY as a caller contract violation).
  5. timed entry (`attn_decode_bench`) + real k/v -> bf16 pool unchanged
     across iterations.

Exit 0 = all cases pass.
"""

import sys

import torch

from lmdeploy.turbomind import _tm

S, D, H, HQ = 64, 64, 2, 4
B, KLEN = 4, 100
BPS = (KLEN + S - 1) // S
NB = B * BPS
LAST = KLEN - 1
MID = 50

gen = torch.Generator(device='cpu').manual_seed(23)
q = (torch.randn(B, HQ, D, generator=gen) * 0.5).cuda().bfloat16()
k = (torch.randn(NB, S, H, D, generator=gen) * 0.5).cuda().bfloat16()
v = (torch.randn(NB, S, H, D, generator=gen) * 0.5).cuda().bfloat16()

blk = lambda b, p: b * BPS + p // S
off = lambda p: p % S


def quant_kv(kv):
    """TM-style per-token-per-head min-max u8 quant; returns (codes, scale,
    min)."""
    vmin = kv.amin(dim=-1, keepdim=True)
    vmax = kv.amax(dim=-1, keepdim=True)
    scale = ((vmax - vmin) / 255.0).clamp(min=1e-6)
    codes = ((kv - vmin) / scale).round().clamp(0, 255).to(torch.uint8)
    return codes, scale, vmin


def build_pool_int8():
    """TM block::Layout int8 pool: per-head [K|V] codes, then per-head [Kp|Vp]
    params (mirrors lmdeploy/pytorch/backends/tm_kv.py TmInt8BlockCacheLayout)."""
    kc, ks, km = quant_kv(k)
    vc, vs, vm = quant_kv(v)
    data_bytes, param_bytes = 2 * H * S * D, 2 * H * S * 4
    pool = torch.empty(NB, data_bytes + param_bytes, dtype=torch.uint8)
    codes = pool[:, :data_bytes].view(NB, H, 2, S, D)
    params = pool[:, data_bytes:].view(torch.bfloat16).view(NB, H, 2, S, 2)
    codes[:, :, 0] = kc.permute(0, 2, 1, 3)
    codes[:, :, 1] = vc.permute(0, 2, 1, 3)
    params[:, :, 0, :, 0] = ks.squeeze(-1).permute(0, 2, 1)
    params[:, :, 0, :, 1] = km.squeeze(-1).permute(0, 2, 1)
    params[:, :, 1, :, 0] = vs.squeeze(-1).permute(0, 2, 1)
    params[:, :, 1, :, 1] = vm.squeeze(-1).permute(0, 2, 1)
    return pool.cuda()


def build_pool_bf16():
    nb, s, h, d = k.shape
    pool = torch.empty(nb, h, 2, s, d, dtype=torch.bfloat16, device=k.device)
    pool[:, :, 0] = k.permute(0, 2, 1, 3)
    pool[:, :, 1] = v.permute(0, 2, 1, 3)
    return pool.view(torch.uint8).view(nb, -1)


def ptr_table(pool):
    return torch.tensor([pool.data_ptr() + i * pool.shape[1] for i in range(NB)],
                        dtype=torch.int64, device='cuda')


def last_kv(kl=k, vl=v):
    a = kl[torch.tensor([blk(b, LAST) for b in range(B)], device=kl.device), off(LAST)].reshape(B, -1).contiguous()
    b = vl[torch.tensor([blk(b, LAST) for b in range(B)], device=vl.device), off(LAST)].reshape(B, -1).contiguous()
    return a, b


def decode(pool, kl=None, vl=None, kv_quant=8):
    if kl is None:
        kl, vl = last_kv()
    ptrs = ptr_table(pool)
    cu_block_nums = torch.arange(0, (B + 1) * BPS, BPS, dtype=torch.int32, device='cuda')
    out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')
    _tm.tm_attn_decode(_tm.from_dlpack(q.reshape(B, -1).contiguous()), _tm.from_dlpack(kl),
                       _tm.from_dlpack(vl), _tm.from_dlpack(out),
                       ptrs.data_ptr(), cu_block_nums.data_ptr(), num_heads=HQ, num_kv_heads=H,
                       head_dim=D, batch_size=B, block_len=S, max_k_len=KLEN,
                       kv_quant=kv_quant, stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out


def snapshot_int8(pool, p):
    """Codes+params at position p for every sequence, per block::Layout."""
    codes = pool[:, :2 * H * S * D].view(NB, H, 2, S, D)
    params = pool[:, 2 * H * S * D:].view(torch.bfloat16).view(NB, H, 2, S, 2)
    bi = torch.tensor([blk(b, p) for b in range(B)])
    o = off(p)
    ks = codes[bi, :, 0, o]
    vs = codes[bi, :, 1, o]
    kp = params[bi, :, 0, o]
    vp = params[bi, :, 1, o]
    return ks, vs, kp, vp


def snapshot_bf16(pool, p):
    ks = torch.stack([pool[blk(b, p), :H * S * D * 2].view(torch.bfloat16).view(H, S, D)[:, off(p)]
                      for b in range(B)])
    vs = torch.stack([pool[blk(b, p), H * S * D * 2:].view(torch.bfloat16).view(H, S, D)[:, off(p)]
                      for b in range(B)])
    return ks, vs


def reference(pool, kv_quant):
    """Fp32 decode reference over the pool's (dequantized) contents."""
    if kv_quant == 8:
        codes = pool[:, :2 * H * S * D].view(NB, H, 2, S, D).float()
        params = pool[:, 2 * H * S * D:].view(torch.bfloat16).view(NB, H, 2, S, 2).float()
        kf = codes[:, :, 0] * params[:, :, 0, :, 0:1] + params[:, :, 0, :, 1:2]
        vf = codes[:, :, 1] * params[:, :, 1, :, 0:1] + params[:, :, 1, :, 1:2]
    else:
        bf = pool.view(torch.bfloat16)
        kf = bf.view(NB, H, 2, S, D)[:, :, 0]
        vf = bf.view(NB, H, 2, S, D)[:, :, 1]
    bidx = (torch.arange(B).unsqueeze(1) * BPS + torch.arange(KLEN).unsqueeze(0) // S)
    poff = torch.arange(KLEN) % S
    kk = kf[bidx.to(kf.device), :, poff.to(kf.device)]  # [B, KLEN, H, D]
    vv = vf[bidx.to(vf.device), :, poff.to(vf.device)]
    group = HQ // H
    kk = kk.repeat_interleave(group, dim=2).float()
    vv = vv.repeat_interleave(group, dim=2).float()
    scores = torch.einsum('bhd,bkhd->bhk', q.float(), kk) / D**0.5
    p = scores.softmax(-1)
    return torch.einsum('bhk,bkhd->bhd', p, vv)


failures = []


def check(name, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    print(f'[{tag}] {name}' + (f'  ({detail})' if detail else ''))
    if not cond:
        failures.append(name)


# ---- Case 1: bf16 pool, real k/v -> bit-identical pool, correct output ----
pool = build_pool_bf16()
before = pool.clone()
out = decode(pool, kv_quant=0)
ref = reference(pool, 0)
err = (out.view(B, HQ, D).float() - ref).abs().max().item()
check('bf16: pool byte-identical after decode', bool(torch.equal(pool, before)))
check('bf16: output vs fp32 reference', err < 2e-2, f'max_abs={err:.5f}')

# ---- Case 2: int8 pool, real k/v -> idempotent re-store within rounding ----
pool = build_pool_int8()
before = pool.clone()
out = decode(pool, kv_quant=8)
after = snapshot_int8(pool, LAST)
b4 = snapshot_int8(before, LAST)
mid_b, mid_a = snapshot_int8(before, MID), snapshot_int8(pool, MID)
code_delta = (after[0].int() - b4[0].int()).abs().max().item()
code_delta_v = (after[1].int() - b4[1].int()).abs().max().item()
param_ok = (torch.allclose(after[2].float(), b4[2].float(), rtol=1e-2, atol=1e-3)
            and torch.allclose(after[3].float(), b4[3].float(), rtol=1e-2, atol=1e-3))
# Kernel vs torch rounding chains differ (bf16 stats + cvt.rni vs fp32 round):
# codes agree within +-2/255 of the value range; the dequant-output check is
# the semantic gate.
check('int8: last-slot K codes within +-2', code_delta <= 2, f'max_code_delta={code_delta}')
check('int8: last-slot V codes within +-2', code_delta_v <= 2, f'max_code_delta={code_delta_v}')
check('int8: last-slot params ~equal (scale,min)', param_ok)
check('int8: mid slot untouched', all(torch.equal(a, b) for a, b in zip(mid_b, mid_a)))
ref = reference(pool, 8)
err = (out.view(B, HQ, D).float() - ref).abs().max().item()
check('int8: output vs dequant reference', err < 0.25, f'max_abs={err:.5f}')

# ---- Case 3: int8 pool + scaled k/v -> store is live and caller-driven ----
pool = build_pool_int8()
before = pool.clone()
kl2, vl2 = last_kv(2 * k, 2 * v)
decode(pool, kl2, vl2, kv_quant=8)
kc2, ks2, km2 = quant_kv(2 * k[torch.tensor([blk(b, LAST) for b in range(B)]), off(LAST)])
after = snapshot_int8(pool, LAST)
before_s = snapshot_int8(before, LAST)
delta = (after[0].int() - kc2.int()).abs().max().item()
scale_diff = (after[2][..., 0].float() - 2 * before_s[2][..., 0].float()).abs().max().item()
min_diff = (after[2][..., 1].float() - 2 * before_s[2][..., 1].float()).abs().max().item()
changed = not torch.equal(pool, before)
check('int8: pool changed under scaled k/v', changed)
check('int8: last-slot codes == quant(2*k_last)', delta <= 2, f'max_code_delta={delta}')
check('int8: last-slot scale ~= 2x original', scale_diff < 0.05, f'max_abs_diff={scale_diff:.5f}')
check('int8: last-slot min ~= 2x original', min_diff < 0.06, f'max_abs_diff={min_diff:.5f}')

# ---- Case 4: zero k/v -> the reported zero-write, as a caller violation ----
pool = build_pool_int8()
decode(pool, torch.zeros(B, H * D, dtype=torch.bfloat16, device='cuda'),
       torch.zeros(B, H * D, dtype=torch.bfloat16, device='cuda'), kv_quant=8)
after = snapshot_int8(pool, LAST)
zeroed = (int(after[0].sum()) == 0 and int(after[1].sum()) == 0
          and float(after[2].float().abs().sum()) == 0.0
          and float(after[3].float().abs().sum()) == 0.0)
check('zero k/v: last slot zeroed (contract violation only)', zeroed)

# ---- Case 5: timed entry keeps the pool stable across iterations ----
# `attn_decode_bench` allocates from the ambient TurboMind context (no
# internal ContextGuard), so it runs under `create_device_context` like
# benchmark/bench_attn_decode.py does.
pool = build_pool_bf16()
before = pool.clone()
with _tm.create_device_context(torch.cuda.current_stream().cuda_stream):
    ptrs = ptr_table(pool)
    cu_block_nums = torch.arange(0, (B + 1) * BPS, BPS, dtype=torch.int32, device='cuda')
    out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')
    kl, vl = last_kv()
    _tm.attn_decode_bench(_tm.from_dlpack(q.reshape(B, -1).contiguous()), _tm.from_dlpack(kl),
                          _tm.from_dlpack(vl), _tm.from_dlpack(out),
                          ptrs.data_ptr(), cu_block_nums.data_ptr(), num_heads=HQ, num_kv_heads=H,
                          head_dim=D, batch_size=B, block_len=S, max_k_len=KLEN,
                          kv_quant=0, iters=10, warmup=3,
                          stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
check('bench: bf16 pool byte-identical after 13 dispatches', bool(torch.equal(pool, before)))

print()
if failures:
    print(f'FAILED: {failures}')
    sys.exit(1)
print('ALL CASES PASSED')
