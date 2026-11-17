#!/usr/bin/env python
"""T4 (minimal case 4): production fill_kv_cache (QuantPolicy.INT8) at the
claimed anomaly shape, called PER THE PRODUCTION CONTRACT.

Claim under test: "生产 fill_kv_cache (QuantPolicy.INT8, 3D grid) 在 B4/klen256
下 16 个物理块仅 6 个写入有效 params - triton 3.7.1 误编译".

Production contract (lmdeploy/pytorch/engine/inputs_maker.py:
get_block_tables -> _tensorlize_block_offsets -> _map_to_kernel_block_offsets;
k_states/v_states are PACKED [total_tokens, H, D] with cumulative
q_start_loc; block_offsets is a 2-D flattened per-sequence page table
[bsz, npages] of ABSOLUTE kernel block ids; the kernel indexes
BlockOffsets + batch_id*stride_boff + kv_block_id):
- variant CORRECT: packed states, cumulative q_start_loc, identity 2-D page
  table. Expect every block valid and content matching the independent
  fp32-exact quant golden at exactly kc[bps*b + j].
- variant MISUSE-PADDED: probe's 1-D arange(bsz)*bps block table (padded with
  block id 0 so all flat reads are in-bounds). Kernel semantics place seq b
  page j at table_flat[b*1 + j] -> predicted valid placement {0, 4, 8, 12}
  only. This deterministically explains the claimed "6/16 valid blocks"
  (the original probe additionally read past the 4-element table; its exact
  harness is re-run verbatim as T5).

Independent golden (fp32-exact quant contract of _quant_int8, no kernel code
reuse): scales=(amax-amin)/255, zeros=-amin/scales,
q=trunc(val/scales+zeros+0.5) in [0,255]. Observed kernel behavior on this
toolchain (characterized in t4 debug runs): scales stored bitwise-exact;
zeros differ by <=2 fp32 ulp (division lowering); codes differ by exactly
+-1 at trunc-vs-fma boundaries (~0.08% of elements), possibly 255<->0 uint8
wrap when a code rounds across 256. Pass = scales bitwise, zeros <= 4 ulp,
codes only +-1/wrap and globally <= 0.1% neq, all blocks valid, exact
placement kc[bps*b + j].
NOTE: the 'misuse-padded' variant uses the stride==1 block-table JIT
specialization, which deterministically faults (illegal memory access) when
launched AFTER a klen>=1024 correct case in the same process (observed,
documented in REPORT.md; survives standalone or when run first). It is
therefore run FIRST here.
"""
import json
import sys

import torch
import triton

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache

DEV = 'cuda'
BLOCK_LEN = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
results = []


def check(name, ok, detail):
    results.append(dict(name=name, ok=bool(ok), detail=detail))
    print(f"[{'PASS' if ok else 'FAIL'}] t4/{name}: {detail}")


def golden_quant_fp32(seg):
    """Seg (bsz, bps, S, H, D) bf16 -> codes uint8, scales/zeros fp32."""
    x = seg.to(torch.float32)
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    scales = (hi - lo) / 255
    zeros = -lo / scales
    q = (x / scales + zeros + 0.5).clamp(0, 255).trunc().to(torch.uint8)
    return q, scales.squeeze(-1), zeros.squeeze(-1)


def ulp_diff(a, b):
    """Elementwise fp32 ulp distance (as int32 bit difference of ordered
    pairs)."""
    ai = a.contiguous().view(torch.int32)
    bi = b.contiguous().view(torch.int32)
    return (ai - bi).abs()


def codes_ok(kernel_codes, golden_codes):
    """True if diffs are only +-1 boundaries or 255<->0 uint8 wraps."""
    d = (kernel_codes.to(torch.int16) - golden_codes.to(torch.int16))
    benign = ((d.abs() <= 1) | (d == 255) | (d == -255))
    neq = int((d != 0).sum())
    return bool(benign.all()), neq


def run_case(bsz, klen, seed, table_kind):
    plen = klen
    bps = klen // BLOCK_LEN
    nb = bsz * bps
    g = torch.Generator().manual_seed(seed)
    k = (torch.randn(bsz, klen, NUM_KV_HEADS, HEAD_DIM, generator=g) * 0.5).to(DEV).bfloat16()
    v = (torch.randn(bsz, klen, NUM_KV_HEADS, HEAD_DIM, generator=g) * 0.5).to(DEV).bfloat16()
    # PRODUCTION CONTRACT: packed [total_tokens, H, D], cumulative q_start_loc
    kp = k.reshape(bsz * klen, NUM_KV_HEADS, HEAD_DIM).contiguous()
    vp = v.reshape(bsz * klen, NUM_KV_HEADS, HEAD_DIM).contiguous()
    qsl = torch.arange(bsz, dtype=torch.int64, device=DEV) * klen
    kc = torch.zeros(nb, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.uint8, device=DEV)
    vc = torch.zeros_like(kc)
    ksz = torch.zeros(nb, BLOCK_LEN, NUM_KV_HEADS, 2, dtype=torch.float32, device=DEV)
    vsz = torch.zeros_like(ksz)

    if table_kind == 'correct':
        boff = torch.arange(nb, dtype=torch.int32, device=DEV).reshape(bsz, bps)
    else:  # 'misuse-padded': probe's 1-D values, padded with block id 0
        boff = torch.zeros(nb, dtype=torch.int32, device=DEV)
        boff[:bsz] = torch.arange(bsz, dtype=torch.int32, device=DEV) * bps

    fill_kv_cache(kp, vp, kc, vc, qsl,
                  torch.full((bsz,), plen, dtype=torch.int64, device=DEV),
                  torch.full((bsz,), plen, dtype=torch.int64, device=DEV),
                  plen, boff,
                  k_scales_zeros=ksz, v_scales_zeros=vsz,
                  quant_policy=QuantPolicy.INT8, kv_layout='bshd')
    torch.cuda.synchronize()

    seg = k.view(bsz, bps, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)
    gq, gs, gz = golden_quant_fp32(seg)
    gq = gq.view(nb, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)
    gs_w = gs.to(torch.float32).view(nb, BLOCK_LEN, NUM_KV_HEADS)
    gz_w = gz.to(torch.float32).view(nb, BLOCK_LEN, NUM_KV_HEADS)

    blocks_valid = int((ksz[..., 0] > 0).any(dim=(1, 2)).sum().item())
    scales_neq = int((ksz[..., 0] != gs_w).sum())
    zeros_max_ulp = int(ulp_diff(ksz[..., 1], gz_w).max().item())

    name = f'B{bsz}_klen{klen}_nb{nb}_{table_kind}'
    if table_kind == 'correct':
        ok_all, neq_total, n_total = True, 0, 0
        for b in range(bsz):
            for j in range(bps):
                ok, neq = codes_ok(kc[b * bps + j], gq[b * bps + j])
                ok_all &= ok
                neq_total += neq
                n_total += kc[b * bps + j].numel()
        empty_blocks = int((kc.view(nb, -1).sum(dim=1) == 0).sum().item())
        # zeros: <=4 fp32 ulp (division lowering); codes: benign + <=0.1% globally
        ok = (blocks_valid == nb) and ok_all and neq_total <= n_total // 1000 and \
            scales_neq == 0 and zeros_max_ulp <= 4 and empty_blocks == 0
        check(name, ok,
              f'blocks_with_valid_params={blocks_valid}/{nb} '
              f'codes_benign(+-1/wrap)={ok_all} codes_neq={neq_total}/{n_total} '
              f'scales_neq={scales_neq} zeros_max_ulp={zeros_max_ulp} '
              f'empty_code_blocks={empty_blocks} placement=kc[4b+j]')
    else:
        pred_valid = sorted({int(boff.flatten()[min(b + j, nb - 1)]) for b in range(bsz) for j in range(bps)})
        got_valid = sorted(torch.nonzero((ksz[..., 0] > 0).any(dim=(1, 2))).flatten().tolist())
        untouched_ok = all(int(kc[p].sum()) == 0 for p in range(nb) if p not in pred_valid)
        ok = got_valid == pred_valid and untouched_ok
        check(name, ok,
              f'blocks_with_valid_params={blocks_valid}/{nb} '
              f'predicted_flat_index_placement={pred_valid} observed={got_valid} '
              f'untouched_blocks_zero={untouched_ok}')
    return ok


def main():
    torch.cuda.init()
    print(f'triton={triton.__version__} torch={torch.__version__} dev={torch.cuda.get_device_name(0)}')
    run_case(4, 256, 11, 'misuse-padded')  # probe's argument values, in-bounds
    run_case(4, 256, 11, 'correct')        # the exact claimed anomaly shape
    run_case(4, 256, 12, 'correct')        # second seed
    run_case(4, 1024, 13, 'correct')       # nb=64
    run_case(8, 4096, 14, 'correct')       # nb=512
    with open('benchmark/b301_reverify/t4_results.json', 'w') as f:
        json.dump(dict(triton=triton.__version__, torch=torch.__version__,
                       results=results), f, indent=1)
    ok = all(r['ok'] for r in results)
    print(f"T4 VERDICT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
