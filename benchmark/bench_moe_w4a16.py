# Copyright (c) OpenMMLab. All rights reserved.
"""Batch-2 benchmark 1: TM grouped-GEMM W4A16 MoE vs PT triton fused_moe_w4a16.

GLM-5.3-Flash sparse-layer geometry: 288 experts, top-8 normalized routing,
moe_intermediate 2048, hidden 4096; m in {1, 8, 32} decode tokens.

Layer-level timing on both sides, token-direct input, no state pools:
- TM: grouped gate_up GEMM (fused kGatedSilu epilogue when the plan supports
  it on this arch, else two grouped GEMMs + activation) -> grouped down GEMM
  -> invokeMoeCombine (token-major weighted sum).
- PT: triton fused_moe_w4a16 (route sort + gate_up GEMM + silu_and_mul + down
  GEMM + weighted scatter), the production W4A16 MoE path.

Each backend uses its native W4A16 format: TM u4 group-128
(CompressedTensorFormat), PT int4 group-32 (compressed-tensors kernel). Same
source geometry and routing; quant granularity is a format property, called
out in the report. Weights are generated and quantized expert-by-expert to
stay under the 24 GB budget.
"""

import argparse
import json
import random
import statistics

import torch
import torch.nn.functional as F

from lmdeploy.turbomind import _tm
from lmdeploy.turbomind.linear import get_linear
from lmdeploy.turbomind.weight_format import CompressedTensorFormat

NUM_EXPERTS = 288
TOP_K = 8
INTER = 2048
HIDDEN = 4096
GROUP_TM = 128
GROUP_PT = 32


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


def build_routing(m: int, seed: int = 5489, device='cuda'):
    """topk_ids/weights [m, topk] plus TM expert-major slot packing.

    TM layout (batch-1 fixture convention):   f2n[slot] = token; en2f[e * m + tok] = slot; offsets[e] prefix counts;
    scales[e * m + tok] = routing weight.
    """
    rng = random.Random(seed)
    ids = torch.tensor([rng.sample(range(NUM_EXPERTS), TOP_K) for _ in range(m)], dtype=torch.int64)
    w = torch.rand(m, TOP_K, generator=torch.Generator().manual_seed(seed + 1)) + 0.1
    w = (w / w.sum(-1, keepdim=True)).float()

    slots = m * TOP_K
    counts = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int64)
    for e in ids.flatten().tolist():
        counts[e + 1] += 1
    offsets = counts.cumsum(0)
    cursor = offsets.clone()
    f2n = torch.empty(slots, dtype=torch.int64)
    en2f = torch.empty(slots, dtype=torch.int64)
    scales = torch.empty(slots, dtype=torch.float32)
    for t in range(m):
        for j, e in enumerate(ids[t].tolist()):
            slot = int(cursor[e])
            cursor[e] += 1
            f2n[slot] = t
            en2f[j * m + t] = slot
            scales[j * m + t] = w[t, j]

    to = lambda t, dt: t.to(dt).to(device)  # noqa: E731
    return dict(
        topk_ids=to(ids, torch.int32),
        topk_weights=to(w, torch.float32),
        f2n=to(f2n, torch.int32),
        en2f=to(en2f, torch.int32),
        offsets=to(offsets, torch.int32),
        scales=to(scales, torch.float32))


def quant_tm_expert(source: torch.Tensor, group: int):
    """Per-expert u4 group-wise quant via TM kernels -> packed int32, scales,
    zeros."""
    tm = _tm
    out_dim, in_dim = source.shape
    raw = torch.empty((out_dim, in_dim // 8), dtype=torch.int32, device=source.device)
    scales = torch.empty((out_dim, in_dim // group), dtype=source.dtype, device=source.device)
    zeros = torch.empty_like(scales)
    dequant = torch.empty_like(source)
    tm.QuantizeGroupwise(quant=tm.from_dlpack(raw).reinterpret(tm.DataType.TYPE_UINT4, [out_dim, in_dim]),
                         scales=tm.from_dlpack(scales),
                         zeros=tm.from_dlpack(zeros),
                         dequant=tm.from_dlpack(dequant),
                         src=tm.from_dlpack(source),
                         group_size=group)
    return raw, scales, zeros


def quant_pt_expert(source_nk: torch.Tensor, group: int):
    """Per-expert int4 g32 quant with PT kernel math: dequant = (code - 8) * scale.

    source_nk: [N, K] -> packed int32 [N, K/8] (nibble i = channel 8w + i),
    scales bf16 [N, K/g].
    """
    n, k = source_nk.shape
    wg = source_nk.view(n, k // group, group).float()
    wmin = wg.amin(-1, keepdim=True)
    wmax = wg.amax(-1, keepdim=True)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    codes = torch.round((wg - wmin) / scale + 8.0).clamp(0, 15).to(torch.int64)
    codes = codes.view(n, k)
    shifts = (torch.arange(8, device=source_nk.device) * 4)
    packed = (codes.view(n, k // 8, 8) << shifts).sum(-1).to(torch.int32)
    scales = scale.expand(n, k // group, group)[..., 0].to(torch.float16)
    return packed, scales


def gen_source(n: int, k: int, seed: int, device) -> torch.Tensor:
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randn(n, k, generator=gen) * (0.1 / k**0.5)).to(device=device, dtype=torch.float16)


class TmMoeLayer:
    """TM grouped W4A16 gate_up (+silu) and down weights with exec plans."""

    def __init__(self, fuse_silu: bool, device):
        self.fuse_silu = fuse_silu
        self.linear = get_linear()
        self.plan13 = self.linear.get_weight_plan(
            weight_format=CompressedTensorFormat(block_in=GROUP_TM),
            dtype=torch.float16,
            input_dtype=torch.float16,
            grouped=True,
            fusion_type='silu' if fuse_silu else None)
        self.plan2 = self.linear.get_weight_plan(
            weight_format=CompressedTensorFormat(block_in=GROUP_TM),
            dtype=torch.float16,
            input_dtype=torch.float16,
            grouped=True,
            fusion_type=None)

        gate_pack, up_pack, _, _, _, _ = [], [], [], [], [], []
        for e in range(NUM_EXPERTS):
            gate_pack.append(quant_tm_expert(gen_source(INTER, HIDDEN, 100 + e, device), GROUP_TM))
            up_pack.append(quant_tm_expert(gen_source(INTER, HIDDEN, 200 + e, device), GROUP_TM))
        gate = [p[0] for p in gate_pack]
        up = [p[0] for p in up_pack]
        gate_scales = [p[1] for p in gate_pack]
        up_scales = [p[1] for p in up_pack]
        gate_zeros = [p[2] for p in gate_pack]
        up_zeros = [p[2] for p in up_pack]

        if fuse_silu:
            self.w13 = self.linear.fuse_weight((gate, up), plan=self.plan13,
                                               scales=(gate_scales, up_scales),
                                               zeros=(gate_zeros, up_zeros))
        else:
            self.w_gate = self.linear.prepare_weight(gate, plan=self.plan13,
                                                     scales=gate_scales, zeros=gate_zeros)
            self.w_up = self.linear.prepare_weight(up, plan=self.plan13,
                                                   scales=up_scales, zeros=up_zeros)

        w2_pack, _, _ = [], [], []
        for e in range(NUM_EXPERTS):
            w2_pack.append(quant_tm_expert(gen_source(HIDDEN, INTER, 300 + e, device), GROUP_TM))
        self.w2 = self.linear.prepare_weight([p[0] for p in w2_pack], plan=self.plan2,
                                             scales=[p[1] for p in w2_pack],
                                             zeros=[p[2] for p in w2_pack])

    def exec_plans(self, x, routing):
        """Exec plans for one token count (gate_up indexed, down non-
        indexed)."""
        exec13 = self.linear.get_exec_plan(x, self.w13 if self.fuse_silu else self.w_gate,
                                           offsets=routing['offsets'], indices=routing['f2n'])
        if not self.fuse_silu:
            # gate and up share the plan shape; build the up plan to warm its cache
            self.linear.get_exec_plan(x, self.w_up, offsets=routing['offsets'], indices=routing['f2n'])
        slots = x.shape[0] * TOP_K
        inter = torch.empty(slots, INTER, dtype=x.dtype, device=x.device)
        exec2 = self.linear.get_exec_plan(inter, self.w2, offsets=routing['offsets'])
        return exec13, exec2


class PtMoeLayer:
    """PT triton W4A16 gate_up / down packed weights."""

    def __init__(self, device):
        gate_up_packed = torch.empty(NUM_EXPERTS, 2 * INTER, HIDDEN // 8, dtype=torch.int32, device=device)
        gate_up_scale = torch.empty(NUM_EXPERTS, 2 * INTER, HIDDEN // GROUP_PT, dtype=torch.bfloat16, device=device)
        down_packed = torch.empty(NUM_EXPERTS, HIDDEN, INTER // 8, dtype=torch.int32, device=device)
        down_scale = torch.empty(NUM_EXPERTS, HIDDEN, INTER // GROUP_PT, dtype=torch.bfloat16, device=device)
        for e in range(NUM_EXPERTS):
            src = gen_source(2 * INTER, HIDDEN, 400 + e, device)
            gate_up_packed[e], gate_up_scale[e] = quant_pt_expert(src, GROUP_PT)
            del src
            src = gen_source(HIDDEN, INTER, 500 + e, device)
            down_packed[e], down_scale[e] = quant_pt_expert(src, GROUP_PT)
            del src
        self.gate_up_packed, self.gate_up_scale = gate_up_packed, gate_up_scale
        self.down_packed, self.down_scale = down_packed, down_scale

    def forward(self, x, routing):
        from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16
        return fused_moe_w4a16(
            x, self.gate_up_packed, self.gate_up_scale, self.down_packed, self.down_scale,
            routing['topk_weights'], routing['topk_ids'], TOP_K,
            renormalize=False, num_bits=4, group_size=GROUP_PT)


def tm_layer_forward(layer: TmMoeLayer, x, routing, exec13, exec2, bufs):
    linear = layer.linear
    r_f2n, r_en2f, r_off = routing['f2n'], routing['en2f'], routing['offsets']
    if layer.fuse_silu:
        inter, _ = linear.forward_moe(x, layer.w13, exec_plan=exec13,
                                      offsets=r_off, indices=r_f2n, out=bufs['inter'])
    else:
        gating, _ = linear.forward_moe(x, layer.w_gate, exec_plan=exec13,
                                       offsets=r_off, indices=r_f2n, out=bufs['gate'])
        up, _ = linear.forward_moe(x, layer.w_up, exec_plan=exec13,
                                   offsets=r_off, indices=r_f2n, out=bufs['up'])
        F.silu(gating, inplace=True)
        inter = bufs['inter'] * up
    temp, _ = linear.forward_moe(inter, layer.w2, exec_plan=exec2, offsets=r_off, out=bufs['temp'])
    _tm.invokeMoeCombine(_tm.from_dlpack(bufs['out']),
                         _tm.from_dlpack(temp),
                         scales=_tm.from_dlpack(routing['scales']),
                         en2f=_tm.from_dlpack(r_en2f),
                         experts_per_token=TOP_K)
    return bufs['out']


def tm_layer_timed(layer, x, routing, exec13, exec2, bufs, iters, warmup):
    linear = layer.linear
    r_f2n, r_off = routing['f2n'], routing['offsets']
    scales_d = _tm.from_dlpack(routing['scales'])
    en2f_d = _tm.from_dlpack(routing['en2f'])
    out_d = _tm.from_dlpack(bufs['out'])
    temp_d = _tm.from_dlpack(bufs['temp'])
    _ = _tm.from_dlpack(bufs['inter'])

    def call():
        if layer.fuse_silu:
            linear.forward_moe(x, layer.w13, exec_plan=exec13, offsets=r_off,
                               indices=r_f2n, out=bufs['inter'])
            linear.forward_moe(bufs['inter'], layer.w2, exec_plan=exec2, offsets=r_off,
                               out=bufs['temp'])
        else:
            linear.forward_moe(x, layer.w_gate, exec_plan=exec13, offsets=r_off,
                               indices=r_f2n, out=bufs['gate'])
            linear.forward_moe(x, layer.w_up, exec_plan=exec13, offsets=r_off,
                               indices=r_f2n, out=bufs['up'])
            F.silu(bufs['gate'], inplace=True)
            inter = bufs['inter'] * bufs['up']
            linear.forward_moe(inter, layer.w2, exec_plan=exec2, offsets=r_off,
                               out=bufs['temp'])
        _tm.invokeMoeCombine(out_d, temp_d, scales=scales_d, en2f=en2f_d, experts_per_token=TOP_K)

    return median_ms(call, iters, warmup)


def correctness_gate(device):
    """Small-geometry functional gate: 8 experts, m=4.

    TM route (fused + separate) and PT route each run against their own
    dequantized-weight einsum reference; catches routing/combine plumbing
    bugs. W4 quant noise makes cross-backend comparison loose, so only
    self-reference agreement is gated.
    """
    from tests.turbomind.linear.fixture import moe_reference

    e_num, m = 8, 4

    def small_routing():
        rng = random.Random(99)
        ids = torch.tensor([rng.sample(range(e_num), TOP_K) for _ in range(m)], dtype=torch.int64)
        w = torch.rand(m, TOP_K, generator=torch.Generator().manual_seed(100)) + 0.1
        w = (w / w.sum(-1, keepdim=True)).float()
        slots = m * TOP_K
        counts = torch.zeros(e_num + 1, dtype=torch.int64)
        for e in ids.flatten().tolist():
            counts[e + 1] += 1
        offsets = counts.cumsum(0)
        cursor = offsets.clone()
        f2n = torch.empty(slots, dtype=torch.int64)
        en2f = torch.empty(slots, dtype=torch.int64)
        scales = torch.empty(slots, dtype=torch.float32)
        for t in range(m):
            for j, e in enumerate(ids[t].tolist()):
                slot = int(cursor[e])
                cursor[e] += 1
                f2n[slot] = t
                en2f[j * m + t] = slot
                scales[j * m + t] = w[t, j]
        to = lambda t, dt: t.to(dt).to(device)  # noqa: E731
        return dict(topk_ids=to(ids, torch.int32), topk_weights=to(w, torch.float32),
                    f2n=to(f2n, torch.int32), en2f=to(en2f, torch.int32),
                    offsets=to(offsets, torch.int32), scales=to(scales, torch.float32))

    routing = small_routing()
    x = torch.randn(m, HIDDEN, device=device, dtype=torch.float16) * 0.5

    # ---- TM route: 8-expert layer with dequant copies kept for reference
    linear = get_linear()
    plan13 = linear.get_weight_plan(weight_format=CompressedTensorFormat(block_in=GROUP_TM),
                                    dtype=torch.float16, input_dtype=torch.float16,
                                    grouped=True, fusion_type='silu')
    plan2 = linear.get_weight_plan(weight_format=CompressedTensorFormat(block_in=GROUP_TM),
                                   dtype=torch.float16, input_dtype=torch.float16,
                                   grouped=True, fusion_type=None)
    gate_w, gate_deq, up_w, up_deq, w2_p, w2_deq = [], [], [], [], [], []
    for e in range(e_num):
        pp, d = quant_tm_full(gen_source(INTER, HIDDEN, 900 + e, device), GROUP_TM)
        gate_w.append(pp), gate_deq.append(d.t())
        pp, d = quant_tm_full(gen_source(INTER, HIDDEN, 950 + e, device), GROUP_TM)
        up_w.append(pp), up_deq.append(d.t())
        pp, d = quant_tm_full(gen_source(HIDDEN, INTER, 975 + e, device), GROUP_TM)
        w2_p.append(pp), w2_deq.append(d.t())
    w2 = linear.prepare_weight([p[0] for p in w2_p], plan=plan2,
                               scales=[p[1] for p in w2_p], zeros=[p[2] for p in w2_p])
    w_gate = linear.prepare_weight([p[0] for p in gate_w], plan=plan13,
                                   scales=[p[1] for p in gate_w], zeros=[p[2] for p in gate_w])
    w_up = linear.prepare_weight([p[0] for p in up_w], plan=plan13,
                                 scales=[p[1] for p in up_w], zeros=[p[2] for p in up_w])

    ref_gate = moe_reference(x, gate_deq, routing['f2n'], routing['offsets'], routing['scales'],
                             routing['en2f'], TOP_K, False)
    ref_up = moe_reference(x, up_deq, routing['f2n'], routing['offsets'], routing['scales'],
                           routing['en2f'], TOP_K, False)
    silu = F.silu(ref_gate.float()) * ref_up.float()
    ref = moe_reference(silu.to(torch.float16), w2_deq, None, routing['offsets'],
                        routing['scales'], routing['en2f'], TOP_K, True)

    def combine(temp):
        out = torch.empty(m, HIDDEN, dtype=torch.float16, device=device)
        _tm.invokeMoeCombine(_tm.from_dlpack(out), _tm.from_dlpack(temp),
                             scales=_tm.from_dlpack(routing['scales']),
                             en2f=_tm.from_dlpack(routing['en2f']),
                             experts_per_token=TOP_K)
        return out

    # fused kGatedSilu route (may be unsupported on this arch)
    tm_err = float('nan')
    try:
        w13 = linear.fuse_weight(([p[0] for p in gate_w], [p[0] for p in up_w]), plan=plan13,
                                 scales=([p[1] for p in gate_w], [p[1] for p in up_w]),
                                 zeros=([p[2] for p in gate_w], [p[2] for p in up_w]))
        exec13 = linear.get_exec_plan(x, w13, offsets=routing['offsets'], indices=routing['f2n'])
        inter, _ = linear.forward_moe(x, w13, exec_plan=exec13, offsets=routing['offsets'],
                                      indices=routing['f2n'])
        exec2 = linear.get_exec_plan(inter, w2, offsets=routing['offsets'])
        temp, _ = linear.forward_moe(inter, w2, exec_plan=exec2, offsets=routing['offsets'])
        tm_err = (combine(temp).float() - ref.float()).abs().max().item()
    except NotImplementedError as e:
        print(f'[gate][moe] fused silu route unsupported: {e}')

    # separate w1/w3 route (the production TM path on this arch)
    exec13s = linear.get_exec_plan(x, w_gate, offsets=routing['offsets'], indices=routing['f2n'])
    gating, _ = linear.forward_moe(x, w_gate, exec_plan=exec13s, offsets=routing['offsets'],
                                   indices=routing['f2n'])
    up, _ = linear.forward_moe(x, w_up, exec_plan=exec13s, offsets=routing['offsets'],
                               indices=routing['f2n'])
    inter2 = F.silu(gating.float()) * up.float()
    exec2s = linear.get_exec_plan(inter2.to(torch.float16), w2, offsets=routing['offsets'])
    temp2, _ = linear.forward_moe(inter2.to(torch.float16), w2, exec_plan=exec2s,
                                  offsets=routing['offsets'])
    tm_sep_err = (combine(temp2).float() - ref.float()).abs().max().item()

    # ---- PT route: same small layer, dequant reference from PT kernel math
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16
    gu_packed = torch.empty(e_num, 2 * INTER, HIDDEN // 8, dtype=torch.int32, device=device)
    gu_scale = torch.empty(e_num, 2 * INTER, HIDDEN // GROUP_PT, dtype=torch.bfloat16, device=device)
    dn_packed = torch.empty(e_num, HIDDEN, INTER // 8, dtype=torch.int32, device=device)
    dn_scale = torch.empty(e_num, HIDDEN, INTER // GROUP_PT, dtype=torch.bfloat16, device=device)
    gu_deq, dn_deq = [], []
    for e in range(e_num):
        src = gen_source(2 * INTER, HIDDEN, 800 + e, device)
        gu_packed[e], gu_scale[e], dq = quant_pt_full(src, GROUP_PT)
        gu_deq.append(dq.T)
        del src
        src = gen_source(HIDDEN, INTER, 850 + e, device)
        dn_packed[e], dn_scale[e], dq = quant_pt_full(src, GROUP_PT)
        dn_deq.append(dq.T)
        del src
    pt_out = fused_moe_w4a16(x, gu_packed, gu_scale, dn_packed, dn_scale,
                             routing['topk_weights'], routing['topk_ids'], TOP_K,
                             renormalize=False, num_bits=4, group_size=GROUP_PT)
    pt_ref1 = moe_reference(x, gu_deq, routing['f2n'], routing['offsets'],
                            routing['scales'], routing['en2f'], TOP_K, False)
    silu = F.silu(pt_ref1[:, :INTER].float()) * pt_ref1[:, INTER:].float()
    pt_ref = moe_reference(silu.to(torch.float16), dn_deq, None,
                           routing['offsets'], routing['scales'], routing['en2f'], TOP_K, True)
    pt_err = (pt_out.float() - pt_ref.float()).abs().max().item()

    ok = tm_sep_err < 0.05 and pt_err < 0.05 and (tm_err < 0.05 or tm_err != tm_err)
    print(f'[gate][moe] tm_fused={tm_err:.4f} tm_separate={tm_sep_err:.4f} pt={pt_err:.4f} '
          f'-> {"ok" if ok else "MISMATCH"}')
    return ok


def quant_tm_full(source: torch.Tensor, group: int):
    """Packed, scales, zeros, dequant for the small-geometry gate."""
    tm = _tm
    out_dim, in_dim = source.shape
    raw = torch.empty((out_dim, in_dim // 8), dtype=torch.int32, device=source.device)
    scales = torch.empty((out_dim, in_dim // group), dtype=source.dtype, device=source.device)
    zeros = torch.empty_like(scales)
    dequant = torch.empty_like(source)
    tm.QuantizeGroupwise(quant=tm.from_dlpack(raw).reinterpret(tm.DataType.TYPE_UINT4, [out_dim, in_dim]),
                         scales=tm.from_dlpack(scales),
                         zeros=tm.from_dlpack(zeros),
                         dequant=tm.from_dlpack(dequant),
                         src=tm.from_dlpack(source),
                         group_size=group)
    return (raw, scales, zeros), dequant


def quant_pt_full(source_nk: torch.Tensor, group: int):
    """Packed, scales, dequant (kernel math) for the small-geometry gate."""
    n, k = source_nk.shape
    wg = source_nk.view(n, k // group, group).float()
    wmin = wg.amin(-1, keepdim=True)
    wmax = wg.amax(-1, keepdim=True)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    codes = torch.round((wg - wmin) / scale + 8.0).clamp(0, 15).to(torch.int64)
    shifts = torch.arange(8, device=source_nk.device) * 4
    packed = (codes.view(n, k // 8, 8) << shifts).sum(-1).to(torch.int32)
    scales = scale.expand(n, k // group, group)[..., 0].to(torch.bfloat16)
    dequant = ((codes.view(n, k) - 8).float() * scales.float().repeat_interleave(group, dim=1)).to(torch.float16)
    return packed, scales, dequant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--m', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--skip-pt', action='store_true')
    ap.add_argument('--skip-gate', action='store_true')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    device = torch.device('cuda')
    rows = []
    stream = torch.cuda.current_stream().cuda_stream
    ctx = _tm.create_device_context(stream)

    with ctx:
        if not args.skip_gate and not correctness_gate(device):
            raise SystemExit('moe correctness gate failed')
        run_grid(args, device, rows)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rows, f, indent=1)
        print(f'wrote {args.json_out}')


def run_grid(args, device, rows):
    tm_fused = None
    try:
        tm_fused = TmMoeLayer(True, device)
        # probe: does the kGatedSilu grouped plan actually run on this arch?
        r0 = build_routing(1)
        x0 = torch.randn(1, HIDDEN, device=device, dtype=torch.float16)
        bufs0 = {'inter': torch.empty(TOP_K, INTER, dtype=torch.float16, device=device),
                 'temp': torch.empty(TOP_K, HIDDEN, dtype=torch.float16, device=device),
                 'out': torch.empty(1, HIDDEN, dtype=torch.float16, device=device)}
        e13, e2 = tm_fused.exec_plans(x0, r0)
        tm_layer_forward(tm_fused, x0, r0, e13, e2, bufs0)
        torch.cuda.synchronize()
        print('[moe] TM fused kGatedSilu epilogue: supported')
    except Exception as e:
        print(f'[moe] TM fused kGatedSilu epilogue unavailable: {type(e).__name__}: {e}')
        tm_fused = None

    tm_plain = TmMoeLayer(False, device)

    pt_layer = None
    if not args.skip_pt:
        pt_layer = PtMoeLayer(device)

    for m in args.m:
        x = torch.randn(m, HIDDEN, device=device, dtype=torch.float16)
        routing = build_routing(m)
        slots = m * TOP_K
        bufs = {'inter': torch.empty(slots, INTER, dtype=torch.float16, device=device),
                'temp': torch.empty(slots, HIDDEN, dtype=torch.float16, device=device),
                'out': torch.empty(m, HIDDEN, dtype=torch.float16, device=device),
                'gate': torch.empty(slots, INTER, dtype=torch.float16, device=device),
                'up': torch.empty(slots, INTER, dtype=torch.float16, device=device)}

        row = dict(m=m)
        for name, layer in (('tm_fused', tm_fused), ('tm_plain', tm_plain)):
            if layer is None:
                continue
            exec13, exec2 = layer.exec_plans(x, routing)
            # one functional call first, then timed runs on the cached plans
            tm_layer_forward(layer, x, routing, exec13, exec2, bufs)
            torch.cuda.synchronize()
            row[name] = round(tm_layer_timed(layer, x, routing, exec13, exec2, bufs,
                                             args.iters, args.warmup), 4)
        if pt_layer is not None:
            pt_ms = median_ms(lambda: pt_layer.forward(x, routing), args.iters, args.warmup)
            row['pt'] = round(pt_ms, 4)
        best_tm = min(v for k, v in row.items() if k.startswith('tm'))
        row['pt_over_best_tm'] = round(row.get('pt', float('nan')) / best_tm, 3)
        rows.append(row)
        print('[moe] m={m:>3}  ' + '  '.join(f'{k}={v}' for k, v in row.items()))

        x = None
        del bufs
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
