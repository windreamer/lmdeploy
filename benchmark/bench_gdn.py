# Copyright (c) OpenMMLab. All rights reserved.
"""Batch-2 benchmark 2: TM GatedDeltaRule kernel vs PT fla-chunk / TileLang recurrent.

Sequence-step alignment: the decode case advances the recurrent state one
token per call on both sides (TM recurrent mode vs PT fused_recurrent); the
prefill case processes T tokens in one chunked call (TM chunked mode vs PT
fla chunk_gated_delta_rule). State is fed directly as tensors / pointer
arrays; no state-pool machinery is exercised. Kernel time only.
"""

import argparse
import json
import statistics

import torch

from lmdeploy.turbomind import _tm


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


def make_inputs(bsz, tokens, hv, head_dim, device='cuda'):
    """Q/k/v bf16 (l2-normalized q/k), g fp32 log-decay, beta fp32 sigmoid."""
    gen = torch.Generator(device='cpu').manual_seed(11)
    q = torch.randn(bsz, tokens, hv, head_dim, generator=gen).to(device=device, dtype=torch.bfloat16)
    k = torch.randn(bsz, tokens, hv, head_dim, generator=gen).to(device=device, dtype=torch.bfloat16)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.randn(bsz, tokens, hv, head_dim, generator=gen).to(device=device, dtype=torch.bfloat16)
    g = -torch.rand(bsz, tokens, hv, generator=gen).to(device=device, dtype=torch.float32) * 0.1 - 0.01
    beta = torch.rand(bsz, tokens, hv, generator=gen).to(device=device, dtype=torch.float32) * 0.5 + 0.25
    return q, k, v, g, beta


def tm_state_ptrs(state: torch.Tensor, bsz: int):
    """Device int64 pointer array [1, B, 1] -> per-sequence state slab."""
    slab = state.stride(0) * state.element_size()
    ptrs = torch.tensor([state.data_ptr() + i * slab for i in range(bsz)], dtype=torch.int64, device='cuda')
    return ptrs.view(1, bsz, 1)


def run_tm_step(q, k, v, g, beta, state, mode, iters, warmup, stream, chunk_size=None, state_layout='kv'):
    """One TM call per timed iteration (one decode step or one chunked call).

    `state_layout` selects the recurrent state addressing variant:
    'kv' = TM native [K,V], 'vk' = PT pool [V,K] (zero-copy direct read).
    """
    bsz, tokens = q.shape[:2]
    if mode == 'recurrent':
        plan = _tm.delta_rule_plan(_tm.from_dlpack(q), _tm.from_dlpack(k), _tm.from_dlpack(v),
                                   _tm.from_dlpack(g), _tm.from_dlpack(beta),
                                   state_dtype='f32', mode='recurrent', state_layout=state_layout)
    else:
        q_offsets = torch.arange(0, (bsz + 1) * tokens, tokens, dtype=torch.int32, device='cuda')
        plan = _tm.delta_rule_plan(_tm.from_dlpack(q), _tm.from_dlpack(k), _tm.from_dlpack(v),
                                   _tm.from_dlpack(g), _tm.from_dlpack(beta),
                                   q_offsets=_tm.from_dlpack(q_offsets),
                                   state_dtype='f32', mode='chunked', chunk_size=chunk_size)
    out_shape = tuple(plan['out']['shape'])
    out = torch.empty(out_shape, dtype=q.dtype, device='cuda')
    state_ptrs = tm_state_ptrs(state, bsz)
    q_offsets = torch.arange(0, (bsz + 1) * tokens, tokens, dtype=torch.int32, device='cuda') \
        if mode == 'chunked' else None
    finished = torch.zeros(bsz, dtype=torch.bool, device='cuda')
    args = dict(
        q=_tm.from_dlpack(q), k=_tm.from_dlpack(k), v=_tm.from_dlpack(v),
        g=_tm.from_dlpack(g), beta=_tm.from_dlpack(beta), plan=plan,
        out=_tm.from_dlpack(out),
        stream_ptr=stream, state_ptrs=_tm.from_dlpack(state_ptrs),
        finished=_tm.from_dlpack(finished))
    if plan['workspace']['storage_size'] > 0:
        workspace = torch.empty(int(plan['workspace']['storage_size']), dtype=torch.uint8, device='cuda')
        args['workspace'] = _tm.from_dlpack(workspace)
    if mode == 'chunked':
        args['q_offsets'] = _tm.from_dlpack(q_offsets)

    def call():
        # state may be updated in place per call; advancing it across timing
        # iterations is fine for perf (bounded decay keeps values finite).
        _tm.delta_rule_run(**args)

    return median_ms(call, iters, warmup), out


def fused_recurrent_pt(q, k, v, g, beta, state):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
    return fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=state,
        output_final_state=True, transpose_state_layout=False)


def fla_chunk_pt(q, k, v, g, beta, state):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=state,
                                  output_final_state=True, scale=None)


def pt_recurrent(q, k, v, g, beta, state, iters, warmup):
    def call():
        fused_recurrent_pt(q, k, v, g, beta, state)

    return median_ms(call, iters, warmup)


def pt_chunk(q, k, v, g, beta, state, iters, warmup):
    def call():
        fla_chunk_pt(q, k, v, g, beta, state)

    return median_ms(call, iters, warmup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--bsz', type=int, nargs='+', default=[1, 8, 32])
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--layout-json-out', default=None)
    args = ap.parse_args()

    hv, head_dim = 64, 128
    stream = torch.cuda.current_stream().cuda_stream
    ctx = _tm.create_device_context(stream)
    rows = []

    with ctx:

        # correctness gates: TM output (and in-place state) vs PT
        for mode, tokens in (('recurrent', 1), ('chunked', 128)):
            bsz = 2
            q, k, v, g, beta = make_inputs(bsz, tokens, hv, head_dim)
            state_tm = torch.randn(bsz, hv, head_dim, head_dim, device='cuda', dtype=torch.float32) * 0.1
            state_pt = state_tm.clone()
            # single call per side so the in-place advanced states are comparable
            _, tm_out = run_tm_step(q, k, v, g, beta, state_tm, mode, 1, 0, stream)
            print(f'[gate] {mode} q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} '
                  f'g={tuple(g.shape)} beta={tuple(beta.shape)} state_pt={tuple(state_pt.shape)}')
            if mode == 'recurrent':
                pt_o, pt_state_new = fused_recurrent_pt(q, k, v, g, beta, state_pt)
            else:
                pt_o, pt_state_new = fla_chunk_pt(q, k, v, g, beta, state_pt)
            print(f'[check] {mode} tm_out={tuple(tm_out.shape)} pt_o={tuple(pt_o.shape)} '
                  f'tm_state={tuple(state_tm.shape)} pt_state={tuple(pt_state_new.shape)}')
            o_err = (tm_out.float() - pt_o.float()).abs().max().item()
            s_err = (state_tm - pt_state_new).abs().max().item()
            tol = 0.1
            tag = 'ok' if o_err < tol and s_err < tol else 'MISMATCH'
            print(f'[check] {mode} out_max_abs={o_err:.4f} state_max_abs={s_err:.4f} -> {tag}')
            if tag != 'ok':
                import sys
                sys.exit(f'gdn correctness gate failed at {mode}')

        # transposed-state gate: the [V,K] variant kernel must reproduce the
        # [K,V] kernel bit-exactly (same math, same FP order, different
        # addressing). state_vk holds the same mathematical state as state_kv
        # in PT pool element order [N, H, V, K].
        bsz = 2
        q, k, v, g, beta = make_inputs(bsz, 1, hv, head_dim)
        state_kv = torch.randn(bsz, hv, head_dim, head_dim, device='cuda', dtype=torch.float32) * 0.1
        state_vk = state_kv.transpose(-1, -2).contiguous()
        plan_vk = _tm.delta_rule_plan(_tm.from_dlpack(q), _tm.from_dlpack(k), _tm.from_dlpack(v),
                                      _tm.from_dlpack(g), _tm.from_dlpack(beta),
                                      state_dtype='f32', mode='recurrent', state_layout='vk')
        vk_kernel = plan_vk['kernel'].get('state_layout'), plan_vk['kernel'].get('mode')
        _, out_kv = run_tm_step(q, k, v, g, beta, state_kv.clone(), 'recurrent', 1, 0, stream, state_layout='kv')
        _, out_vk = run_tm_step(q, k, v, g, beta, state_vk.clone(), 'recurrent', 1, 0, stream, state_layout='vk')
        o_err = (out_kv.float() - out_vk.float()).abs().max().item()
        s_err = (state_kv.transpose(-1, -2) - state_vk).abs().max().item()
        tag = 'ok' if o_err == 0.0 and s_err == 0.0 else 'MISMATCH'
        print(f'[gate] state_layout={vk_kernel} out_max_abs={o_err:.3e} final_state_max_abs={s_err:.3e} -> {tag}')
        if tag != 'ok':
            import sys
            sys.exit(f'gdn vk state gate failed: out={o_err} state={s_err}')

        # ---- decode: one token per call, recurrent state advanced per step
        # Three-way comparison per B:
        #   1. original kernel + [K,V] state (batch-2 baseline)
        #   2. [V,K] variant kernel + PT pool layout [N,H,V,K] (zero-copy form)
        #   3. PT fused_recurrent baseline (batch-2 reference)
        layout_rows = []
        for bsz in args.bsz:
            q, k, v, g, beta = make_inputs(bsz, 1, hv, head_dim)
            state_kv = torch.randn(bsz, hv, head_dim, head_dim, device='cuda', dtype=torch.float32) * 0.1
            # [V,K] pool slab per sequence, pointers taken straight off the
            # tensor: the zero-copy target form for scheme B.
            state_vk = state_kv.transpose(-1, -2).contiguous()
            tm_kv_ms, _ = run_tm_step(q, k, v, g, beta, state_kv, 'recurrent', args.iters, args.warmup, stream,
                                      state_layout='kv')
            tm_vk_ms, _ = run_tm_step(q, k, v, g, beta, state_vk, 'recurrent', args.iters, args.warmup, stream,
                                      state_layout='vk')
            pt_ms = pt_recurrent(q, k, v, g, beta, state_kv, args.iters, args.warmup)
            row = dict(mode='recurrent', bsz=bsz, tokens=1, layout='kv', tm_ms=round(tm_kv_ms, 5),
                       pt_ms=round(pt_ms, 5), speedup_pt_over_tm=round(pt_ms / tm_kv_ms, 3))
            rows.append(row)
            layout_rows.append(row)
            row_vk = dict(mode='recurrent', bsz=bsz, tokens=1, layout='vk', tm_ms=round(tm_vk_ms, 5),
                          pt_ms=round(pt_ms, 5), speedup_pt_over_tm=round(pt_ms / tm_vk_ms, 3),
                          transpose_penalty_vs_kv=round(tm_vk_ms / tm_kv_ms, 3))
            rows.append(row_vk)
            layout_rows.append(row_vk)
            print('decode  B={bsz:>3}  TM-kv {tm_ms:.4f} ms  TM-vk {tvk:.4f} ms  PT {pt_ms:.4f} ms  '
                  'vk/kv x{tp}  PT/vk x{sv}'.format(bsz=bsz, tm_ms=tm_kv_ms, tvk=tm_vk_ms, pt_ms=pt_ms,
                                                    tp=row_vk['transpose_penalty_vs_kv'],
                                                    sv=row_vk['speedup_pt_over_tm']))

        # ---- chunked prefill: T tokens in one call
        for bsz, tokens in ((1, 256), (1, 1024), (8, 256)):
            q, k, v, g, beta = make_inputs(bsz, tokens, hv, head_dim)
            state = torch.randn(bsz, hv, head_dim, head_dim, device='cuda', dtype=torch.float32) * 0.1
            tm_ms, _ = run_tm_step(q, k, v, g, beta, state, 'chunked', args.iters, args.warmup, stream)
            pt_ms = pt_chunk(q, k, v, g, beta, state, args.iters, args.warmup)
            row = dict(mode='chunked', bsz=bsz, tokens=tokens, tm_ms=round(tm_ms, 5), pt_ms=round(pt_ms, 5),
                       speedup_pt_over_tm=round(pt_ms / tm_ms, 3))
            rows.append(row)
            print('chunk   B={bsz:>3} T={tokens:>5}  TM {tm_ms:.4f} ms  '
                  'PT {pt_ms:.4f} ms  PT/TM x{speedup_pt_over_tm}'.format(**row))

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rows, f, indent=1)
        print(f'wrote {args.json_out}')
    if args.layout_json_out:
        with open(args.layout_json_out, 'w') as f:
            json.dump(layout_rows, f, indent=1)
        print(f'wrote {args.layout_json_out}')


if __name__ == '__main__':
    main()
