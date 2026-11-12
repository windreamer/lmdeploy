# Copyright (c) OpenMMLab. All rights reserved.
"""Numerical comparison of the TurboMind gated-delta-rule path against the
existing FLA/TileLang stack, on both engine paths: prefill (chunked) and
decode (recurrent).

Requires a CUDA device with a built TurboMind native extension and a
capability inside the supported buckets (pre_sm90 / sm90 / sm120); the
factory table ships no row, so the TM path is forced explicitly here."""

import pytest
import torch

from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaMeta, GatedDeltaRuleBuildSpec

_cuda_available = torch.cuda.is_available()


def _tm_bridge_ready():
    try:
        from lmdeploy import turbomind
    except Exception:
        return False
    if not turbomind.is_available():
        return False
    return all(
        hasattr(turbomind._tm, sym)
        for sym in ('delta_rule_plan', 'delta_rule_run', 'from_dlpack',
                    'delta_rule_prepare_state_tma_descs'))


def _capability_supported():
    if not _cuda_available:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major < 9 or major == 9 or (major, minor) == (12, 0)


pytestmark = [
    pytest.mark.skipif(not _cuda_available, reason='CUDA required'),
    pytest.mark.skipif(not _tm_bridge_ready(), reason='TurboMind native extension required'),
    pytest.mark.skipif(not _capability_supported(),
                       reason='capability outside pre_sm90/sm90/sm120 buckets'),
]

from lmdeploy.pytorch.model_inputs import StepContextManager, set_step_ctx_manager  # noqa: E402

set_step_ctx_manager(StepContextManager())

NUM_K_HEADS = 2
NUM_V_HEADS = 4
HEAD_DIM = 128


def _make_rule(tm: bool):
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import _build_gated_delta_rule
    import os
    old = os.environ.get('LMDEPLOY_TURBOMIND_OPS')
    os.environ['LMDEPLOY_TURBOMIND_OPS'] = 'force=gated_delta' if tm else 'auto'
    try:
        return _build_gated_delta_rule(GatedDeltaRuleBuildSpec(indexed_state=False))
    finally:
        if old is None:
            os.environ.pop('LMDEPLOY_TURBOMIND_OPS', None)
        else:
            os.environ['LMDEPLOY_TURBOMIND_OPS'] = old


def _make_inputs(lengths, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    total = sum(lengths)
    q = ((torch.rand(1, total, NUM_K_HEADS, HEAD_DIM, generator=gen, device=device) - 0.5) *
         2).to(torch.bfloat16)
    k = ((torch.rand(1, total, NUM_K_HEADS, HEAD_DIM, generator=gen, device=device) - 0.5) *
         2).to(torch.bfloat16)
    v = ((torch.rand(1, total, NUM_V_HEADS, HEAD_DIM, generator=gen, device=device) - 0.5) *
         2).to(torch.bfloat16)
    b = torch.rand(1, total, NUM_K_HEADS, NUM_V_HEADS // NUM_K_HEADS,
                   generator=gen, device=device).to(torch.bfloat16)
    a = torch.rand(1, total, NUM_K_HEADS, NUM_V_HEADS // NUM_K_HEADS,
                   generator=gen, device=device).to(torch.bfloat16)
    dt_bias = torch.rand(NUM_V_HEADS, generator=gen, device=device) * 0.2
    a_log_exp = -torch.rand(NUM_V_HEADS, generator=gen, device=device) * 2
    return q, k, v, b, a, dt_bias, a_log_exp


def _prefill_meta(lengths, kv_lengths, state_ids, device):
    total = sum(lengths)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device=device, dtype=torch.int32)
    attn = type('Attn', (), {})()
    attn.is_decoding = False
    attn.cu_seqlens_q = cu
    attn.q_seqlens = torch.tensor(lengths, device=device, dtype=torch.int32)
    attn.kv_seqlens = torch.tensor(kv_lengths, device=device, dtype=torch.int32)
    return GatedDeltaMeta(total, 4, torch.tensor(state_ids, device=device), attn)


def _decode_meta(state_ids, device):
    batch = len(state_ids)
    attn = type('Attn', (), {})()
    attn.is_decoding = True
    attn.cu_seqlens_q = torch.arange(batch + 1, device=device, dtype=torch.int32)
    attn.q_seqlens = torch.ones(batch, device=device, dtype=torch.int32)
    attn.kv_seqlens = torch.full((batch, ), 3, device=device, dtype=torch.int32)
    return GatedDeltaMeta(batch, 4, torch.tensor(state_ids, device=device, dtype=torch.int64), attn)


def _state_pool(device, seed=42, slots=10):
    gen = torch.Generator(device=device).manual_seed(seed)
    return (torch.rand(slots, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, generator=gen, device=device) *
            0.2).to(torch.bfloat16)


class TestTmVsFlaNumerics:

    HEADS = NUM_V_HEADS

    @pytest.fixture(autouse=True)
    def _cuda(self):
        with torch.inference_mode():
            torch.set_default_device('cuda')
            try:
                yield
            finally:
                torch.set_default_device('cpu')

    @pytest.mark.parametrize('lengths,kv_lengths,state_ids', [
        ((5, 9, 1), (0, 0, 0), [2, 5, 7]),  # all fresh
        ((1, 72, 137), (3, 65, 100), [1, 3, 8]),  # chunk-boundary + history
    ])
    def test_prefill_chunk(self, lengths, kv_lengths, state_ids):
        device = 'cuda'
        q, k, v, b, a, dt_bias, a_log_exp = _make_inputs(lengths, device)
        fla = _make_rule(tm=False)
        tm = _make_rule(tm=True)

        pool_fla, pool_tm = _state_pool(device), _state_pool(device)
        meta_fla = _prefill_meta(lengths, kv_lengths, state_ids, device)
        meta_tm = _prefill_meta(lengths, kv_lengths, state_ids, device)

        out_fla = fla.forward(q, k, v, b, a, dt_bias, a_log_exp, pool_fla, meta_fla, 2, True)
        out_tm = tm.forward(q, k, v, b, a, dt_bias, a_log_exp, pool_tm, meta_tm, 2, True)

        assert out_fla.shape == out_tm.shape == (1, sum(lengths), self.HEADS, HEAD_DIM)
        torch.testing.assert_close(out_tm.float(), out_fla.float(), rtol=2e-2, atol=2e-2)
        touched = torch.tensor(state_ids, device=device, dtype=torch.long)
        torch.testing.assert_close(pool_tm[touched].float(),
                                   pool_fla[touched].float(),
                                   rtol=2e-2,
                                   atol=2e-2)

    @pytest.mark.parametrize('state_ids', [(2, 5, 7), (2, 5, -1)])
    def test_decode_recurrent(self, state_ids):
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

        from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaRuleImpl

        device = 'cuda'
        batch = len(state_ids)
        q, k, v, b, a, dt_bias, a_log_exp = _make_inputs((batch, ), device)
        tm = _make_rule(tm=True)

        pool_fla, pool_tm = _state_pool(device), _state_pool(device)
        meta_fla = _decode_meta(state_ids, device)
        meta_tm = _decode_meta(state_ids, device)

        out_tm = tm.forward(q, k, v, b, a, dt_bias, a_log_exp, pool_tm, meta_tm, 2, True)

        # FLA reference on the valid rows, replicating the ABC decode math
        # (the lmdeploy recurrent kernel is TileLang and shares the pre-existing
        # JIT compile failures in this environment, so compare against FLA).
        ids = torch.tensor(state_ids, device=device, dtype=torch.long)
        valid = (ids > 0) & (ids < pool_fla.shape[0])
        prepare = GatedDeltaRuleImpl.prepare_inputs
        qf, kf, gf, betaf, _ = prepare(tm, q, k, b, a, dt_bias, a_log_exp, 2, True, True, None)
        sel = ids[valid].clamp(0).long()

        def unflatten(x):
            return x[0].unflatten(0, (batch, -1)).contiguous()[valid]

        out_fla, final = fused_recurrent_gated_delta_rule(
            unflatten(qf),
            unflatten(kf),
            unflatten(v),
            g=unflatten(gf),
            beta=unflatten(betaf),
            initial_state=pool_fla[sel].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
        )
        pool_fla[sel] = final.to(pool_fla.dtype)

        if out_fla.dim() == 3:  # FLA squeezes the singleton token dim
            out_fla = out_fla.unsqueeze(1)

        assert out_tm.shape == (1, batch, self.HEADS, HEAD_DIM)
        tm_rows = out_tm.reshape(batch, self.HEADS, HEAD_DIM)[valid]
        fla_rows = out_fla.reshape(-1, self.HEADS, HEAD_DIM)
        torch.testing.assert_close(tm_rows.float(), fla_rows.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(pool_tm[sel].float(),
                                   pool_fla[sel].float(),
                                   rtol=2e-2,
                                   atol=2e-2)
