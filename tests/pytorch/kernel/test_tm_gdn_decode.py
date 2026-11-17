# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind GDN decode graft: bridge kernels, routing semantics, S1 parity
against the fla production decode path, and the hot-path net-gain report.

Default suite: random tensors only, no model weights.
"""

from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaRuleBuildSpec


def gpu_ready():
    try:
        import tilelang  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def make_meta(batch: int, input_ids: torch.Tensor, output_ids: torch.Tensor):
    """Minimal indexed decode metadata. The decode branches only touch the
    state ids and the phase flag."""
    state_meta = SimpleNamespace(input_state_ids=input_ids, output_state_ids=output_ids)
    return SimpleNamespace(is_decoding=True, ssm_state_meta=state_meta)


def make_decode_inputs(batch, hv, head_dim, device='cuda'):
    """Production-shape decode inputs: [1, T, H, D] tensors plus scalar
    gate params. Mirrors the model-side GDN call."""
    gen = torch.Generator(device='cpu').manual_seed(7)

    def rand(*shape, dtype):
        x = torch.randn(*shape, generator=gen)
        return x.to(device=device, dtype=dtype)

    q = rand(batch, hv, head_dim, dtype=torch.bfloat16)
    k = rand(batch, hv, head_dim, dtype=torch.bfloat16)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = rand(batch, hv, head_dim, dtype=torch.bfloat16)
    b = torch.rand(batch, hv, generator=gen).to(device=device, dtype=torch.bfloat16)
    a = rand(batch, hv, dtype=torch.bfloat16)

    def lead(x):
        return x.unsqueeze(0)

    dt_bias = rand(hv, dtype=torch.float32)
    a_log_exp = rand(hv, dtype=torch.float32)
    return lead(q), lead(k), lead(v), lead(b), lead(a), dt_bias, a_log_exp


def test_decline_reasons_cpu():
    """Hard constraints reject without any device touch. Each constraint
    yields its own reason fragment. The checks run on CPU."""
    from lmdeploy.pytorch.backends.cuda.tm_gated_delta_rule import tm_gdn_decline_reason
    reasons = [
        tm_gdn_decline_reason(64, 8, torch.bfloat16, torch.float32),
        tm_gdn_decline_reason(128, 0, torch.bfloat16, torch.float32),
        tm_gdn_decline_reason(128, 8, torch.float32, torch.float32),
        tm_gdn_decline_reason(128, 8, torch.bfloat16, torch.float16),
    ]
    assert 'head_dim' in reasons[0]
    assert 'hv' in reasons[1]
    assert 'input dtype' in reasons[2]
    assert 'state dtype' in reasons[3]


def test_routing_env_off(monkeypatch):
    """off routes straight to the fla-backed CUDA impl."""
    import lmdeploy.pytorch.envs as envs
    from lmdeploy.pytorch.backends import get_backend
    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl

    monkeypatch.setattr(envs, 'gdn_backend', 'off')
    impl = get_backend().build_op(GatedDeltaRuleBuildSpec(indexed_state=True))
    assert type(impl) is CudaGatedDeltaRuleImpl


@pytest.mark.skipif(not gpu_ready(), reason='fla/tilelang/CUDA are not available')
class TestTmGdnBridge:

    hv = 8
    head_dim = 128

    def test_bridge_roundtrip(self):
        """gather(scatter(x)) and scatter(gather(x)) preserve pool rows."""
        from lmdeploy.pytorch.backends.cuda.tm_gated_delta_rule import (
            _gdn_state_gather,
            _gdn_state_scatter,
        )
        num_slots = 6
        batch = 3
        pool = torch.randn(num_slots, self.hv, self.head_dim, self.head_dim, device='cuda', dtype=torch.float32)
        ids = torch.tensor([5, 0, 3], device='cuda', dtype=torch.int64)
        compact = torch.empty(batch, self.hv, self.head_dim, self.head_dim, device='cuda', dtype=torch.float32)
        _gdn_state_gather(pool, ids, compact)
        expect = pool[ids.cpu()].transpose(-2, -1).contiguous()
        torch.testing.assert_close(compact, expect)

        pool2 = pool.clone()
        compact2 = torch.randn_like(compact)
        _gdn_state_scatter(pool2, ids, compact2)
        expect2 = compact2.transpose(-2, -1).contiguous()
        for i, slot in enumerate([5, 0, 3]):
            torch.testing.assert_close(pool2[slot], expect2[i])
        # untouched slots keep their rows
        for slot in [1, 2, 4]:
            torch.testing.assert_close(pool2[slot], pool[slot])

    def test_plan_probe_accepts(self):
        from lmdeploy.pytorch.backends.cuda.tm_gated_delta_rule import tm_gdn_decline_reason
        reason = tm_gdn_decline_reason(self.head_dim, self.hv, torch.bfloat16, torch.float32)
        assert reason is None, f'planner declined on this arch: {reason}'

    @pytest.fixture(autouse=True)
    def _impls(self):
        from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaGatedDeltaRuleImpl
        from lmdeploy.pytorch.backends.cuda.tm_gated_delta_rule import TmGatedDeltaRuleImpl
        self.flha_impl = CudaGatedDeltaRuleImpl(indexed_state=True)
        self.tm_impl = TmGatedDeltaRuleImpl(indexed_state=True)

    def _run(self, impl, inputs, pool, input_ids, output_ids, use_qk_l2norm_in_kernel=False):
        q, k, v, b, a, dt_bias, a_log_exp = inputs
        meta = make_meta(q.shape[1], input_ids, output_ids)
        return impl.forward(q, k, v, b, a, dt_bias, a_log_exp, pool, meta, 1, use_qk_l2norm_in_kernel)

    def test_s1_parity_against_fla(self):
        """Same state and inputs: TurboMind decode vs the fla production
        decode kernel, distinct input and output slots per request."""
        num_slots, batch = 12, 4
        inputs = make_decode_inputs(batch, self.hv, self.head_dim)
        pool = torch.randn(num_slots, self.hv, self.head_dim, self.head_dim,
                           device='cuda', dtype=torch.float32) * 0.1
        input_ids = torch.tensor([9, 2, 11, 4], device='cuda', dtype=torch.int64)
        output_ids = torch.tensor([[8], [3], [1], [6]], device='cuda', dtype=torch.int64)

        pool_fla = pool.clone()
        out_fla = self._run(self.flha_impl, inputs, pool_fla, input_ids, output_ids)
        pool_tm = pool.clone()
        out_tm = self._run(self.tm_impl, inputs, pool_tm, input_ids, output_ids)

        assert out_tm.shape == out_fla.shape
        out_err = (out_tm.float() - out_fla.float()).abs().max().item()
        state_err = (pool_tm - pool_fla).abs().max().item()
        print(f'[S1] out max_abs={out_err:.6f} state max_abs={state_err:.6f}')
        assert out_err < 0.02, out_err
        assert state_err < 0.02, state_err

    def test_forced_on_raises_when_declined(self, monkeypatch):
        """LMDEPLOY_GDN_BACKEND=on surfaces a hard constraint as an error."""
        import lmdeploy.pytorch.envs as envs
        inputs = make_decode_inputs(2, self.hv, 64)
        pool = torch.randn(4, self.hv, 64, 64, device='cuda', dtype=torch.float32)
        ids = torch.tensor([1, 2], device='cuda', dtype=torch.int64)
        meta = make_meta(2, ids, ids[:, None])
        monkeypatch.setattr(envs, 'gdn_backend', 'on')
        q, k, v, b, a, dt_bias, a_log_exp = inputs
        with pytest.raises(RuntimeError, match='LMDEPLOY_GDN_BACKEND=on'):
            self.tm_impl.forward(q, k, v, b, a, dt_bias, a_log_exp, pool, meta, 1, False)

    @pytest.mark.parametrize('hv,batch', [(32, 1), (32, 8), (32, 32), (64, 1), (64, 8), (64, 32)])
    def test_decode_net_gain(self, hv: int, batch: int):
        """Hot-path report: bridge overhead plus kernel vs the fla decode.

        Prints the net gain per batch; the fallback criterion (net gain
        below 10 percent triggers the resident-TM-format amendment) is a
        report deliverable, not a hard assertion.
        """
        import statistics

        from lmdeploy.pytorch.backends.cuda.tm_gated_delta_rule import _gdn_state_gather, _gdn_state_scatter

        hv = 32
        num_slots = 64
        inputs = make_decode_inputs(batch, hv, self.head_dim)
        pool = torch.randn(num_slots, hv, self.head_dim, self.head_dim,
                           device='cuda', dtype=torch.float32) * 0.1
        input_ids = torch.arange(batch, device='cuda', dtype=torch.int64) + 4
        output_ids = input_ids[:, None]
        iters, warmup = 50, 10

        def median_ms(fn):
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

        # production GDN layers enable fused q/k l2norm preprocessing
        fla_ms = median_ms(lambda: self._run(self.flha_impl, inputs, pool, input_ids, output_ids, True))
        tm_ms = median_ms(lambda: self._run(self.tm_impl, inputs, pool, input_ids, output_ids, True))

        # bridge floor: the two transpose copies without the TM kernel
        hv_local = inputs[0].shape[-2]
        compact = pool.new_empty((batch, hv_local, self.head_dim, self.head_dim))
        scatter_src = torch.randn_like(compact)
        bridge_ms = median_ms(lambda: (
            _gdn_state_gather(pool, input_ids, compact),
            _gdn_state_scatter(pool, input_ids, scatter_src),
        ))

        gain = (fla_ms - tm_ms) / fla_ms
        print(f'[net] B={batch} fla={fla_ms:.4f} ms tm(bridge incl)={tm_ms:.4f} ms '
              f'bridge copies={bridge_ms:.4f} ms net gain={gain * 100:.1f}%')
