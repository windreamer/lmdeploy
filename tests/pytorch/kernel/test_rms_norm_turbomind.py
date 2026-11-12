# Copyright (c) OpenMMLab. All rights reserved.
"""Numeric and selection-level tests for the TurboMind RMSNorm kernels.

Variant coverage (ticket 04):
- plain rmsnorm and fused add-residual: bound and covered below;
- qk variant: TM supports it via ``invokeQkRMSNorm`` on packed qkv buffers;
  covered below at the raw-binding level (the ``RMSNormImpl`` seam has no
  qk entry point, so production callers keep their existing qk path).
- All other variants (e.g. zero-centered qk layouts not used by the seam)
  stay on Triton and are intentionally untested here.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _triton_rms(x, w, eps):
    xf = x.float()
    var = (xf * xf).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


class TestTurbomindRmsNormNumeric:

    @pytest.fixture(autouse=True, scope='class')
    def initialize(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        yield

    @pytest.fixture(scope='class', params=[torch.float16, torch.bfloat16])
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope='class', params=[512, 4096])
    def hidden_size(self, request):
        yield request.param

    def test_plain_matches_reference(self, dtype, hidden_size):
        from lmdeploy.turbomind import _tm

        x = torch.randn(64, hidden_size, dtype=dtype, device='cuda')
        w = torch.randn(hidden_size, dtype=dtype, device='cuda')
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream().cuda_stream
        _tm.rms_norm(_tm.from_dlpack(out), _tm.from_dlpack(x), _tm.from_dlpack(w), 1e-6, False, stream)
        torch.cuda.synchronize()
        ref = _triton_rms(x, w, 1e-6)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_residual_variant_matches_reference(self, dtype, hidden_size):
        from lmdeploy.turbomind import _tm

        x = torch.randn(64, hidden_size, dtype=dtype, device='cuda')
        residual = torch.randn(64, hidden_size, dtype=dtype, device='cuda')
        w = torch.randn(hidden_size, dtype=dtype, device='cuda')
        x2 = x.clone()
        r2 = residual.clone()
        stream = torch.cuda.current_stream().cuda_stream
        _tm.residual_rms_norm(_tm.from_dlpack(x2), _tm.from_dlpack(r2), _tm.from_dlpack(w), None, 1e-6, False, stream)
        torch.cuda.synchronize()
        sum_ref = x + residual
        torch.testing.assert_close(r2, sum_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x2, _triton_rms(sum_ref, w, 1e-6), rtol=1e-3, atol=1e-3)

    def test_qk_variant_matches_reference(self, dtype):
        from lmdeploy.turbomind import _tm

        token_num, head_dim = 8, 128
        q_head_num, k_head_num, v_head_num = 4, 4, 4
        width = (q_head_num + k_head_num + v_head_num) * head_dim
        qkv = torch.randn(token_num, width, head_dim, dtype=dtype, device='cuda')
        qw = torch.randn(head_dim, dtype=dtype, device='cuda')
        kw = torch.randn(head_dim, dtype=dtype, device='cuda')
        ref = qkv.clone()
        for hi in range(q_head_num):
            ref[:, hi, :] = _triton_rms(qkv[:, hi, :], qw, 1e-6)
        for hi in range(k_head_num):
            ref[:, q_head_num + hi, :] = _triton_rms(qkv[:, q_head_num + hi, :], kw, 1e-6)
        stream = torch.cuda.current_stream().cuda_stream
        _tm.qk_rms_norm(_tm.from_dlpack(qkv), _tm.from_dlpack(qw), _tm.from_dlpack(kw), q_head_num, k_head_num, 1e-6,
                        False, stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(qkv, ref, rtol=1e-3, atol=1e-3)


class TestTurbomindRMSNormImpl:

    @pytest.fixture(autouse=True, scope='class')
    def initialize(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        yield

    @pytest.mark.parametrize('with_residual', [False, True])
    @pytest.mark.parametrize('shape', [(8, 512), (4, 16, 512), (256, 4096)])
    def test_impl_matches_triton_and_does_not_mutate_inputs(self, shape, with_residual):
        from lmdeploy.pytorch.backends.cuda.norm import build_rms_norm
        from lmdeploy.pytorch.kernels.cuda import rms_norm as triton_rms_norm

        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        w = torch.randn(shape[-1], dtype=torch.float16, device='cuda')
        residual = torch.randn(shape, dtype=torch.float16, device='cuda') if with_residual else None
        x_snap, w_snap, r_snap = x.clone(), w.clone(), None if residual is None else residual.clone()

        impl = build_rms_norm(type('Spec', (), {'hidden_size': shape[-1], 'eps': 1e-6})())
        out = impl.forward(x, w, residual)
        torch.cuda.synchronize()

        if with_residual:
            out_x, out_r = out
            ref_x, ref_r = triton_rms_norm(x_snap, w_snap, 1e-6, residual=r_snap)
            torch.testing.assert_close(out_x, ref_x, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(out_r, ref_r, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(out, triton_rms_norm(x_snap, w_snap, 1e-6), rtol=1e-3, atol=1e-3)

        # The Triton implementation never mutates its inputs; the TM
        # implementation must honor that seam contract.
        torch.testing.assert_close(x, x_snap)
        torch.testing.assert_close(w, w_snap)
        if with_residual:
            torch.testing.assert_close(residual, r_snap)


class TestRmsNormSelection:

    def _build(self, monkeypatch, env, hidden_size=512):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', env)
        from lmdeploy.pytorch.backends.cuda.norm import TritonRMSNormImpl, build_rms_norm  # noqa: F401
        from lmdeploy.pytorch.backends.turbomind_norm import TurbomindRMSNormImpl  # noqa: F401
        spec = type('Spec', (), {'hidden_size': hidden_size, 'eps': 1e-6})()
        return type(build_rms_norm(spec)).__name__

    def test_off_selects_triton(self, monkeypatch):
        assert self._build(monkeypatch, 'off') == 'TritonRMSNormImpl'

    def test_auto_selects_triton_without_table_row(self, monkeypatch):
        assert self._build(monkeypatch, 'auto') == 'TritonRMSNormImpl'

    def test_force_selects_turbomind(self, monkeypatch):
        assert self._build(monkeypatch, 'force=rms_norm') == 'TurbomindRMSNormImpl'

    def test_force_with_failing_gate_raises(self, monkeypatch):
        from lmdeploy.pytorch.backends.cuda import norm as norm_mod
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=rms_norm')
        monkeypatch.setattr(norm_mod, 'turbomind_rms_norm_gate_reason', lambda *a, **k: 'stubbed gate failure')
        spec = type('Spec', (), {'hidden_size': 512, 'eps': 1e-6})()
        with pytest.raises(RuntimeError, match='forces TM for family'):
            norm_mod.build_rms_norm(spec)
