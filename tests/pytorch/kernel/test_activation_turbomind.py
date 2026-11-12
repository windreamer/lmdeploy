# Copyright (c) OpenMMLab. All rights reserved.
"""Numeric and selection-level tests for the TurboMind SiluAndMul kernel.

Variant coverage (ticket 04):
- Covered: split gate/up TM ``Activation`` kernel with ``kSilu`` and the
  padded-buffer variant gated by a device-side ``num_valid_tokens`` count.
- Kept on Triton, recorded per ticket (no TM equivalent on this seam):
  - ``silu_and_mul_moe_ep`` (per-row ``mask_m`` mask; the TM kernel takes a
    single device-side valid-token *count*, not a per-row mask);
  - ``silu_and_mul_masked_post_quant_fwd`` (fused FP8 output quantization);
  - ``swiglu_limit`` / ``precise_mul`` options of the Triton kernel.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _ref_silu_and_mul(gate_up):
    gate, up = gate_up.chunk(2, -1)
    return (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(gate_up.dtype)


class TestTurbomindSiluAndMulNumeric:

    @pytest.fixture(autouse=True, scope='class')
    def initialize(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        yield

    @pytest.fixture(scope='class', params=[torch.float16, torch.bfloat16])
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope='class', params=[64, 8192])
    def dim(self, request):
        yield request.param

    def test_matches_reference(self, dtype, dim):
        from lmdeploy.turbomind import _tm

        x = torch.randn(64, 2 * dim, dtype=dtype, device='cuda')
        gate, up = x.chunk(2, -1)
        x_snapshot = x.clone()
        stream = torch.cuda.current_stream().cuda_stream
        _tm.silu_and_mul(_tm.from_dlpack(gate), _tm.from_dlpack(up), None, stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(gate, _ref_silu_and_mul(x_snapshot), rtol=1e-3, atol=1e-3)

    def test_valid_tokens_padded_variant(self, dtype):
        from lmdeploy.turbomind import _tm

        num, valid, dim = 16, 5, 128
        x = torch.randn(num, 2 * dim, dtype=dtype, device='cuda')
        gate, up = x.chunk(2, -1)
        x_snapshot = x.clone()
        gate_snapshot = gate.clone()
        valid_tokens = torch.tensor([valid], dtype=torch.int32, device='cuda')
        stream = torch.cuda.current_stream().cuda_stream
        _tm.silu_and_mul(_tm.from_dlpack(gate), _tm.from_dlpack(up), _tm.from_dlpack(valid_tokens), stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(gate[:valid], _ref_silu_and_mul(x_snapshot[:valid]), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(gate[valid:], gate_snapshot[valid:])


class TestTurbomindSiluAndMulImpl:

    @pytest.fixture(autouse=True, scope='class')
    def initialize(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        yield

    @pytest.mark.parametrize('inplace', [False, True])
    @pytest.mark.parametrize('shape', [(8, 128), (4, 16, 128), (256, 8192)])
    def test_impl_matches_triton(self, inplace, shape):
        from lmdeploy.pytorch.backends.cuda.activation import build_silu_and_mul
        from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul as triton_silu_and_mul

        x = torch.randn(shape, dtype=torch.float16, device='cuda')
        x_snapshot = x.clone()

        impl = build_silu_and_mul(type('Spec', (), {'inplace': inplace})())
        out = impl.forward(x)
        torch.cuda.synchronize()

        triton_out = triton_silu_and_mul(x_snapshot.flatten(0, -2), None)
        triton_out = triton_out.unflatten(0, shape[:-1]) if x.dim() != 2 else triton_out
        torch.testing.assert_close(out, triton_out, rtol=1e-3, atol=1e-3)


class TestSiluAndMulSelection:

    def _build(self, monkeypatch, env):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', env)
        from lmdeploy.pytorch.backends.cuda.activation import build_silu_and_mul
        spec = type('Spec', (), {'inplace': False})()
        return type(build_silu_and_mul(spec)).__name__

    def test_off_selects_triton(self, monkeypatch):
        assert self._build(monkeypatch, 'off') == 'TritonSiluAndMulImpl'

    def test_auto_selects_triton_without_table_row(self, monkeypatch):
        assert self._build(monkeypatch, 'auto') == 'TritonSiluAndMulImpl'

    def test_force_selects_turbomind(self, monkeypatch):
        assert self._build(monkeypatch, 'force=silu_and_mul') == 'TurbomindSiluAndMulImpl'

    def test_force_with_failing_gate_raises(self, monkeypatch):
        from lmdeploy.pytorch.backends.cuda import activation as act_mod
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=silu_and_mul')
        monkeypatch.setattr(act_mod, 'turbomind_silu_and_mul_gate_reason', lambda *a, **k: 'stubbed gate failure')
        spec = type('Spec', (), {'inplace': False})()
        with pytest.raises(RuntimeError, match='forces TM for family'):
            act_mod.build_silu_and_mul(spec)
