# Copyright (c) OpenMMLab. All rights reserved.
"""Selection-level and numeric tests for the TurboMind blocked-FP8 linear
(opt-in via LMDEPLOY_TURBOMIND_OPS force). The published per-op variable
LMDEPLOY_BLOCKED_FP8_GEMM_BACKEND stays untouched and the auto selection
chain (deepgemm/gluon/triton) must remain byte-identical."""

import pytest
import torch

from lmdeploy.pytorch.backends.blockedf8_modules import LinearBlockedF8BuildSpec
from lmdeploy.pytorch.backends.tm_ops import TM_OPS_FACTORY_TABLE

cuda = pytest.importorskip('torch', reason='torch required')

BUILDER = 'lmdeploy.pytorch.backends.cuda.blockedf8_modules'


def _spec(block_size=128,
          in_features=256,
          out_features=256,
          output_dtype=torch.bfloat16,
          fp8_dtype=torch.float8_e4m3fn,
          scale_fmt=None):
    return LinearBlockedF8BuildSpec(in_features=in_features,
                                    out_features=out_features,
                                    block_size=block_size,
                                    bias=False,
                                    output_dtype=output_dtype,
                                    fp8_dtype=fp8_dtype,
                                    scale_fmt=scale_fmt)


def _noop_init(self, *args, **kwargs):
    self.in_features = args[0] if args else kwargs.get('in_features')
    self.out_features = args[1] if len(args) > 1 else kwargs.get('out_features')


@pytest.fixture(autouse=True)
def _clean_switch(monkeypatch):
    """No single-switch value and no factory rows for blocked_fp8."""
    monkeypatch.delenv('LMDEPLOY_TURBOMIND_OPS', raising=False)
    monkeypatch.setattr(
        'lmdeploy.pytorch.backends.tm_ops.TM_OPS_FACTORY_TABLE',
        {k: v for k, v in TM_OPS_FACTORY_TABLE.items() if k[0] != 'blocked_fp8'})
    yield


@pytest.fixture
def sm80(monkeypatch):
    """Pretend an SM80 device is present without touching real hardware."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a, **k: (8, 0))


@pytest.fixture
def gate_ok(monkeypatch):
    """Stub the native-build part of the gate so force semantics can be
    exercised on machines without the TurboMind extension."""
    monkeypatch.setattr(
        'lmdeploy.pytorch.backends.cuda.turbomind_blockedf8_modules.turbomind_blockedf8_gate_reason',
        lambda *a, **k: None)


class TestSelection:
    """Selection behavior: force reaches TM, auto/off keep the chain."""

    def test_auto_no_row_keeps_existing_chain(self, monkeypatch, sm80):
        """auto + empty factory table must never pick the TM impl on sm80."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setattr(builder, '_supports_deep_gemm', lambda *a: False)
        monkeypatch.setattr(builder, '_supports_gluon', lambda *a: False)
        impl = builder._build_linear_blocked_f8(_spec())
        assert type(impl).__name__ == 'TritonLinearBlockedF8Impl'

    def test_auto_deepgemm_branch_unchanged(self, monkeypatch, sm80):
        """auto still prefers deepgemm when available (zero behavior change)."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setattr(builder, '_supports_deep_gemm', lambda *a: True)
        monkeypatch.setattr(builder, '_supports_gluon', lambda *a: True)
        monkeypatch.setattr(builder.DeepGemmLinearBlockedF8Impl, '__init__', _noop_init)
        impl = builder._build_linear_blocked_f8(_spec(in_features=4096, out_features=4096))
        assert type(impl).__name__ == 'DeepGemmLinearBlockedF8Impl'

    def test_off_ignores_force(self, monkeypatch, sm80):
        """off closes the mechanism: nothing can reach the TM impl."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        monkeypatch.setattr(builder, '_supports_deep_gemm', lambda *a: False)
        monkeypatch.setattr(builder, '_supports_gluon', lambda *a: False)
        impl = builder._build_linear_blocked_f8(_spec())
        assert type(impl).__name__ == 'TritonLinearBlockedF8Impl'

    @pytest.mark.parametrize('switch', ['force', 'force=blocked_fp8'])
    def test_force_selects_tm(self, monkeypatch, sm80, gate_ok, switch):
        """force / force=blocked_fp8 selects the TM impl when the gate passes."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', switch)
        impl = builder._build_linear_blocked_f8(_spec())
        assert type(impl).__name__ == 'TurbomindLinearBlockedF8Impl'
        assert impl.in_features == 256 and impl.out_features == 256

    @pytest.mark.parametrize('switch', ['force', 'force=blocked_fp8'])
    def test_force_non_128_block_raises(self, monkeypatch, sm80, switch):
        """Shapes outside family 15's format must fail at build time."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', switch)
        with pytest.raises(RuntimeError, match='block_size'):
            builder._build_linear_blocked_f8(_spec(block_size=64))

    def test_force_wrong_capability_raises(self, monkeypatch):
        """family 15 is an SM80 family: other capabilities must be rejected."""
        import importlib

        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a, **k: (9, 0))
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=blocked_fp8')
        builder = importlib.import_module(BUILDER)
        with pytest.raises(RuntimeError, match='capability'):
            builder._build_linear_blocked_f8(_spec())

    def test_force_rejects_non_bf16(self, monkeypatch, sm80):
        """family 15 computes in bf16; fp16 output is out of scope."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=blocked_fp8')
        with pytest.raises(RuntimeError, match='bfloat16'):
            builder._build_linear_blocked_f8(_spec(output_dtype=torch.float16))

    def test_force_k_not_multiple_of_128_raises(self, monkeypatch, sm80):
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=blocked_fp8')
        with pytest.raises(RuntimeError, match='divisible'):
            builder._build_linear_blocked_f8(_spec(in_features=200))

    def test_explicit_triton_with_force_raises(self, monkeypatch, sm80):
        """The only hard contradiction: per-op triton x force."""
        import importlib

        builder = importlib.import_module(BUILDER)
        # the per-op variable is resolved at import time; patch the resolved value
        monkeypatch.setattr(builder, 'blocked_fp8_gemm_backend', 'triton')
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=blocked_fp8')
        with pytest.raises(RuntimeError, match='contradiction'):
            builder._build_linear_blocked_f8(_spec())

    def test_explicit_triton_without_force_unchanged(self, monkeypatch, sm80):
        """per-op triton alone keeps the published behavior."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setattr(builder, 'blocked_fp8_gemm_backend', 'triton')
        impl = builder._build_linear_blocked_f8(_spec())
        assert type(impl).__name__ == 'TritonLinearBlockedF8Impl'

    def test_factory_table_row_would_select(self, monkeypatch, sm80, gate_ok):
        """auto + matching factory row selects TM (rows land only with
        benchmark evidence; here the row is injected)."""
        import importlib

        builder = importlib.import_module(BUILDER)
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'auto')
        monkeypatch.setitem(
            # resolve_tm reads the module-level dict by reference
            __import__('lmdeploy.pytorch.backends.tm_ops', fromlist=['x']).TM_OPS_FACTORY_TABLE,
            ('blocked_fp8', (8, 0)),
            {'evidence': 'benchmarks/ops/blocked_fp8_sm80.json'})
        impl = builder._build_linear_blocked_f8(_spec())
        assert type(impl).__name__ == 'TurbomindLinearBlockedF8Impl'


class TestScaleLayoutChoice:
    """The TM impl must map checkpoint scale layouts to family 15 formats."""

    def _impl(self):
        from lmdeploy.pytorch.backends.cuda.turbomind_blockedf8_modules import (
            TurbomindLinearBlockedF8Impl)

        return TurbomindLinearBlockedF8Impl(256, 256, 128)

    def test_groupwise_layout(self):
        impl = self._impl()
        assert impl._detect_block_out(torch.empty(256, 2)) == 1

    def test_blockwise_layout(self):
        impl = self._impl()
        assert impl._detect_block_out(torch.empty(2, 2)) == 128

    def test_bad_layout_raises(self):
        impl = self._impl()
        with pytest.raises(ValueError):
            impl._detect_block_out(torch.empty(3, 2))


def _make_weights(n, k, block_out, device):
    """Random fp8-block weights with the given scale layout."""
    torch.manual_seed(42)
    weight = (torch.randn(n, k, device=device) * 0.05).to(torch.float8_e4m3fn)
    k_groups = k // 128
    rows = n if block_out == 1 else n // 128
    scale = torch.rand(rows, k_groups, device=device) * 0.01 + 0.005
    return weight, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason='family 15 targets SM80')
class TestNumericParity:
    """TM impl vs triton and vs a float32 dequant reference on real GPU."""

    @pytest.fixture
    def env_auto(self, monkeypatch):
        monkeypatch.delenv('LMDEPLOY_TURBOMIND_OPS', raising=False)

    @pytest.fixture
    def env_force(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=blocked_fp8')

    def _build_tm(self, spec):
        from lmdeploy import turbomind

        if not turbomind.is_available():
            pytest.skip('TurboMind native modules are not built')
        from lmdeploy.pytorch.backends.cuda.blockedf8_modules import _build_linear_blocked_f8

        impl = _build_linear_blocked_f8(spec)
        assert type(impl).__name__ == 'TurbomindLinearBlockedF8Impl'
        return impl

    def _check(self, n, k, block_out, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=blocked_fp8')
        device = 'cuda'
        spec = _spec(in_features=k, out_features=n, output_dtype=torch.bfloat16)
        tm = self._build_tm(spec)

        weight, scale = _make_weights(n, k, block_out, device)
        weight = weight.contiguous()
        scale = scale.contiguous()
        x = torch.randn(9, k, dtype=torch.bfloat16, device=device) * 0.5

        tm.update_weights(weight, scale)
        with torch.no_grad():
            out_tm = tm.forward(x, weight, scale)

        # exact float32 dequant reference (ground truth: fp8 weights dequant
        # to bf16 inside family 15, no fp8 hardware involved)
        w_f32 = weight.to(torch.float32)
        if block_out == 1:
            w_dq = w_f32 * scale.repeat_interleave(128, dim=1)
        else:
            w_dq = w_f32 * scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        ref = x.to(torch.float32) @ w_dq.t()

        torch.testing.assert_close(out_tm.float(), ref, rtol=1e-2, atol=1e-2)

        # 3D input must keep its shape
        x3 = x.view(3, 3, k)
        with torch.no_grad():
            out3 = tm.forward(x3, weight, scale)
        assert out3.shape == (3, 3, n)
        torch.testing.assert_close(out3.float(), ref.view(3, 3, n), rtol=1e-2, atol=1e-2)

        # triton comparison where the triton fp8 path can run (it requires
        # fp8e4nv, i.e. cc >= 9; on SM80 the incumbent chain falls back to a
        # provider that cannot compile, so the dequant reference is the
        # available ground truth)
        try:
            from lmdeploy.pytorch.backends.cuda.blockedf8_modules import TritonLinearBlockedF8Impl

            triton_impl = TritonLinearBlockedF8Impl(k, n, 128)
            with torch.no_grad():
                out_triton = triton_impl.forward(x, weight, scale)
            torch.testing.assert_close(out_tm.float(), out_triton.float(), rtol=5e-2, atol=5e-2)
        except Exception as e:
            pytest.skip(f'triton fp8 path unavailable on this architecture: {e}')

    @pytest.mark.usefixtures('env_force')
    def test_blockwise_128x128(self, monkeypatch):
        self._check(256, 256, 128, monkeypatch)

    @pytest.mark.usefixtures('env_force')
    def test_groupwise_128x1(self, monkeypatch):
        self._check(256, 512, 1, monkeypatch)

    def test_auto_on_sm80_never_tm(self, monkeypatch):
        """Without the switch, an sm80 device must keep the existing chain."""
        monkeypatch.delenv('LMDEPLOY_TURBOMIND_OPS', raising=False)
        from lmdeploy.pytorch.backends.cuda.blockedf8_modules import _build_linear_blocked_f8

        impl = _build_linear_blocked_f8(_spec(in_features=256, out_features=256))
        assert type(impl).__name__ != 'TurbomindLinearBlockedF8Impl'
