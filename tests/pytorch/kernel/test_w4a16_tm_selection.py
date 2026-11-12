# Copyright (c) OpenMMLab. All rights reserved.
"""Selection-level and gate tests for the promoted W4A16 TurboMind
implementation. Pure CPU logic; no CUDA required.

Seams under test:
- ``_w4a16_tm_feasibility_reason`` (relaxed gate: major in {8, 9})
- ``_turbomind_support_reason``    (legacy SM90-only gate, off path)
- ``_build_linear_w4a16``          (provider x switch selection)
- ``TM_OPS_FACTORY_TABLE``         (w4a16 rows for (8, 0) and (9, 0))
"""

import sys
import types
from pathlib import Path

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import awq_modules
from lmdeploy.pytorch.backends.tm_ops import TM_OPS_FACTORY_TABLE

TM_STUB_MODULE = 'lmdeploy.pytorch.backends.cuda.turbomind_awq_modules'


class _StubTurbomindImpl:
    """Stand-in for TurbomindAwqLinearW4A16Impl so selection tests stay CPU
    only; the real class is exercised by the numeric tests on real hardware."""

    def __init__(self, in_features, out_features, w_bit, group_size):
        self.args = (in_features, out_features, w_bit, group_size)


@pytest.fixture(autouse=True)
def _stub_turbomind(monkeypatch):
    module = types.ModuleType(TM_STUB_MODULE)
    module.TurbomindAwqLinearW4A16Impl = _StubTurbomindImpl
    monkeypatch.setitem(sys.modules, TM_STUB_MODULE, module)


@pytest.fixture
def set_capability(monkeypatch):
    """Pin torch.cuda device capability and CUDA availability."""

    def _set(capability, available=True):
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
        monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a, **k: capability)

    return _set


@pytest.fixture
def set_provider(monkeypatch):
    def _set(provider):
        monkeypatch.setattr(awq_modules._envs, 'w4a16_gemm_backend', provider)

    return _set


@pytest.fixture
def allow_tm_gate(monkeypatch):
    """Bypass environment-dependent gate parts (CUDA + native modules)."""
    monkeypatch.setattr('lmdeploy.turbomind.is_available', lambda: True)


def _build(in_features=512, out_features=256, w_bit=4, group_size=128, dtype=torch.float16):
    from lmdeploy.pytorch.backends.awq_modules import LinearW4A16BuildSpec

    spec = LinearW4A16BuildSpec(in_features=in_features,
                                out_features=out_features,
                                w_bit=w_bit,
                                group_size=group_size,
                                bias=False,
                                output_dtype=dtype)
    return awq_modules._build_linear_w4a16(spec)


class TestRelaxedFeasibilityGate:
    """The promoted gate accepts every major {8, 9} capability."""

    @pytest.mark.parametrize('capability', [(8, 0), (8, 6), (8, 7), (9, 0)])
    def test_major_8_and_9_pass(self, monkeypatch, set_capability, allow_tm_gate, capability):
        set_capability(capability)
        reason = awq_modules._w4a16_tm_feasibility_reason(512, 256, 4, 128, torch.float16)
        assert reason is None

    @pytest.mark.parametrize('capability', [(7, 5), (10, 0), (12, 0)])
    def test_other_majors_fail(self, monkeypatch, set_capability, capability):
        set_capability(capability)
        reason = awq_modules._w4a16_tm_feasibility_reason(512, 256, 4, 128, torch.float16)
        assert reason is not None and 'capability' in reason

    def test_shape_constraints_kept(self, monkeypatch, set_capability, allow_tm_gate):
        set_capability((8, 0))
        assert awq_modules._w4a16_tm_feasibility_reason(512, 256, 8, 128, torch.float16) is not None
        assert awq_modules._w4a16_tm_feasibility_reason(512, 256, 4, 32, torch.float16) is not None
        assert awq_modules._w4a16_tm_feasibility_reason(130, 256, 4, 128, torch.float16) is not None
        assert awq_modules._w4a16_tm_feasibility_reason(512, 33, 4, 128, torch.float16) is not None
        assert awq_modules._w4a16_tm_feasibility_reason(512, 256, 4, 128, torch.bfloat16) is not None


class TestLegacyGateUnchanged:
    """The off-path legacy gate stays SM90-only."""

    def test_sm90_only(self, monkeypatch, set_capability, allow_tm_gate):
        set_capability((9, 0))
        assert awq_modules._turbomind_support_reason(512, 256, 4, 128, torch.float16) is None
        set_capability((8, 0))
        reason = awq_modules._turbomind_support_reason(512, 256, 4, 128, torch.float16)
        assert reason is not None and 'capability' in reason

    def test_sm90_undef(self, monkeypatch, set_capability):
        set_capability((9, 0))
        assert awq_modules._is_turbomind_gemm_capability_supported((9, 0))
        assert not awq_modules._is_turbomind_gemm_capability_supported((8, 0))


class TestFactoryTableRows:
    """Factory table ships exactly the two w4a16 rows with evidence."""

    def test_rows_present(self):
        assert ('w4a16', (8, 0)) in TM_OPS_FACTORY_TABLE
        assert ('w4a16', (9, 0)) in TM_OPS_FACTORY_TABLE
        for key, row in TM_OPS_FACTORY_TABLE.items():
            assert row.get('evidence'), key

    def test_evidence_files_exist(self):
        repo_root = Path(__file__).resolve().parents[3]
        for key, row in TM_OPS_FACTORY_TABLE.items():
            evidence = repo_root / row['evidence']
            assert evidence.is_file(), f'{key} evidence missing: {evidence}'

    def test_no_other_family_has_rows(self):
        other = [k for k in TM_OPS_FACTORY_TABLE if k[0] != 'w4a16']
        assert other == []


class TestAutoSelection:
    """auto: table row decides; unaligned shapes fall back to awq."""

    def test_sm80_auto_goes_tm(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 0))
        set_provider('auto')
        impl = _build()
        assert isinstance(impl, _StubTurbomindImpl)

    def test_sm86_auto_stays_awq(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 6))
        set_provider('auto')
        impl = _build()
        assert isinstance(impl, awq_modules.AwqLinearW4A16Impl)

    def test_sm90_auto_goes_tm(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((9, 0))
        set_provider('auto')
        impl = _build()
        assert isinstance(impl, _StubTurbomindImpl)

    def test_unaligned_k_auto_falls_back(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 0))
        set_provider('auto')
        impl = _build(in_features=130)
        assert isinstance(impl, awq_modules.AwqLinearW4A16Impl)

    def test_group32_auto_falls_back(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 0))
        set_provider('auto')
        impl = _build(group_size=32)
        assert isinstance(impl, awq_modules.AwqLinearW4A16Impl)


class TestOffSelection:
    """off: legacy SM90-only behavior, byte-for-byte status quo."""

    def test_off_sm90_auto_tm(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((9, 0))
        set_provider('auto')
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        impl = _build()
        assert isinstance(impl, _StubTurbomindImpl)

    def test_off_sm80_auto_awq(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 0))
        set_provider('auto')
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        impl = _build()
        assert isinstance(impl, awq_modules.AwqLinearW4A16Impl)


class TestExplicitProvider:
    """Explicit per-op values keep their published semantics."""

    def test_explicit_turbomind_on_sm80(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 0))
        set_provider('turbomind')
        impl = _build()
        assert isinstance(impl, _StubTurbomindImpl)

    def test_explicit_turbomind_gate_fail_raises(self, monkeypatch, set_capability, set_provider):
        set_capability((7, 5))
        set_provider('turbomind')
        with pytest.raises(RuntimeError, match='incompatible'):
            _build()

    def test_explicit_triton_stays_triton(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 0))
        set_provider('triton')
        impl = _build()
        assert isinstance(impl, awq_modules.AwqLinearW4A16Impl)

    def test_explicit_triton_with_force_raises(self, monkeypatch, set_capability, set_provider):
        set_capability((8, 0))
        set_provider('triton')
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=w4a16')
        with pytest.raises(RuntimeError, match='contradict'):
            _build()


class TestForce:
    """force enables TM on gate-passing non-table capabilities."""

    def test_force_sm86_goes_tm(self, monkeypatch, set_capability, set_provider, allow_tm_gate):
        set_capability((8, 6))
        set_provider('auto')
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=w4a16')
        impl = _build()
        assert isinstance(impl, _StubTurbomindImpl)

    def test_force_gate_fail_raises(self, monkeypatch, set_capability, set_provider):
        set_capability((8, 6))
        set_provider('auto')
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=w4a16')
        with pytest.raises(RuntimeError, match='forces'):
            _build(in_features=130)
