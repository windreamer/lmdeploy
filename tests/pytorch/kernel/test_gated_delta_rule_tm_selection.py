# Copyright (c) OpenMMLab. All rights reserved.
"""Selection-level tests for the TurboMind gated-delta-rule (GDN) build.

Pure build-path logic: the TM branch is only reachable through the single
switch (``LMDEPLOY_TURBOMIND_OPS=force`` / ``force=gated_delta``); the
factory table ships no ``gated_delta`` row, so ``auto``/``off`` keep the
existing FLA/TileLang selection untouched. No CUDA device required."""

import pytest
import torch

from lmdeploy.pytorch.backends.gated_delta_rule import GatedDeltaRuleBuildSpec
from lmdeploy.pytorch.backends.tm_ops import TM_OPS_FACTORY_TABLE

gated_delta_rule_cuda = pytest.importorskip(
    'lmdeploy.pytorch.backends.cuda.gated_delta_rule')

CudaGatedDeltaRuleImpl = gated_delta_rule_cuda.CudaGatedDeltaRuleImpl
_build_gated_delta_rule = gated_delta_rule_cuda._build_gated_delta_rule


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Pristine switch value and factory table per test."""
    monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'auto')
    monkeypatch.delenv('LMDEPLOY_W4A16_GEMM_BACKEND', raising=False)
    saved = dict(TM_OPS_FACTORY_TABLE)
    TM_OPS_FACTORY_TABLE.clear()
    yield
    TM_OPS_FACTORY_TABLE.clear()
    TM_OPS_FACTORY_TABLE.update(saved)


@pytest.fixture
def tm_gate_ok(monkeypatch):
    """Force the TM gate to pass without requiring the native extension."""
    calls = []

    def fake_gate(spec):
        calls.append(spec)
        return None

    monkeypatch.setattr(gated_delta_rule_cuda, '_tm_gated_delta_gate_reason', fake_gate)
    return calls


def _spec(indexed_state=False):
    return GatedDeltaRuleBuildSpec(indexed_state=indexed_state)


class TestDefaultSelectionUnchanged:

    def test_auto_no_row_builds_existing_impl(self, tm_gate_ok):
        impl = _build_gated_delta_rule(_spec())
        assert type(impl) is CudaGatedDeltaRuleImpl
        assert impl.indexed_state is False

    def test_off_builds_existing_impl(self, monkeypatch, tm_gate_ok):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        impl = _build_gated_delta_rule(_spec(indexed_state=True))
        assert type(impl) is CudaGatedDeltaRuleImpl
        assert impl.indexed_state is True

    def test_auto_gate_never_evaluated_without_row(self, monkeypatch):
        calls = []

        def spy(spec):
            calls.append(spec)
            return None

        monkeypatch.setattr(gated_delta_rule_cuda, '_tm_gated_delta_gate_reason', spy)
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (8, 0))
        _build_gated_delta_rule(_spec())
        assert calls == []  # no factory-table row -> gate must not run


class TestForce:

    def test_force_csv_gated_delta_selects_tm(self, monkeypatch, tm_gate_ok):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=gated_delta')
        from lmdeploy.pytorch.backends.cuda.turbomind_gated_delta_rule import (
            TmGatedDeltaRuleImpl)
        impl = _build_gated_delta_rule(_spec())
        assert type(impl) is TmGatedDeltaRuleImpl
        assert impl.indexed_state is False
        assert tm_gate_ok == [_spec()]  # lazy gate was evaluated exactly once

    def test_bare_force_selects_tm_for_gated_delta(self, monkeypatch, tm_gate_ok):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force')
        from lmdeploy.pytorch.backends.cuda.turbomind_gated_delta_rule import (
            TmGatedDeltaRuleImpl)
        assert type(_build_gated_delta_rule(_spec())) is TmGatedDeltaRuleImpl

    def test_force_other_family_leaves_gdn_alone(self, monkeypatch, tm_gate_ok):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=w4a16,rms_norm')
        assert type(_build_gated_delta_rule(_spec())) is CudaGatedDeltaRuleImpl
        assert tm_gate_ok == []  # gated_delta not forced -> gate untouched

    def test_force_gate_failure_raises_build_time(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=gated_delta')
        monkeypatch.setattr(gated_delta_rule_cuda, '_tm_gated_delta_gate_reason',
                            lambda spec: 'native bridge unavailable')
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (10, 1))
        with pytest.raises(RuntimeError, match='feasibility gate'):
            _build_gated_delta_rule(_spec())

    def test_force_indexed_state_rejected(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=gated_delta')
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (8, 0))
        from lmdeploy.pytorch.backends import tm_ops
        reason = gated_delta_rule_cuda._tm_gated_delta_gate_reason(_spec(indexed_state=True))
        assert reason is not None and 'indexed' in reason
        # And through resolve_tm the same gate failure becomes a build error.
        with pytest.raises(RuntimeError):
            tm_ops.resolve_tm('gated_delta', (8, 0),
                              lambda: gated_delta_rule_cuda._tm_gated_delta_gate_reason(
                                  _spec(indexed_state=True)))


class TestFactoryTable:

    def test_auto_row_with_passing_gate_selects_tm(self, monkeypatch, tm_gate_ok):
        TM_OPS_FACTORY_TABLE[('gated_delta', (8, 0))] = {'evidence': 'benchmarks/ops/x.json'}
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (8, 0))
        from lmdeploy.pytorch.backends.cuda.turbomind_gated_delta_rule import (
            TmGatedDeltaRuleImpl)
        impl = _build_gated_delta_rule(_spec())
        assert type(impl) is TmGatedDeltaRuleImpl

    def test_auto_row_gate_failure_falls_back(self, monkeypatch):
        TM_OPS_FACTORY_TABLE[('gated_delta', (9, 0))] = {'evidence': 'benchmarks/ops/x.json'}
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'auto')
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (9, 0))
        monkeypatch.setattr(gated_delta_rule_cuda, '_tm_gated_delta_gate_reason',
                            lambda spec: 'native bridge unavailable')
        impl = _build_gated_delta_rule(_spec())
        assert type(impl) is CudaGatedDeltaRuleImpl

    def test_auto_row_other_capability_keeps_default(self, monkeypatch, tm_gate_ok):
        TM_OPS_FACTORY_TABLE[('gated_delta', (9, 0))] = {'evidence': 'benchmarks/ops/x.json'}
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (8, 6))
        assert type(_build_gated_delta_rule(_spec())) is CudaGatedDeltaRuleImpl


class TestFeasibilityGate:

    def test_gate_buckets(self, monkeypatch):
        # pre_sm90 / sm90 / sm120 pass the capability bucket check when the
        # bridge is present; other capabilities are rejected.
        monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda: (8, 0))
        reason = gated_delta_rule_cuda._tm_gated_delta_gate_reason(_spec())
        if gated_delta_rule_cuda._tm_native_bridge_available():
            assert reason is None
        else:
            assert reason is not None

        for cap in ((8, 6), (9, 0), (10, 1), (12, 0)):
            monkeypatch.setattr(gated_delta_rule_cuda, '_device_capability', lambda cap=cap: cap)
            reason = gated_delta_rule_cuda._tm_gated_delta_gate_reason(_spec())
            if cap == (12, 0) and gated_delta_rule_cuda._tm_native_bridge_available():
                assert reason is None
            else:
                assert reason is not None


def test_gate_is_lazy_on_non_tm_path(monkeypatch):
    """Without force or a table row, neither the gate nor the native bridge is touched."""
    import lmdeploy.pytorch.backends.cuda.gated_delta_rule as mod

    calls = []

    def spy_gate(spec):
        calls.append(spec)
        return None

    monkeypatch.setattr(mod, '_tm_gated_delta_gate_reason', spy_gate)
    monkeypatch.setattr(mod, '_tm_native_bridge_available',
                        lambda: (_ for _ in ()).throw(AssertionError('bridge must not be probed')))
    monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'auto')
    assert type(_build_gated_delta_rule(_spec())) is CudaGatedDeltaRuleImpl
    assert calls == []
