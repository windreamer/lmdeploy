# Copyright (c) OpenMMLab. All rights reserved.
"""Selection-level tests for the LMDEPLOY_TURBOMIND_OPS single switch and
factory table infrastructure. Pure CPU logic; no CUDA required."""

import pytest

from lmdeploy.pytorch.backends.tm_ops import TM_OPS_FACTORY_TABLE, resolve_tm
from lmdeploy.pytorch.envs import parse_turbomind_ops


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure a pristine switch value and factory table per test."""
    monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'auto')
    yield


class TestParseTurbomindOps:

    def test_default_auto(self, monkeypatch):
        monkeypatch.delenv('LMDEPLOY_TURBOMIND_OPS', raising=False)
        cfg = parse_turbomind_ops(None)
        assert cfg.mode == 'auto'
        assert cfg.forced == frozenset()

    def test_off(self):
        cfg = parse_turbomind_ops('off')
        assert cfg.mode == 'off'
        assert cfg.forced == frozenset()

    def test_force_all_families(self):
        cfg = parse_turbomind_ops('force')
        assert cfg.mode == 'force'
        assert cfg.forced == frozenset({
            'w4a16', 'gated_delta', 'rms_norm', 'silu_and_mul', 'blocked_fp8', 'copy_cache'
        })

    def test_force_csv(self):
        cfg = parse_turbomind_ops('force=w4a16,copy_cache')
        assert cfg.mode == 'force'
        assert cfg.forced == frozenset({'w4a16', 'copy_cache'})

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            parse_turbomind_ops('yes')

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            parse_turbomind_ops('force=w4a16,page_attn')


class TestResolveOff:
    """off = mechanism closed: injected rows are ignored, gates untouched."""

    def test_row_ignored_under_off(self, monkeypatch):
        monkeypatch.setitem(TM_OPS_FACTORY_TABLE, ('rms_norm', (8, 0)), {'evidence': 'test'})
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        calls = []

        decision = resolve_tm('rms_norm', (8, 0),
                              gate_reason_fn=lambda: calls.append('x') or None)

        assert decision.selected is False
        assert decision.forced is False
        assert decision.source == 'off'
        assert calls == []  # gate never evaluated


class TestResolveAuto:
    """auto: empty table = default impl; row + passing gate = TM; fallback on gate."""

    def test_empty_table_selects_default(self):
        decision = resolve_tm('rms_norm', (8, 0), gate_reason_fn=lambda: None)
        assert decision.selected is False
        assert decision.forced is False
        assert decision.source == 'no-row'

    def test_row_and_passing_gate_selects_tm(self, monkeypatch):
        monkeypatch.setitem(TM_OPS_FACTORY_TABLE, ('rms_norm', (8, 0)), {'evidence': 'test'})
        decision = resolve_tm('rms_norm', (8, 0), gate_reason_fn=lambda: None)
        assert decision.selected is True
        assert decision.forced is False
        assert decision.source == 'factory-table'

    def test_row_and_failing_gate_falls_back(self, monkeypatch):
        monkeypatch.setitem(TM_OPS_FACTORY_TABLE, ('rms_norm', (8, 0)), {'evidence': 'test'})
        decision = resolve_tm('rms_norm', (8, 0),
                              gate_reason_fn=lambda: 'missing kernel')
        assert decision.selected is False
        assert decision.source == 'gate-fallback'
        assert decision.gate_reason == 'missing kernel'

    def test_row_other_capability_not_matched(self, monkeypatch):
        monkeypatch.setitem(TM_OPS_FACTORY_TABLE, ('rms_norm', (8, 0)), {'evidence': 'test'})
        decision = resolve_tm('rms_norm', (8, 6), gate_reason_fn=lambda: None)
        assert decision.selected is False
        assert decision.source == 'no-row'

    def test_unknown_capability_family_no_row(self):
        decision = resolve_tm('rms_norm', None, gate_reason_fn=lambda: None)
        assert decision.selected is False
        assert decision.source == 'no-row'

    def test_gate_lazy_without_row(self):
        calls = []
        resolve_tm('rms_norm', (8, 0), gate_reason_fn=lambda: calls.append('x'))
        assert calls == []  # no row -> gate must not be evaluated


class TestResolveForce:
    """force: gate must pass or build-time error; contradicts explicit triton."""

    def test_force_and_passing_gate(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force')
        decision = resolve_tm('silu_and_mul', (8, 6), gate_reason_fn=lambda: None)
        assert decision.selected is True
        assert decision.forced is True
        assert decision.source == 'force'

    def test_force_and_failing_gate_raises(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force')
        with pytest.raises(RuntimeError, match='silu_and_mul'):
            resolve_tm('silu_and_mul', (8, 6), gate_reason_fn=lambda: 'no kernel')

    def test_force_not_covering_family_is_auto(self):
        decision = resolve_tm('silu_and_mul', (8, 6), gate_reason_fn=lambda: None)
        assert decision.selected is False

    def test_hard_contradiction_explicit_triton_vs_force(self, monkeypatch):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=w4a16')
        with pytest.raises(RuntimeError, match='contradic'):
            resolve_tm('w4a16', (8, 0),
                       gate_reason_fn=lambda: None,
                       per_op_backend='triton')

    def test_explicit_triton_without_force_selects_default(self, monkeypatch):
        monkeypatch.setitem(TM_OPS_FACTORY_TABLE, ('w4a16', (8, 0)), {'evidence': 'test'})
        decision = resolve_tm('w4a16', (8, 0),
                              gate_reason_fn=lambda: None,
                              per_op_backend='triton')
        assert decision.selected is False
        assert decision.source == 'per-op'

    def test_explicit_turbomind_overrides_off(self, monkeypatch):
        """off only closes the new mechanism, not published per-op opt-ins."""
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        decision = resolve_tm('w4a16', (8, 0),
                              gate_reason_fn=lambda: 'missing kernel',
                              per_op_backend='turbomind')
        assert decision.selected is True
        assert decision.source == 'per-op'

    def test_explicit_turbomind_overrides_auto_no_row(self):
        decision = resolve_tm('copy_cache', (8, 0),
                              gate_reason_fn=lambda: 'missing kernel',
                              per_op_backend='turbomind')
        assert decision.selected is True
        assert decision.source == 'per-op'

    def test_explicit_backend_not_turbomind_or_triton_ignored(self, monkeypatch):
        monkeypatch.setitem(TM_OPS_FACTORY_TABLE, ('w4a16', (8, 0)), {'evidence': 'test'})
        decision = resolve_tm('w4a16', (8, 0),
                              gate_reason_fn=lambda: None,
                              per_op_backend='deepgemm')
        assert decision.selected is True  # falls through to single-switch logic
        assert decision.source == 'factory-table'


class TestDynamicEnvRead:
    """resolve_tm must re-read the env on every call."""

    def test_env_change_between_calls(self, monkeypatch):
        decision = resolve_tm('rms_norm', (8, 0), gate_reason_fn=lambda: None)
        assert decision.selected is False

        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force')
        decision = resolve_tm('rms_norm', (8, 0), gate_reason_fn=lambda: None)
        assert decision.selected is True

        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'off')
        decision = resolve_tm('rms_norm', (8, 0), gate_reason_fn=lambda: None)
        assert decision.selected is False
