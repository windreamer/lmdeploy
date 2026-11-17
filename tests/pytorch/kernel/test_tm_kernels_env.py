# Copyright (c) OpenMMLab. All rights reserved.
"""Unified big-env ``LMDEPLOY_TM_KERNELS`` parse matrix and routing.

Unit cases without a CUDA dependency where possible. The parse matrix
covers the grammar of SPEC-B3: global tokens (off/auto/all), class-level
CSV overrides, legacy env fallback, and conflict warnings. The routing
cases re-check the migrated consumers (dense W4A16 builder, grouped MoE
route) under the big env and under the patched legacy attribute, keeping
the ticket 01/02 probe/fallback/forced semantics.

The ``attn_int8_kv`` consumer stays in ``backends/tm_kv.py``; the last
section pins the agreed boundary between the two readers: class entries
resolve identically, global tokens are SPEC semantics here while tm_kv
keeps ignoring them.
"""

import pytest
import torch

import lmdeploy.pytorch.envs as envs
from lmdeploy.pytorch.backends.cuda.tm_moe_modules import tm_moe_route
from lmdeploy.pytorch.backends.tm_kv import _parse_knob

TM_CLASSES = ('w4a16', 'moe_grouped', 'gdn', 'attn_int8_kv')

SUPPORTED = dict(in_features=4096, out_features=4096, w_bit=4, group_size=128)
UNSUPPORTED = dict(in_features=4096, out_features=4094, w_bit=4, group_size=128)

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.fixture(autouse=True)
def _clean_tm_env(monkeypatch):
    """Isolate the big env, the legacy envs, and the warning dedupe."""
    monkeypatch.delenv('LMDEPLOY_TM_KERNELS', raising=False)
    monkeypatch.delenv('LMDEPLOY_W4A16_GEMM_BACKEND', raising=False)
    monkeypatch.delenv('LMDEPLOY_TM_ATTN_INT8_KV', raising=False)
    envs._tm_kernel_conflict_warned.clear()


def _set_legacy(monkeypatch, value):
    """Patch the legacy provider attribute the consumers historically read."""
    monkeypatch.setattr(envs, 'w4a16_gemm_backend', value)


# ---------------------------------------------------------------------------
# Parse matrix: global tokens x class entries x malformed input.
# ---------------------------------------------------------------------------


def test_parse_unset_leaves_all_classes_open():
    """Without the big env every class stays open for its legacy env."""
    assert envs.parse_tm_kernels('') == dict.fromkeys(TM_CLASSES)


@pytest.mark.parametrize('token,state', [('off', 'off'), ('auto', 'auto'), ('all', 'on')])
def test_parse_global_token_sets_every_class(token, state):
    """Off/auto/all map onto per-class states for all classes."""
    assert envs.parse_tm_kernels(token) == dict.fromkeys(TM_CLASSES, state)


def test_parse_class_entry_overrides_global():
    """A class entry beats the global token; the global fills the rest."""
    assert envs.parse_tm_kernels('all,w4a16=off') == {
        'w4a16': 'off', 'moe_grouped': 'on', 'gdn': 'on', 'attn_int8_kv': 'on'}
    assert envs.parse_tm_kernels('off,gdn=auto') == {
        'w4a16': 'off', 'moe_grouped': 'off', 'gdn': 'auto', 'attn_int8_kv': 'off'}


def test_parse_global_after_class_entry_keeps_class_choice():
    """A global token behind a class entry must not undo the entry."""
    assert envs.parse_tm_kernels('w4a16=on,off')['w4a16'] == 'on'
    assert envs.parse_tm_kernels('w4a16=on,off')['moe_grouped'] == 'off'


def test_parse_first_entry_wins():
    """Duplicate globals and duplicate class entries keep the first value."""
    assert envs.parse_tm_kernels('off,all') == dict.fromkeys(TM_CLASSES, 'off')
    assert envs.parse_tm_kernels('w4a16=on,w4a16=off')['w4a16'] == 'on'


def test_parse_accepts_semicolons_and_whitespace():
    """';' separates entries too; blanks and spaces around tokens are fine."""
    parsed = envs.parse_tm_kernels('  w4a16=on ; moe_grouped = off ,,')
    assert parsed['w4a16'] == 'on'
    assert parsed['moe_grouped'] == 'off'
    assert parsed['gdn'] is None


def test_parse_skips_unknown_classes_silently(caplog):
    """Unknown class names stay silent for forward compatibility."""
    with caplog.at_level('WARNING', logger='lmdeploy'):
        parsed = envs.parse_tm_kernels('whatever=on,w4a16=off')
    assert parsed['w4a16'] == 'off'
    assert parsed['moe_grouped'] is None
    assert not caplog.records


def test_parse_state_is_case_insensitive_but_class_name_is_exact():
    """States lowercase; class names match exactly (tm_kv reading)."""
    assert envs.parse_tm_kernels('w4a16=ON')['w4a16'] == 'on'
    assert envs.parse_tm_kernels('W4A16=on')['w4a16'] is None


def test_parse_invalid_global_warns_and_continues(caplog):
    """A bare token outside off/auto/all warns and lets entries decide."""
    with caplog.at_level('WARNING', logger='lmdeploy'):
        parsed = envs.parse_tm_kernels('banana,w4a16=on')
    assert parsed['w4a16'] == 'on'
    assert any('Ignoring invalid LMDEPLOY_TM_KERNELS value banana' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    'value,cls,expected',
    [
        # Invalid state leaves the named class unresolved even when a later
        # entry retries it.
        ('attn_int8_kv=bogus,attn_int8_kv=on', 'attn_int8_kv', None),
        # Other classes still parse: one broken entry must not poison them.
        ('w4a16=bogus,moe_grouped=on', 'moe_grouped', 'on'),
    ],
    ids=['retry-ignored', 'other-class-parses'],
)
def test_parse_invalid_class_state_warns_and_skips_class(value, cls, expected, caplog):
    """Invalid states mirror the tm_kv reading: warn once, class open."""
    with caplog.at_level('WARNING', logger='lmdeploy'):
        assert envs.parse_tm_kernels(value)[cls] == expected
    assert any('Ignoring invalid LMDEPLOY_TM_KERNELS entry' in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Resolution: class entry > global token > legacy env > auto.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('attr,state', [('auto', 'auto'), ('turbomind', 'on'), ('triton', 'off'), ('off', 'off')])
def test_legacy_attribute_maps_onto_states(monkeypatch, attr, state):
    """With the big env silent, the patched legacy provider decides."""
    _set_legacy(monkeypatch, attr)
    assert envs.tm_kernel_state('w4a16') == state
    assert envs.tm_kernel_state('moe_grouped') == state


@pytest.mark.parametrize('token,state', [('off', 'off'), ('auto', 'auto'), ('all', 'on')])
def test_global_token_overrides_legacy_attribute(monkeypatch, token, state):
    """Any explicit big-env value beats the patched legacy attribute."""
    _set_legacy(monkeypatch, 'turbomind')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', token)
    assert envs.tm_kernel_state('w4a16') == state


def test_class_entries_are_isolated(monkeypatch):
    """Naming one class leaves the other classes on their legacy path."""
    _set_legacy(monkeypatch, 'off')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'moe_grouped=on')
    assert envs.tm_kernel_state('moe_grouped') == 'on'
    assert envs.tm_kernel_state('w4a16') == 'off'


def test_legacy_environ_only_applies_through_import(monkeypatch):
    """The w4a16 legacy env is read at import; runtime behavior is the import-
    time attribute.

    Setting the environ var alone must not flip a decision (the attribute is the single runtime source).
    """
    monkeypatch.setenv('LMDEPLOY_W4A16_GEMM_BACKEND', 'turbomind')
    _set_legacy(monkeypatch, 'auto')
    assert envs.tm_kernel_state('w4a16') == 'auto'


def test_attn_int8_kv_slot_resolution(monkeypatch):
    """The attn slot resolves class entry, legacy env, then auto."""
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'attn_int8_kv=on')
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'off')
    assert envs.tm_kernel_state('attn_int8_kv') == 'on'
    monkeypatch.delenv('LMDEPLOY_TM_KERNELS')
    assert envs.tm_kernel_state('attn_int8_kv') == 'off'
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'banana')
    assert envs.tm_kernel_state('attn_int8_kv') == 'auto'


def test_gdn_slot_defaults_to_auto(monkeypatch):
    """The gdn slot has no legacy env; only the big env can set it."""
    assert envs.tm_kernel_state('gdn') == 'auto'
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'gdn=on')
    assert envs.tm_kernel_state('gdn') == 'on'


def test_unknown_class_raises():
    """An unknown class name is a caller error, not a silent auto."""
    with pytest.raises(ValueError, match='Unknown TM kernel class'):
        envs.tm_kernel_state('nonsense')


def test_provider_mapping_roundtrips_legacy_values():
    """The import-time derivation is an identity on legacy values."""
    for raw in ('auto', 'turbomind', 'triton'):
        state = envs._TM_LEGACY_PROVIDER_STATES.get(raw, 'off')
        assert envs._TM_STATE_PROVIDERS[state] == raw


# ---------------------------------------------------------------------------
# Conflict warnings: explicit big env vs explicitly set legacy env.
# ---------------------------------------------------------------------------


def test_conflict_warns_once(monkeypatch, caplog):
    """A disagreeing legacy env warns once; repeats stay silent."""
    monkeypatch.setenv('LMDEPLOY_W4A16_GEMM_BACKEND', 'triton')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=on')
    with caplog.at_level('WARNING', logger='lmdeploy'):
        assert envs.tm_kernel_state('w4a16') == 'on'
        assert len(caplog.records) == 1
        assert 'w4a16=on' in caplog.records[0].getMessage()
        assert 'LMDEPLOY_W4A16_GEMM_BACKEND=triton' in caplog.records[0].getMessage()
        envs.tm_kernel_state('w4a16')
        assert len(caplog.records) == 1


def test_conflict_shared_legacy_env_warns_per_class(monkeypatch, caplog):
    """A global token conflicts with the legacy env of each consumer class."""
    monkeypatch.setenv('LMDEPLOY_W4A16_GEMM_BACKEND', 'triton')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'all')
    with caplog.at_level('WARNING', logger='lmdeploy'):
        envs.tm_kernel_state('w4a16')
        envs.tm_kernel_state('moe_grouped')
    messages = [r.getMessage() for r in caplog.records]
    assert any('w4a16=on' in m for m in messages)
    assert any('moe_grouped=on' in m for m in messages)


@pytest.mark.parametrize('legacy,token', [('turbomind', 'w4a16=on'), ('triton', 'w4a16=off')])
def test_agreement_never_warns(monkeypatch, caplog, legacy, token):
    """A legacy env that maps onto the same state stays silent."""
    monkeypatch.setenv('LMDEPLOY_W4A16_GEMM_BACKEND', legacy)
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', token)
    with caplog.at_level('WARNING', logger='lmdeploy'):
        envs.tm_kernel_state('w4a16')
    assert not caplog.records


def test_unset_legacy_never_warns(monkeypatch, caplog):
    """Without an explicitly set legacy env there is nothing to conflict."""
    _set_legacy(monkeypatch, 'turbomind')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=off')
    with caplog.at_level('WARNING', logger='lmdeploy'):
        assert envs.tm_kernel_state('w4a16') == 'off'
    assert not caplog.records


def test_default_posture_legacy_conflicts_warn(monkeypatch, caplog):
    """An explicit legacy auto against a concrete big-env state warns too:
    the old var no longer decides, and the user must know."""
    monkeypatch.setenv('LMDEPLOY_W4A16_GEMM_BACKEND', 'auto')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=off')
    with caplog.at_level('WARNING', logger='lmdeploy'):
        assert envs.tm_kernel_state('w4a16') == 'off'
    assert len(caplog.records) == 1


# ---------------------------------------------------------------------------
# Routing behavior: grouped MoE route and the dense W4A16 builder.
# ---------------------------------------------------------------------------


def _moe_route(w_bit=4):
    return tm_moe_route(group_size=128, w_bit=w_bit, dtype=torch.float16, hidden_dim=4096, ffn_dim=2048)


def test_moe_route_big_env_off_routes_triton(monkeypatch):
    """The moe_grouped class entry keeps the triton fused_moe route."""
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'moe_grouped=off')
    reason = _moe_route()
    assert reason is not None and 'off' in reason


def test_moe_route_big_env_forced_raises(monkeypatch):
    """On forces the grouped route and fails loudly on a bad geometry."""
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'moe_grouped=on')
    with pytest.raises(RuntimeError, match='TurboMind grouped MoE'):
        _moe_route(w_bit=3)


def test_moe_route_big_env_auto_probes(monkeypatch):
    """Auto probes and reports the probe reason on a bad geometry."""
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'moe_grouped=auto')
    reason = _moe_route(w_bit=3)
    assert reason is not None and 'w_bit' in reason


def test_moe_route_global_all_forces(monkeypatch):
    """The global all token forces the grouped route too."""
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'all')
    with pytest.raises(RuntimeError, match='TurboMind grouped MoE'):
        _moe_route(w_bit=3)


def test_moe_route_global_all_with_class_off_routes_triton(monkeypatch):
    """A class entry carves one class out of the global all token."""
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'all,moe_grouped=off')
    reason = _moe_route()
    assert reason is not None and 'off' in reason


@pytest.mark.parametrize('attr,expect', [('off', 'off'), ('auto', 'w_bit'), ('turbomind', 'raise')])
def test_moe_route_legacy_attribute_still_decides(monkeypatch, attr, expect):
    """Without the big env the patched legacy attribute keeps its semantics."""
    _set_legacy(monkeypatch, attr)
    if expect == 'raise':
        with pytest.raises(RuntimeError, match='TurboMind grouped MoE'):
            _moe_route(w_bit=3)
    else:
        reason = _moe_route(w_bit=3)
        assert reason is not None and expect in reason


def _dense_spec(**overrides):
    from lmdeploy.pytorch.backends.awq_modules import LinearW4A16BuildSpec
    params = dict(bias=False, output_dtype=torch.float16)
    params.update(overrides)
    return LinearW4A16BuildSpec(**params)


def test_dense_route_big_env_off_keeps_awq_triton(monkeypatch):
    """The w4a16 class entry off builds the historical AWQ Triton impl."""
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl, _build_linear_w4a16
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=off')
    impl = _build_linear_w4a16(_dense_spec(**SUPPORTED))
    assert isinstance(impl, AwqLinearW4A16Impl)


@pytest.mark.parametrize('value', ['w4a16=off', 'off'])
def test_dense_route_off_states_agree(monkeypatch, value):
    """The global off token and the class entry pick the same impl."""
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl, _build_linear_w4a16
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', value)
    impl = _build_linear_w4a16(_dense_spec(**SUPPORTED))
    assert isinstance(impl, AwqLinearW4A16Impl)


@cuda_required
def test_dense_route_big_env_on_builds_tm(monkeypatch):
    """The w4a16 class entry on builds the TM impl on a supported shape."""
    from lmdeploy.pytorch.backends.cuda.awq_modules import _build_linear_w4a16
    from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=on')
    impl = _build_linear_w4a16(_dense_spec(**SUPPORTED))
    assert isinstance(impl, TurbomindGemmW4A16Impl)


@cuda_required
def test_dense_route_big_env_on_raises_on_unsupported(monkeypatch):
    """On raises on a rejected shape instead of slowing down."""
    from lmdeploy.pytorch.backends.cuda.awq_modules import _build_linear_w4a16
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=on')
    with pytest.raises(RuntimeError, match='unavailable or incompatible'):
        _build_linear_w4a16(_dense_spec(**UNSUPPORTED))


@cuda_required
def test_dense_route_big_env_auto_falls_back(monkeypatch):
    """Auto falls back per layer when the registry rejects the shape."""
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl, _build_linear_w4a16
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'w4a16=auto')
    impl = _build_linear_w4a16(_dense_spec(**UNSUPPORTED))
    assert isinstance(impl, AwqLinearW4A16Impl)


@cuda_required
def test_dense_route_ignores_other_class_entries(monkeypatch):
    """A big env naming only other classes leaves dense on its legacy path."""
    from lmdeploy.pytorch.backends.cuda.awq_modules import _build_linear_w4a16
    from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
    _set_legacy(monkeypatch, 'auto')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', 'attn_int8_kv=on,moe_grouped=off')
    impl = _build_linear_w4a16(_dense_spec(**SUPPORTED))
    assert isinstance(impl, TurbomindGemmW4A16Impl)


# ---------------------------------------------------------------------------
# Boundary with backends/tm_kv.py (the attn_int8_kv consumer).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'value',
    [
        'attn_int8_kv=on',
        'attn_int8_kv=off',
        'attn_int8_kv=auto',
        'attn_int8_kv=bogus,attn_int8_kv=on',
        'attn_int8_kv=on,attn_int8_kv=bogus',
        'w4a16=on,attn_int8_kv=off',
        'w4a16=bogus,attn_int8_kv=on',
        'banana,attn_int8_kv=on',
        'attn_int8_kv=on;gdn=off',
        'ATTN_INT8_KV=on',
        'attn_int8_kv=ON',
        ' attn_int8_kv = on ',
    ],
)
def test_class_entries_match_tm_kv_reading(monkeypatch, value):
    """Class entries resolve identically in both readers."""
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'off')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', value)
    assert _parse_knob() == envs.tm_kernel_state('attn_int8_kv')


@pytest.mark.parametrize('token,state', [('off', 'off'), ('auto', 'auto'), ('all', 'on')])
def test_global_tokens_are_spec_semantics_here(monkeypatch, token, state):
    """Global tokens apply per SPEC; tm_kv still ignores them (legacy env
    fallback), the one documented divergence between the two readers."""
    monkeypatch.delenv('LMDEPLOY_TM_ATTN_INT8_KV', raising=False)
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', token)
    assert envs.tm_kernel_state('attn_int8_kv') == state
    assert _parse_knob() == 'auto'
