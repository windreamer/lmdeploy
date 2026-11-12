# Copyright (c) OpenMMLab. All rights reserved.
"""Single-switch + factory-table infrastructure for TurboMind extracted
kernels in the PyTorch engine.

The switch value ``LMDEPLOY_TURBOMIND_OPS`` (``auto`` by default) and the
valid family keys are defined in :mod:`lmdeploy.pytorch.envs`; this module
holds the built-in factory table and the per-family selection helper used by
the device builders.

Selection priority per family (spec: tm-kernel-integration):

1. per-op env explicit value (published per-op variables stay authoritative)
2. single-switch ``force`` / ``force=<csv>`` covering the family
3. ``auto`` + factory-table row whose capability matches (gate not passing ->
   fall back to the original implementation)
4. no row -> original implementation

The only hard contradiction is an explicit per-op ``triton`` combined with
``force`` covering the same family; that raises at build time. Everything
else is silently resolved by the priority above.
"""
from dataclasses import dataclass

from lmdeploy.pytorch.envs import TURBOMIND_OPS_FAMILIES, TurbomindOpsConfig, parse_turbomind_ops

__all__ = [
    'TM_OPS_FACTORY_TABLE',
    'TM_OPS_FAMILIES',
    'TmDecision',
    'resolve_tm',
    'current_turbomind_ops_config',
]

# Re-exported so builders only need this module.
TM_OPS_FAMILIES = TURBOMIND_OPS_FAMILIES

# Built-in factory table: key = (family, exact (major, minor) capability),
# value = dict with an 'evidence' field referencing benchmark records under
# benchmarks/ops/. Rows are only added by review PRs carrying benchmark
# evidence. w4a16 rows landed by ticket 02: sm80 benchmark record plus the
# sm90 record documenting the f745840-era default behavior inherited by the
# promoted stack.
TM_OPS_FACTORY_TABLE: dict = {
    ('w4a16', (8, 0)): {
        'evidence': 'benchmarks/ops/w4a16_sm80.json',
    },
    ('w4a16', (9, 0)): {
        'evidence': 'benchmarks/ops/w4a16_sm90.json',
    },
}


def current_turbomind_ops_config() -> TurbomindOpsConfig:
    """Re-read LMDEPLOY_TURBOMIND_OPS on every call.

    Keeps selection monkeypatch- and runtime-overridable: builders evaluate
    the switch at op build time, not at import time.
    """
    import os

    return parse_turbomind_ops(os.environ.get('LMDEPLOY_TURBOMIND_OPS', 'auto'))


@dataclass(frozen=True)
class TmDecision:
    """Outcome of the single-switch + factory-table selection for a family.

    Attributes:
        family: the TM op family being resolved.
        selected: whether the TurboMind implementation should be built.
        forced: whether selection came from the force semantics (gate must
            pass; failing it raises) rather than the factory table.
        source: diagnostic origin of the decision ('per-op', 'force',
            'factory-table', 'gate-fallback', 'no-row', 'off').
        gate_reason: gate failure reason when the gate blocked selection.
    """
    family: str
    selected: bool
    forced: bool
    source: str
    gate_reason: str | None = None


def resolve_tm(family: str,
               capability,
               gate_reason_fn,
               per_op_backend: str | None = None) -> TmDecision:
    """Resolve whether a family should use its TurboMind implementation.

    Args:
        family: one of TM_OPS_FAMILIES.
        capability: device (major, minor) capability tuple, or None when
            unknown (no factory-table row can match).
        gate_reason_fn: lazy zero-arg callable returning None when the
            feasibility gate passes, or a reason string otherwise. Only
            invoked when a decision actually depends on it (force, or
            auto + matching table row), so that the gate never triggers
            avoidable imports for families that will not go TM.
        per_op_backend: explicit per-op env value if the family has one
            (e.g. 'auto'/'triton'/'turbomind' for w4a16); None otherwise.

    Raises:
        RuntimeError: force covers the family but the gate fails, or the
            hard contradiction (explicit per-op 'triton' x force) holds.
    """
    if family not in TM_OPS_FAMILIES:
        raise ValueError(
            f'Unknown TM op family: {family!r}. Valid families: {TM_OPS_FAMILIES}')

    cfg = current_turbomind_ops_config()
    forced_here = cfg.mode == 'force' and family in cfg.forced

    # Priority 1: explicit per-op env value.
    if per_op_backend is not None:
        if per_op_backend == 'turbomind':
            return TmDecision(family=family, selected=True, forced=False,
                              source='per-op')
        if per_op_backend == 'triton':
            if forced_here:
                raise RuntimeError(
                    f'Configuration contradiction for TM op family {family!r}: '
                    'the per-op env explicitly selects "triton" while '
                    f'LMDEPLOY_TURBOMIND_OPS={cfg.mode} forces it. '
                    'Unset one of them.')
            return TmDecision(family=family, selected=False, forced=False,
                              source='per-op')
        # Other explicit values (e.g. 'deepgemm', 'gluon') keep their own
        # semantics; fall through to single-switch resolution.

    # Priority 0: off closes the whole mechanism.
    if cfg.mode == 'off':
        return TmDecision(family=family, selected=False, forced=False, source='off')

    # Priority 2: force.
    if forced_here:
        gate_reason = gate_reason_fn()
        if gate_reason is not None:
            raise RuntimeError(
                f'LMDEPLOY_TURBOMIND_OPS forces TM for family {family!r} but the '
                f'feasibility gate failed on capability {capability}: {gate_reason}')
        return TmDecision(family=family, selected=True, forced=True, source='force')

    # Priority 3: auto + factory-table row.
    if cfg.mode == 'auto' and capability is not None:
        row = TM_OPS_FACTORY_TABLE.get((family, tuple(capability)))
        if row is not None:
            gate_reason = gate_reason_fn()
            if gate_reason is not None:
                return TmDecision(family=family, selected=False, forced=False,
                                  source='gate-fallback', gate_reason=gate_reason)
            return TmDecision(family=family, selected=True, forced=False,
                              source='factory-table')

    # Priority 4: no row -> original implementation.
    return TmDecision(family=family, selected=False, forced=False, source='no-row')
