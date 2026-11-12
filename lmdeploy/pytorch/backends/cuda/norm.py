# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.tm_ops import resolve_tm
from lmdeploy.pytorch.kernels.cuda import rms_norm

from ..norm import RMSNormBuildSpec, RMSNormImpl


class TritonRMSNormImpl(RMSNormImpl):
    """Triton RMS norm implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        if residual is None:
            x = rms_norm(x, weight, self.eps)
            return x
        else:
            x, residual = rms_norm(x, weight, self.eps, residual=residual)
            return x, residual


def turbomind_rms_norm_gate_reason(hidden_size: int) -> str | None:
    """Return a reason string when the TurboMind RMSNorm kernels cannot run
    this problem, else None.

    The TM kernels vectorize with 16-byte loads (vec = 16 / sizeof(dtype))
    and have no per-element tail mask, so the hidden size must be a
    multiple of the vector width (8 elements at fp16/bf16, the dtypes the
    kernels actually dispatch to; fp32 inputs raise "unsupported type: f32"
    inside ``invokeRMSNorm``).
    """
    from lmdeploy import turbomind

    if not turbomind.is_available():
        return f'TurboMind extension unavailable: {turbomind._import_error}'
    if hidden_size % 8 != 0:
        return f'hidden_size {hidden_size} not a multiple of the TM vector width 8'
    return None


def build_rms_norm(spec: RMSNormBuildSpec) -> RMSNormImpl:
    """Build the selected CUDA RMS norm implementation.

    Selection priority (spec): per-op env (none for this family) > single-
    switch force > auto + factory-table row > no row -> triton. No factory
    table row ships for ``rms_norm``, so ``auto``/``off`` keep the Triton
    path byte-identical.
    """
    capability = torch.cuda.get_device_capability()
    decision = resolve_tm('rms_norm',
                          capability,
                          gate_reason_fn=lambda: turbomind_rms_norm_gate_reason(spec.hidden_size))
    if decision.selected:
        from ..turbomind_norm import TurbomindRMSNormImpl
        impl_cls = TurbomindRMSNormImpl
    else:
        impl_cls = TritonRMSNormImpl

    return impl_cls(spec.hidden_size, spec.eps)
