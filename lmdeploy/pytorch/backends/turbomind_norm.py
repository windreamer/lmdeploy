# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind RMSNorm implementation (opt-in via LMDEPLOY_TURBOMIND_OPS).

Variant coverage conclusion (recorded per ticket 04):

- Plain rmsnorm (``invokeRMSNorm``) and the fused add-residual variant
  (``invokeResidualBiasRMSNorm``) are bound and covered: their semantics
  match the Triton ``rms_norm``/``add_rms_norm`` kernels
  (``residual <- x + residual``; ``out <- norm(residual) * w``).
- The TM **qk** variant (``invokeQkRMSNorm``) normalizes packed qkv heads
  in place; the PyTorch engine's ``RMSNormImpl`` seam has no qk-shaped
  entry point, so it stays exposed as a raw binding only and is exercised
  by the numeric parity test.
- Inputs are cloned before the in-place TM kernels so the seam keeps the
  Triton implementation's non-mutating contract.
"""
import torch

from .norm import RMSNormImpl


class TurbomindRMSNormImpl(RMSNormImpl):
    """RMS norm via the TurboMind ``invokeRMSNorm`` /
    ``invokeResidualBiasRMSNorm`` kernels."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        from lmdeploy.turbomind import _tm

        stream = torch.cuda.current_stream(x.device).cuda_stream
        x2 = x.reshape(-1, x.shape[-1]).clone()
        if residual is None:
            out = torch.empty_like(x2)
            _tm.rms_norm(_tm.from_dlpack(out), _tm.from_dlpack(x2), _tm.from_dlpack(weight), self.eps, False, stream)
            return out.reshape(x.shape)

        residual2 = residual.reshape(-1, residual.shape[-1]).clone()
        _tm.residual_rms_norm(
            _tm.from_dlpack(x2),
            _tm.from_dlpack(residual2),
            _tm.from_dlpack(weight),
            None,
            self.eps,
            False,
            stream,
        )
        return x2.reshape(x.shape), residual2.reshape(residual.shape)
