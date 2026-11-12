# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.tm_ops import resolve_tm
from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

from ..activation import SiluAndMulBuildSpec, SiluAndMulImpl


class TritonSiluAndMulImpl(SiluAndMulImpl):
    """Silu + multiple residual fused implementation."""

    def __init__(self, inplace: bool):
        self.inplace = inplace

    def forward(self, x):
        """forward."""
        out = None
        x_shape = None
        if x.dim() != 2:
            x_shape = x.shape
            x = x.flatten(0, -2)
        if self.inplace:
            out = x.chunk(2, -1)[0]

        out = silu_and_mul(x, out)

        if x_shape is not None:
            out = out.unflatten(0, x_shape[:-1])
        return out


def turbomind_silu_and_mul_gate_reason(dtype: torch.dtype) -> str | None:
    """Return a reason string when the TurboMind activation kernel cannot run
    this problem, else None.

    The TM kernel vectorizes with 16-byte loads (vec = 4 elements for
    fp16/bf16) without a per-element tail mask, so the activation width must
    be a multiple of the vector size; the moe_ep / masked-post-quant variants
    are covered by ``turbomind_activation.py``'s module docstring.
    """
    from lmdeploy import turbomind

    if not turbomind.is_available():
        return f'TurboMind extension unavailable: {turbomind._import_error}'
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f'unsupported dtype {dtype} (TM kernel supports fp16/bf16/fp32)'
    return None


def build_silu_and_mul(spec: SiluAndMulBuildSpec) -> SiluAndMulImpl:
    """Build the selected CUDA silu-and-mul implementation.

    Selection priority (spec): per-op env (none for this family) > single-
    switch force > auto + factory-table row > no row -> triton. No factory
    table row ships for ``silu_and_mul``, so ``auto``/``off`` keep the
    Triton path byte-identical.
    """
    capability = torch.cuda.get_device_capability()
    decision = resolve_tm('silu_and_mul',
                          capability,
                          gate_reason_fn=lambda: turbomind_silu_and_mul_gate_reason(torch.get_default_dtype()))
    if decision.selected:
        from .turbomind_activation import TurbomindSiluAndMulImpl
        impl_cls = TurbomindSiluAndMulImpl
    else:
        impl_cls = TritonSiluAndMulImpl

    return impl_cls(spec.inplace)
