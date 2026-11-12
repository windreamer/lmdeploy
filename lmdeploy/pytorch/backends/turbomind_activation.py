# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind SiluAndMul implementation (opt-in via LMDEPLOY_TURBOMIND_OPS).

Variant coverage conclusion (recorded per ticket 04):

- Covered: the split gate/up form of the TM ``Activation`` kernel with
  ``ActivationType::kSilu``, including the padded-buffer variant gated by
  a device-side ``num_valid_tokens`` count (bound as ``valid_tokens``).
- Kept on Triton (no TM equivalent on this seam):
  - ``silu_and_mul_moe_ep`` (per-row ``mask_m`` mask — the TM kernel takes
    a single device-side valid-token *count*, not a per-row mask);
  - ``silu_and_mul_masked_post_quant_fwd`` (fused FP8 quantization — the
    TM kernel has no quantized-output mode);
  - ``swiglu_limit`` / ``precise_mul`` options of the Triton kernel.
- The TM kernel writes in place into the gate half; the implementation
  clones it unless ``inplace`` matches the Triton implementation's
  contract.
"""
import torch

from .activation import SiluAndMulImpl


class TurbomindSiluAndMulImpl(SiluAndMulImpl):
    """Silu + multiply via the TurboMind ``Activation`` kernel."""

    def __init__(self, inplace: bool):
        self.inplace = inplace

    def forward(self, x):
        """forward."""
        from lmdeploy.turbomind import _tm

        x_shape = None
        if x.dim() != 2:
            x_shape = x.shape
            x = x.flatten(0, -2)

        gate, up = x.chunk(2, -1)
        if self.inplace:
            # The kernel writes into the gate half of x and reads both halves
            # with the gate row stride, which matches here (both are views of
            # x). This mirrors the Triton inplace contract.
            out = gate
        else:
            # The TM kernel uses a single row stride for gate and up, so the
            # two halves must have equal strides; clone both into contiguous
            # buffers to keep the seam's non-mutating contract.
            out = gate.clone()
            up = up.clone()

        _tm.silu_and_mul(_tm.from_dlpack(out), _tm.from_dlpack(up), None,
                         torch.cuda.current_stream(x.device).cuda_stream)

        if x_shape is not None:
            out = out.unflatten(0, x_shape[:-1])
        return out
