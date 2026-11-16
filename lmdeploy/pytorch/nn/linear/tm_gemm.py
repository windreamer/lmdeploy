# Copyright (c) OpenMMLab. All rights reserved.
"""W4A16 linear layer backed by the TurboMind GEMM channel."""

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
from lmdeploy.pytorch.backends.cuda.tm_weight_release import TorchWeightSpill
from lmdeploy.pytorch.backends.tm_gemm_modules import LinearTmGemmBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context

from .awq import AwqLinear


class TmGemmLinear(AwqLinear):
    """W4a16 linear that prefers the TurboMind GEMM kernel.

    Falls back to the AWQ Triton kernel per layer when the registry probe
    rejects the shape, mirroring the ``LMDEPLOY_W4A16_GEMM_BACKEND``
    semantics.

    ``release_torch_weights`` spills the raw torch weights to exact CPU
    copies after packing and frees the GPU storage. Hot updates restore the
    module from the copies through the wrapped loaders, then re-pack. Build
    modules that feed ``get_unquantized_weight`` (MLA absorption such as
    kv_b_proj) with ``release_torch_weights=False``; they keep their torch
    copies (see SPEC.md weight release, MLA exemption clause).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_bit: int,
        group_size: int,
        bias: bool,
        device: torch.device | None = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        layer_type: str = 'attn',
        release_torch_weights: bool = True,
    ):
        self.release_torch_weights = release_torch_weights
        self._spill: TorchWeightSpill | None = None
        super(AwqLinear, self).__init__(
            dtype=torch.float16,
            device=device,
            colwise=colwise,
            is_tp=is_tp,
            all_reduce=all_reduce,
            layer_type=layer_type,
        )
        if self.is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, w_bit, group_size, colwise)
        qweight, scales, qzeros, bias = self.create_weights(
            in_features, out_features, w_bit, group_size, bias, self.dtype, self.device
        )
        device_guard = torch.cuda.device(scales.device) if scales.is_cuda else __import__('contextlib').nullcontext()
        with device_guard:
            self.impl = get_backend().build_op(
                LinearTmGemmBuildSpec(
                    in_features=in_features,
                    out_features=out_features,
                    w_bit=w_bit,
                    group_size=group_size,
                    bias=bias is not None,
                    output_dtype=scales.dtype,
                    release_torch_weights=self.release_torch_weights,
                ),
                enable_deterministic=get_build_model_context().enable_deterministic,
            )
        self.register_all_parameters(qweight, scales, qzeros, bias)

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.elem_per_int = 32 // w_bit

    def update_weights(self):
        """Pack into the TurboMind channel, then spill the torch weights."""
        qweight, scales, qzeros, bias = self.impl.update_weights(self.qweight, self.scales, self.qzeros, self.bias)
        self.register_all_parameters(qweight, scales, qzeros, bias)
        if not isinstance(self.impl, TurbomindGemmW4A16Impl):
            # Triton fallback keeps its torch weights; nothing to spill.
            return
        if self._spill is None:
            self._spill = TorchWeightSpill()
        # Bias is read in python on every forward, so it stays on the GPU.
        self._spill.spill(
            dict(
                qweight=self.qweight,
                scales=self.scales,
                qzeros=self.qzeros,
            )
        )
