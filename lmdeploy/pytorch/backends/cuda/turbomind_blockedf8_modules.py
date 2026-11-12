# Copyright (c) OpenMMLab. All rights reserved.
"""Optional blocked-FP8 linear backend using the TurboMind Linear API
(family 15, SM80): fp8-block weights are served by the TurboMind kernel,
which dequantizes the fp8 weight to bf16 in-pipeline; no fp8 hardware is
required."""

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.utils import get_logger

from ..blockedf8_modules import LinearBlockedF8Impl

logger = get_logger('lmdeploy')


def turbomind_blockedf8_gate_reason(in_features: int,
                                    out_features: int,
                                    block_size: int,
                                    output_dtype: torch.dtype = None,
                                    fp8_dtype: torch.dtype = None) -> str | None:
    """Return why the bundled TurboMind provider cannot serve this layer."""
    if block_size != 128:
        return f'block_size must be 128, but got {block_size}'
    if fp8_dtype not in (None, torch.float8_e4m3fn):
        return f'fp8_dtype must be float8_e4m3fn, but got {fp8_dtype}'
    if output_dtype not in (None, torch.bfloat16):
        return f'output_dtype must be bfloat16, but got {output_dtype}'
    if in_features % 128 != 0:
        return f'K must be divisible by 128, but got {in_features}'
    if out_features % 8 != 0:
        return f'N must be divisible by 8, but got {out_features}'
    if not torch.cuda.is_available():
        return 'CUDA is unavailable'

    capability = torch.cuda.get_device_capability()
    if capability[0] != 8:
        return (f'CUDA capability {capability} is unsupported: '
                'the blocked-FP8 family targets SM80 devices')

    from lmdeploy import turbomind
    if not turbomind.is_available():
        return 'TurboMind native modules are not built'
    return None


class TurbomindLinearBlockedF8Impl(LinearBlockedF8Impl):
    """Adapt canonical blocked-FP8 parameters to the TurboMind Linear API."""

    def __init__(self, in_features: int, out_features: int, block_size: int = 128):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self._linear: 'Linear | None' = None
        self._weight: 'Weight | None' = None

    def _detect_block_out(self, scale: torch.Tensor) -> int:
        """Map the checkpoint scale layout to the family 15 format."""
        k_groups = self.in_features // self.block_size
        if scale.shape[-1] != k_groups:
            raise ValueError(f'unsupported blocked-FP8 scale shape {tuple(scale.shape)} '
                             f'for K={self.in_features}, block_size={self.block_size}')
        if scale.shape[-2] == self.out_features:
            return 1
        if scale.shape[-2] == (self.out_features + self.block_size - 1) // self.block_size:
            return self.block_size
        raise ValueError(f'unsupported blocked-FP8 scale shape {tuple(scale.shape)} '
                         f'for N={self.out_features}, K={self.in_features}')

    def _release(self):
        weight, self._weight = self._weight, None
        if weight is None:
            self._linear = None
            return
        # Forward is asynchronous and may use any engine stream. Release the
        # prepared weight only after all device work completes.
        torch.cuda.synchronize()
        weight.close()
        self._linear = None

    def __del__(self):
        try:
            linear = self._linear
            if linear is None:
                return
            with torch.cuda.device(linear.device):
                self._release()
        except Exception:
            pass

    def update_weights(self,
                       weight: torch.Tensor,
                       scale: torch.Tensor,
                       bias: torch.Tensor | None = None):
        """Prepare the fp8-block weight for the TurboMind family 15 kernel."""
        block_out = self._detect_block_out(scale)
        self._release()

        from lmdeploy.turbomind.linear import get_linear
        from lmdeploy.turbomind.weight_format import FP8Format

        linear = get_linear()
        plan = linear.get_weight_plan(weight_format=FP8Format(block_out=block_out), dtype=torch.bfloat16)
        prepared = linear.prepare_weight(weight, plan=plan, scales=scale)
        # Weight preparation is asynchronous on the loading stream, while the
        # engine executes on another stream. Publish only after packing
        # completes.
        torch.cuda.current_stream().synchronize()

        self._linear = linear
        self._weight = prepared
        logger.debug(f'Prepared blocked-FP8 weight with FP8Format(block_out={block_out}).')
        return weight, scale, bias

    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: torch.distributed.ProcessGroup | None = None,
                rank: int = 0,
                scatter_size: list[int] = None):
        linear = self._linear
        prepared = self._weight
        if prepared is None:
            raise RuntimeError('update_weights must be called before forward')

        x_shape = x.shape
        x = x.flatten(0, -2)
        exec_plan = linear.get_exec_plan(x, prepared)
        out, _ = linear(x, prepared, exec_plan=exec_plan)

        if bias is not None:
            out = out + bias
        out = out.unflatten(0, x_shape[:-1])

        if all_reduce:
            if scatter_size is not None:
                out = dist.reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out
