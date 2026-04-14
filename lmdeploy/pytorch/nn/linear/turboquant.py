# Copyright (c) OpenMMLab. All rights reserved.
"""TurboQuant Linear layer with on-the-fly 4-bit dequantization."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
    get_lloyd_max_codebook,
    hadamard_rotate,
    hadamard_rotate_inv,
)
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

from ..utils import get_distribute_size
from .base import LinearBase


def pack_4bit(indices: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8, 2 per byte.

    Sequential layout: within each group, first half → lo nibble,
    second half → hi nibble.

    packed[i] = indices[g*GS + i] | (indices[g*GS + HALF + i] << 4)
    """
    K = indices.shape[-1]
    assert K % 2 == 0, f'Last dim must be even for 4-bit packing, got {K}'
    gs = group_size if group_size > 0 else K
    assert K % gs == 0, f'K={K} not divisible by group_size={gs}'
    half = gs // 2

    chunks = []
    for g_start in range(0, K, gs):
        lo = indices[..., g_start: g_start + half].to(torch.uint8)
        hi = indices[..., g_start + half: g_start + gs].to(torch.uint8)
        chunks.append(lo | (hi << 4))
    return torch.cat(chunks, dim=-1)

def unpack_4bit(packed: torch.Tensor, n: int, group_size: int = 0) -> torch.Tensor:
    """Unpack uint8 → 4-bit indices (sequential layout).

    Reverses pack_4bit: lo nibble = first half of group,
    hi nibble = second half. concat(lo, hi) per group = original order.
    """
    gs = group_size if group_size > 0 else n
    packed_gs = gs // 2

    chunks = []
    packed_k = packed.shape[-1]
    for pg_start in range(0, packed_k, packed_gs):
        block = packed[..., pg_start: pg_start + packed_gs]
        lo = (block & 0x0F).to(torch.int32)
        hi = ((block >> 4) & 0x0F).to(torch.int32)
        chunks.append(lo)
        chunks.append(hi)
    result = torch.cat(chunks, dim=-1)
    return result[..., :n]

class TurboQuantLinear(LinearBase):
    """TurboQuant Linear layer with on-the-fly 4-bit dequantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        colwise: bool = True,
        is_tp: bool = False,
        all_reduce: bool = True,
        bit_width: int = 4,
        group_size: int = 128,
        layer_type: str = 'attn',
    ):
        super().__init__(dtype=dtype,
                         device=device,
                         colwise=colwise,
                         is_tp=is_tp,
                         all_reduce=all_reduce,
                         layer_type=layer_type)

        self.bit_width = bit_width
        self.group_size = group_size

        if self.is_tp:
            in_features, out_features = self._get_io_features(in_features, out_features, colwise)

        codebook, boundaries = get_lloyd_max_codebook(d=in_features, bits=bit_width, device=self.device)
        weight, weight_norms, bias = self._create_weights(in_features, out_features, bias, self.device)
        self._register_all_parameters(weight, weight_norms, bias, codebook, boundaries)

        self.in_features = in_features
        self.out_features = out_features

    def _create_weights(self, in_features: int, out_features: int, bias: bool, device: torch.device):
        packed_in_features = in_features // 2
        weight = torch.empty((out_features, packed_in_features), dtype=torch.uint8, device=device)
        n_groups = (in_features + self.group_size - 1) // self.group_size
        weight_norms = torch.empty((out_features, n_groups), dtype=torch.float32, device=device)
        bias_tensor = torch.empty((out_features,), dtype=torch.float32, device=device) if bias else None
        return weight, weight_norms, bias_tensor

    def _register_all_parameters(
        self,
        weight: torch.Tensor,
        weight_norms: torch.Tensor,
        bias: torch.Tensor | None = None,
        codebook: torch.Tensor | None = None,
        boundaries: torch.Tensor | None = None,
    ):
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight_norms = torch.nn.Parameter(weight_norms, requires_grad=False)
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)

        self.register_parameter('weight', weight)
        self.register_parameter('weight_norms', weight_norms)
        self.register_parameter('bias', bias)

        # Register codebook and boundaries as buffers (non-learnable constants)
        if codebook is not None:
            self.register_buffer('_codebook', codebook)
        if boundaries is not None:
            self.register_buffer('_boundaries', boundaries)

        self.setup_loaders()

    def setup_loaders(self):
        """Setup weight loaders."""
        self.weight.weight_loader = self._weight_loader_with_quant
        self.weight_norms.weight_loader = self._weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    @property
    def codebook(self) -> torch.Tensor | None:
        return getattr(self, '_codebook', None)

    @property
    def boundaries(self) -> torch.Tensor | None:
        return getattr(self, '_boundaries', None)

    def _get_io_features(self, in_features: int, out_features: int, colwise: bool):
        world_size, rank = self.get_tp_world_rank()
        if colwise:
            out_features = get_distribute_size(out_features, world_size, rank)
        else:
            in_features = get_distribute_size(in_features, world_size, rank)
        return in_features, out_features

    def _weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def _weight_loader_tp_colwise(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        rank: int,
        world_size: int,
    ):
        weight = loaded_weight.chunk(world_size, 0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        rank: int,
        world_size: int,
    ):
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.to(param.device)
            weight = loaded_weight.chunk(world_size, 1)[rank]
            return default_weight_loader(param, weight)
        else:
            if rank != 0:
                loaded_weight = torch.zeros_like(loaded_weight)
            return default_weight_loader(param, loaded_weight)

    def _weight_loader_with_quant(self,
                                  param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor,
                                  shard_id: Any = None):
        if loaded_weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            loaded_weight = loaded_weight.to(param.device)
            quant_result = self._quantize_weight(loaded_weight)
            indices_packed, weight_norms = quant_result['indices_packed'], quant_result['norms']
            self._weight_loader(self.weight, indices_packed)
            self._weight_loader(self.weight_norms, weight_norms)
        else:
            return self._weight_loader(param, loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = None):
        return self._weight_loader_with_quant(param, loaded_weight, shard_id)

    def _quantize_weight(self, w: torch.Tensor) -> dict:
        w = w.float()
        out_features, in_features = w.shape

        codebook = self.codebook
        boundaries = self.boundaries

        all_norms = []
        all_indices = []

        for g_start in range(0, in_features, self.group_size):
            g_end = min(g_start + self.group_size, in_features)
            g_dim = g_end - g_start
            w_g = w[:, g_start:g_end]

            norms = w_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
            w_norm = w_g / norms
            all_norms.append(norms.squeeze(1))

            y = hadamard_rotate(w_norm)

            sigma = 1.0 / math.sqrt(in_features)
            scale = sigma * math.sqrt(g_dim)
            y_scaled = y * scale

            indices = torch.searchsorted(boundaries, y_scaled.reshape(-1))
            indices = indices.clamp(0, len(codebook) - 1).reshape(out_features, g_dim)
            all_indices.append(indices)

        full_indices = torch.cat(all_indices, dim=1)
        norms_out = torch.stack(all_norms, dim=1)

        if in_features % 2 != 0:
            full_indices = nn.functional.pad(full_indices, (0, 1), value=0)

        packed = pack_4bit(full_indices, self.group_size)
        return {'indices_packed': packed, 'norms': norms_out}

    def _dequantize_weight(self) -> torch.Tensor:
        out_features, in_features = self.out_features, self.in_features
        device = self.weight.device
        codebook = self.codebook
        indices = unpack_4bit(self.weight.data, in_features, self.group_size)
        w = torch.zeros(out_features, in_features, dtype=torch.float32, device=device)

        for g_start in range(0, in_features, self.group_size):
            g_end = min(g_start + self.group_size, in_features)
            g_dim = g_end - g_start

            y = codebook[indices[:, g_start:g_end].long()]

            sigma = 1.0 / math.sqrt(in_features)
            scale = sigma * math.sqrt(g_dim)
            y = y / scale

            w_g = hadamard_rotate_inv(y)
            g_idx = g_start // self.group_size
            w_g = w_g * self.weight_norms[:, g_idx].unsqueeze(1)
            w[:, g_start:g_end] = w_g
        return w

    def _forward_fused(self, x, all_reduce: bool, tp_sizes: list[int]):
        """Fused forward path (single Triton launch, no weight
        materialization)."""
        from lmdeploy.pytorch.kernels.cuda.turbo_quant_fused import (
            turboquant_fused_forward,
        )
        out = turboquant_fused_forward(
            x=x,
            weight_packed=self.weight.data,
            codebook=self.codebook.data,
            weight_norms=self.weight_norms.data,
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
            bias=self.bias,
        )
        if all_reduce and self.tp_group is not None:
            if not (self.tp_mode == TPMode.DP_TP
                    and tp_sizes is not None):
                torch.distributed.all_reduce(out, group=self.tp_group)
        return out

    def _forward_default(self, x, all_reduce: bool, tp_sizes: list[int]):
        """Unfused fallback (CPU / testing)."""
        w = self._dequantize_weight().to(x.dtype)
        out = torch.matmul(x, w.t())
        if self.bias is not None:
            out = out + self.bias
        if all_reduce and self.tp_group is not None:
            if not (self.tp_mode == TPMode.DP_TP
                    and tp_sizes is not None):
                torch.distributed.all_reduce(out, group=self.tp_group)
        return out

    def forward(self, x, all_reduce: bool = True,
                tp_sizes: list[int] = None):
        if x.is_cuda:
            return self._forward_fused(x, all_reduce, tp_sizes)
        return self._forward_default(x, all_reduce, tp_sizes)

    def update_weights(self):
        """Update weights."""
        self.register_all_parameters(self.weight, self.weight_norms, self.bias)

    def register_all_parameters(self,
                                weight: torch.Tensor,
                                weight_norms: torch.Tensor,
                                bias: torch.Tensor | None = None):
        """Register all parameters."""
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight_norms = torch.nn.Parameter(weight_norms, requires_grad=False)
        if bias is not None:
            bias = torch.nn.Parameter(bias, requires_grad=False)
        self.register_parameter('weight', weight)
        self.register_parameter('weight_norms', weight_norms)
        self.register_parameter('bias', bias)
        self.setup_loaders()

    def get_dequantized_weight(self) -> torch.Tensor:
        return self._dequantize_weight().to(torch.bfloat16)


class TurboQuantQKVLinear(LinearBase):
    """Combined QKV linear layer using separate TurboQuantLinear modules."""

    def __init__(self,
                 q_proj: TurboQuantLinear,
                 k_proj: TurboQuantLinear,
                 v_proj: TurboQuantLinear,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 head_size_v: int,
                 is_tp: bool = False,
                 all_reduce: bool = False):
        super().__init__(dtype=q_proj.dtype if q_proj else None,
                         device=q_proj.device if q_proj else None,
                         colwise=True,
                         is_tp=is_tp,
                         all_reduce=all_reduce,
                         layer_type='attn')

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        self.layers = [q_proj, k_proj, v_proj]
        self.out_names = ['q', 'k', 'v']
        self.out_names_map = {'q': 0, 'k': 1, 'v': 2}

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.head_size_v = head_size_v

        self.in_features = q_proj.in_features if q_proj else 0
        self.out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features

        packed_in_features = self.in_features // 2
        merged_weight = torch.empty((self.out_features, packed_in_features),
                                    dtype=torch.uint8,
                                    device=self.device)

        n_groups = (self.in_features + q_proj.group_size - 1) // q_proj.group_size
        merged_weight_norms = torch.empty((self.out_features, n_groups),
                                          dtype=torch.float32,
                                          device=self.device)

        self.register_parameter('weight', torch.nn.Parameter(merged_weight, requires_grad=False))
        self.register_parameter('weight_norms', torch.nn.Parameter(merged_weight_norms, requires_grad=False))

        self.codebook = q_proj.codebook
        self.boundaries = q_proj.boundaries
        self.group_size = q_proj.group_size
        self.bit_width = q_proj.bit_width

        self.setup_loaders()

    def setup_loaders(self):
        """Setup weight loaders."""
        self.weight.weight_loader = self.weight_loader
        self.weight_norms.weight_loader = self.weight_loader

    def _dequantize_weight(self) -> torch.Tensor:
        out_features, in_features = self.out_features, self.in_features
        device = self.weight.device

        codebook = self.codebook
        indices = unpack_4bit(self.weight.data, in_features, self.group_size)

        w = torch.zeros(out_features, in_features, dtype=torch.float32, device=device)

        for g_start in range(0, in_features, self.group_size):
            g_end = min(g_start + self.group_size, in_features)
            g_dim = g_end - g_start

            y = codebook[indices[:, g_start:g_end].long()]

            sigma = 1.0 / math.sqrt(in_features)
            scale = sigma * math.sqrt(g_dim)
            y = y / scale

            w_g = hadamard_rotate_inv(y)
            g_idx = g_start // self.group_size
            w_g = w_g * self.weight_norms[:, g_idx].unsqueeze(1)
            w[:, g_start:g_end] = w_g

        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from lmdeploy.pytorch.kernels.cuda.turbo_quant_fused import (
                turboquant_fused_forward,
            )
            return turboquant_fused_forward(
                x=x,
                weight_packed=self.weight.data,
                codebook=self.codebook.data,
                weight_norms=self.weight_norms.data,
                in_features=self.in_features,
                out_features=self.out_features,
                group_size=self.group_size,
                bias=None,
            )
        w = self._dequantize_weight().to(x.dtype)
        return torch.matmul(x, w.t())

    def split_qkv(self, qkv_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split packed qkv tensor to the same contract as QKVBaseLinear."""
        q_out_dim = self.q_proj.out_features
        k_out_dim = self.k_proj.out_features
        v_out_dim = self.v_proj.out_features

        query_states = qkv_states[..., :q_out_dim]
        key_states = qkv_states[..., q_out_dim:q_out_dim + k_out_dim]
        value_states = qkv_states[..., q_out_dim + k_out_dim:q_out_dim + k_out_dim + v_out_dim]

        query_states = query_states.unflatten(-1, (self.num_q_heads, self.head_size))
        key_states = key_states.unflatten(-1, (self.num_kv_heads, self.head_size))
        value_states = value_states.unflatten(-1, (self.num_kv_heads, self.head_size_v))

        return query_states, key_states, value_states

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = None):
        if shard_id is None:
            raise ValueError('shard_id is required for TurboQuantQKVLinear')

        if isinstance(shard_id, int):
            if shard_id < len(self.out_names):
                shard_idx = shard_id
            else:
                raise ValueError(f'shard_id {shard_id} out of range, max is {len(self.out_names) - 1}')
        else:
            shard_idx = self.out_names_map.get(shard_id)
            if shard_idx is None:
                raise ValueError(f'Unknown shard_id: {shard_id}, available: {list(self.out_names_map.keys())}')

        layer = self.layers[shard_idx]
        start_idx = sum(l.out_features for l in self.layers[:shard_idx])
        end_idx = start_idx + layer.out_features

        if loaded_weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            loaded_weight = loaded_weight.to(self.weight.device)
            quant_result = layer._quantize_weight(loaded_weight)
            indices_packed = quant_result['indices_packed']
            weight_norms = quant_result['norms']

            # Always update both tensors together for float weights
            self.weight.data[start_idx:end_idx].copy_(indices_packed)
            self.weight_norms.data[start_idx:end_idx].copy_(weight_norms)
        else:
            if param is self.weight:
                self.weight.data[start_idx:end_idx].copy_(loaded_weight)
            elif param is self.weight_norms:
                self.weight_norms.data[start_idx:end_idx].copy_(loaded_weight)

    def update_weights(self):
        """Update weights."""
        self.register_all_parameters(self.weight, self.weight_norms)

    def register_all_parameters(self,
                                weight: torch.Tensor,
                                weight_norms: torch.Tensor):
        """Register all parameters."""
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight_norms = torch.nn.Parameter(weight_norms, requires_grad=False)
        self.register_parameter('weight', weight)
        self.register_parameter('weight_norms', weight_norms)
        self.setup_loaders()

    def __repr__(self):
        return f'TurboQuantQKVLinear(q_proj={self.q_proj}, k_proj={self.k_proj}, v_proj={self.v_proj})'


class MergedTurboQuantLinear(LinearBase):
    """Merged TurboQuant linear for gate+up projections."""

    def __init__(
        self,
        layers: list[TurboQuantLinear],
        out_names: list[str] = None,
        is_tp: bool = False,
        all_reduce: bool = False,
        layer_type: str = 'mlp',
    ):
        super().__init__(dtype=layers[0].dtype if layers else None,
                         device=layers[0].device if layers else None,
                         colwise=True,
                         is_tp=is_tp,
                         all_reduce=all_reduce,
                         layer_type=layer_type)

        self.layers = nn.ModuleList(layers)
        if out_names is None:
            out_names = [f'out_{i}' for i in range(len(layers))]
        self.out_names = out_names
        self.out_names_map = {name: idx for idx, name in enumerate(out_names)}

        self.in_features = layers[0].in_features if layers else 0
        self.out_features = sum(layer.out_features for layer in layers)
        total_out_features = self.out_features

        packed_in_features = self.in_features // 2
        merged_weight = torch.empty((total_out_features, packed_in_features),
                                    dtype=torch.uint8,
                                    device=self.device)

        n_groups = (self.in_features + layers[0].group_size - 1) // layers[0].group_size
        merged_weight_norms = torch.empty((total_out_features, n_groups),
                                          dtype=torch.float32,
                                          device=self.device)

        self.register_parameter('weight', torch.nn.Parameter(merged_weight, requires_grad=False))
        self.register_parameter('weight_norms', torch.nn.Parameter(merged_weight_norms, requires_grad=False))

        self.group_size = layers[0].group_size
        self.bit_width = layers[0].bit_width

        # Register codebook and boundaries as buffers (non-learnable constants)
        codebook = layers[0].codebook
        boundaries = layers[0].boundaries
        if codebook is not None:
            self.register_buffer('_codebook', codebook)
        if boundaries is not None:
            self.register_buffer('_boundaries', boundaries)

        self.setup_loaders()

    def setup_loaders(self):
        """Setup weight loaders."""
        self.weight.weight_loader = self.weight_loader
        self.weight_norms.weight_loader = self.weight_loader

    @property
    def codebook(self) -> torch.Tensor | None:
        return getattr(self, '_codebook', None)

    @property
    def boundaries(self) -> torch.Tensor | None:
        return getattr(self, '_boundaries', None)

    def _dequantize_weight(self) -> torch.Tensor:
        out_features, in_features = self.out_features, self.in_features
        device = self.weight.device
        codebook = self.codebook
        indices = unpack_4bit(self.weight.data, in_features, self.group_size)
        w = torch.zeros(out_features, in_features, dtype=torch.float32, device=device)

        for g_start in range(0, in_features, self.group_size):
            g_end = min(g_start + self.group_size, in_features)
            g_dim = g_end - g_start

            y = codebook[indices[:, g_start:g_end].long()]

            sigma = 1.0 / math.sqrt(in_features)
            scale = sigma * math.sqrt(g_dim)
            y = y / scale

            w_g = hadamard_rotate_inv(y)
            g_idx = g_start // self.group_size
            w_g = w_g * self.weight_norms[:, g_idx].unsqueeze(1)
            w[:, g_start:g_end] = w_g
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from lmdeploy.pytorch.kernels.cuda.turbo_quant_fused import (
                turboquant_fused_forward,
            )
            return turboquant_fused_forward(
                x=x,
                weight_packed=self.weight.data,
                codebook=self.codebook.data,
                weight_norms=self.weight_norms.data,
                in_features=self.in_features,
                out_features=self.out_features,
                group_size=self.group_size,
                bias=None,
            )
        w = self._dequantize_weight().to(x.dtype)
        return torch.matmul(x, w.t())

    def _weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        if not self.is_tp:
            return default_weight_loader(param, loaded_weight)

        world_size, rank = self.get_tp_world_rank()
        if self.colwise:
            return self._weight_loader_tp_colwise(param, loaded_weight, rank, world_size)
        else:
            return self._weight_loader_tp_rowwise(param, loaded_weight, rank, world_size)

    def _weight_loader_tp_colwise(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        rank: int,
        world_size: int,
    ):
        weight = loaded_weight.chunk(world_size, 0)[rank]
        return default_weight_loader(param, weight)

    def _weight_loader_tp_rowwise(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        rank: int,
        world_size: int,
    ):
        if loaded_weight.dim() == 2:
            weight = loaded_weight.chunk(world_size, 1)[rank]
        else:
            weight = loaded_weight
        return default_weight_loader(param, weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shard_id: Any = None):
        if shard_id is None:
            raise ValueError('shard_id is required for MergedTurboQuantLinear')

        if isinstance(shard_id, int):
            if shard_id < len(self.out_names):
                shard_idx = shard_id
            else:
                raise ValueError(f'shard_id {shard_id} out of range, max is {len(self.out_names) - 1}')
        else:
            shard_idx = self.out_names_map.get(shard_id)
            if shard_idx is None:
                raise ValueError(f'Unknown shard_id: {shard_id}, available: {list(self.out_names_map.keys())}')

        layer = self.layers[shard_idx]
        start_idx = sum(l.out_features for l in self.layers[:shard_idx])
        end_idx = start_idx + layer.out_features

        if loaded_weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
            loaded_weight = loaded_weight.to(self.weight.device)
            quant_result = layer._quantize_weight(loaded_weight)
            indices_packed = quant_result['indices_packed']
            weight_norms = quant_result['norms']

            # Always update both tensors together for float weights
            self.weight.data[start_idx:end_idx].copy_(indices_packed)
            self.weight_norms.data[start_idx:end_idx].copy_(weight_norms)
        else:
            if param is self.weight:
                self.weight.data[start_idx:end_idx].copy_(loaded_weight)
            elif param is self.weight_norms:
                self.weight_norms.data[start_idx:end_idx].copy_(loaded_weight)

    def update_weights(self):
        """Update weights."""
        self.register_all_parameters(self.weight, self.weight_norms)

    def register_all_parameters(self,
                                weight: torch.Tensor,
                                weight_norms: torch.Tensor):
        """Register all parameters."""
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight_norms = torch.nn.Parameter(weight_norms, requires_grad=False)
        self.register_parameter('weight', weight)
        self.register_parameter('weight_norms', weight_norms)
        self.setup_loaders()

    def __repr__(self):
        return f'MergedTurboQuantLinear(layers={self.layers})'
