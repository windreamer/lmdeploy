# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in TurboMind int8-KV block layout: opt-in decision, pool geometry,
and the triton-free KV write adapter.

Pool geometry (one physical kernel block, int8 KV, per-token-per-head
asymmetric uint8 quantization; contract validated end to end by
tests/pytorch/kernel/test_tm_int8_kv.py and by the B3-04 bridge probes):

    [codes  (Hkv, 2, S, D) u8 ]  K at slot 0, V at slot 1, per head
    [params (Hkv, 2, S, 2) T  ]  (scale, min) per token per head

with T = model dtype (bf16/fp16) and dequant ``code * scale + min`` — the
`block::Layout` arrangement that `invokeProcessKV_v2` writes and the TM
decode kernel reads. The PyTorch convention stores (scale, zero) with
``(code - zero) * scale``; the two agree on identical codes once ``min =
-zero * scale``, so torch-side readers only need the second slot rewritten.

Block-cache tensor views keep the standard ``(S, H, ...)`` token-major
contract via strided views over the head-major TM storage, so stock PyTorch
readers (paged attention, flatten) keep working unchanged. Everything else
in the engine (block copy, swap, migration) moves whole pool entries and is
layout agnostic.

Opt-in gating lives in three pieces that must agree:

- ``turbomind_int8_kv_reason``: geometry/availability probe (pure function).
- ``record_tm_int8_kv_capability``: attention-build side records whether every
  paged-attention layer can serve the TM decode path. Model builds run before
  cache plans, so the cache plan can consume the verdict.
- ``tm_int8_kv_layout_requested``: quant policy INT8 + knob + probe + recorded
  capability. Records accumulate with AND semantics so one incapable layer
  disables the layout for the whole model; wrong-enable is impossible.
"""

import os
from dataclasses import dataclass

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.utils import get_logger

from ..engine.cache_engine.layout import (
    CacheAllocation,
    CachePool,
    CompositeBlockCacheLayout,
)

logger = get_logger('lmdeploy')

# Head dims dispatchable by the TM attention kernels (see
# `invokeProcessKV_v2`); 576 is the share-KV MLA dim and stays excluded.
TM_INT8_KV_HEAD_DIMS = (64, 128, 192, 256)
# The decode kernel iterates 64-token CTAs; the pool geometry follows.
TM_INT8_KV_BLOCK_LEN = 64

_TM_INT8_KV_CLASS = 'attn_int8_kv'
_TM_STATES = ('auto', 'on', 'off')


def _parse_knob() -> str:
    """Resolve the opt-in state: LMDEPLOY_TM_KERNELS class entry first, then
    the dedicated LMDEPLOY_TM_ATTN_INT8_KV, else auto."""
    big = os.getenv('LMDEPLOY_TM_KERNELS', '')
    for item in big.replace(';', ',').split(','):
        item = item.strip()
        if '=' not in item:
            continue
        name, _, state = item.partition('=')
        if name.strip() == _TM_INT8_KV_CLASS:
            state = state.strip().lower()
            if state not in _TM_STATES:
                logger.warning('Ignoring invalid LMDEPLOY_TM_KERNELS entry %s=%s', name, state)
                break
            return state
    return os.getenv('LMDEPLOY_TM_ATTN_INT8_KV', 'auto').lower()


def tm_int8_kv_knob() -> str:
    """Return the opt-in state: auto / on / off."""
    return _parse_knob()


def turbomind_int8_kv_reason(model_config, cache_config) -> str | None:
    """Return why the TM int8-KV route cannot serve this model, or None.

    Pure geometry/availability probe shared by the attention build and the cache plan so both sides always reach the
    same verdict.
    """
    if cache_config.quant_policy != QuantPolicy.INT8:
        return f'quant_policy must be INT8, but got {cache_config.quant_policy}'
    if cache_config.kernel_block_size != TM_INT8_KV_BLOCK_LEN:
        return (f'kernel_block_size must be {TM_INT8_KV_BLOCK_LEN}, '
                f'but got {cache_config.kernel_block_size}')
    head_dim = model_config.k_head_dim or model_config.head_dim
    v_head_dim = model_config.v_head_dim or model_config.head_dim
    if head_dim != v_head_dim:
        return f'asymmetric kv head dims are unsupported: k={head_dim} v={v_head_dim}'
    return _tm_int8_kv_geometry_reason(head_dim, model_config.num_key_value_heads)


def turbomind_int8_kv_attention_reason(spec, sliding_window) -> str | None:
    """Capability probe at attention-build time.

    Quant-policy agnostic on purpose: the cache plan combines this verdict
    with the INT8 requirement. MLA layers do not consume the standard KV pool
    and stay neutral (the caller skips recording for them).
    """
    if getattr(spec, 'use_flash_mla', False):
        return 'MLA layers stay on FlashMLA'
    if getattr(spec, 'alibi', False):
        return 'alibi bias is unsupported by the TM decode kernel'
    if getattr(spec, 'learnable_sink', False):
        return 'learnable sink is unsupported by the TM decode kernel'
    softcap = getattr(spec, 'logit_softcapping', 0.0) or 0.0
    if softcap > 0:
        return 'logit softcapping is unsupported by the TM decode kernel'
    if tuple(sliding_window) != (-1, -1):
        return 'sliding window is unsupported by the TM decode kernel'
    if not getattr(spec, 'causal', True):
        return 'non-causal decode is unsupported by the TM decode kernel'
    if getattr(spec, 'block_sparse_size', 1) != 1:
        return 'block sparse attention is unsupported by the TM decode kernel'
    return _tm_int8_kv_geometry_reason(spec.head_dim, spec.num_kv_heads,
                                       v_head_dim=spec.v_head_dim,
                                       num_heads=spec.num_heads)


def _tm_int8_kv_geometry_reason(head_dim: int, num_kv_heads: int, v_head_dim: int | None = None,
                                num_heads: int | None = None) -> str | None:
    """Shared native/geometry checks for both probes."""
    if v_head_dim is not None and v_head_dim != head_dim:
        return f'asymmetric kv head dims are unsupported: k={head_dim} v={v_head_dim}'
    if head_dim not in TM_INT8_KV_HEAD_DIMS:
        return f'head_dim must be one of {TM_INT8_KV_HEAD_DIMS}, but got {head_dim}'
    if num_kv_heads is not None and num_kv_heads < 1:
        return f'num_key_value_heads must be positive, but got {num_kv_heads}'
    if num_heads is not None and num_heads % num_kv_heads != 0:
        return f'num_heads {num_heads} is not divisible by num_kv_heads {num_kv_heads}'
    if not torch.cuda.is_available():
        return 'CUDA is unavailable'
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return f'CUDA compute capability {major}.x is below the required 8.0'
    try:
        from lmdeploy.turbomind import is_available
        if not is_available():
            return 'TurboMind native modules are not built'
    except Exception as e:  # pragma: no cover - import guard
        return f'TurboMind native modules are unavailable: {e}'
    return None


# AND-accumulating verdict recorded by paged-attention builds (model build)
# and consumed by cache plans (which always run after the model build).
_TM_INT8_KV_CAPABLE: bool | None = None


def record_tm_int8_kv_capability(capable: bool) -> None:
    """Accumulate one attention layer's capability with AND semantics."""
    global _TM_INT8_KV_CAPABLE
    if _TM_INT8_KV_CAPABLE is None:
        _TM_INT8_KV_CAPABLE = capable
    else:
        _TM_INT8_KV_CAPABLE = _TM_INT8_KV_CAPABLE and capable


def reset_tm_int8_kv_capability() -> None:
    """Clear accumulated verdicts (test isolation only)."""
    global _TM_INT8_KV_CAPABLE
    _TM_INT8_KV_CAPABLE = None


def tm_int8_kv_capability() -> bool | None:
    """Return the accumulated attention-side verdict, None if never built."""
    return _TM_INT8_KV_CAPABLE


def tm_int8_kv_layout_requested(model_config, cache_config) -> bool:
    """Whether the KV pool must use the TM int8 block layout.

    Requires the knob not off, the geometry probe passing, INT8 KV policy, and at least one capable paged-attention
    build without any incapable one.
    """
    if tm_int8_kv_knob() == 'off':
        return False
    if _TM_INT8_KV_CAPABLE is not True:
        return False
    return turbomind_int8_kv_reason(model_config, cache_config) is None


@dataclass(frozen=True)
class TmInt8BlockCacheLayout:
    """Allocate one packed uint8 block pool per layer in TM int8 geometry.

    `tensor_specs` is the standard KV prefix [k_cache, v_cache, quant_0,
    quant_1]; views are returned in spec order with the standard (S, H, ...)
    shapes, backed by strided views of the head-major TM storage.
    """

    tensor_specs: tuple
    num_layers: int

    def __post_init__(self):
        names = [spec.name for spec in self.tensor_specs]
        if names[:2] != ['k_cache', 'v_cache'] or len(names) != 4:
            raise ValueError(f'Expected [k_cache, v_cache, quant_0, quant_1], got {names}')
        k_shape = self.tensor_specs[0].desc.shape
        v_shape = self.tensor_specs[1].desc.shape
        q_shapes = [spec.desc.shape for spec in self.tensor_specs[2:]]
        if k_shape != v_shape or len(k_shape) != 3:
            raise ValueError(f'kv block shapes must match and be 3D, got {k_shape} vs {v_shape}')
        for qs in q_shapes:
            if len(qs) != 3 or qs[:-1] != k_shape[:-1] or qs[-1] != 2:
                raise ValueError(f'quant block shape {qs} does not match kv shape {k_shape}')

    @property
    def block_len(self) -> int:
        """Tokens per physical kernel block."""
        return self.tensor_specs[0].desc.shape[0]

    @property
    def num_kv_heads(self) -> int:
        """KV heads per rank in one block."""
        return self.tensor_specs[0].desc.shape[1]

    @property
    def head_dim(self) -> int:
        """Head dim of one block."""
        return self.tensor_specs[0].desc.shape[2]

    @property
    def param_dtype(self) -> torch.dtype:
        """Storage dtype of the per-token (scale, min) slots."""
        return self.tensor_specs[2].desc.dtype

    @property
    def block_nbytes(self) -> int:
        """Bytes of one physical block."""
        h, s, d = self.num_kv_heads, self.block_len, self.head_dim
        return 2 * h * s * d + 2 * h * s * 2 * self.param_dtype.itemsize

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the packed pool plus standard-contract strided views."""
        num_layers = self.num_layers
        h, s, d = self.num_kv_heads, self.block_len, self.head_dim
        data_bytes = 2 * h * s * d
        param_bytes = 2 * h * s * 2 * self.param_dtype.itemsize
        pool = torch.zeros((num_layers, num_blocks, self.block_nbytes), dtype=torch.uint8, device=device)

        # block::Layout arrangement: per head a [K | V] code pair followed by
        # a [K | V] param pair region across all heads.
        codes = pool[:, :, :data_bytes].view(num_layers, num_blocks, h, 2, s, d)
        params = pool[:, :, data_bytes:].view(self.param_dtype).view(num_layers, num_blocks, h, 2, s, 2)

        # Standard contract keeps tokens before heads; permute is a view.
        tensor_views = (
            codes[:, :, :, 0].permute(0, 1, 3, 2, 4),
            codes[:, :, :, 1].permute(0, 1, 3, 2, 4),
            params[:, :, :, 0].permute(0, 1, 3, 2, 4),
            params[:, :, :, 1].permute(0, 1, 3, 2, 4),
        )
        return CacheAllocation(pools=(CachePool(pool, entry_axis=1), ), tensor_views=tensor_views)


def build_tm_int8_block_layout(tensor_specs, num_layers: int, default_builder):
    """Select the TM int8 layout for the KV prefix, default layout for the
    rest, preserving spec order when extra block-cache tensors exist."""
    tensor_specs = tuple(tensor_specs)
    prefix = tensor_specs[:4]
    rest = tensor_specs[4:]
    tm_layout = TmInt8BlockCacheLayout(prefix, num_layers=num_layers)
    if not rest:
        return tm_layout
    return CompositeBlockCacheLayout((tm_layout, default_builder(rest, num_layers=num_layers)))


def quant_kv_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize (T, H, D) tokens per token per head: uint8 codes plus
    (scale, min) in fp32, contract ``code = clamp(round((x - min)/scale))``.
    """
    xf = x.float()
    lo = xf.amin(-1)
    scale = ((xf.amax(-1) - lo) / 255.0).clamp(min=1e-6)
    codes = torch.round((xf - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, 255).to(torch.uint8)
    return codes, scale, lo


def fill_kv_cache_tm(key: torch.Tensor,
                     value: torch.Tensor,
                     k_cache: torch.Tensor,
                     v_cache: torch.Tensor,
                     k_sz: torch.Tensor,
                     v_sz: torch.Tensor,
                     q_start_loc: torch.Tensor,
                     q_seqlens: torch.Tensor,
                     kv_seqlens: torch.Tensor,
                     block_offsets: torch.Tensor) -> None:
    """Append key/value tokens into the TM int8 block pool without triton.

    Quantizes per token per head with the host contract above and scatters
    codes plus (scale, min) params through the standard-contract strided
    views, replicating `fill_kv_cache` placement: batch b's tokens land at
    positions [kv_seqlens[b] - q_seqlens[b], kv_seqlens[b]) inside the blocks
    listed by `block_offsets[b]`.

    Args:
        key: New tokens, (sum_q, Hkv, D) token-major.
        value: New tokens, (sum_q, Hkv, D).
        k_cache: K codes view (num_blocks, S, Hkv, D).
        v_cache: V codes view (num_blocks, S, Hkv, D).
        k_sz: K param view (num_blocks, S, Hkv, 2).
        v_sz: V param view (num_blocks, S, Hkv, 2).
        q_start_loc: Packed token offsets, (B,).
        q_seqlens: New token counts, (B,).
        kv_seqlens: Total lengths after the append, (B,).
        block_offsets: Physical block ids per sequence, (B, max_blocks).
    """
    block_len = k_cache.size(1)
    device = key.device
    q_start_loc = q_start_loc.to(device=device, dtype=torch.long)
    q_seqlens = q_seqlens.to(device=device, dtype=torch.long)
    kv_seqlens = kv_seqlens.to(device=device, dtype=torch.long)

    num_tokens = key.shape[0]
    token_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
    # Fixed-shape batch resolution; no host sync (output shape is known).
    batch_ids = torch.searchsorted(torch.cumsum(q_seqlens, 0), token_ids, right=True)
    batch_ids = batch_ids.clamp(max=q_seqlens.shape[0] - 1)
    in_q_ids = token_ids - q_start_loc[batch_ids]
    # Positions may be negative or beyond the appended block table when a
    # row is invalid; clamp for safe indexing and mask below.
    positions = (kv_seqlens - q_seqlens)[batch_ids] + in_q_ids
    max_position = block_offsets.shape[1] * block_len
    safe_positions = positions.clamp(min=0, max=max_position - 1)
    valid = (in_q_ids < q_seqlens[batch_ids]) & (positions >= 0) & (positions < max_position)
    block_ids = block_offsets[batch_ids, safe_positions.div(block_len, rounding_mode='floor')]
    token_in_block = safe_positions % block_len

    # Capture-safe masked scatter: advanced indexing with plain (dense)
    # index tensors, then `where` on the gathered old values. Invalid rows
    # rewrite their (clamped) slot with its own current content, so the
    # pool is untouched for them and no boolean filtering (host sync,
    # `cudaErrorStreamCaptureInvalidated` under graph capture) happens.
    keep = valid.view(-1, 1, 1)
    for x, cache, params in ((key, k_cache, k_sz), (value, v_cache, v_sz)):
        codes, scale, lo = quant_kv_int8(x)
        old_codes = cache[block_ids, token_in_block]
        cache[block_ids, token_in_block] = torch.where(keep, codes, old_codes)
        old_params = params[block_ids, token_in_block]
        param_pair = torch.stack((scale, lo), dim=-1).to(params.dtype)
        params[block_ids, token_in_block] = torch.where(keep, param_pair, old_params)


def tm_min_to_zero(sz: torch.Tensor) -> torch.Tensor:
    """Convert (scale, min) param slots to the PT (scale, zero) convention.

    Args:
        sz: (..., S, H, 2) param view with slot 1 carrying min.

    Returns:
        Contiguous (..., S, H, 2) tensor with slot 1 carrying zero. Degenerate
        constant rows (scale 0) keep zero at 0 so readers stay finite.
    """
    scale = sz[..., 0]
    mn = sz[..., 1]
    zero = torch.where(scale > 0, -mn / scale.clamp(min=1e-6), torch.zeros_like(mn))
    return torch.stack((scale, zero), dim=-1).contiguous()


def tm_block_pointer_table(block_offsets: torch.Tensor, pool_base: int,
                           block_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the TM binding's sequence-major block pointer table.

    The bindings index blocks as `base + local` with `base =
    cu_block_base[batch]`; a row-major table over `block_offsets` (B, C) with
    bases `arange(B + 1) * C` matches that convention. Padded columns yield
    unused pointer values, which the kernels never dereference.

    Args:
        block_offsets: Physical block ids per sequence, (B, C) int device.
        pool_base: Device address of this layer's pool (bytes).
        block_bytes: Bytes of one physical block.

    Returns:
        (pointer_table int64 (B*C,), cu_block_base int32 (B+1)) on the same
        device.
    """
    offsets = block_offsets.to(torch.int64).reshape(-1)
    ptrs = offsets * block_bytes + pool_base
    num_seqs, num_cols = block_offsets.shape
    bases = torch.arange(num_seqs + 1, dtype=torch.int32, device=offsets.device) * num_cols
    return ptrs, bases
