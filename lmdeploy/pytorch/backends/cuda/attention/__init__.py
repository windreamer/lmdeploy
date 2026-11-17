# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
from lmdeploy.pytorch.backends.tm_kv import (
    record_tm_int8_kv_capability,
    tm_int8_kv_knob,
    turbomind_int8_kv_attention_reason,
)
from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl
from .default import TritonAttentionMetadata as TritonAttentionMetadata

logger = get_logger('lmdeploy')

use_fa3 = False
try:
    # flash-attention supports FA3 for sm80+ (Ampere and above) && cuda >= 12.3
    _cuda_ver = tuple(int(x) for x in torch.version.cuda.split('.')[:2]) if torch.version.cuda else (0, 0)
    if (torch.cuda.get_device_capability()[0] >= 8) and _cuda_ver >= (12, 3):
        import lmdeploy.pytorch.third_party.flash_attn_interface  # noqa: F401
        assert torch.ops.flash_attn_3 is not None
        use_fa3 = True
except Exception:
    logger.debug('For higher performance, please install FlashAttention-3 '
                 'https://github.com/Dao-AILab/flash-attention')


def require_fa3_for_speculative_decoding() -> None:
    """Require the FA3 backend used by multi-token speculative decode."""
    if use_fa3:
        return

    sm = torch.cuda.get_device_capability()
    cuda_ver = torch.version.cuda or 'N/A'
    raise RuntimeError(
        f'Speculative decoding on CUDA requires FlashAttention-3 (FA3), '
        f'which needs SM80+ (Ampere and above) with CUDA >= 12.3 and '
        f'flash-attn installed. Detected: SM{sm[0]}.{sm[1]}, CUDA {cuda_ver}. '
        f'Please ensure your GPU meets SM80+, CUDA >= 12.3, and flash-attn '
        f'is installed, or disable speculative decoding.')


@functools.lru_cache
def use_fa3_warning():
    if use_fa3:
        return True
    logger.warning('For higher performance, please install FlashAttention-3 '
                   'https://github.com/Dao-AILab/flash-attention')
    return False


@functools.lru_cache
def _enable_fa3(alibi: bool, learnable_sink: bool, block_sparse_size: int, head_size: int) -> bool:
    """Check if FA3 should be enabled.

    FA3 is enabled when:
    - No alibi
    - No learnable sink
    - block_sparse_size == 1
    - FA3 is available (checked by use_fa3_warning)

    Returns:
        True if FA3 should be enabled, False otherwise.
    """
    enable = not alibi and not learnable_sink and block_sparse_size == 1 and head_size <= 256
    if enable and not use_fa3_warning():
        enable = False
    return enable


def _build_tm_int8_kv_attention(spec: PagedAttentionBuildSpec, common_args: dict):
    """Build the TM int8-KV paged attention, or None to keep the stock impl.

    The verdict is recorded for the cache plan (model builds run before cache
    plans). `on` turns probe failures into a hard error; `auto` falls back.
    MLA layers never consume the standard KV pool, so they stay neutral
    (no verdict recorded) and cannot disable the layout for sibling layers.
    """
    if spec.use_flash_mla:
        return None

    knob = tm_int8_kv_knob()
    if knob == 'off':
        return None

    reason = turbomind_int8_kv_attention_reason(spec, sliding_window=_normalize_sliding_window(spec.sliding_window))
    if reason is not None:
        if knob == 'on':
            raise RuntimeError(f'TM int8-KV attention is forced on but unavailable: {reason}')
        record_tm_int8_kv_capability(False)
        logger.debug(f'TM int8-KV attention unavailable, falling back: {reason}')
        return None

    record_tm_int8_kv_capability(True)
    logger.debug('Build TmInt8KVAttentionImpl Attention')
    from .int8_kv import TmInt8KVAttentionImpl
    return TmInt8KVAttentionImpl(block_sparse_size=spec.block_sparse_size, **common_args)


def _normalize_sliding_window(sliding_window):
    """Normalize sliding window to tuple format.

    Args:
        sliding_window: None, int, or tuple of (left, right).

    Returns:
        Tuple of (left, right) or (-1, -1) if None.
    """
    if sliding_window is None:
        return (-1, -1)
    if isinstance(sliding_window, int):
        return (sliding_window, sliding_window)
    return sliding_window


def _build_paged_attention(spec: PagedAttentionBuildSpec) -> TritonAttentionImpl:
    """Build the selected CUDA paged-attention implementation.

    Selection order:
    1. tm_int8_kv: TM decode + TM int8 pool when the opt-in probe passes
    2. use_flash_mla: Use dense or sparse FlashMLA for MLA models
    3. enable_fa3: Use FA3Impl if FA3 is available and supported
    4. Default: Use TritonAttentionImpl as fallback
    """
    sliding_window = _normalize_sliding_window(spec.sliding_window)
    common_args = dict(
        num_heads=spec.num_heads,
        head_size=spec.head_dim,
        scale=spec.scale,
        num_kv_heads=spec.num_kv_heads,
        v_head_size=spec.v_head_dim,
        alibi=spec.alibi,
        sliding_window=sliding_window,
        logit_softcapping=spec.logit_softcapping,
        causal=spec.causal,
    )

    tm_impl = _build_tm_int8_kv_attention(spec, common_args)
    if tm_impl is not None:
        return tm_impl

    enable_fa3 = _enable_fa3(spec.alibi, spec.learnable_sink, spec.block_sparse_size, spec.head_dim)

    if spec.use_flash_mla is True:
        if spec.mla_index_topk is not None:
            if _envs.sparse_mla_backend == 'tilelang':
                logger.debug('Build TileLangSparseMLAImpl Attention')
                from .sparse_mla import TileLangSparseMLAImpl
                return TileLangSparseMLAImpl(
                    mla_index_topk=spec.mla_index_topk,
                    use_fa3=use_fa3,
                    **common_args,
                )
            logger.debug('Build FlashMLASparseImpl Attention')
            from .sparse_mla import FlashMLASparseImpl
            return FlashMLASparseImpl(
                mla_index_topk=spec.mla_index_topk,
                use_fa3=use_fa3,
                **common_args,
            )
        logger.debug('Build FlashMLAImpl Attention')
        from .mla import FlashMLAImpl
        return FlashMLAImpl(use_fa3=use_fa3, **common_args)
    elif enable_fa3:
        logger.debug('Build FA3Impl Attention')
        from .fa3 import FA3Impl
        return FA3Impl(**common_args)
    else:
        logger.debug('Build TritonAttentionImpl Attention')
        return TritonAttentionImpl(block_sparse_size=spec.block_sparse_size, **common_args)
