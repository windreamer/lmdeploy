# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind GenericCopy-based logical-block copy primitive.

Opt-in only: selected when ``LMDEPLOY_TURBOMIND_OPS`` is ``force`` or
``force=copy_cache`` (no factory-table row ships; ``auto`` keeps the
Triton path byte-identical).

dtype coverage conclusion (recorded per ticket 06):

The PyTorch engine resolves standard KV/state pool storage dtypes in
``engine/cache_engine/schema.py::_resolve_kv_cache_dtype`` to one of
``torch.float16``/``torch.bfloat16`` (model dtype), ``torch.uint8``
(INT4/INT8/TURBO_QUANT policies), ``torch.float8_e4m3fn`` (FP8 policy
and MLA fp8 cache), or ``torch.float8_e5m2`` (FP8_E5M2 policy).

TurboMind ``GenericCopy`` dispatches purely on element byte size
(1/2/4/8 all instantiated) and receives tensors through DLPack, so
dtype reachability is decided by the DLPack mapping in
``src/turbomind/python/bind.cpp::getDataType``: fp16, bf16, fp32, fp64,
int/uint 8..64 bit, bool, and ``kDLFloat8_e4m3fn`` are mapped; there is
**no** ``kDLFloat8_e5m2`` case. Hence:

- fp16 / bf16 / fp8_e4m3fn / uint8 pools -> TM GenericCopy is safe.
- ``torch.float8_e5m2`` pools -> the DLPack conversion raises
  "unsupported DLPack dtype"; that variant must stay on the Triton
  path. Under force the gate below fails so the build errors out
  (spec user story 7) instead of silently degrading.
"""
import torch

from ..cache import BlockCacheCopy

# dtypes whose DLPack -> TurboMind DataType mapping does not exist.
_TM_COPY_UNSUPPORTED_DTYPES = frozenset({torch.float8_e5m2})


def turbomind_copy_gate_reason(pools) -> str | None:
    """Return a reason string when any pool dtype is not reachable by
    TurboMind GenericCopy, else None."""
    for pool in pools:
        dtype = pool.tensor.dtype
        if dtype in _TM_COPY_UNSUPPORTED_DTYPES:
            return (f'dtype {dtype} has no TurboMind DLPack mapping '
                    '(only float8_e4m3fn is bound); this variant stays on triton')
    return None


class TurbomindBlockCacheCopy(BlockCacheCopy):
    """TurboMind GenericCopy logical-block copy across allocation pools.

    One ``generic_copy_on_stream`` launch per (pool, block pair); the
    Triton primitive stays one launch per pool, so this path trades
    launch count for the TM kernel and is strictly opt-in.
    """

    def __init__(self, allocation, num_logical_blocks: int, pages_per_block: int):
        super().__init__(allocation, num_logical_blocks, pages_per_block)
        if any(not pool.tensor.is_contiguous() for pool in self.pools):
            raise ValueError('CUDA block-cache copy requires contiguous allocation pools.')

    def copy(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        """Copy every scheduler-sized block pair of every owning pool on
        the current stream."""
        from lmdeploy.turbomind import _tm

        stream = torch.cuda.current_stream(self.device).cuda_stream
        for pool in self.pools:
            blocks = pool.tensor.unflatten(pool.entry_axis,
                                           (self.num_logical_blocks, self.pages_per_block))
            for src, dst in zip(src_block_offsets.tolist(), dst_block_offsets.tolist()):
                _tm.generic_copy_on_stream(_tm.from_dlpack(blocks.select(pool.entry_axis, src)),
                                           _tm.from_dlpack(blocks.select(pool.entry_axis, dst)), stream)
