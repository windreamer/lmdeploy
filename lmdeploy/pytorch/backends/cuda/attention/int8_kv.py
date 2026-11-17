# Copyright (c) OpenMMLab. All rights reserved.
"""Paged attention that pairs the TM int8 decode kernel with the TM block pool.

The pool follows the `block::Layout` int8 geometry (see
`lmdeploy/pytorch/backends/tm_kv.py`). New KV tokens are appended by the TM
quantizing write kernel (`_tm.tm_process_kv`, null rope: the model pre-rotates
K outside attention); decode steps dispatch the TM decode kernel
(`_tm.tm_attn_decode`). PyTorch paths remain in charge everywhere else:

- prefill / chunked prefill attention runs the stock Triton path on the
  layout's strided views, with (scale, min) slots converted to the PT
  (scale, zero) convention per step;
- decode steps dispatch the TM decode kernel (`_tm.tm_attn_decode`); under
  CUDA graph capture the graph-safe binding entry (`_tm.tm_attn_decode_graph`)
  runs instead: persistent caller-owned buffers (device cu_q_len/cu_k_len,
  one workspace) make replay follow per-step buffer content bit-exactly, so
  TM decode participates in full-graph and piecewise capture; only a missing
  native module falls back to the capture-safe Triton paged-attention path;
- non-INT8 runtime quant policies behave exactly like `TritonAttentionImpl`
  (the pool then uses the default layout, matched by the same opt-in
  decision that built this impl).
"""
import os

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl, TritonAttentionMetadata

logger = get_logger('lmdeploy')


def _is_tm_int8_policy(quant_policy: QuantPolicy) -> bool:
    """Whether the runtime KV policy selects the TM int8 pool semantics."""
    return quant_policy == QuantPolicy.INT8


class TmInt8KVAttentionImpl(TritonAttentionImpl):
    """Triton attention with TM int8 decode and TM write bridging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tm = None

    def _tm_module(self):
        """Import the native module lazily; None when unavailable."""
        if self._tm is None:
            try:
                from lmdeploy.turbomind import _tm
                self._tm = _tm
            except ImportError as e:
                logger.warning(f'TurboMind native module unavailable, TM int8-KV disabled: {e}')
                self._tm = False
        return self._tm or None

    def _pool_geometry(self, k_cache: torch.Tensor) -> tuple[int, int, int, int, int]:
        """(pool_base, block_bytes, block_len, num_kv_heads, head_dim) from one
        layer's K codes view (num_blocks, S, Hkv, D)."""
        block_len, num_kv_heads, head_dim = k_cache.shape[-3:]
        return k_cache.data_ptr(), k_cache.stride(0), block_len, num_kv_heads, head_dim

    def _pointer_tables(self, k_cache: torch.Tensor, attn_metadata: TritonAttentionMetadata):
        """Block pointer table plus per-sequence bases for the TM bindings."""
        from ...tm_kv import tm_block_pointer_table

        pool_base, block_bytes, *_ = self._pool_geometry(k_cache)
        return tm_block_pointer_table(attn_metadata.block_offsets, pool_base, block_bytes)

    def _fill_kv_cache_impl(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ):
        """Append new tokens: TM write kernel for the int8 pool, stock Triton
        fill otherwise."""
        if not _is_tm_int8_policy(attn_metadata.quant_policy):
            return super()._fill_kv_cache_impl(key, value, k_cache, v_cache, attn_metadata, max_q_seqlen)

        if k_scales_zeros is None or v_scales_zeros is None:
            raise RuntimeError('TM int8 pool fill requires the quant scale views.')
        from ...tm_kv import fill_kv_cache_tm

        # Default: the triton-free ATen convert+scatter writer. The TM write
        # kernel (`_tm.tm_process_kv`) stays available under
        # LMDEPLOY_TM_ATTN_INT8_BRIDGE=1; a q_len=1 illegal access in that
        # path is still under investigation, so correctness ships first.
        if os.getenv('LMDEPLOY_TM_ATTN_INT8_BRIDGE', '0') == '1' and self._tm_module() is not None:
            _, _, block_len, num_kv_heads, head_dim = self._pool_geometry(k_cache)
            num_seqs = attn_metadata.kv_seqlens.shape[0]
            cu_q = torch.nn.functional.pad(torch.cumsum(attn_metadata.q_seqlens, 0, dtype=torch.int32), (1, 0))
            cu_k = torch.nn.functional.pad(torch.cumsum(attn_metadata.kv_seqlens, 0, dtype=torch.int32), (1, 0))
            ptrs, cu_block_base = self._pointer_tables(k_cache, attn_metadata)
            stride_rows = key.stride(0) // head_dim
            stride_head = key.stride(1) // head_dim
            max_q = attn_metadata.max_q_seqlen or (key.shape[0] + num_seqs - 1) // num_seqs
            self._tm.tm_process_kv(
                self._wrap(key.contiguous()),
                self._wrap(value.contiguous()),
                ptrs.data_ptr(),
                cu_q.data_ptr(),
                cu_k.data_ptr(),
                cu_block_base.data_ptr(),
                stride_c=stride_rows,
                stride_s=stride_rows,
                stride_h=stride_head,
                max_q_len=max_q,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                batch_size=num_seqs,
                block_len=block_len,
                kv_quant=8,
                stream_ptr=torch.cuda.current_stream(key.device).cuda_stream,
            )
            return

        fill_kv_cache_tm(key,
                         value,
                         k_cache,
                         v_cache,
                         k_scales_zeros,
                         v_scales_zeros,
                         attn_metadata.q_start_loc,
                         attn_metadata.q_seqlens,
                         attn_metadata.kv_seqlens,
                         attn_metadata.block_offsets)

    def _wrap(self, tensor: torch.Tensor):
        """DLPack-wrap a torch tensor for the TurboMind bindings."""
        return self._tm.from_dlpack(tensor)

    def _graph_state(self, num_seqs: int, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device):
        """Persistent buffers for the graph-captured TM decode binding.

        The binding contract (`tm_attn_decode_graph`) requires every plan
        buffer to live at a stable address across capture and replay:

        - `cu_q_len`/`cu_k_len`: cumulative device lengths, rewritten by
          captured ops each step (device-side cumsum of the runner-filled
          `q_seqlens`/`kv_seqlens` buffers) — per-sequence lengths then
          reach the kernel inside the graph (binding defect ③ fixed);
        - `workspace`: one persistent byte buffer holding the binding's
          plan state (fused qkv staging, finished/readonly/split flags,
          split-K partials) at binding-defined offsets — capture records no
          allocation/free nodes (defect ② fixed).

        Buffers are keyed by capture shape so different graphs (batch
        sizes) own disjoint persistent state.
        """
        key = (num_seqs, num_heads, num_kv_heads, head_dim, device)
        state = getattr(self, '_tm_graph_state', None)
        if state is not None and state['key'] == key:
            return state
        ws_bytes = self._tm.tm_decode_workspace_bytes(num_seqs, num_heads, num_kv_heads, head_dim,
                                                      query.element_size())
        state = {
            'key': key,
            'cu_q_len': torch.zeros(num_seqs + 1, dtype=torch.int32, device=device),
            'cu_k_len': torch.zeros(num_seqs + 1, dtype=torch.int32, device=device),
            'workspace': torch.zeros(ws_bytes, dtype=torch.uint8, device=device),
        }
        self._tm_graph_state = state
        return state

    def _forward_tm_decode_graph(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        """One decode step through the graph-safe TM decode binding.

        Runs identically for eager steps and graph capture/replay: every per-step input is content of a persistent
        buffer (capture-pool tensors during capture), rewritten by captured device ops, so replay follows new
        q/k/v/seqlen content bit-exactly.
        """
        tm = self._tm_module()
        pool_base, block_bytes, block_len, num_kv_heads, head_dim = self._pool_geometry(k_cache)
        num_seqs = attn_metadata.kv_seqlens.shape[0]
        num_heads = query.shape[-2]
        gs = self._graph_state(num_seqs, num_heads, num_kv_heads, head_dim, query.device)

        q = query.reshape(num_seqs, -1).contiguous()
        out = torch.empty((num_seqs, num_heads * head_dim), dtype=query.dtype, device=query.device)
        ptrs, cu_block_base = self._pointer_tables(k_cache, attn_metadata)

        # Device-side cumulative lengths from the (runner-filled) metadata
        # buffers; captured as ops, re-executed on every replay.
        gs['cu_q_len'][1:].copy_(torch.cumsum(attn_metadata.q_seqlens.to(torch.int32), 0))
        gs['cu_q_len'][0].zero_()
        gs['cu_k_len'][1:].copy_(torch.cumsum(attn_metadata.kv_seqlens.to(torch.int32), 0))
        gs['cu_k_len'][0].zero_()

        # Capacity bound frozen at capture: grid/split schedule derives from
        # it, while true per-sequence lengths come from `cu_k_len` on device
        # (masking and store placement follow the real lengths, so padded
        # capacity never pollutes softmax or the pool tail).
        max_k_len = attn_metadata.block_offsets.shape[1] * block_len

        # The binding re-stores the newest-token K/V rows into the pool, so
        # it needs the [B, Hkv*D] rows of this step's new token per sequence;
        # forward stashes them in _tm_last_new_kv before the pool fill.
        key, value = getattr(self, '_tm_last_new_kv', (None, None))
        if key is None or value is None:
            raise RuntimeError('TM int8 decode: no new-token K/V captured for this step')
        rows = attn_metadata.q_start_loc.to(torch.long)
        k_rows = key.index_select(0, rows).reshape(num_seqs, num_kv_heads * head_dim).contiguous()
        v_rows = value.index_select(0, rows).reshape(num_seqs, num_kv_heads * head_dim).contiguous()
        tm.tm_attn_decode_graph(
            self._wrap(q),
            self._wrap(k_rows),
            self._wrap(v_rows),
            self._wrap(out),
            ptrs.data_ptr(),
            cu_block_base.data_ptr(),
            gs['cu_q_len'].data_ptr(),
            gs['cu_k_len'].data_ptr(),
            gs['workspace'].data_ptr(),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            batch_size=num_seqs,
            block_len=block_len,
            max_k_len=max_k_len,
            kv_quant=8,
            stream_ptr=torch.cuda.current_stream(query.device).cuda_stream,
        )
        return out.view(num_seqs, num_heads, head_dim)

    def _forward_tm_decode(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        """One decode step through the TM decode kernel."""
        tm = self._tm_module()
        _, _, block_len, num_kv_heads, head_dim = self._pool_geometry(k_cache)
        num_seqs = attn_metadata.kv_seqlens.shape[0]
        num_heads = query.shape[-2]
        q = query.reshape(num_seqs, -1).contiguous()
        out = torch.empty((num_seqs, num_heads * head_dim), dtype=query.dtype, device=query.device)
        ptrs, cu_block_base = self._pointer_tables(k_cache, attn_metadata)
        # No D2H sync here: a `.max()` fallback on kv_seqlens would invalidate
        # graph capture; fall back to the Triton path instead.
        max_k_len = attn_metadata.max_kv_seqlen
        if torch.cuda.is_current_stream_capturing():
            return self._forward_tm_decode_graph(query, k_cache, v_cache, attn_metadata)
        if max_k_len is None:
            return self._forward_triton_decode(query, k_cache, v_cache, attn_metadata)
        # The binding re-stores the newest-token K/V rows into the pool, so it
        # needs the [B, Hkv*D] rows of this step's new token per sequence;
        # forward stashes them in _tm_last_new_kv before the pool fill.
        key, value = getattr(self, '_tm_last_new_kv', (None, None))
        if key is None or value is None:
            raise RuntimeError('TM int8 decode: no new-token K/V captured for this step')
        rows = attn_metadata.q_start_loc.to(torch.long)
        k_rows = key.index_select(0, rows).reshape(num_seqs, num_kv_heads * head_dim).contiguous()
        v_rows = value.index_select(0, rows).reshape(num_seqs, num_kv_heads * head_dim).contiguous()
        tm.tm_attn_decode(
            self._wrap(q),
            self._wrap(k_rows),
            self._wrap(v_rows),
            self._wrap(out),
            ptrs.data_ptr(),
            cu_block_base.data_ptr(),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            batch_size=num_seqs,
            block_len=block_len,
            max_k_len=max_k_len,
            kv_quant=8,
            stream_ptr=torch.cuda.current_stream(query.device).cuda_stream,
        )
        return out.view(num_seqs, num_heads, head_dim)

    def _forward_triton_decode(self, query, k_cache, v_cache, attn_metadata):
        """Capture-safe Triton decode fallback over the TM pool views."""
        return TritonAttentionImpl._forward_decoding(
            self, query, k_cache, v_cache, attn_metadata, None,
            k_scales_zeros=self._pt_scales(getattr(self, '_tm_k_scales_zeros', None)),
            v_scales_zeros=self._pt_scales(getattr(self, '_tm_v_scales_zeros', None)))

    def _can_tm_decode(self, attn_metadata: TritonAttentionMetadata) -> bool:
        """TM decode covers single-token decode steps, eager and captured.

        During graph capture the graph-safe binding entry
        (`tm_attn_decode_graph`) is used: all plan state lives in persistent
        caller-owned buffers (device cu_q_len/cu_k_len, one workspace) and
        per-step content reaches replay via captured device-side fills, so
        capture no longer needs the Triton fallback. Capture falls back to
        Triton only when the native module is unavailable.
        """
        if attn_metadata.max_q_seqlen is not None and attn_metadata.max_q_seqlen != 1:
            return False
        if not torch.cuda.is_current_stream_capturing():
            return True
        return self._tm_module() is not None

    def _pt_scales(self, scales: torch.Tensor) -> torch.Tensor | None:
        """Rewrite (scale, min) slots to (scale, zero) for Triton readers."""
        if scales is None:
            return None
        from ...tm_kv import tm_min_to_zero

        return tm_min_to_zero(scales)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Fill the pool, then dispatch: TM decode for int8 single-token steps,
        stock Triton paths everywhere else."""
        if _is_tm_int8_policy(attn_metadata.quant_policy):
            if key is not None and value is not None:
                # The TM decode binding needs this step's newest-token rows;
                # stash them before the fill (works identically under graph
                # capture: index_select on the captured input buffers is
                # replayed).
                self._tm_last_new_kv = (key, value)
                self._tm_k_scales_zeros = k_scales_zeros
                self._tm_v_scales_zeros = v_scales_zeros
                self._fill_kv_cache_impl(key, value, k_cache, v_cache, attn_metadata,
                                         max_q_seqlen=self._get_max_q_seqlen(query, attn_metadata),
                                         k_scales_zeros=k_scales_zeros,
                                         v_scales_zeros=v_scales_zeros)
            if attn_metadata.is_decoding and self._can_tm_decode(attn_metadata) \
                    and self._tm_module() is not None:
                return self._forward_tm_decode(query, k_cache, v_cache, attn_metadata)
            # Torch readers need the PT (scale, zero) convention.
            k_scales_zeros = self._pt_scales(k_scales_zeros)
            v_scales_zeros = self._pt_scales(v_scales_zeros)

        return super().forward(query,
                               None if _is_tm_int8_policy(attn_metadata.quant_policy) else key,
                               None if _is_tm_int8_policy(attn_metadata.quant_policy) else value,
                               k_cache,
                               v_cache,
                               attn_metadata,
                               k_scales_zeros=k_scales_zeros,
                               v_scales_zeros=v_scales_zeros,
                               learnable_sink=learnable_sink,
                               inplace=inplace,
                               **kwargs)
