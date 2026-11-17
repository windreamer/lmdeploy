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

The KV write defaults to the TM fused write kernel (`_tm.tm_process_kv`
via `tm_kv.tm_process_kv_fill`): quantize plus block write in one launch,
for both prefill appends and one-token decode appends. The ATen
convert+scatter writer (`tm_kv.fill_kv_cache_tm`) stays as the fallback
for non-qualifying shapes (non-bf16/fp16 rows) or a missing native module.
"""

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.utils import get_logger

from ...tm_kv import tm_int8_kv_knob
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
        from ...tm_kv import fill_kv_cache_tm, tm_process_kv_fill

        # Default: the TM fused write kernel — quantize and block write in one
        # launch (measured 2.8-5.9x over the ATen writer on decode appends,
        # 2.1-4.1x on prefill; benchmark/bench_tm_write.py). The historical
        # q_len=1 illegal access was a harness defect (int64 cumsum promotion
        # feeding a `const int*` binding), not a kernel defect; `int8_kv` and
        # `tm_kv` build cumulative lengths int32 end to end. The ATen
        # convert+scatter writer stays as the fallback for non-qualifying
        # shapes or a missing native module.
        if self._can_tm_write(key):
            max_q = attn_metadata.max_q_seqlen
            if not max_q:
                # Grid bound without a host sync: total new tokens cover any
                # ragged per-sequence count; the kernel masks against cu_q_len.
                max_q = max(key.shape[0], 1)
            tm_process_kv_fill(key,
                               value,
                               k_cache,
                               v_cache,
                               attn_metadata.q_seqlens,
                               attn_metadata.kv_seqlens,
                               attn_metadata.block_offsets,
                               max_q_len=max_q)
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

    def _can_tm_write(self, key: torch.Tensor) -> bool:
        """Whether this append qualifies for the TM fused write kernel.

        The knob stays the single opt-in (`LMDEPLOY_TM_ATTN_INT8_KV` on/auto:
        write and read share the TM path); the kernel binds bf16/fp16 only.
        """
        if tm_int8_kv_knob() == 'off':
            return False
        if self._tm_module() is None:
            return False
        return key.dtype in (torch.bfloat16, torch.float16)

    def _wrap(self, tensor: torch.Tensor):
        """DLPack-wrap a torch tensor for the TurboMind bindings."""
        return self._tm.from_dlpack(tensor)

    def _graph_state(self, num_seqs: int, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device,
                     elem_size: int):
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
        sizes) own disjoint persistent state; an existing entry is never
        replaced (a captured graph would keep reading buffers we freed).
        `reset_tm_graph_state` releases every entry when the graph
        lifecycle ends.
        """
        key = (num_seqs, num_heads, num_kv_heads, head_dim, device)
        states = getattr(self, '_tm_graph_states', None)
        if states is None:
            states = {}
            self._tm_graph_states = states
        state = states.get(key)
        if state is not None:
            return state
        ws_bytes = self._tm.tm_decode_workspace_bytes(num_seqs, num_heads, num_kv_heads, head_dim, elem_size)
        state = {
            'cu_q_len': torch.zeros(num_seqs + 1, dtype=torch.int32, device=device),
            'cu_k_len': torch.zeros(num_seqs + 1, dtype=torch.int32, device=device),
            'workspace': torch.zeros(ws_bytes, dtype=torch.uint8, device=device),
        }
        states[key] = state
        return state

    def reset_tm_graph_state(self):
        """Release all persistent graph-capture buffers (paired with the
        graph lifecycle: call when every captured graph of this impl has
        been dropped)."""
        self._tm_graph_states = {}

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
        gs = self._graph_state(num_seqs, num_heads, num_kv_heads, head_dim, query.device, query.element_size())

        q = query.reshape(num_seqs, -1).contiguous()
        out = torch.empty((num_seqs, num_heads * head_dim), dtype=query.dtype, device=query.device)
        ptrs, cu_block_base = self._pointer_tables(k_cache, attn_metadata)

        # Device-side cumulative lengths from the (runner-filled) metadata
        # buffers; captured as ops, re-executed on every replay.
        gs['cu_q_len'][1:].copy_(torch.cumsum(attn_metadata.q_seqlens.to(torch.int32), 0, dtype=torch.int32))
        gs['cu_q_len'][0].zero_()
        gs['cu_k_len'][1:].copy_(torch.cumsum(attn_metadata.kv_seqlens.to(torch.int32), 0, dtype=torch.int32))
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
        # Real per-sequence cumulative lengths on device: without them the
        # binding bakes constant lengths from max_k_len (padded capacity),
        # pushing padded keys into softmax and re-storing rows at
        # ti = max_k_len - 1 beyond the true last slot (pool-tail pollution).
        cu_q_len = torch.zeros(num_seqs + 1, dtype=torch.int32, device=query.device)
        cu_k_len = torch.zeros(num_seqs + 1, dtype=torch.int32, device=query.device)
        cu_q_len[1:].copy_(torch.cumsum(attn_metadata.q_seqlens.to(torch.int32), 0, dtype=torch.int32))
        cu_k_len[1:].copy_(torch.cumsum(attn_metadata.kv_seqlens.to(torch.int32), 0, dtype=torch.int32))
        tm.tm_attn_decode(
            self._wrap(q),
            self._wrap(k_rows),
            self._wrap(v_rows),
            self._wrap(out),
            ptrs.data_ptr(),
            cu_block_base.data_ptr(),
            cu_q_len.data_ptr(),
            cu_k_len.data_ptr(),
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
