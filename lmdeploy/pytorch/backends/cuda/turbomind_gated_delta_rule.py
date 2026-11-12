# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind delta-rule implementation of the gated-delta-rule seam.

Selected only through the ``LMDEPLOY_TURBOMIND_OPS`` single switch
(``force`` / ``force=gated_delta``); the factory table ships no
``gated_delta`` row, so the default remains the FLA/TileLang stack
(:class:`CudaGatedDeltaRuleImpl`).

The TurboMind native stack exposes ``delta_rule_plan`` / ``delta_rule_run``
(imported from the TM linear_attn kernels) and expects

- q/k/v: bf16, layout (batch, tokens, heads, 128);
- g/beta: float32 gates with arbitrary batch/token strides;
- state: one contiguous (heads, 128, 128) block per sequence, addressed
  through a device tensor of raw pointers, state written back in place
  unless the matching ``finished`` entry is set;
- ``q_offsets``: int32 cu_seqlens over token slots (chunked mode).

The engine stores recurrent states as (slots, hv, V, K) bf16 rows addressed
by slot id, so this implementation materializes per-sequence K-major state
blocks around each ``delta_rule_run`` call and scatters them back.

Not supported by this path (feasibility gate rejects at build time):
indexed state caches; spec-decode ring state offsets and circular buffers;
non-bf16 inputs.
"""
from functools import lru_cache

import torch

from ..gated_delta_rule import GatedDeltaRuleBuildSpec, GatedDeltaRuleImpl
from .gated_delta_rule import GatedDeltaStepMetaUpdater, register_step_metadata_impl

TM_DELTA_RULE_SYMBOLS = (
    'delta_rule_plan',
    'delta_rule_run',
    'from_dlpack',
    'delta_rule_prepare_state_tma_descs',
)

_REQUIRED_BRIDGE_MESSAGE = (
    'TurboMind native delta-rule bridge is unavailable; build the '
    '`lmdeploy.turbomind._turbomind` extension (needs delta_rule_plan, '
    'delta_rule_run, delta_rule_prepare_state_tma_descs, from_dlpack)')


@lru_cache
def _tm_native_bridge_available() -> bool:
    try:
        from lmdeploy import turbomind
    except Exception:
        return False
    if not turbomind.is_available() or turbomind._tm is None:
        return False
    return all(hasattr(turbomind._tm, sym) for sym in TM_DELTA_RULE_SYMBOLS)


class TmGatedDeltaStepMetaUpdater(GatedDeltaStepMetaUpdater):
    """Step metadata for the TM rule: it plans from cu_seqlens itself, so no
    FLA chunk-index preparation is needed for prefill."""

    def update(self, step_context, sequence_metadata) -> None:
        if self.indexed_state:
            if not step_context.is_decoding and step_context.ssm_prefill_meta is None:
                raise ValueError('Compact prefill requires prebuilt host-length metadata.')
            return


class TmGatedDeltaRuleImpl(GatedDeltaRuleImpl):
    """Gated delta rule backed by the TurboMind native delta_rule stack."""

    def __init__(self, indexed_state: bool = False) -> None:
        self.indexed_state = indexed_state
        self._tm = None
        register_step_metadata_impl(self)

    # -- bridge helpers ----------------------------------------------------

    def _bridge(self):
        if self._tm is None:
            if not _tm_native_bridge_available():
                raise RuntimeError(_REQUIRED_BRIDGE_MESSAGE)
            from lmdeploy import turbomind
            self._tm = turbomind._tm
        return self._tm

    @staticmethod
    def _tensor(tm, x: torch.Tensor | None):
        return None if x is None else tm.from_dlpack(x)

    @staticmethod
    def _stream_ptr(tensor: torch.Tensor) -> int:
        return int(torch.cuda.current_stream(tensor.device).cuda_stream)

    @staticmethod
    def _empty_from_plan(plan: dict, *, device, dtype) -> torch.Tensor:
        shape = tuple(plan['shape'])
        stride = tuple(plan.get('stride', ()))
        storage_size = plan.get('storage_size')
        if stride and storage_size is not None and int(storage_size) > 1 + sum(
                (dim - 1) * step for dim, step in zip(shape, stride)):
            storage = torch.empty(int(storage_size), device=device, dtype=dtype)
            return torch.as_strided(storage, shape, stride)
        if stride:
            return torch.empty_strided(shape, stride, device=device, dtype=dtype)
        return torch.empty(shape, device=device, dtype=dtype)

    @staticmethod
    def _row_pointers(block: torch.Tensor) -> torch.Tensor:
        """Device int64 pointers to each per-sequence state block."""
        row_bytes = block.stride(0) * block.element_size()
        return (torch.arange(block.shape[0], device=block.device, dtype=torch.int64) *
                row_bytes + block.data_ptr())

    @staticmethod
    def _align_gates(g: torch.Tensor, beta: torch.Tensor):
        """Cast the float gates to float32 (engine beta arrives in the model
        dtype) and pad the head dim to a 4-element-aligned stride when the
        engine layout is not aligned; otherwise pass through."""
        if g.dtype != torch.float32:
            g = g.float()
        if beta.dtype != torch.float32:
            beta = beta.float()
        if (g.stride(2) == 1 and beta.stride(2) == 1 and g.stride(1) % 4 == 0
                and beta.stride(1) % 4 == 0 and g.stride(0) % 4 == 0 and beta.stride(0) % 4 == 0):
            return g, beta
        batch, tokens = g.shape[0], g.shape[1]

        def pad(x):
            hv = x.shape[2]
            gate_stride = max(4, hv)
            if gate_stride % 4:
                gate_stride += 4 - gate_stride % 4
            padded = x.new_zeros((batch, tokens, gate_stride))
            padded[..., :hv].copy_(x)
            return padded

        return pad(g), pad(beta)

    def _gather_state_blocks(self, recurrent_state: torch.Tensor, state_ids: torch.Tensor):
        """Materialize TM-expected (B, hv, K, V) state blocks plus validity.

        Returns (blocks, write_rows, write_ids): ``write_rows`` are the
        positions within ``blocks`` whose sequences may write state back and
        ``write_ids`` are the corresponding slot ids in the pool (slot 0 is
        reserved read-only, mirroring the TileLang recurrent kernel
        semantics).
        """
        num_slots = recurrent_state.shape[0]
        ids = state_ids.to(torch.int64)
        valid = (ids >= 0) & (ids < num_slots)
        read_ids = ids.clamp(0)
        blocks = recurrent_state.index_select(0, read_ids).transpose(-2, -1).contiguous()
        blocks[~valid] = 0
        writable = valid & (ids > 0)
        write_rows = writable.nonzero(as_tuple=True)[0]
        write_ids = ids[writable]
        return blocks, write_rows, write_ids

    def _scatter_state_blocks(self, recurrent_state: torch.Tensor, blocks: torch.Tensor,
                              write_rows: torch.Tensor, write_ids: torch.Tensor) -> None:
        if write_rows.numel() == 0:
            return
        recurrent_state.index_copy_(0, write_ids, blocks[write_rows].transpose(-2, -1))

    def _plan_and_run(self,
                      q,
                      k,
                      v,
                      g,
                      beta,
                      blocks,
                      finished,
                      *,
                      mode,
                      q_offsets=None):
        tm = self._bridge()
        g, beta = self._align_gates(g.contiguous(), beta.contiguous())
        tensors_q = tm.from_dlpack(q)
        tensors_k = tm.from_dlpack(k)
        tensors_v = tm.from_dlpack(v)
        tensors_g = tm.from_dlpack(g)
        tensors_beta = tm.from_dlpack(beta)
        state_ptrs = self._row_pointers(blocks)
        execution_state_ptrs = state_ptrs[:, None] if mode == 'recurrent' else state_ptrs
        plan = tm.delta_rule_plan(
            tensors_q,
            tensors_k,
            tensors_v,
            tensors_g,
            tensors_beta,
            q_offsets=self._tensor(tm, q_offsets),
            state_dtype='bf16',
            mode=mode,
            chunk_size=None,
            cp_level='off',
            num_head_groups=1,
            heads_per_block=v.shape[2],
        )
        state_tma_descs = None
        if mode == 'recurrent' and plan['kernel']['arch'] != 'pre_sm90':
            prepared = torch.empty(
                (1, execution_state_ptrs.shape[-2], plan['problem']['num_head_groups'], 128),
                device=q.device,
                dtype=torch.uint8,
            )
            tm.delta_rule_prepare_state_tma_descs(
                tm.from_dlpack(execution_state_ptrs),
                tm.from_dlpack(prepared),
                plan,
                layer_groups=1,
                layers_per_block=1,
                stream_ptr=self._stream_ptr(q),
            )
            state_tma_descs = prepared
        out = self._empty_from_plan(plan['out'], device=q.device, dtype=q.dtype)
        workspace = None
        if plan['workspace_bytes']:
            workspace = torch.empty(plan['workspace_bytes'], device=q.device, dtype=torch.uint8)
        tm.delta_rule_run(
            tensors_q,
            tensors_k,
            tensors_v,
            tensors_g,
            tensors_beta,
            plan,
            out=tm.from_dlpack(out),
            workspace=self._tensor(tm, workspace),
            stream_ptr=self._stream_ptr(q),
            state_ptrs=tm.from_dlpack(execution_state_ptrs),
            state_tma_descs=self._tensor(tm, state_tma_descs),
            q_offsets=self._tensor(tm, q_offsets),
            finished=tm.from_dlpack(finished),
            state_layer_offset=0,
        )
        # Keep native-side views alive until the (stream-ordered) copies back
        # below have been enqueued.
        out._delta_rule_refs = (workspace, execution_state_ptrs, state_tma_descs, q_offsets,
                                finished)
        return out

    # -- seam API ----------------------------------------------------------

    def get_step_metadata_provider(self):
        return TmGatedDeltaStepMetaUpdater(indexed_state=self.indexed_state)

    def chunk_gated_delta_rule(self,
                               q,
                               k,
                               v,
                               g=None,
                               beta=None,
                               initial_state=None,
                               state_indices=None,
                               scale=None,
                               use_qk_l2norm_in_kernel=False,
                               cu_seqlens=None,
                               output_final_state=False,
                               spec_state_offsets=None,
                               transpose_state_layout=False):
        assert initial_state is not None and state_indices is not None
        if spec_state_offsets is not None:
            raise NotImplementedError(
                'TurboMind gated delta rule does not support spec-decode ring state offsets')
        if not transpose_state_layout:
            raise NotImplementedError(
                'TurboMind gated delta rule requires the transposed state layout')
        if scale is not None and scale != q.shape[-1]**-0.5:
            raise NotImplementedError(
                f'TurboMind gated delta rule applies scale=head_dim**-0.5 internally, got {scale}')
        if cu_seqlens is None:
            raise ValueError('chunk_gated_delta_rule requires cu_seqlens')
        if use_qk_l2norm_in_kernel:
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        blocks, write_rows, write_ids = self._gather_state_blocks(initial_state, state_indices)
        sequence_num = state_indices.shape[0]
        finished = torch.zeros(sequence_num, device=q.device, dtype=torch.bool)
        out = self._plan_and_run(
            q,
            k,
            v,
            g,
            beta,
            blocks,
            finished,
            mode='chunked',
            q_offsets=cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous(),
        )
        # TM wrote final states into `blocks` (stream-ordered); scatter back.
        self._scatter_state_blocks(initial_state, blocks, write_rows, write_ids)
        return out, initial_state

    def fused_recurrent_gated_delta_rule(self,
                                         q,
                                         k,
                                         v,
                                         g=None,
                                         beta=None,
                                         initial_state=None,
                                         state_indices=None,
                                         scale=None,
                                         use_qk_l2norm_in_kernel=False,
                                         output_final_state=False,
                                         cache_seqlens=None,
                                         transpose_state_layout=False):
        assert initial_state is not None and state_indices is not None
        if cache_seqlens is not None:
            raise NotImplementedError(
                'TurboMind gated delta rule does not support circular-buffer decode '
                '(spec-decode state ring)')
        if not transpose_state_layout:
            raise NotImplementedError(
                'TurboMind gated delta rule requires the transposed state layout')
        if scale is not None and scale != q.shape[-1]**-0.5:
            raise NotImplementedError(
                f'TurboMind gated delta rule applies scale=head_dim**-0.5 internally, got {scale}')
        if use_qk_l2norm_in_kernel:
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        blocks, write_rows, write_ids = self._gather_state_blocks(initial_state, state_indices)
        # finished: no write-back for invalid ids and reserved slot 0.
        num_slots = initial_state.shape[0]
        ids = state_indices.to(torch.int64)
        finished = ~((ids > 0) & (ids < num_slots))
        out = self._plan_and_run(
            q,
            k,
            v,
            g,
            beta,
            blocks,
            finished,
            mode='recurrent',
        )
        self._scatter_state_blocks(initial_state, blocks, write_rows, write_ids)
        batch = q.shape[0]
        return out.reshape(batch, 1, v.shape[2], v.shape[3]), None


def _build_gated_delta_rule_tm_only(spec: GatedDeltaRuleBuildSpec) -> TmGatedDeltaRuleImpl:
    return TmGatedDeltaRuleImpl(indexed_state=spec.indexed_state)
