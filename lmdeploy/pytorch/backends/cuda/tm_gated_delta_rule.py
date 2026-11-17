# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind recurrent decode path for the CUDA gated delta rule op.

The TurboMind delta-rule kernel consumes per-sequence state slabs of shape
[heads, K, V]. It addresses them through a device pointer array. The
PyTorch indexed state pool stores transposed [heads, V, K] rows. The
bridge gathers the pool rows of the current batch into a compact
contiguous buffer and transposes the last two dims back to [K, V]. The
TurboMind recurrent kernel then advances that buffer in place, and the
bridge scatters the states back into the pool. Every step of the bridge
is a device-side kernel or a host-scalar pointer computation. No device
synchronization happens on the hot path.

Hard constraints mirror the TurboMind GDN layer. head_dim must be 128.
The state dtype must be float32 or equal to the input dtype. The
registered planner must accept the workload (architecture and dtype
dependent). Anything else declines the route, and the caller falls back
to the fla production kernels. Chunked prefill is never routed here.
Batch-2 measurements showed fla ahead on that regime, so the chunked
path stays untouched.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from .gated_delta_rule import CudaGatedDeltaRuleImpl

logger = get_logger('lmdeploy')


@triton.jit
def _gdn_state_gather_kernel(
    pool_ptr,
    out_ptr,
    ids_ptr,
    stride_pool_n,
    stride_pool_h,
    stride_out_n,
    stride_out_h,
    num_heads,
    NUM_STATES,
    HEAD_DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    """Gather compact [B, H, K, V] state rows from a [N, H, V, K] pool.

    Pool rows are [V, K]. Compact rows are [K, V]. Invalid state ids
    yield zero rows so padded requests read the immutable zero state.
    """
    pid = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    batch = pid // num_heads
    head = pid % num_heads

    state_id = tl.load(ids_ptr + batch)
    valid = (state_id >= 0) & (state_id < NUM_STATES)
    pool_base = pool_ptr + state_id * stride_pool_n + head * stride_pool_h
    out_base = out_ptr + batch * stride_out_n + head * stride_out_h

    tiles = HEAD_DIM // TILE
    tk = tile // tiles
    tv = tile % tiles
    v_offs = tv * TILE + tl.arange(0, TILE)[None, :]
    k_offs = tk * TILE + tl.arange(0, TILE)[:, None]
    src = pool_base + v_offs * HEAD_DIM + k_offs
    data = tl.load(src, mask=valid, other=0.0)
    tl.store(out_base + k_offs * HEAD_DIM + v_offs, data)


@triton.jit
def _gdn_state_scatter_kernel(
    pool_ptr,
    src_ptr,
    ids_ptr,
    stride_pool_n,
    stride_pool_h,
    stride_src_n,
    stride_src_h,
    num_heads,
    NUM_STATES,
    HEAD_DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    """Scatter compact [B, H, K, V] state rows back into a [N, H, V, K]
    pool.

    The write reverses the gather transpose. Invalid state ids are
    ignored. Padded requests therefore cannot overwrite a live cache row.
    """
    pid = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    batch = pid // num_heads
    head = pid % num_heads

    state_id = tl.load(ids_ptr + batch)
    valid = (state_id >= 0) & (state_id < NUM_STATES)
    pool_base = pool_ptr + state_id * stride_pool_n + head * stride_pool_h
    src_base = src_ptr + batch * stride_src_n + head * stride_src_h

    tiles = HEAD_DIM // TILE
    tk = tile // tiles
    tv = tile % tiles
    v_offs = tv * TILE + tl.arange(0, TILE)[None, :]
    k_offs = tk * TILE + tl.arange(0, TILE)[:, None]

    data = tl.load(src_base + k_offs * HEAD_DIM + v_offs, mask=valid, other=0.0)
    tl.store(pool_base + v_offs * HEAD_DIM + k_offs, data, mask=valid)


def _gdn_state_gather(pool: torch.Tensor, state_ids: torch.Tensor, compact: torch.Tensor) -> None:
    """compact[b] = transpose(pool[state_ids[b]], last two dims)."""
    num_heads, head_dim = pool.shape[1], pool.shape[-1]
    grid = (compact.shape[0] * num_heads, (head_dim // 32) * (head_dim // 32))
    _gdn_state_gather_kernel[grid](
        pool,
        compact,
        state_ids,
        pool.stride(0),
        pool.stride(1),
        compact.stride(0),
        compact.stride(1),
        num_heads,
        pool.shape[0],
        HEAD_DIM=head_dim,
        TILE=32,
    )


def _gdn_state_scatter(pool: torch.Tensor, state_ids: torch.Tensor, compact: torch.Tensor) -> None:
    """pool[state_ids[b]] += transpose(compact[b], last two dims) (assign)."""
    num_heads, head_dim = pool.shape[1], pool.shape[-1]
    grid = (compact.shape[0] * num_heads, (head_dim // 32) * (head_dim // 32))
    _gdn_state_scatter_kernel[grid](
        pool,
        compact,
        state_ids,
        pool.stride(0),
        pool.stride(1),
        compact.stride(0),
        compact.stride(1),
        num_heads,
        pool.shape[0],
        HEAD_DIM=head_dim,
        TILE=32,
    )


@lru_cache(maxsize=None)
def _tm_plan_decline(hv: int, input_dtype: str, state_dtype: str) -> str | None:
    """Probe whether the TurboMind planner accepts a recurrent workload.

    Returns None when a plan can be built, else the decline reason. The
    probe covers architecture-dependent availability (for example no path
    exists on sm100) through the registered planner itself.
    """
    try:
        from lmdeploy.turbomind import _tm
    except Exception as exc:  # pragma: no cover - depends on the build
        return f'turbomind python module unavailable: {exc}'
    if not torch.cuda.is_available():
        return 'CUDA is unavailable'
    try:
        device = torch.device('cuda')
        q = torch.zeros(2, 1, hv, 128, device=device, dtype=getattr(torch, input_dtype))
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        g = torch.zeros(2, 1, hv, device=device, dtype=torch.float32)
        beta = torch.zeros(2, 1, hv, device=device, dtype=torch.float32)
        stream = torch.cuda.current_stream(device).cuda_stream
        context = _tm.create_device_context(stream)
        with context:
            _tm.delta_rule_plan(
                _tm.from_dlpack(q),
                _tm.from_dlpack(k),
                _tm.from_dlpack(v),
                _tm.from_dlpack(g),
                _tm.from_dlpack(beta),
                state_dtype=state_dtype,
                mode='recurrent',
            )
    except Exception as exc:
        return f'TurboMind planner declined the workload: {exc}'
    return None


def tm_gdn_decline_reason(head_dim: int, hv: int, input_dtype: torch.dtype,
                          state_dtype: torch.dtype) -> str | None:
    """Return why the TurboMind recurrent decode cannot serve this shape.

    Hard constraints follow the TurboMind GDN layer: head_dim must be 128,
    the state dtype must be float32 or equal to the input dtype, and the
    registered planner must accept the dtype pair on this architecture. The
    k/v head divisibility constraint (heads divisible by attn_tp x cp) is
    enforced upstream when heads are sharded; the kernel only needs
    hv >= 1 here.
    """
    if head_dim != 128:
        return f'head_dim={head_dim}, TurboMind GDN requires 128'
    if hv < 1:
        return f'hv={hv}, no value heads'
    if input_dtype not in (torch.bfloat16, torch.float16):
        return f'input dtype {input_dtype} is not f16/bf16'
    if state_dtype != torch.float32 and state_dtype != input_dtype:
        return f'state dtype {state_dtype} is neither f32 nor the input dtype'
    return _tm_plan_decline(hv, str(input_dtype).removeprefix('torch.'), str(state_dtype).removeprefix('torch.'))


@lru_cache(maxsize=1)
def build_level_decline_reason() -> str | None:
    """Architecture-level probe for the op build step.

    Runs one generic plan probe (8 heads, bf16 input, f32 state).
    Per-layer geometry is rechecked on the first decode step.
    """
    return tm_gdn_decline_reason(128, 8, torch.bfloat16, torch.float32)


class TmGatedDeltaRuleImpl(CudaGatedDeltaRuleImpl):
    """CUDA gated delta rule with the TurboMind recurrent decode kernel.

    Prefill (chunked) steps and non-indexed pools keep the fla production
    path unchanged; only indexed decode steps route to TurboMind.
    """

    def __init__(self, indexed_state: bool = False):
        super().__init__(indexed_state=indexed_state)
        if not indexed_state:
            raise ValueError('TurboMind GDN decode requires the indexed state pool.')
        self._tm_plans = {}
        self._tm_buffers = {}
        self._tm_contexts = {}

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log_exp: torch.Tensor,
        recurrent_state: torch.Tensor,
        gated_delta_meta,
        kv_ratio: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> torch.Tensor:
        meta = gated_delta_meta
        if not self.indexed_state or not meta.is_decoding:
            return super().forward(query, key, value, b, a, dt_bias, a_log_exp, recurrent_state, meta, kv_ratio,
                                   use_qk_l2norm_in_kernel)
        ids = meta.ssm_state_meta
        batch, width = ids.output_state_ids.shape
        head_dim, hv = query.shape[-1], query.shape[-2]
        reason = tm_gdn_decline_reason(head_dim, hv, query.dtype, recurrent_state.dtype)
        if reason is None and width != 1:
            reason = (f'width={width} candidates share one input state; TurboMind advances slabs in place '
                      'and cannot serve MTP decode')
        if reason is not None:
            if _envs.gdn_backend == 'on':
                raise RuntimeError(f'TurboMind GDN decode was forced (LMDEPLOY_GDN_BACKEND=on) but declined: {reason}.')
            return super().forward(query, key, value, b, a, dt_bias, a_log_exp, recurrent_state, meta, kv_ratio,
                                   use_qk_l2norm_in_kernel)
        return self._tm_decode(query, key, value, b, a, dt_bias, a_log_exp, recurrent_state, ids, batch, kv_ratio,
                               use_qk_l2norm_in_kernel)

    def _tm_decode(self, query, key, value, b, a, dt_bias, a_log_exp, recurrent_state, ids, batch, kv_ratio,
                   use_qk_l2norm_in_kernel):
        """One TurboMind recurrent step over the indexed state pool."""
        q, k, g, beta, normalized = self.prepare_inputs(
            query,
            key,
            b,
            a,
            dt_bias,
            a_log_exp,
            kv_ratio,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_decoding=True,
            init_token_mask=None,
        )
        head_dim, hv = q.shape[-1], q.shape[-2]
        # The TurboMind kernel reads g and beta as float32; the production
        # preprocess keeps beta in the input dtype.
        g = g.float()
        beta = beta.float()
        q, k, v, g, beta = [x[0].unflatten(0, (batch, 1)).contiguous() for x in (q, key, value, g, beta)]
        if use_qk_l2norm_in_kernel and not normalized:
            # The TurboMind kernel has no fused q/k l2 normalization.
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        pool = recurrent_state
        compact = pool.new_empty((batch, hv, head_dim, head_dim))
        _gdn_state_gather(pool, ids.input_state_ids.reshape(-1), compact)
        out = self._run_tm_recurrent(q, k, v, g, beta, compact)
        _gdn_state_scatter(pool, ids.output_state_ids.reshape(-1), compact)
        return out.flatten(0, 1).unsqueeze(0)

    def _tm_context(self, stream_ptr: int):
        """Cache one native context guard per Torch stream (linear.py
        precedent)."""
        context = self._tm_contexts.get(stream_ptr)
        if context is None:
            from lmdeploy.turbomind import _tm
            context = _tm.create_device_context(stream_ptr)
            self._tm_contexts[stream_ptr] = context
        return context

    def _run_tm_recurrent(self, q, k, v, g, beta, state):
        """Run one TurboMind recurrent step with the state updated in place."""
        from lmdeploy.turbomind import _tm

        batch, hv, head_dim = state.shape[0], state.shape[1], state.shape[-1]
        plan_key = (batch, hv, head_dim, q.dtype, state.dtype)
        entry = self._tm_plans.get(plan_key)
        if entry is None:
            stream = torch.cuda.current_stream(q.device)
            context = self._tm_context(stream.cuda_stream)
            with context:
                plan = _tm.delta_rule_plan(
                    _tm.from_dlpack(q),
                    _tm.from_dlpack(k),
                    _tm.from_dlpack(v),
                    _tm.from_dlpack(g),
                    _tm.from_dlpack(beta),
                    state_dtype='f32' if state.dtype == torch.float32 else
                    ('bf16' if state.dtype == torch.bfloat16 else 'f16'),
                    mode='recurrent',
                )
                buffers = {
                    'finished': torch.zeros(batch, dtype=torch.bool, device=q.device),
                    'ptr_offsets': torch.arange(batch, dtype=torch.int64, device=q.device) *
                    (state.stride(0) * state.element_size()),
                }
                workspace_size = int(plan['workspace']['storage_size'])
                if workspace_size > 0:
                    buffers['workspace'] = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)
                self._tm_plans[plan_key] = (plan, buffers)
                entry = (plan, buffers)

        plan, buffers = entry
        out = torch.empty(tuple(plan['out']['shape']), dtype=q.dtype, device=q.device)
        state_ptrs = buffers['ptr_offsets'] + state.data_ptr()
        args = dict(
            q=_tm.from_dlpack(q),
            k=_tm.from_dlpack(k),
            v=_tm.from_dlpack(v),
            g=_tm.from_dlpack(g),
            beta=_tm.from_dlpack(beta),
            plan=plan,
            out=_tm.from_dlpack(out),
            stream_ptr=torch.cuda.current_stream(q.device).cuda_stream,
            state_ptrs=_tm.from_dlpack(state_ptrs),
            finished=_tm.from_dlpack(buffers['finished']),
        )
        workspace = buffers.get('workspace')
        if workspace is not None:
            args['workspace'] = _tm.from_dlpack(workspace)
        with self._tm_context(args['stream_ptr']):
            _tm.delta_rule_run(**args)
        return out
