# Copyright (c) OpenMMLab. All rights reserved.
"""TurboMind recurrent decode path for the CUDA gated delta rule op.

The TurboMind recurrent kernel addresses per-sequence state slabs through a
device pointer array, and the transposed ``kVK`` variant reads the PyTorch
pool layout directly: a pool row ``[heads, V, K]`` is exactly the per-head
``[V, K]`` slab the variant expects. Decode therefore runs zero copy. A
per-step pointer array points at the destination pool rows; when a decoded
destination differs from its source row (checkpoint rebinding plans a
private slot, fresh history reads the zero unit) a small relocation kernel
copies or zeroes that row once, in pool layout, with no transpose. The
states never leave the pool, so checkpointing, PD transfer and eviction
keep the standard storage untouched.

Hard constraints mirror the TurboMind GDN layer. head_dim must be 128.
The state dtype must be float32 or equal to the input dtype. The
registered planner must accept the workload (architecture and dtype
dependent). Anything else declines the route, and the caller falls back
to the fla production kernels. Chunked prefill is never routed here:
batch-2 measurements showed fla ahead on that regime.
"""

from functools import cache, lru_cache

import torch
import triton
import triton.language as tl

from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import get_logger

from .gated_delta_rule import CudaGatedDeltaRuleImpl

logger = get_logger('lmdeploy')


@triton.jit
def _gdn_pack_kernel(
    q_src_ptr,
    k_src_ptr,
    v_src_ptr,
    g_src_ptr,
    beta_src_ptr,
    q_dst_ptr,
    k_dst_ptr,
    v_dst_ptr,
    g_dst_ptr,
    beta_dst_ptr,
    pool_ptr,
    in_ids_ptr,
    out_ids_ptr,
    ptrs_ptr,
    finished_ptr,
    num_states,
    pool_base,
    pool_row_elems,
    pool_row_bytes,
    head_dim,
    hq,
    kv_ratio,
    NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fill the persistent decode step buffers and prepare the state rows in
    one launch.

    q and k rows are l2 normalized when the production preprocess deferred that to the fla kernel; v, g and beta
    transfer as-is. The same program aligns its head's state slab with the decode source in pool layout: checkpoint
    rebinding copies once, fresh history zero-fills, the common case (destination equals source) touches nothing.
    Programs at head zero fill the byte-address pointer array the recurrent kernel consumes; invalid destinations clamp
    to slot zero under a finished flag that keeps the store off.
    """
    seq = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    flat_v = seq * (hq * kv_ratio) + head
    qk_head = head // kv_ratio
    flat_qk = seq * hq + qk_head
    qk_mask = (qk_head < hq) & mask
    if NORMALIZE:
        q = tl.load(q_src_ptr + flat_qk * head_dim + offs, mask=qk_mask, other=0.0).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-12)
        tl.store(q_dst_ptr + flat_qk * head_dim + offs, q.to(q_dst_ptr.dtype.element_ty), mask=qk_mask)
        k = tl.load(k_src_ptr + flat_qk * head_dim + offs, mask=qk_mask, other=0.0).to(tl.float32)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-12)
        tl.store(k_dst_ptr + flat_qk * head_dim + offs, k.to(k_dst_ptr.dtype.element_ty), mask=qk_mask)
    else:
        q = tl.load(q_src_ptr + flat_qk * head_dim + offs, mask=qk_mask, other=0.0)
        tl.store(q_dst_ptr + flat_qk * head_dim + offs, q, mask=qk_mask)
        k = tl.load(k_src_ptr + flat_qk * head_dim + offs, mask=qk_mask, other=0.0)
        tl.store(k_dst_ptr + flat_qk * head_dim + offs, k, mask=qk_mask)
    v = tl.load(v_src_ptr + flat_v * head_dim + offs, mask=mask, other=0.0)
    tl.store(v_dst_ptr + flat_v * head_dim + offs, v, mask=mask)
    g_value = tl.load(g_src_ptr + flat_v)
    tl.store(g_dst_ptr + flat_v, g_value.to(g_dst_ptr.dtype.element_ty))
    beta_value = tl.load(beta_src_ptr + flat_v)
    tl.store(beta_dst_ptr + flat_v, beta_value.to(beta_dst_ptr.dtype.element_ty))

    # state slab relocation, one head slab per program
    src = tl.load(in_ids_ptr + seq)
    dst = tl.load(out_ids_ptr + seq)
    src_valid = (src >= 0) & (src < num_states)
    dst_valid = (dst >= 0) & (dst < num_states)
    do_copy = dst_valid & src_valid & (src != dst)
    do_zero = dst_valid & (src_valid == 0)
    slab_elems = pool_row_elems // (hq * kv_ratio)
    slab_base = head.to(tl.int64) * slab_elems
    for i in range(0, tl.cdiv(slab_elems, BLOCK)):
        offset = slab_base + i * BLOCK + offs
        in_range = (i * BLOCK + offs) < slab_elems
        data = tl.load(pool_ptr + src * pool_row_elems + offset, mask=do_copy & in_range, other=0.0)
        tl.store(pool_ptr + dst * pool_row_elems + offset, data, mask=(do_copy | do_zero) & in_range)
    if head == 0:
        store_ptr = tl.where(dst_valid, pool_base + dst * pool_row_bytes, pool_base)
        tl.store(ptrs_ptr + seq, store_ptr)
        tl.store(finished_ptr + seq, (dst_valid == 0).to(tl.int8))


def _gdn_pack(entry, q, k, v, g, beta, recurrent_state, input_ids, output_ids, batch, normalize: bool) -> None:
    """Copy one decode step's preprocessed inputs into the persistent buffers,
    normalizing q/k on the way when the caller asks, and prepare the state rows
    plus pointer array."""
    bufs = entry['bufs']
    hq, hv, head_dim = bufs['q'].shape[2], bufs['v'].shape[2], bufs['q'].shape[3]
    row_elems = recurrent_state.stride(0)
    _gdn_pack_kernel[(batch, hv)](
        q,
        k,
        v,
        g,
        beta,
        bufs['q'],
        bufs['k'],
        bufs['v'],
        bufs['g'],
        bufs['beta'],
        recurrent_state,
        input_ids,
        output_ids,
        bufs['ptrs'],
        bufs['finished'],
        recurrent_state.shape[0],
        recurrent_state.data_ptr(),
        row_elems,
        row_elems * recurrent_state.element_size(),
        head_dim,
        hq,
        hv // hq,
        NORMALIZE=normalize,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@cache
def _tm_plan_decline(hv: int, input_dtype: str, state_dtype: str) -> str | None:
    """Probe whether the TurboMind planner accepts a recurrent workload.

    Returns None when a plan can be built, else the decline reason. The probe covers architecture-dependent availability
    (for example no path exists on sm100) through the registered planner itself.
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

    Runs one generic plan probe (8 heads, bf16 input, f32 state). Per-layer geometry is rechecked on the first decode
    step.
    """
    return tm_gdn_decline_reason(128, 8, torch.bfloat16, torch.float32)


class TmGatedDeltaRuleImpl(CudaGatedDeltaRuleImpl):
    """CUDA gated delta rule with the TurboMind recurrent decode kernel.

    Decode runs the transposed-layout TurboMind variant straight on the
    indexed state pool: no gather, no transpose, no scatter. Prefill
    (chunked) steps keep the fla production path unchanged.
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
        """One zero-copy TurboMind recurrent step over the indexed state
        pool."""
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
        # The TurboMind kernel reads g and beta as float32; the production
        # preprocess keeps beta in the input dtype. The pack kernel walks
        # the preprocessed tensors flat, so contiguous layouts are assumed
        # (production preprocess and projections produce contiguous decode
        # activations).
        # The TurboMind kernel has no fused q/k l2 normalization, so the
        # pack below normalizes when the production preprocess deferred
        # that to the fla kernel.
        head_dim, hv = query.shape[-1], query.shape[-2]
        hq = q.shape[-2]
        normalize = bool(use_qk_l2norm_in_kernel and not normalized)
        state_dtype = {
            torch.float32: 'f32',
            torch.bfloat16: 'bf16',
            torch.float16: 'f16',
        }[recurrent_state.dtype]
        entry = self._tm_entry(batch, hq, hv, head_dim, query.dtype, state_dtype, recurrent_state.device)
        bufs = entry['bufs']
        _gdn_pack(entry, q[0], k[0], value[0], g[0], beta[0], recurrent_state, ids.input_state_ids,
                  ids.output_state_ids[:, 0], batch, normalize)
        from lmdeploy.turbomind import _tm

        plan = entry['plan']
        out = torch.empty(tuple(plan['out']['shape']), dtype=query.dtype, device=query.device)
        args = entry['args']
        args['out'] = _tm.from_dlpack(out)
        with self._tm_context(torch.cuda.current_stream(query.device).cuda_stream):
            _tm.delta_rule_run(**args)
        return out.flatten(0, 1).unsqueeze(0)

    def _tm_entry(self, batch: int, hq: int, hv: int, head_dim: int, dtype: torch.dtype, state_dtype: str,
                  device: torch.device) -> dict:
        """Plan, persistent step buffers and prewrapped dlpack handles for one
        workload signature.

        dlpack wrapping is pure host work on fixed buffers; caching the handles keeps the decode step free of it.
        """
        from lmdeploy.turbomind import _tm

        plan_key = (batch, hq, hv, head_dim, str(dtype), state_dtype, str(device))
        entry = self._tm_plans.get(plan_key)
        if entry is not None:
            return entry
        stream = torch.cuda.current_stream(device)
        context = self._tm_context(stream.cuda_stream)
        with context:
            q = torch.zeros(batch, 1, hq, head_dim, device=device, dtype=dtype)
            v = torch.zeros(batch, 1, hv, head_dim, device=device, dtype=dtype)
            g = torch.zeros(batch, 1, hv, device=device, dtype=torch.float32)
            plan = _tm.delta_rule_plan(
                _tm.from_dlpack(q),
                _tm.from_dlpack(q),
                _tm.from_dlpack(v),
                _tm.from_dlpack(g),
                _tm.from_dlpack(g),
                state_dtype=state_dtype,
                mode='recurrent',
                state_layout='vk',
            )
            bufs = {
                'q': torch.empty_like(q),
                'k': torch.empty_like(q),
                'v': torch.empty_like(v),
                'g': torch.empty_like(g),
                'beta': torch.empty_like(g),
                'ptrs': torch.empty(batch, dtype=torch.int64, device=device),
                'finished': torch.empty(batch, dtype=torch.bool, device=device),
            }
            workspace_size = int(plan['workspace']['storage_size'])
            args = dict(
                q=_tm.from_dlpack(bufs['q']),
                k=_tm.from_dlpack(bufs['k']),
                v=_tm.from_dlpack(bufs['v']),
                g=_tm.from_dlpack(bufs['g']),
                beta=_tm.from_dlpack(bufs['beta']),
                plan=plan,
                stream_ptr=0,
                state_ptrs=_tm.from_dlpack(bufs['ptrs']),
                finished=_tm.from_dlpack(bufs['finished']),
            )
            if workspace_size > 0:
                bufs['workspace'] = torch.empty(workspace_size, dtype=torch.uint8, device=device)
                args['workspace'] = _tm.from_dlpack(bufs['workspace'])
            entry = {'plan': plan, 'bufs': bufs, 'args': args}
            self._tm_plans[plan_key] = entry
            return entry

    def _tm_context(self, stream_ptr: int):
        """Cache one native context guard per Torch stream (linear.py
        precedent)."""
        context = self._tm_contexts.get(stream_ptr)
        if context is None:
            from lmdeploy.turbomind import _tm
            context = _tm.create_device_context(stream_ptr)
            self._tm_contexts[stream_ptr] = context
        return context
