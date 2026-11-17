# Copyright (c) OpenMMLab. All rights reserved.
"""S1 checks for the opt-in TM int8-KV block layout (ticket B3-04).

Covers the pool geometry, the TM write bridge, the TM decode dispatch, the
stock PyTorch readers on the same pool, the mixed prefill/decode chain, the
triton-free ATen fallback writer, block copy, and the opt-in routing matrix.

Gates follow ticket 09 batch-1 S1 with the B3-01 amendment: KL <= 1e-2
against the bf16 reference, no NaN, TM max_abs budget 0.25. Argmax match is
reported but not asserted (near-tie output rows make it uninformative for
int8 KV).
"""

from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import QuantPolicy

BLOCK_LEN = 64
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
BATCH = 4
PREFILL_LEN = 192
DECODE_STEPS = 64
KLEN = PREFILL_LEN + DECODE_STEPS
NUM_BLOCKS = BATCH * (KLEN // BLOCK_LEN)
NUM_LAYERS = 2
AMP = 0.5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')


def _make_data(seed=11):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    q = (torch.randn(BATCH, NUM_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    k = (torch.randn(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    v = (torch.randn(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    return q, k, v


def _block_offsets():
    return (torch.arange(BATCH, device='cuda')[:, None] * (KLEN // BLOCK_LEN) +
            torch.arange(KLEN // BLOCK_LEN, device='cuda')[None, :]).int()


def _cu(vals):
    return torch.nn.functional.pad(torch.cumsum(torch.tensor(vals, dtype=torch.int32), 0), (1, 0)).cuda()


def _allocate_layout():
    from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout
    from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

    kv_shape = [BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM]
    specs = (
        CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
        CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
        CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.bfloat16)),
        CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.bfloat16)),
    )
    return TmInt8BlockCacheLayout(specs, num_layers=NUM_LAYERS)


def _gather_by_block_table(values, klen):
    """Reorder pool-order rows into (B, klen) sequence order via the block
    table: pool row = offsets[b, pos//64] * 64 + pos % 64."""
    offsets = _block_offsets().long()
    pos = torch.arange(klen, device='cuda', dtype=torch.long)
    src = offsets[:, pos.div(BLOCK_LEN, rounding_mode='floor')] * BLOCK_LEN + pos % BLOCK_LEN
    return values[src]


def _dequant_pool(pool, layer=0):
    """Ground truth from pool bytes under the block.h arrangement, gathered
    into (B, klen, H, D) sequence order."""
    h2sd = 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM
    codes = pool[layer, :, :h2sd].view(NUM_BLOCKS, NUM_KV_HEADS, 2, BLOCK_LEN, HEAD_DIM).float()
    params = pool[layer, :, h2sd:].view(torch.bfloat16).view(NUM_BLOCKS, NUM_KV_HEADS, 2, BLOCK_LEN, 2).float()
    k = (codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2]).permute(0, 2, 1, 3)
    v = (codes[:, :, 1] * params[:, :, 1][..., 0:1] + params[:, :, 1][..., 1:2]).permute(0, 2, 1, 3)
    k = k.reshape(NUM_BLOCKS * BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)
    v = v.reshape(NUM_BLOCKS * BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)
    return _gather_by_block_table(k, KLEN), _gather_by_block_table(v, KLEN)


def _reference(q, k, v, klen):
    group = NUM_HEADS // NUM_KV_HEADS
    kk = k[:, :klen].float().repeat_interleave(group, dim=2)
    vv = v[:, :klen].float().repeat_interleave(group, dim=2)
    scores = torch.einsum('bhd,bkhd->bhk', q.float(), kk) / HEAD_DIM**0.5
    p = scores.softmax(-1)
    return torch.einsum('bhk,bkhd->bhd', p, vv)


def _metrics(out, ref):
    diff = (out.float() - ref.float()).abs()
    p = out.float().softmax(-1).clamp_min(1e-12)
    r = ref.float().softmax(-1).clamp_min(1e-12)
    kl = (r * (r / p).log()).sum(-1).mean().item()
    argmatch = (out.float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
    return diff.max().item(), kl, argmatch


def _assert_gate(out, ref, label):
    assert not out.isnan().any().item(), f'{label}: NaN in output'
    max_abs, kl, argmatch = _metrics(out, ref)
    assert kl <= 1e-2, f'{label}: KL {kl:.3e} exceeds 1e-2'
    assert max_abs <= 0.25, f'{label}: max_abs {max_abs:.3f} exceeds 0.25 budget'
    return max_abs, kl, argmatch


def _bridge_fill(k_slice, v_slice, kv_seqlens, pool, layer=0):
    """One TM write-kernel append of new tokens (token-major slices)."""
    from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table
    from lmdeploy.turbomind import _tm

    k_cache, v_cache, k_sz, v_sz = _layer_views(pool, layer)
    pool_base, block_bytes, _, num_kv_heads, head_dim = (
        k_cache.data_ptr(), k_cache.stride(0), k_cache.shape[1], k_cache.shape[2], k_cache.shape[3])
    ptrs, cu_block_base = tm_block_pointer_table(_block_offsets(), pool_base, block_bytes)
    num_tokens = k_slice.shape[0]
    q_len = num_tokens // BATCH
    cu_q = _cu([q_len] * BATCH)
    cu_k = _cu(kv_seqlens)
    _tm.tm_process_kv(
        _tm.from_dlpack(k_slice.contiguous()),
        _tm.from_dlpack(v_slice.contiguous()),
        ptrs.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(), cu_block_base.data_ptr(),
        stride_c=num_kv_heads, stride_s=num_kv_heads, stride_h=1,
        max_q_len=q_len, num_kv_heads=num_kv_heads, head_dim=head_dim,
        batch_size=BATCH, block_len=BLOCK_LEN, kv_quant=8,
        stream_ptr=torch.cuda.current_stream().cuda_stream)


def _tm_decode(q, pool, klen, k_last, v_last, layer=0):
    from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table
    from lmdeploy.turbomind import _tm

    k_cache = _layer_views(pool, layer)[0]
    ptrs, cu_block_base = tm_block_pointer_table(_block_offsets(), k_cache.data_ptr(), k_cache.stride(0))
    # The pointer table is row-major over block_offsets columns (C wide); the
    # kernel's cu_block_base must use that stride, not klen // BLOCK_LEN.
    # Passing a per-sequence block-count cumsum shifts every batch b > 0's
    # base by b, so its ti = history_len store lands in the wrong block.
    cu = _cu([_block_offsets().shape[1]] * BATCH).int()
    out = torch.empty(BATCH, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    _tm.tm_attn_decode(
        _tm.from_dlpack(q.reshape(BATCH, -1).contiguous()),
        _tm.from_dlpack(k_last.reshape(BATCH, -1).contiguous()),
        _tm.from_dlpack(v_last.reshape(BATCH, -1).contiguous()),
        _tm.from_dlpack(out),
        ptrs.data_ptr(), cu.data_ptr(),
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        batch_size=BATCH, block_len=BLOCK_LEN, max_k_len=klen,
        kv_quant=8, stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out.view(BATCH, NUM_HEADS, HEAD_DIM)


def _layer_views(allocation_or_pool, layer=0):
    """Per-layer (k, v, k_sz, v_sz) views, PT contract shapes."""
    from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation

    if isinstance(allocation_or_pool, CacheAllocation):
        views = allocation_or_pool.tensor_views
    else:
        views = allocation_or_pool
    return tuple(v[layer] for v in views)


def _pt_decode(q, pool, klen, layer=0):
    """Stock Triton paged attention over the TM pool via strided views."""
    from lmdeploy.pytorch.backends.tm_kv import tm_min_to_zero
    from lmdeploy.pytorch.kernels.cuda.pagedattention import flash_attn_with_kvcache

    k_cache, v_cache, k_sz, v_sz = _layer_views(pool, layer)
    cache_seqlens = torch.full((BATCH, ), klen, dtype=torch.int32, device='cuda')
    page_table = _block_offsets()
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens, page_table,
        k_scales_zeros=tm_min_to_zero(k_sz), v_scales_zeros=tm_min_to_zero(v_sz),
        quant_policy=QuantPolicy.INT8, kv_layout='bshd')


def test_layout_views_and_writer_roundtrip():
    """Views expose the standard contract over TM storage; the ATen writer
    fills codes and (scale, min) params at the contract positions."""
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm

    layout = _allocate_layout()
    assert layout.block_nbytes == 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM + \
        2 * NUM_KV_HEADS * BLOCK_LEN * 2 * 2

    allocation = layout.allocate(NUM_BLOCKS, 'cuda')
    assert allocation.pools[0].tensor.shape == (NUM_LAYERS, NUM_BLOCKS, layout.block_nbytes)
    assert allocation.pools[0].entry_axis == 1
    k_cache, v_cache, k_sz, v_sz = _layer_views(allocation, 1)
    for view, shape in ((k_cache, (NUM_BLOCKS, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)),
                        (v_cache, (NUM_BLOCKS, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)),
                        (k_sz, (NUM_BLOCKS, BLOCK_LEN, NUM_KV_HEADS, 2)),
                        (v_sz, (NUM_BLOCKS, BLOCK_LEN, NUM_KV_HEADS, 2))):
        assert tuple(view.shape) == shape

    _, k, v = _make_data()
    q_seqlens = [PREFILL_LEN] * BATCH
    # Writer input rows are tightly packed in batch order (the fill_kv_cache
    # contract): batch b occupies rows [b * q_seqlens[b], (b + 1) * q_seqlens[b]).
    # Truncate each sequence to its append length BEFORE flattening; slicing a
    # flat b-major reshape instead interleaves other batches' rows into every
    # batch, the exact B3-04 multi-batch placement symptom.
    fill_kv_cache_tm(k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                     v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                     k_cache, v_cache, k_sz, v_sz,
                     _cu(q_seqlens)[:-1].long(),
                     torch.tensor(q_seqlens, dtype=torch.long, device='cuda'),
                     torch.tensor(q_seqlens, dtype=torch.long, device='cuda'),
                     _block_offsets())
    torch.cuda.synchronize()

    kq, vq = _dequant_pool(allocation.pools[0].tensor, layer=1)
    kt = k.float().reshape(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM)
    vt = v.float().reshape(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM)
    assert (kq[:, :PREFILL_LEN] - kt[:, :PREFILL_LEN]).abs().max().item() <= 0.05
    assert (vq[:, :PREFILL_LEN] - vt[:, :PREFILL_LEN]).abs().max().item() <= 0.05
    # untouched tail stays zero-filled
    assert kq[:, PREFILL_LEN:].abs().max().item() == 0.0


def test_bridge_write_and_tm_decode_matches_reference():
    """TM write kernel -> TM pool -> TM decode vs the bf16 reference (S1 gate)
    and vs the pool's own dequantization."""
    _, k, v = _make_data()
    q, _, _ = _make_data(seed=3)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    pool_t = pool.pools[0].tensor
    _bridge_fill(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM), v.reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 [PREFILL_LEN] * BATCH, pool)
    torch.cuda.synchronize()

    out = _tm_decode(q, pool, PREFILL_LEN, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1])
    ref = _reference(q, k, v, PREFILL_LEN)
    max_abs, kl, argmatch = _assert_gate(out, ref, 'tm-decode/prefill')

    kq, vq = _dequant_pool(pool_t)
    ref_q = _reference(q, kq, vq, PREFILL_LEN)
    diff_q = (out.float() - ref_q).abs().max().item()
    print(f'\ntm-decode/prefill: max_abs={max_abs:.4f} KL={kl:.3e} argmatch={argmatch:.4f} '
          f'max_abs_vs_dequant={diff_q:.4f}')


def test_pt_reader_on_tm_pool():
    """Stock Triton paged attention reads the TM pool through the strided views
    with (scale, min) -> (scale, zero) conversion."""
    _, k, v = _make_data()
    q, _, _ = _make_data(seed=5)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    _bridge_fill(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM), v.reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 [PREFILL_LEN] * BATCH, pool)
    torch.cuda.synchronize()

    out_pt = _pt_decode(q, pool, PREFILL_LEN)
    ref = _reference(q, k, v, PREFILL_LEN)
    max_abs, kl, _ = _assert_gate(out_pt, ref, 'pt-reader/prefill')

    out_tm = _tm_decode(q, pool, PREFILL_LEN, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1])
    cross = (out_pt.float() - out_tm.float()).abs().max().item()
    print(f'\npt-reader/prefill: max_abs={max_abs:.4f} KL={kl:.3e} vs-tm={cross:.4f}')
    assert cross <= 0.25, 'both readers must stay within the int8 max_abs budget'


def test_mixed_chain_prefill_then_decode():
    """B3-01's core assertion on the production path: one pool receives a
    prefill append then real one-token decode appends, and both readers stay
    correct over the full length.

    The decode binding (src/turbomind/python/attn_bench_bind.cpp,
    BuildDecodePlan) fills params.k/v with the real K/V segments and
    readonly_block_num with a valid zero buffer (production convention,
    commit 45e9322c), so the universal kernel's kProcessKV branch
    (src/turbomind/kernels/attention/attention_universal.h:59 — kProcessKV is
    hardcoded true for DecodingCtaMap) re-stores the token's own converted
    K/V at ti = history_len: idempotent on the pool (the re-store writes
    the token's own converted K/V back).
    """
    _, k, v = _make_data()
    q, _, _ = _make_data(seed=7)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    pool_t = pool.pools[0].tensor
    # Packed rows in batch order: truncate to the append length BEFORE
    # flattening (see the roundtrip test), and the bridge reads batch b's
    # tokens at rows [b * q_len, (b + 1) * q_len) for its cu_q bases.
    kt = k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    vt = v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)

    _bridge_fill(kt, vt, [PREFILL_LEN] * BATCH, pool)
    torch.cuda.synchronize()
    snapshot = pool_t.clone()

    # Production fill path: the triton-free ATen writer appends the decode
    # steps into the same pool the bridge-prefilled half lives in. One new
    # token per sequence: the (B, Hkv, D) grid at one position IS the packed
    # layout for q_seqlens = [1] * BATCH.
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm
    for step in range(DECODE_STEPS):
        kv_now = PREFILL_LEN + step + 1
        fill_kv_cache_tm(k[:, PREFILL_LEN + step], v[:, PREFILL_LEN + step],
                         *_layer_views(pool, 0),
                         _cu([1] * BATCH)[:-1].long(),
                         torch.tensor([1] * BATCH, dtype=torch.long, device='cuda'),
                         torch.tensor([kv_now] * BATCH, dtype=torch.long, device='cuda'),
                         _block_offsets())
        torch.cuda.synchronize()
        if step in (0, DECODE_STEPS - 1):
            out = _tm_decode(q, pool, kv_now, k[:, kv_now - 1], v[:, kv_now - 1])
            ref = _reference(q, k, v, kv_now)
            _assert_gate(out, ref, f'tm-decode/decode-step-{step}')

    # decode steps really wrote new positions (mixed provenance, not a no-op)
    assert (pool_t != snapshot).any().item()
    kq, _ = _dequant_pool(pool_t)
    kt_full = k.float().reshape(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM)
    tail_err = (kq[:, PREFILL_LEN:] - kt_full[:, PREFILL_LEN:]).abs().max().item()
    assert tail_err <= 0.05, 'decode-appended tail must round-trip'

    out = _tm_decode(q, pool, KLEN, k[:, KLEN - 1], v[:, KLEN - 1])
    _assert_gate(out, _reference(q, k, v, KLEN), 'tm-decode/mixed-full')
    out_pt = _pt_decode(q, pool, KLEN)
    _assert_gate(out_pt, _reference(q, k, v, KLEN), 'pt-reader/mixed-full')
    cross = (out.float() - out_pt.float()).abs().max().item()
    print(f'\nmixed-chain KLEN={KLEN}: tm-vs-pt={cross:.4f}')
    assert cross <= 0.02


def test_aten_writer_pool_decodes():
    """The triton-free torch writer produces a pool the TM decode kernel and
    the Triton reader both consume (fallback correctness)."""
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm

    _, k, v = _make_data()
    q, _, _ = _make_data(seed=9)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    k_cache, v_cache, k_sz, v_sz = _layer_views(pool, 0)
    fill_kv_cache_tm(k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                     v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                     k_cache, v_cache, k_sz, v_sz,
                     _cu([PREFILL_LEN] * BATCH)[:-1].long(),
                     torch.full((BATCH, ), PREFILL_LEN, dtype=torch.long, device='cuda'),
                     torch.full((BATCH, ), PREFILL_LEN, dtype=torch.long, device='cuda'),
                     _block_offsets())
    torch.cuda.synchronize()

    # The decode gates below cannot tell a dropped batch from a filled one
    # (B3-04: a wrong q_start_loc left batches 1..3 zero yet still passed);
    # assert the pool content itself so every batch is pinned down.
    kq, vq = _dequant_pool(pool.pools[0].tensor, layer=0)
    kt = k.float().reshape(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM)
    vt = v.float().reshape(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM)
    assert (kq[:, :PREFILL_LEN] - kt[:, :PREFILL_LEN]).abs().max().item() <= 0.05
    assert (vq[:, :PREFILL_LEN] - vt[:, :PREFILL_LEN]).abs().max().item() <= 0.05
    assert kq[:, PREFILL_LEN:].abs().max().item() == 0.0

    out = _tm_decode(q, pool, PREFILL_LEN, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1])
    _assert_gate(out, _reference(q, k, v, PREFILL_LEN), 'tm-decode/aten-writer')
    out_pt = _pt_decode(q, pool, PREFILL_LEN)
    _assert_gate(out_pt, _reference(q, k, v, PREFILL_LEN), 'pt-reader/aten-writer')


def test_aten_writer_fill_is_capture_safe():
    """B3-04 defect-4 regression: the ATen pool fill must run inside CUDA
    graph capture without a boolean-filtering host sync (previously
    `cudaErrorStreamCaptureInvalidated` at engine start), and the captured
    fill must place bytes exactly like an eager fill."""
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm

    _, k, v = _make_data()
    k_slice = k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM).contiguous()
    v_slice = v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM).contiguous()
    q_loc = _cu([PREFILL_LEN] * BATCH)[:-1].long()
    q_len = torch.full((BATCH, ), PREFILL_LEN, dtype=torch.long, device='cuda')

    def fill(kc, vc, ksz, vsz):
        fill_kv_cache_tm(k_slice, v_slice, kc, vc, ksz, vsz, q_loc, q_len, q_len, _block_offsets())

    pool_ref = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    fill(*_layer_views(pool_ref, 0))

    pool_cap = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    views = _layer_views(pool_cap, 0)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fill(*views)  # warmup outside capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        fill(*views)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(pool_cap.pools[0].tensor, pool_ref.pools[0].tensor), \
        'captured fill diverges from the eager fill'


def test_aten_writer_multi_batch_packed_placement():
    """B3-04 regression: with correctly packed input the ATen writer places
    every token at its contract slot in multi-batch appends, and slots
    outside the append range stay untouched.

    Constant per-token values plus fp32 param slots make the round trip
    exact, so each pool row's provenance is byte-identifiable: a single
    misplaced token cannot hide inside a tolerance gate.
    """
    from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout, fill_kv_cache_tm
    from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

    def run_case(tag, batch, q_lens, kv_lens):
        s, h, d = 64, 2, 8
        ncols = max(-(-l // s) for l in kv_lens)
        kv_shape = [s, h, d]
        specs = (
            CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
            CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
            CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.float32)),
            CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.float32)),
        )
        layout = TmInt8BlockCacheLayout(specs, num_layers=1)
        allocation = layout.allocate(batch * ncols, 'cuda')
        pool = allocation.pools[0].tensor
        k_cache, v_cache, k_sz, v_sz = tuple(v[0] for v in allocation.tensor_views)

        # Packed rows in batch order; a unique constant value per token makes
        # the dequantized pool content byte-identifiable.
        vals, per_batch = [], []
        row = 0
        for bi in range(batch):
            n = q_lens[bi]
            base = (torch.arange(n, dtype=torch.float32) + row + 1) * 0.01
            vals.append(base[:, None, None].expand(n, h, d))
            per_batch.append(base)
            row += n
        key = torch.cat(vals).cuda() if vals else torch.zeros(0, h, d).cuda()
        value = -key

        q_seqlens = torch.tensor(q_lens, dtype=torch.long, device='cuda')
        kv_seqlens = torch.tensor(kv_lens, dtype=torch.long, device='cuda')
        q_start_loc = torch.nn.functional.pad(torch.cumsum(q_seqlens.cpu(), 0), (1, 0))[:-1].cuda().long()
        block_offsets = (torch.arange(batch, device='cuda')[:, None] * ncols +
                         torch.arange(ncols, device='cuda')[None, :]).int()
        fill_kv_cache_tm(key, value, k_cache, v_cache, k_sz, v_sz,
                         q_start_loc, q_seqlens, kv_seqlens, block_offsets)
        torch.cuda.synchronize()

        # Exact readback: constant tokens quantize to code 0 with min = value.
        num_blocks = batch * ncols
        h2sd = 2 * h * s * d
        codes = pool[0, :, :h2sd].view(num_blocks, h, 2, s, d).float()
        params = pool[0, :, h2sd:].view(torch.float32).view(num_blocks, h, 2, s, 2)
        kdeq = codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2]
        rows = kdeq.permute(0, 2, 1, 3).reshape(num_blocks * s, h, d)[:, 0, 0].cpu()
        offsets = block_offsets.cpu()

        touched = torch.zeros(num_blocks * s, dtype=torch.bool)
        for bi in range(batch):
            for j in range(q_lens[bi]):
                pos = kv_lens[bi] - q_lens[bi] + j
                want = offsets[bi, pos // s].item() * s + pos % s
                touched[want] = True
                assert rows[want].item() == per_batch[bi][j].item(), \
                    f'{tag}: token (b{bi}, idx {j}) expected at pool row {want} with value ' \
                    f'{per_batch[bi][j].item()}, found {rows[want].item()}'
        stray = (~touched) & (rows != 0)
        assert not stray.any().item(), \
            f'{tag}: {stray.sum().item()} pool rows written outside the append range'

    # single batch stays bit-exact (B=1 makes any batch order coincide)
    run_case('single-batch control', 1, [192], [192])
    # ragged multi-batch with a zero-length row
    run_case('ragged multi-batch', 4, [5, 0, 3, 2], [5, 7, 3, 2])
    # the mixed-chain shape: one new token per sequence past position 192
    run_case('multi-batch decode step', 4, [1, 1, 1, 1], [193, 193, 193, 193])


def test_block_copy_moves_whole_entries():
    """Logical-block copy on the TM pool is layout agnostic: whole entries
    move byte-exactly through the standard entry-axis copy primitive."""
    layout = _allocate_layout()
    allocation = layout.allocate(NUM_BLOCKS, 'cuda')
    pool = allocation.pools[0].tensor
    pool[0, 2].random_()
    src, dst = 2, 5
    before = pool[0, src].clone()
    pool[0, dst] = 0
    pool.index_copy_(1, torch.tensor([dst], device='cuda'), pool[:, src:src + 1].clone())
    torch.cuda.synchronize()
    assert (pool[0, dst] == before).all().item()
    # the CUDA backend's copy primitive accepts the layout's allocation
    from lmdeploy.pytorch.backends.cuda.cache import CudaCacheBackend
    copier = CudaCacheBackend.build_block_copy(allocation, NUM_BLOCKS, 1)
    assert copier is not None


def test_opt_in_routing_matrix(monkeypatch):
    """Knob x quant-policy x capability decides both impl selection and pool
    layout; non-opt-in models are untouched."""
    from lmdeploy.pytorch.backends.tm_kv import (
        record_tm_int8_kv_capability,
        reset_tm_int8_kv_capability,
        tm_int8_kv_layout_requested,
    )

    model_cfg = SimpleNamespace(num_key_value_heads=NUM_KV_HEADS,
                                head_dim=HEAD_DIM,
                                k_head_dim=HEAD_DIM,
                                v_head_dim=HEAD_DIM)
    cache_cfg_int8 = SimpleNamespace(quant_policy=QuantPolicy.INT8, kernel_block_size=BLOCK_LEN)
    cache_cfg_bf16 = SimpleNamespace(quant_policy=QuantPolicy.NONE, kernel_block_size=BLOCK_LEN)

    def spec(**overrides):
        base = dict(num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                    v_head_dim=HEAD_DIM, alibi=False, learnable_sink=False,
                    logit_softcapping=0.0, causal=True, block_sparse_size=1,
                    use_flash_mla=False, sliding_window=None)
        base.update(overrides)
        return SimpleNamespace(**base)

    from lmdeploy.pytorch.backends.cuda.attention import _build_tm_int8_kv_attention

    common = dict(num_heads=NUM_HEADS, head_size=HEAD_DIM, scale=0.1,
                  num_kv_heads=NUM_KV_HEADS, v_head_size=HEAD_DIM,
                  alibi=False, sliding_window=(-1, -1), logit_softcapping=0.0,
                  causal=True)

    # knob off: no TM impl, no TM layout, even for an INT8 model
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'off')
    reset_tm_int8_kv_capability()
    assert _build_tm_int8_kv_attention(spec(), common) is None
    assert not tm_int8_kv_layout_requested(model_cfg, cache_cfg_int8)

    # knob auto + capable attention + INT8: TM impl and TM layout
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'auto')
    reset_tm_int8_kv_capability()
    impl = _build_tm_int8_kv_attention(spec(), common)
    assert type(impl).__name__ == 'TmInt8KVAttentionImpl'
    assert tm_int8_kv_layout_requested(model_cfg, cache_cfg_int8)
    # non-INT8 models keep the default pool even with a capable attention
    assert not tm_int8_kv_layout_requested(model_cfg, cache_cfg_bf16)

    # one incapable layer disables the layout for the whole model (AND rule)
    reset_tm_int8_kv_capability()
    assert _build_tm_int8_kv_attention(spec(alibi=True), common) is None
    assert not tm_int8_kv_layout_requested(model_cfg, cache_cfg_int8)

    # MLA layers stay neutral: they neither enable nor disable the layout
    reset_tm_int8_kv_capability()
    assert _build_tm_int8_kv_attention(spec(use_flash_mla=True), common) is None
    assert not tm_int8_kv_layout_requested(model_cfg, cache_cfg_int8)
    record_tm_int8_kv_capability(True)
    assert tm_int8_kv_layout_requested(model_cfg, cache_cfg_int8)

    # forced knob with an unsupported geometry must fail loudly
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'on')
    reset_tm_int8_kv_capability()
    with pytest.raises(RuntimeError, match='head_dim'):
        _build_tm_int8_kv_attention(spec(head_dim=96, v_head_dim=96), common)

    # wrong block size is rejected at the layout probe
    monkeypatch.setenv('LMDEPLOY_TM_ATTN_INT8_KV', 'auto')
    reset_tm_int8_kv_capability()
    record_tm_int8_kv_capability(True)
    cache_cfg_bad = SimpleNamespace(quant_policy=QuantPolicy.INT8, kernel_block_size=32)
    assert not tm_int8_kv_layout_requested(model_cfg, cache_cfg_bad)
    reset_tm_int8_kv_capability()


# ---------------------------------------------------------------------------
# B3-04 redo: TM int8 decode under CUDA graph capture/replay.
#
# Mandatory methodology (see benchmark/b304_graph_reverify/report.md): plain
# tolerance gates are proven false-green under replay (stale output passes
# the 0.25 int8 budget). Verification below is bitwise replay-vs-eager plus
# content-follow and host-interference invariance probes.
# ---------------------------------------------------------------------------


class _GraphDecodeCase:
    """Graph-captured TM decode step over persistent buffers, mirroring
    `TmInt8KVAttentionImpl._forward_tm_decode_graph` without engine
    machinery: every per-step input (q/k/v rows, seqlens) is content of a
    persistent buffer rewritten before each replay."""

    def __init__(self, pool, layer=0):
        from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table
        from lmdeploy.turbomind import _tm

        self.tm = _tm
        k_cache = _layer_views(pool, layer)[0]
        self.pool_base = k_cache.data_ptr()
        self.block_bytes = k_cache.stride(0)
        self.cols = _block_offsets().shape[1]
        self.capacity = self.cols * BLOCK_LEN  # frozen max_k_len at capture
        self.cu_block_nums = _cu([self.cols] * BATCH).int()
        b, hkv = BATCH, NUM_KV_HEADS
        self.q_b = torch.zeros(b, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        self.k_b = torch.zeros(b, hkv * HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        self.v_b = torch.zeros(b, hkv * HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        self.out_b = torch.zeros(b, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        self.kv_seqlens_buf = torch.zeros(b, dtype=torch.int32, device='cuda')
        self.cu_q = torch.zeros(b + 1, dtype=torch.int32, device='cuda')
        self.cu_k = torch.zeros(b + 1, dtype=torch.int32, device='cuda')
        ws_bytes = _tm.tm_decode_workspace_bytes(b, NUM_HEADS, hkv, HEAD_DIM, 2)
        self.ws = torch.zeros(ws_bytes, dtype=torch.uint8, device='cuda')
        self.ptrs, self.bases = tm_block_pointer_table(_block_offsets(), self.pool_base, self.block_bytes)
        self.graph = torch.cuda.CUDAGraph()

    def set_inputs(self, q, k_last, v_last, kv_seqlens):
        """Host-side content rewrite of the persistent step buffers."""
        self.q_b.copy_(q.reshape(BATCH, -1))
        self.k_b.copy_(k_last.reshape(BATCH, -1))
        self.v_b.copy_(v_last.reshape(BATCH, -1))
        self.kv_seqlens_buf.copy_(torch.tensor(kv_seqlens, dtype=torch.int32, device='cuda'))

    def _captured_ops(self):
        # Per-step device fills (captured): cumulative lengths follow the
        # simulated runner seqlen buffer; the binding reads them on device.
        self.cu_q[1:].copy_(torch.arange(1, BATCH + 1, dtype=torch.int32, device='cuda'))
        self.cu_q[0].zero_()
        self.cu_k[1:].copy_(torch.cumsum(self.kv_seqlens_buf, 0))
        self.cu_k[0].zero_()
        self.tm.tm_attn_decode_graph(
            self.tm.from_dlpack(self.q_b),
            self.tm.from_dlpack(self.k_b),
            self.tm.from_dlpack(self.v_b),
            self.tm.from_dlpack(self.out_b),
            self.ptrs.data_ptr(), self.cu_block_nums.data_ptr(),
            self.cu_q.data_ptr(), self.cu_k.data_ptr(), self.ws.data_ptr(),
            num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            batch_size=BATCH, block_len=BLOCK_LEN, max_k_len=self.capacity,
            kv_quant=8, stream_ptr=torch.cuda.current_stream().cuda_stream)

    def capture(self):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._captured_ops()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self._captured_ops()

    def replay(self):
        self.graph.replay()
        torch.cuda.synchronize()
        return self.out_b.view(BATCH, NUM_HEADS, HEAD_DIM).clone()

    def eager(self):
        """Same graph-entry dispatch, run eagerly (control path)."""
        self._captured_ops()
        torch.cuda.synchronize()
        return self.out_b.view(BATCH, NUM_HEADS, HEAD_DIM).clone()


def _graph_fixture():
    """Two identically prefilled pools: replay runs on pool_g, the eager
    control on pool_e; every step keeps them state-synchronized."""
    _, k, v = _make_data()
    kt = k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    vt = v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    pools = []
    for _ in range(2):
        pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
        _bridge_fill(kt, vt, [PREFILL_LEN] * BATCH, pool)
        pools.append(pool)
    torch.cuda.synchronize()
    return pools[0], pools[1], k, v


def _eager_control(case, pool, q, k_last, v_last, kv_seqlens):
    """Mirror one graph step on the eager pool with the same contents."""
    ctrl = _GraphDecodeCase(pool)
    ctrl.set_inputs(q, k_last, v_last, kv_seqlens)
    return ctrl.eager()


def test_graph_capture_replay_bitwise_and_content_follow():
    """Capture the graph-safe TM decode, then prove replay follows content:

    identical replay is bitwise vs the eager control; new q/k/v rows and grown/unequal per-seq lengths replay bitwise vs
    the eager recomputation on a state-mirrored pool.
    """
    pool_g, pool_e, k, v = _graph_fixture()
    q, _, _ = _make_data(seed=31)

    case = _GraphDecodeCase(pool_g)
    case.set_inputs(q, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1], [PREFILL_LEN] * BATCH)
    case.capture()

    # R0: same-content replay vs eager control (bitwise).
    ctrl = _GraphDecodeCase(pool_e)
    ctrl.set_inputs(q, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1], [PREFILL_LEN] * BATCH)
    out_e0 = ctrl.eager()
    out_r0 = case.replay()
    assert torch.equal(out_r0, out_e0), \
        f'replay vs eager not bitwise at capture lengths: {(out_r0 - out_e0).abs().max().item()}'

    gen = torch.Generator(device='cpu').manual_seed(77)
    q2 = (torch.randn(BATCH, NUM_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    k_last2 = (torch.randn(BATCH, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    v_last2 = (torch.randn(BATCH, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()

    # R1: content-follow — mutate q/k/v rows and grow every sequence to 200.
    case.set_inputs(q2, k_last2, v_last2, [PREFILL_LEN + 8] * BATCH)
    out_r1 = case.replay()
    delta = (out_r1.float() - out_r0.float()).abs().max().item()
    assert delta >= 0.005, f'content-follow probe failed: replay output did not follow (delta={delta:.5f})'
    out_e1 = _eager_control(ctrl, pool_e, q2, k_last2, v_last2, [PREFILL_LEN + 8] * BATCH)
    assert torch.equal(out_r1, out_e1), \
        f'replay vs eager not bitwise after mutation: {(out_r1 - out_e1).abs().max().item()}'

    # R2: unequal per-seq lengths under the frozen capacity bound (masking
    # must come from device cu_k_len, not from max_k_len).
    lens = [PREFILL_LEN + 1, PREFILL_LEN + 16, PREFILL_LEN, PREFILL_LEN + 33]
    case.set_inputs(q, k[:, PREFILL_LEN + 32], v[:, PREFILL_LEN + 32], lens)
    out_r2 = case.replay()
    out_e2 = _eager_control(ctrl, pool_e, q, k[:, PREFILL_LEN + 32], v[:, PREFILL_LEN + 32], lens)
    assert torch.equal(out_r2, out_e2), \
        f'replay vs eager not bitwise with unequal lengths: {(out_r2 - out_e2).abs().max().item()}'

    # Ref sanity on the last step: int8 gate on the eager control output.
    ref = _reference(q, k, v, max(lens))
    _assert_gate(out_e2, ref, 'graph-step/unequal-lengths')


def test_graph_replay_host_interference_invariance():
    """Host/device interference probe (report P2/P3): an unrelated eager TM
    dispatch between two replays must not change replay output bitwise — this
    is what failed when plan buffers lived in the capture region."""
    pool_g, pool_e, k, v = _graph_fixture()
    q, _, _ = _make_data(seed=41)
    pool_i = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    _bridge_fill(k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 [PREFILL_LEN] * BATCH, pool_i)
    torch.cuda.synchronize()

    case = _GraphDecodeCase(pool_g)
    case.set_inputs(q, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1], [PREFILL_LEN] * BATCH)
    case.capture()
    out_a = case.replay()

    # Interference: eager TM decode on a different pool + a padded max_k_len
    # eager dispatch (historically poisoned the replayed plan buffers).
    _tm_decode(q, pool_i, PREFILL_LEN, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1])
    _eager_control(_GraphDecodeCase(pool_e), pool_e, q,
                   k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1], [PREFILL_LEN] * BATCH)
    torch.cuda.synchronize()

    out_b = case.replay()
    assert torch.equal(out_a, out_b), \
        f'replay changed under host/device interference: {(out_a - out_b).abs().max().item()}'


def test_graph_capture_guard_routes_tm():
    """Capture guard: TM decode is eligible while capturing when the native
    module is present (graph-safe entry), eager steps unchanged."""
    from lmdeploy.pytorch.backends.cuda.attention.int8_kv import TmInt8KVAttentionImpl

    impl = TmInt8KVAttentionImpl(NUM_HEADS, HEAD_DIM, num_kv_heads=NUM_KV_HEADS)
    meta = SimpleNamespace(max_q_seqlen=1)
    assert impl._can_tm_decode(meta), 'eager single-token steps must take TM'
    meta = SimpleNamespace(max_q_seqlen=3)
    assert not impl._can_tm_decode(meta), 'multi-token steps must not take TM'
    if impl._tm_module() is None:
        pytest.skip('native TM module unavailable')
    torch.cuda.synchronize()  # ensure no capture is active
    meta = SimpleNamespace(max_q_seqlen=1)
    assert impl._can_tm_decode(meta), 'capture must take TM when buffers can be prepared'
