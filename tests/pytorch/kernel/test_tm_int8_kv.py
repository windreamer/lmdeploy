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
    from lmdeploy.turbomind import _tm
    from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table

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


def _tm_decode(q, pool, klen, layer=0):
    from lmdeploy.turbomind import _tm
    from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table

    k_cache = _layer_views(pool, layer)[0]
    ptrs, cu_block_base = tm_block_pointer_table(_block_offsets(), k_cache.data_ptr(), k_cache.stride(0))
    cu = _cu([klen // BLOCK_LEN] * BATCH)
    out = torch.empty(BATCH, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    _tm.tm_attn_decode(
        _tm.from_dlpack(q.reshape(BATCH, -1).contiguous()),
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
    # The writer treats input rows as packed (sum of q_seqlens); pass only
    # the real prefill tokens.
    fill_kv_cache_tm(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:BATCH * PREFILL_LEN],
                     v.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:BATCH * PREFILL_LEN],
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
    """TM write kernel -> TM pool -> TM decode vs the bf16 reference (S1
    gate) and vs the pool's own dequantization."""
    _, k, v = _make_data()
    q, _, _ = _make_data(seed=3)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    pool_t = pool.pools[0].tensor
    _bridge_fill(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM), v.reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 [PREFILL_LEN] * BATCH, pool)
    torch.cuda.synchronize()

    out = _tm_decode(q, pool, PREFILL_LEN)
    ref = _reference(q, k, v, PREFILL_LEN)
    max_abs, kl, argmatch = _assert_gate(out, ref, 'tm-decode/prefill')

    kq, vq = _dequant_pool(pool_t)
    ref_q = _reference(q, kq, vq, PREFILL_LEN)
    diff_q = (out.float() - ref_q).abs().max().item()
    print(f'\ntm-decode/prefill: max_abs={max_abs:.4f} KL={kl:.3e} argmatch={argmatch:.4f} '
          f'max_abs_vs_dequant={diff_q:.4f}')


def test_pt_reader_on_tm_pool():
    """Stock Triton paged attention reads the TM pool through the strided
    views with (scale, min) -> (scale, zero) conversion."""
    _, k, v = _make_data()
    q, _, _ = _make_data(seed=5)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    _bridge_fill(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM), v.reshape(-1, NUM_KV_HEADS, HEAD_DIM),
                 [PREFILL_LEN] * BATCH, pool)
    torch.cuda.synchronize()

    out_pt = _pt_decode(q, pool, PREFILL_LEN)
    ref = _reference(q, k, v, PREFILL_LEN)
    max_abs, kl, _ = _assert_gate(out_pt, ref, 'pt-reader/prefill')

    out_tm = _tm_decode(q, pool, PREFILL_LEN)
    cross = (out_pt.float() - out_tm.float()).abs().max().item()
    print(f'\npt-reader/prefill: max_abs={max_abs:.4f} KL={kl:.3e} vs-tm={cross:.4f}')
    assert cross <= 0.25, 'both readers must stay within the int8 max_abs budget'


def test_mixed_chain_prefill_then_decode():
    """B3-01's core assertion on the production path: one pool receives a
    prefill append then real one-token decode appends, and both readers stay
    correct over the full length."""
    _, k, v = _make_data()
    q, _, _ = _make_data(seed=7)
    pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
    pool_t = pool.pools[0].tensor
    kt = k.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    vt = v.reshape(-1, NUM_KV_HEADS, HEAD_DIM)

    _bridge_fill(kt[:BATCH * PREFILL_LEN], vt[:BATCH * PREFILL_LEN], [PREFILL_LEN] * BATCH, pool)
    torch.cuda.synchronize()
    snapshot = pool_t.clone()

    # Production fill path: the triton-free ATen writer appends the decode
    # steps into the same pool the bridge-prefilled half lives in.
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm
    for step in range(DECODE_STEPS):
        begin = BATCH * (PREFILL_LEN + step)
        kv_now = PREFILL_LEN + step + 1
        fill_kv_cache_tm(kt[begin:begin + BATCH], vt[begin:begin + BATCH],
                         *_layer_views(pool, 0),
                         _cu([1] * BATCH)[:-1].long(),
                         torch.tensor([1] * BATCH, dtype=torch.long, device='cuda'),
                         torch.tensor([kv_now] * BATCH, dtype=torch.long, device='cuda'),
                         _block_offsets())
        torch.cuda.synchronize()
        if step in (0, DECODE_STEPS - 1):
            out = _tm_decode(q, pool, kv_now)
            ref = _reference(q, k, v, kv_now)
            _assert_gate(out, ref, f'tm-decode/decode-step-{step}')

    # decode steps really wrote new positions (mixed provenance, not a no-op)
    assert (pool_t != snapshot).any().item()
    kq, _ = _dequant_pool(pool_t)
    kt_full = k.float().reshape(BATCH, KLEN, NUM_KV_HEADS, HEAD_DIM)
    tail_err = (kq[:, PREFILL_LEN:] - kt_full[:, PREFILL_LEN:]).abs().max().item()
    assert tail_err <= 0.05, 'decode-appended tail must round-trip'

    out = _tm_decode(q, pool, KLEN)
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
    fill_kv_cache_tm(k.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:BATCH * PREFILL_LEN],
                     v.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:BATCH * PREFILL_LEN],
                     k_cache, v_cache, k_sz, v_sz,
                     torch.zeros(BATCH, dtype=torch.long, device='cuda'),
                     torch.full((BATCH, ), PREFILL_LEN, dtype=torch.long, device='cuda'),
                     torch.full((BATCH, ), PREFILL_LEN, dtype=torch.long, device='cuda'),
                     _block_offsets())
    torch.cuda.synchronize()

    out = _tm_decode(q, pool, PREFILL_LEN)
    _assert_gate(out, _reference(q, k, v, PREFILL_LEN), 'tm-decode/aten-writer')
    out_pt = _pt_decode(q, pool, PREFILL_LEN)
    _assert_gate(out_pt, _reference(q, k, v, PREFILL_LEN), 'pt-reader/aten-writer')


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
    pool.index_copy_(0, torch.tensor([dst], device='cuda'), pool[0:1, src:src + 1].clone())
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
