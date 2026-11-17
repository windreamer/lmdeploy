# Shared geometry/helpers for the B3-04 graph-capture re-verification matrix.
# Mirrors tests/pytorch/kernel/test_tm_int8_kv.py conventions (S1 gate budget).
import json
import os

import torch

BLOCK_LEN = 64
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
BATCH = 4
NUM_LAYERS = 2
AMP = 0.5
PREFILL_LEN = 192  # 3 blocks / sequence
KLEN = 256  # window cap: 4 blocks / sequence
C = KLEN // BLOCK_LEN
NUM_BLOCKS = BATCH * C
MAX_ABS_BUDGET = 0.25

_HERE = os.path.dirname(os.path.abspath(__file__))


def make_layout():
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


def allocate_pool():
    return make_layout().allocate(NUM_BLOCKS, 'cuda')


def layer_views(allocation, layer=0):
    return tuple(v[layer] for v in allocation.tensor_views)


def block_offsets():
    """(B, C) int32 identity block ids."""
    return (torch.arange(BATCH, device='cuda')[:, None] * C +
            torch.arange(C, device='cuda')[None, :]).int()


def cu(vals):
    # torch.cumsum promotes int32 to int64 unless pinned; the TM bindings
    # read `const int*` and an int64 array interleaves garbage (B3-07).
    return torch.nn.functional.pad(
        torch.cumsum(torch.tensor(vals, dtype=torch.int32), 0, dtype=torch.int32), (1, 0)).cuda()


def make_qkv(seed, klen):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    q = (torch.randn(BATCH, NUM_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    k = (torch.randn(BATCH, klen, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    v = (torch.randn(BATCH, klen, NUM_KV_HEADS, HEAD_DIM, generator=gen) * AMP).cuda().bfloat16()
    return q, k, v


def gather_by_block_table(values, klen):
    offsets = block_offsets().long()
    pos = torch.arange(klen, device='cuda', dtype=torch.long)
    src = offsets[:, pos.div(BLOCK_LEN, rounding_mode='floor')] * BLOCK_LEN + pos % BLOCK_LEN
    return values[src]


def dequant_pool(pool_tensor, layer=0):
    """Ground truth from pool bytes under the block.h arrangement, gathered
    into (B, klen, Hkv, D) sequence order."""
    h2sd = 2 * NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM
    codes = pool_tensor[layer, :, :h2sd].view(NUM_BLOCKS, NUM_KV_HEADS, 2, BLOCK_LEN, HEAD_DIM).float()
    params = pool_tensor[layer, :, h2sd:].view(torch.bfloat16).view(
        NUM_BLOCKS, NUM_KV_HEADS, 2, BLOCK_LEN, 2).float()
    k = (codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2]).permute(0, 2, 1, 3)
    v = (codes[:, :, 1] * params[:, :, 1][..., 0:1] + params[:, :, 1][..., 1:2]).permute(0, 2, 1, 3)
    k = k.reshape(NUM_BLOCKS * BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)
    v = v.reshape(NUM_BLOCKS * BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM)
    return gather_by_block_table(k, KLEN), gather_by_block_table(v, KLEN)


def reference(q, k, v, klen):
    group = NUM_HEADS // NUM_KV_HEADS
    kk = k[:, :klen].float().repeat_interleave(group, dim=2)
    vv = v[:, :klen].float().repeat_interleave(group, dim=2)
    scores = torch.einsum('bhd,bkhd->bhk', q.float(), kk) / HEAD_DIM**0.5
    p = scores.softmax(-1)
    return torch.einsum('bhk,bkhd->bhd', p, vv)


def metrics(out, ref):
    diff = (out.float() - ref.float()).abs()
    p = out.float().softmax(-1).clamp_min(1e-12)
    r = ref.float().softmax(-1).clamp_min(1e-12)
    kl = (r * (r / p).log()).sum(-1).mean().item()
    argmatch = (out.float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
    return diff.max().item(), kl, argmatch


RESULTS = {}


def record(key, **kw):
    RESULTS.setdefault(key, []).append(kw)
    print(f'[record] {key}: {kw}', flush=True)


def gate(key, out, ref, label, budget=MAX_ABS_BUDGET):
    assert not out.isnan().any().item(), f'{label}: NaN in output'
    max_abs, kl, argmatch = metrics(out, ref)
    ok = max_abs <= budget and kl <= 1e-2
    record(key, label=label, max_abs=round(max_abs, 6), kl=f'{kl:.3e}', argmatch=argmatch, ok=ok)
    assert ok, f'{label}: max_abs={max_abs:.4f} kl={kl:.3e} exceeds budget'
    return max_abs


def save_results(name):
    path = os.path.join(_HERE, name)
    with open(path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f'[results] -> {path}', flush=True)


def bridge_fill(k_slice, v_slice, kv_seqlens, pool, layer=0):
    """One TM write-kernel append (token-major slices), as in the S1 test."""
    from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table
    from lmdeploy.turbomind import _tm

    k_cache, v_cache, k_sz, v_sz = layer_views(pool, layer)
    del v_cache, k_sz, v_sz
    pool_base, block_bytes = k_cache.data_ptr(), k_cache.stride(0)
    num_kv_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    ptrs, cu_block_base = tm_block_pointer_table(block_offsets(), pool_base, block_bytes)
    num_tokens = k_slice.shape[0]
    q_len = num_tokens // BATCH
    cu_q = cu([q_len] * BATCH)
    cu_k = cu(kv_seqlens)
    _tm.tm_process_kv(
        _tm.from_dlpack(k_slice.contiguous()),
        _tm.from_dlpack(v_slice.contiguous()),
        ptrs.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(), cu_block_base.data_ptr(),
        stride_c=num_kv_heads, stride_s=num_kv_heads, stride_h=1,
        max_q_len=q_len, num_kv_heads=num_kv_heads, head_dim=head_dim,
        batch_size=BATCH, block_len=BLOCK_LEN, kv_quant=8,
        stream_ptr=torch.cuda.current_stream().cuda_stream)


def writer_append(k_new, v_new, kv_seqlens, pool, layer=0):
    """ATen masked-scatter append of 1 token/sequence (capture-safe writer,
    used eagerly between replays).

    k_new/v_new: (B, Hkv, D) or (1, B, Hkv, D).
    """
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm

    if k_new.dim() == 4:
        k_new = k_new.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
        v_new = v_new.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    k_cache, v_cache, k_sz, v_sz = layer_views(pool, layer)
    q_loc = torch.arange(BATCH, device='cuda', dtype=torch.long)
    q_len = torch.ones(BATCH, dtype=torch.long, device='cuda')
    kv = torch.full((BATCH, ), kv_seqlens, dtype=torch.long, device='cuda')
    fill_kv_cache_tm(k_new, v_new, k_cache, v_cache, k_sz, v_sz,
                     q_loc, q_len, kv, block_offsets())


def writer_overwrite(k_new, v_new, kv_seqlens, q_len, pool, layer=0):
    """Overwrite the slots [(kv_seqlens - q_len), kv_seqlens) per sequence."""
    from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm

    k_new = k_new.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    v_new = v_new.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    k_cache, v_cache, k_sz, v_sz = layer_views(pool, layer)
    q_loc = torch.arange(BATCH, device='cuda', dtype=torch.long) * q_len
    q_seqlens = torch.full((BATCH, ), q_len, dtype=torch.long, device='cuda')
    kv = torch.full((BATCH, ), kv_seqlens, dtype=torch.long, device='cuda')
    fill_kv_cache_tm(k_new, v_new, k_cache, v_cache, k_sz, v_sz,
                     q_loc, q_seqlens, kv, block_offsets())


def pointer_tables(k_cache):
    from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table
    return tm_block_pointer_table(block_offsets(), k_cache.data_ptr(), k_cache.stride(0))


def tm_decode(q2d, k2d, v2d, out2d, ptrs, cu_base, max_k_len):
    """One TM int8 decode dispatch on caller-owned buffers.

    No sync (must be legal inside CUDA graph capture). Buffers must be 2-D contiguous bf16.
    """
    from lmdeploy.turbomind import _tm

    _tm.tm_attn_decode(
        _tm.from_dlpack(q2d),
        _tm.from_dlpack(k2d),
        _tm.from_dlpack(v2d),
        _tm.from_dlpack(out2d),
        ptrs.data_ptr(), cu_base.data_ptr(),
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        batch_size=BATCH, block_len=BLOCK_LEN, max_k_len=max_k_len,
        kv_quant=8, stream_ptr=torch.cuda.current_stream().cuda_stream)


def rows2d(t):
    """(B, Hkv, D) -> (B, Hkv*D) contiguous."""
    return t.reshape(BATCH, -1).contiguous()


def q2d(t):
    return t.reshape(BATCH, -1).contiguous()
