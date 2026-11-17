# Copyright (c) OpenMMLab. All rights reserved.
"""[DEBUG-b304] Multi-batch compute-sanitizer entry for the B3-04 writer path.

Exercises, on ONE pool with B=4: the TM write-kernel prefill append (bridge),
two triton-free ATen decode-step appends with correctly packed feeds, then the
TM decode and the Triton paged-attention reader. Minimal shapes so the
sanitizer overhead stays tolerable. Memory-safety entry only — numeric gates
live in tests/pytorch/kernel/test_tm_int8_kv.py.
"""

import torch

from lmdeploy.pytorch.backends.tm_kv import (
    TmInt8BlockCacheLayout,
    fill_kv_cache_tm,
    tm_block_pointer_table,
)
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

S, D, H, HQ = 64, 64, 2, 4
B, NL = 4, 1
PREFILL, STEPS = 100, 2
KLEN = PREFILL + STEPS
BPS = 3  # block-table columns (padded past ceil(102/64)=2)
NB = B * BPS

gen = torch.Generator(device='cpu').manual_seed(5)
q = (torch.randn(B, HQ, D, generator=gen) * 0.5).cuda().bfloat16()
k = (torch.randn(B, KLEN, H, D, generator=gen) * 0.5).cuda().bfloat16()
v = (torch.randn(B, KLEN, H, D, generator=gen) * 0.5).cuda().bfloat16()

kv_shape = [S, H, D]
specs = (CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
         CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
         CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.bfloat16)),
         CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.bfloat16)))
alloc = TmInt8BlockCacheLayout(specs, num_layers=NL).allocate(NB, 'cuda')
pool = alloc.pools[0].tensor
k_cache, v_cache, k_sz, v_sz = tuple(x[0] for x in alloc.tensor_views)

boff = (torch.arange(B, device='cuda')[:, None] * BPS + torch.arange(BPS, device='cuda')[None, :]).int()
ptrs, cu_block_base = tm_block_pointer_table(boff, k_cache.data_ptr(), k_cache.stride(0))


def cu(vals):
    return torch.nn.functional.pad(torch.cumsum(torch.tensor(vals, dtype=torch.int32), 0), (1, 0)).cuda()


# writer 1: TM write kernel (bridge), packed prefill rows
from lmdeploy.turbomind import _tm

kt = k[:, :PREFILL].reshape(-1, H, D)
vt = v[:, :PREFILL].reshape(-1, H, D)
_tm.tm_process_kv(_tm.from_dlpack(kt.contiguous()), _tm.from_dlpack(vt.contiguous()),
                  ptrs.data_ptr(), cu([PREFILL] * B).data_ptr(), cu([PREFILL] * B).data_ptr(),
                  cu_block_base.data_ptr(), stride_c=H, stride_s=H, stride_h=1,
                  max_q_len=PREFILL, num_kv_heads=H, head_dim=D,
                  batch_size=B, block_len=S, kv_quant=8,
                  stream_ptr=torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

# writer 2: triton-free ATen convert+scatter, two decode-step appends
for step in range(STEPS):
    fill_kv_cache_tm(k[:, PREFILL + step], v[:, PREFILL + step],
                     k_cache, v_cache, k_sz, v_sz,
                     cu([1] * B)[:-1].long(),
                     torch.ones(B, dtype=torch.long, device='cuda'),
                     torch.full((B, ), PREFILL + step + 1, dtype=torch.long, device='cuda'),
                     boff)
    torch.cuda.synchronize()

# reader 1: TM decode (note: exercises the known kernel-side last-slot
# zero-write; in-bounds, so sanitizer-clean but numerically destructive)
out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')
_tm.tm_attn_decode(_tm.from_dlpack(q.reshape(B, -1).contiguous()), _tm.from_dlpack(out),
                   ptrs.data_ptr(), cu([BPS] * B).data_ptr(), num_heads=HQ, num_kv_heads=H,
                   head_dim=D, batch_size=B, block_len=S, max_k_len=KLEN,
                   kv_quant=8, stream_ptr=torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

# reader 2: stock Triton paged attention over the strided views
from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.kernels.cuda.pagedattention import flash_attn_with_kvcache

codes = pool[0, :, :2 * H * S * D].view(NB, H, 2, S, D)
params = pool[0, :, 2 * H * S * D:].view(torch.bfloat16).view(NB, H, 2, S, 2)


def min_to_zero(p):
    scale, mn = p[..., 0], p[..., 1]
    zero = torch.where(scale > 0, -mn / scale.clamp(min=1e-6), torch.zeros_like(mn))
    return torch.stack((scale, zero), -1).contiguous()


flash_attn_with_kvcache(q, codes[:, :, 0].permute(0, 2, 1, 3), codes[:, :, 1].permute(0, 2, 1, 3),
                        torch.full((B, ), KLEN, dtype=torch.int32, device='cuda'), boff,
                        k_scales_zeros=min_to_zero(params[:, :, 0].permute(0, 2, 1, 3)),
                        v_scales_zeros=min_to_zero(params[:, :, 1].permute(0, 2, 1, 3)),
                        quant_policy=QuantPolicy.INT8, kv_layout='bshd')
torch.cuda.synchronize()
print('SANITIZER ENTRY B304 DONE')
