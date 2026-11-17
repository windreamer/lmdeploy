# Copyright (c) OpenMMLab. All rights reserved.
"""Small compute-sanitizer entry for the B3-04 int8-KV path.

Exercises both writers into one pool (TM write kernel prefill append, then a triton-free ATen decode-step append), the
TM decode dispatch, and the Triton paged-attention reader on the same pool. Shapes are minimal so the sanitizer overhead
stays tolerable.
"""

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm
from lmdeploy.turbomind import _tm

S, D, H = 64, 128, 4
HQ = 8
B, PREFILL, KLEN = 1, 128, 129 + 1  # one decode step past a non-aligned length
BPS = KLEN // S + 1
NB = B * BPS
HSD = H * S * D
BYTES = 2 * HSD + 2 * H * S * 2 * 2

gen = torch.Generator(device='cpu').manual_seed(3)
q = (torch.randn(B, HQ, D, generator=gen) * 0.5).cuda().bfloat16()
k = (torch.randn(B, KLEN, H, D, generator=gen) * 0.5).cuda().bfloat16()
v = (torch.randn(B, KLEN, H, D, generator=gen) * 0.5).cuda().bfloat16()

pool = torch.zeros(NB, BYTES, dtype=torch.uint8, device='cuda')
ptrs = torch.tensor([pool.data_ptr() + i * BYTES for i in range(NB)], dtype=torch.int64, device='cuda')
cu_blk = torch.arange(0, (B + 1) * BPS, BPS, dtype=torch.int32, device='cuda')

stream = torch.cuda.current_stream().cuda_stream
ctx = _tm.create_device_context(stream)

def cu(vals):
    return torch.nn.functional.pad(torch.cumsum(torch.tensor(vals, dtype=torch.int32), 0), (1, 0)).cuda()

def min_to_zero(p):
    scale, mn = p[..., 0], p[..., 1]
    zero = torch.where(scale > 0, -mn / scale.clamp(min=1e-6), torch.zeros_like(mn))
    return torch.stack((scale, zero), -1).contiguous()

def readers(klen):
    out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')
    with ctx:
        _tm.tm_attn_decode(_tm.from_dlpack(q.reshape(B, -1).contiguous()), _tm.from_dlpack(out),
                           ptrs.data_ptr(), cu_blk.data_ptr(),
                           num_heads=HQ, num_kv_heads=H, head_dim=D,
                           batch_size=B, block_len=S, max_k_len=klen,
                           kv_quant=8, stream_ptr=stream)
    torch.cuda.synchronize()
    tm = out.view(B, HQ, D).float()

    codes = pool[:, :2 * HSD].view(NB, H, 2, S, D)
    params = pool[:, 2 * HSD:].view(torch.bfloat16).view(NB, H, 2, S, 2)
    from lmdeploy.pytorch.kernels.cuda.pagedattention import flash_attn_with_kvcache
    pt = flash_attn_with_kvcache(
        q, codes[:, :, 0].permute(0, 2, 1, 3), codes[:, :, 1].permute(0, 2, 1, 3),
        torch.full((B, ), klen, dtype=torch.int32, device='cuda'),
        torch.arange(NB, device='cuda').int().reshape(B, BPS),
        k_scales_zeros=min_to_zero(params[:, :, 0].permute(0, 2, 1, 3)),
        v_scales_zeros=min_to_zero(params[:, :, 1].permute(0, 2, 1, 3)),
        quant_policy=QuantPolicy.INT8, kv_layout='bshd').float()
    print(f'klen={klen}: tm max_abs={tm.abs().max().item():.3f} pt max_abs={pt.abs().max().item():.3f} '
          f'cross={(tm - pt).abs().max().item():.4f}')

# writer 1: TM write kernel, prefill-shaped append (128 tokens)
with ctx:
    _tm.tm_process_kv(_tm.from_dlpack(k[:, :PREFILL].contiguous()), _tm.from_dlpack(v[:, :PREFILL].contiguous()),
                      ptrs.data_ptr(), cu([PREFILL]).data_ptr(), cu([PREFILL]).data_ptr(), cu_blk.data_ptr(),
                      stride_c=H, stride_s=H, stride_h=1,
                      max_q_len=PREFILL, num_kv_heads=H, head_dim=D,
                      batch_size=B, block_len=S, kv_quant=8, stream_ptr=stream)
torch.cuda.synchronize()
readers(PREFILL)

# writer 2: triton-free ATen convert+scatter, decode-step append (1 token)

codes_v = pool[:, :2 * HSD].view(NB, H, 2, S, D)
params_v = pool[:, 2 * HSD:].view(torch.bfloat16).view(NB, H, 2, S, 2)
k_codes = codes_v[:, :, 0].permute(0, 2, 1, 3)
v_codes = codes_v[:, :, 1].permute(0, 2, 1, 3)
k_params = params_v[:, :, 0].permute(0, 2, 1, 3)
v_params = params_v[:, :, 1].permute(0, 2, 1, 3)
fill_kv_cache_tm(k[:, PREFILL:KLEN].reshape(-1, H, D), v[:, PREFILL:KLEN].reshape(-1, H, D),
                 k_codes, v_codes, k_params, v_params,
                 torch.zeros(B, dtype=torch.long, device='cuda'),
                 torch.tensor([KLEN - PREFILL], dtype=torch.long, device='cuda'),
                 torch.tensor([KLEN], dtype=torch.long, device='cuda'),
                 torch.arange(NB, device='cuda').int().reshape(B, BPS))
torch.cuda.synchronize()
readers(KLEN)
print('SANITIZER ENTRY DONE')
