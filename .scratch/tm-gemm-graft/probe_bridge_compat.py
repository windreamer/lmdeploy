# Copyright (c) OpenMMLab. All rights reserved.
"""Bridge compatibility probe: write a pool with the new `_tm.tm_process_kv`
(invokeProcessKV_v2) and read it back with the TM decode binding, against a
bf16 reference. Decides the B3-04 write path with hard evidence."""

import torch

from lmdeploy.turbomind import _tm

S, D, H = 64, 128, 8  # kv heads
HQ = 16
B, KLEN = 2, 128
BPS = KLEN // S
DATA = H * S * D          # 65536 bytes per K (or V) region
PARAM = H * S * 2 * 2     # 4096 bytes per K (or V) param region
BYTES = 2 * DATA + 2 * PARAM

gen = torch.Generator(device='cpu').manual_seed(11)
q = (torch.randn(B, HQ, D, generator=gen) * 0.5).cuda().bfloat16()
k = (torch.randn(B, KLEN, H, D, generator=gen) * 0.5).cuda().bfloat16()
v = (torch.randn(B, KLEN, H, D, generator=gen) * 0.5).cuda().bfloat16()

nb = B * BPS
pool = torch.zeros(nb, BYTES, dtype=torch.uint8, device='cuda')
ptrs = torch.tensor([pool.data_ptr() + i * BYTES for i in range(nb)], dtype=torch.int64, device='cuda')

# full-context prefill append: cu_q = cu_k = [0, KLEN, 2*KLEN]
cu_q = torch.arange(0, (B + 1) * KLEN, KLEN, dtype=torch.int32, device='cuda')
cu_blk = torch.arange(0, (B + 1) * BPS, BPS, dtype=torch.int32, device='cuda')

stream = torch.cuda.current_stream().cuda_stream
ctx = _tm.create_device_context(stream)
with ctx:
    _tm.tm_process_kv(_tm.from_dlpack(k), _tm.from_dlpack(v),
                      ptrs.data_ptr(), cu_q.data_ptr(), cu_q.data_ptr(), cu_blk.data_ptr(),
                      stride_c=H, stride_s=H, stride_h=1,
                      max_q_len=KLEN, num_kv_heads=H, head_dim=D,
                      batch_size=B, block_len=S, kv_quant=8,
                      stream_ptr=stream)
    torch.cuda.synchronize()

# ---- readers
def dequant_flat(pool):
    nb = pool.shape[0]
    kd = pool[:, :DATA].view(nb, H, S, D).float()
    vd = pool[:, DATA:2 * DATA].view(nb, H, S, D).float()
    kp = pool[:, 2 * DATA:2 * DATA + PARAM].view(torch.bfloat16).view(nb, H, S, 2).float()
    vp = pool[:, 2 * DATA + PARAM:].view(torch.bfloat16).view(nb, H, S, 2).float()
    kq = (kd * kp[..., 0:1] + kp[..., 1:2]).permute(0, 2, 1, 3).reshape(B, KLEN, H, D)
    vq = (vd * vp[..., 0:1] + vp[..., 1:2]).permute(0, 2, 1, 3).reshape(B, KLEN, H, D)
    return kq, vq

def reference(q, k, v):
    group = HQ // H
    kk = k.float().repeat_interleave(group, dim=2)
    vv = v.float().repeat_interleave(group, dim=2)
    scores = torch.einsum('bhd,bkhd->bhk', q.float(), kk) / D**0.5
    p = scores.softmax(-1)
    return torch.einsum('bhk,bkhd->bhd', p, vv)

with ctx:
    out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')
    _tm.tm_attn_decode(_tm.from_dlpack(q), _tm.from_dlpack(out),
                       ptrs.data_ptr(), cu_blk.data_ptr(),
                       num_heads=HQ, num_kv_heads=H, head_dim=D,
                       batch_size=B, block_len=S, max_k_len=KLEN,
                       kv_quant=8, stream_ptr=stream)
    torch.cuda.synchronize()

ref = reference(q, k, v)
o = out.view(B, HQ, D).float()
print('ProcessKV-written pool -> TM decode vs bf16 reference:')
print('  max_abs =', (o - ref).abs().max().item())
kq, vq = dequant_flat(pool)
ref_q = reference(q, kq, vq)
print('  max_abs vs flat-dequant =', (o - ref_q).abs().max().item())

# where did the params land? compare pool param region against host-computed
# (scale, min) at flat positions
xf = k.float().reshape(B, BPS, S, H, D).permute(0, 1, 3, 2, 4)  # b, bp, h, s, d
lo = xf.amin(-1)
scale = ((xf.amax(-1) - lo) / 255.0)
kp_host = torch.stack([scale, lo], -1).bfloat16()  # b, bp, h, s, 2
kp_pool = pool[:, 2 * DATA:2 * DATA + PARAM].view(torch.bfloat16).view(B, BPS, H, S, 2)
kp_pool_intl = pool[:, 2 * DATA:].view(torch.bfloat16).view(B, BPS, H, 2, S, 2)[:, :, :, 0]
print('  params match flat arrangement:     ', bool((kp_pool[0] != 0).any().item()),
      '| nonzero flat param slots:', int((kp_pool.abs().sum(-1) > 0).sum().item()), '/', kp_pool[0].numel() // 2)
print('  params match interleaved arrangement:', int((kp_pool_intl.abs().sum(-1) > 0).sum().item()))
