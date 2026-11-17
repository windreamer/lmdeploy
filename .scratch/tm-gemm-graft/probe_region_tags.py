# Copyright (c) OpenMMLab. All rights reserved.
"""Region-tag probe: decode output reveals which byte regions the TM int8
decode kernel reads as head h's K/V.

Every head slot stores one constant value: codes=200, min=0, scale=tag/200
-> dequant = tag everywhere in that slot. Attention over constant scores
returns the mean of V tags, so out[h] identifies the V slot the kernel used.
"""

import torch

from lmdeploy.turbomind import _tm

S, D, H = 64, 128, 4
HQ = 4
B, KLEN = 1, 64
CS = S * D      # per-head code slot bytes: 8192
PS = S * 2 * 2  # per-head param slot bytes: 256
KD, VD, KP, VP = H * CS, H * CS, H * PS, H * PS  # flat region sizes
BYTES = KD + VD + KP + VP

pool = torch.zeros(B, BYTES, dtype=torch.uint8, device='cuda')


def fill_slot(tag):
    """One head slot: codes=200 plus (scale=tag/200, min=0)."""
    codes = torch.zeros(CS, dtype=torch.uint8, device='cuda')
    params = torch.zeros(PS, dtype=torch.uint8, device='cuda')
    codes.fill_(200)
    params.view(torch.bfloat16).view(S, 2)[:, 0] = tag / 200.0
    params.view(torch.bfloat16).view(S, 2)[:, 1] = 0.0
    return codes, params


for h in range(H):
    kc, kp = fill_slot(10 + h)
    vc, vp = fill_slot(100 + h)
    # flat writer: all K codes, then all V codes, then all K params, then
    # all V params
    pool[:, h * CS:(h + 1) * CS] = kc
    pool[:, KD + h * CS:KD + (h + 1) * CS] = vc
    pool[:, KD + VD + h * PS:KD + VD + (h + 1) * PS] = kp
    pool[:, KD + VD + KP + h * PS:KD + VD + KP + (h + 1) * PS] = vp

q = (torch.randn(B, HQ, D) * 0.1).cuda().bfloat16()
ptrs = torch.tensor([pool.data_ptr()], dtype=torch.int64, device='cuda')
cu = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')

stream = torch.cuda.current_stream().cuda_stream
ctx = _tm.create_device_context(stream)
with ctx:
    _tm.attn_decode_bench(_tm.from_dlpack(q), _tm.from_dlpack(out),
                          ptrs.data_ptr(), cu.data_ptr(),
                          num_heads=HQ, num_kv_heads=H, head_dim=D,
                          batch_size=B, block_len=S, max_k_len=KLEN,
                          kv_quant=8, iters=1, warmup=0,
                          stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

o = out.view(B, HQ, D).float()
print('out per q head (should equal the V tag where the kernel found V):')
for h in range(HQ):
    print(f'  head {h}: mean={o[0, h].mean().item():9.3f} '
          f'min={o[0, h].min().item():9.3f} max={o[0, h].max().item():9.3f}')
print('V tags by flat hypothesis:     [100, 101, 102, 103]')
print('K tags (if V read from K zone): [10..13]')
