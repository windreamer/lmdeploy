# Copyright (c) OpenMMLab. All rights reserved.
"""2x2 region probe: codes arrangement x params arrangement.

Out[h] = 200 * (the scale the kernel reads for head h's V) whenever codes
are constant. Tags: K slot h scale=(10+h)/200, V slot h scale=(100+h)/200.
"""

import torch

from lmdeploy.turbomind import _tm

S, D, H = 64, 128, 4
HQ = 4
B, KLEN = 1, 64
CS = S * D      # 8192 per-head codes
PS = S * 2 * 2  # 256 per-head params
KD, VD, KP, VP = H * CS, H * CS, H * PS, H * PS
BYTES = KD + VD + KP + VP


def codes_buf(tag):
    c = torch.zeros(CS, dtype=torch.uint8, device='cuda')
    c.fill_(200)
    return c


def params_buf(tag):
    p = torch.zeros(PS, dtype=torch.uint8, device='cuda')
    p.view(torch.bfloat16).view(S, 2)[:, 0] = tag / 200.0
    p.view(torch.bfloat16).view(S, 2)[:, 1] = 0.0
    return p


def build(codes_mode, params_mode):
    pool = torch.zeros(B, BYTES, dtype=torch.uint8, device='cuda')
    for h in range(H):
        kc, vc = codes_buf(10 + h), codes_buf(100 + h)
        kp, vp = params_buf(10 + h), params_buf(100 + h)
        if codes_mode == 'flat':
            pool[:, h * CS:(h + 1) * CS] = kc
            pool[:, KD + h * CS:KD + (h + 1) * CS] = vc
        else:  # interleaved: per head [K_h | V_h]
            pool[:, h * 2 * CS:h * 2 * CS + CS] = kc
            pool[:, h * 2 * CS + CS:(h + 1) * 2 * CS] = vc
        if params_mode == 'flat':
            pool[:, KD + VD + h * PS:KD + VD + (h + 1) * PS] = kp
            pool[:, KD + VD + KP + h * PS:KD + VD + KP + (h + 1) * PS] = vp
        else:  # interleaved: per head [Kp_h | Vp_h]
            base = KD + VD + h * 2 * PS
            pool[:, base:base + PS] = kp
            pool[:, base + PS:base + 2 * PS] = vp
    return pool


q = (torch.randn(B, HQ, D) * 0.1).cuda().bfloat16()
ptrs = torch.empty(1, dtype=torch.int64, device='cuda')
cu = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')

stream = torch.cuda.current_stream().cuda_stream
ctx = _tm.create_device_context(stream)
with ctx:
    for codes_mode in ('flat', 'interleaved'):
        for params_mode in ('flat', 'interleaved'):
            pool = build(codes_mode, params_mode)
            ptrs[0] = pool.data_ptr()
            _tm.attn_decode_bench(_tm.from_dlpack(q), _tm.from_dlpack(out),
                                  ptrs.data_ptr(), cu.data_ptr(),
                                  num_heads=HQ, num_kv_heads=H, head_dim=D,
                                  batch_size=B, block_len=S, max_k_len=KLEN,
                                  kv_quant=8, iters=1, warmup=0,
                                  stream_ptr=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            o = out.view(B, HQ, D).float()[0].mean(-1)
            print(f'codes={codes_mode:11s} params={params_mode:11s} '
                  f'out=', [round(v.item(), 2) for v in o])
