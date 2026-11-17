# Copyright (c) OpenMMLab. All rights reserved.
"""[B3-04 repro] Empirical reproduction of the decode last-slot KV wipe on the
UNFIXED `_turbomind` build (old `tm_attn_decode` API, no k/v args).

Mechanism under test: `attention_universal.h` `kProcessKV` is compile-time true
for DecodingCtaMap, so every decode dispatch re-stores `convert(params.k/v)`
into the block pool at `ti = history_len` (the last valid position). The
unfixed bench binding points `params.k/v` at a Q-only fused-qkv workspace
(K/V sections uninitialized), and passes `readonly_block_num = nullptr`, so
the store executes with garbage. Expected: every sequence's last valid position
(codes AND quant params) gets overwritten; other positions are untouched.

Exit code 0 = defect reproduced (last slot rewritten), 1 = not reproduced.

Run on the pre-fix tree ONLY (post-fix the API requires k/v and this script
fails at the call, which is itself the contract fix taking effect).
"""

import sys

import torch

from lmdeploy.turbomind import _tm

S, D, H, HQ = 64, 64, 2, 4  # block_len, head_dim, kv_heads, q_heads
B, KLEN = 4, 100
BPS = (KLEN + S - 1) // S  # 2
NB = B * BPS

gen = torch.Generator(device='cpu').manual_seed(11)
# Non-zero, dequant-consistent pool content so any rewrite is visible.
codes_k = torch.randint(1, 255, (NB, H, S, D), generator=gen, dtype=torch.int64).to(torch.uint8)
codes_v = torch.randint(1, 255, (NB, H, S, D), generator=gen, dtype=torch.int64).to(torch.uint8)
scale = (torch.rand(NB, H, S, 1, generator=gen) * 0.05 + 0.01).bfloat16()
vmin = (torch.rand(NB, H, S, 1, generator=gen) * 2 - 1).bfloat16()

data_bytes = H * S * D
param_bytes = H * S * 4
pool = torch.empty(NB, 2 * data_bytes + 2 * param_bytes, dtype=torch.uint8)
kd = pool[:, :data_bytes].view(NB, H, S, D)
vd = pool[:, data_bytes:2 * data_bytes].view(NB, H, S, D)
kp = pool[:, 2 * data_bytes:2 * data_bytes + param_bytes].view(torch.bfloat16).view(NB, H, S, 2)
vp = pool[:, 2 * data_bytes + param_bytes:].view(torch.bfloat16).view(NB, H, S, 2)
kd.copy_(codes_k)
vd.copy_(codes_v)
kp[..., 0] = scale.squeeze(-1)
kp[..., 1] = vmin.squeeze(-1)
vp[..., 0] = scale.squeeze(-1).clone()
vp[..., 1] = vmin.squeeze(-1).clone()
pool = pool.cuda()

q = (torch.randn(B, HQ, D, generator=gen) * 0.5).cuda().bfloat16()
ptrs = torch.tensor([pool.data_ptr() + i * pool.shape[1] for i in range(NB)],
                    dtype=torch.int64, device='cuda')
cu_block_nums = torch.arange(0, (B + 1) * BPS, BPS, dtype=torch.int32, device='cuda')

LAST = KLEN - 1          # 99: last valid position per sequence
MID = 50                 # interior position, must stay untouched
blk = lambda b, p: b * BPS + p // S
off = lambda p: p % S


def snapshot(p):
    """Codes+params at position p for every sequence -> cpu clones."""
    ks = torch.stack([pool[blk(b, p), :data_bytes].view(H, S, D)[:, off(p)] for b in range(B)])
    vs = torch.stack([pool[blk(b, p), data_bytes:2 * data_bytes].view(H, S, D)[:, off(p)] for b in range(B)])
    kp_ = torch.stack([pool[blk(b, p), 2 * data_bytes:2 * data_bytes + param_bytes]
                       .view(torch.bfloat16).view(H, S, 2)[:, off(p)] for b in range(B)])
    vp_ = torch.stack([pool[blk(b, p), 2 * data_bytes + param_bytes:]
                       .view(torch.bfloat16).view(H, S, 2)[:, off(p)] for b in range(B)])
    return ks, vs, kp_, vp_


before_last = snapshot(LAST)
before_mid = snapshot(MID)

out = torch.empty(B, HQ * D, dtype=torch.bfloat16, device='cuda')
_tm.tm_attn_decode(_tm.from_dlpack(q.reshape(B, -1).contiguous()), _tm.from_dlpack(out),
                   ptrs.data_ptr(), cu_block_nums.data_ptr(), num_heads=HQ, num_kv_heads=H,
                   head_dim=D, batch_size=B, block_len=S, max_k_len=KLEN,
                   kv_quant=8, stream_ptr=torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()

after_last = snapshot(LAST)
after_mid = snapshot(MID)


def diff(name, a, b):
    ks, vs, kp_, vp_ = a
    ks2, vs2, kp2, vp2 = b
    n = int((ks != ks2).sum() + (vs != vs2).sum())
    p = int((kp_ != kp2).sum() + (vp_ != vp2).sum())
    print(f'{name}: code bytes changed = {n}/{ks.numel() * 2}, param values changed = {p}/{kp_.numel() * 2}')
    return n, p


n_last, p_last = diff('last slot  (ti=99) ', before_last, after_last)
n_mid, p_mid = diff('mid slot   (ti=50) ', before_mid, after_mid)

zeroed = (int(after_last[0].sum()) == 0 and int(after_last[1].sum()) == 0
          and float(after_last[2].float().abs().sum()) == 0.0
          and float(after_last[3].float().abs().sum()) == 0.0) if p_last else False
print(f'last slot fully zeroed after call: {zeroed}')

reproduced = (n_last > 0 or p_last > 0) and n_mid == 0 and p_mid == 0
print('DEFECT REPRODUCED' if reproduced else 'NOT REPRODUCED')
sys.exit(0 if reproduced else 1)
