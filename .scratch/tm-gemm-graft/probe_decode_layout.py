# Copyright (c) OpenMMLab. All rights reserved.
"""Empirical probe: which byte layout does the TM int8 decode kernel read?

Two candidate layouts for one physical block (H kv heads, S tokens, D dims):
- flat:      [K all heads][V all heads][Kp all][Vp all]  (B3-01 prototype)
- interleaved: per head [K_h][V_h], params per head [Kp_h][Vp_h]  (block.h
  L(H2SDQ+H2S2T): v_data = k_data + S*D, v_param = k_param + S*2)
"""

import torch

from lmdeploy.turbomind import _tm

BLOCK_LEN = 64
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
BSZ = 2
KLEN = 128  # two blocks per sequence

SD = NUM_KV_HEADS * BLOCK_LEN * HEAD_DIM  # 65536
PSP = NUM_KV_HEADS * BLOCK_LEN * 2 * 2  # params bytes per kv tensor


def make_data(seed=7):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    q = (torch.randn(BSZ, NUM_HEADS, HEAD_DIM, generator=gen) * 0.5).cuda().bfloat16()
    k = (torch.randn(BSZ, KLEN, NUM_KV_HEADS, HEAD_DIM, generator=gen) * 0.5).cuda().bfloat16()
    v = (torch.randn(BSZ, KLEN, NUM_KV_HEADS, HEAD_DIM, generator=gen) * 0.5).cuda().bfloat16()
    return q, k, v


def write_pool(pool, kind, k, v):
    """Quantize per token per head and scatter into `kind` layout."""
    pool.fill_(0)
    nb = pool.shape[0]
    kd = pool[:, :SD].view(nb, NUM_KV_HEADS, BLOCK_LEN, HEAD_DIM)
    vd = pool[:, SD:2 * SD].view(nb, NUM_KV_HEADS, BLOCK_LEN, HEAD_DIM)
    kp = pool[:, 2 * SD:2 * SD + PSP].view(torch.bfloat16).view(nb, NUM_KV_HEADS, BLOCK_LEN, 2)
    vp = pool[:, 2 * SD + PSP:].view(torch.bfloat16).view(nb, NUM_KV_HEADS, BLOCK_LEN, 2)
    bps = KLEN // BLOCK_LEN
    kd = kd.view(BSZ, bps, NUM_KV_HEADS, BLOCK_LEN, HEAD_DIM)
    vd = vd.view(BSZ, bps, NUM_KV_HEADS, BLOCK_LEN, HEAD_DIM)
    kp = kp.view(BSZ, bps, NUM_KV_HEADS, BLOCK_LEN, 2)
    vp = vp.view(BSZ, bps, NUM_KV_HEADS, BLOCK_LEN, 2)
    seg_k = k.reshape(BSZ, bps, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4)
    seg_v = v.reshape(BSZ, bps, BLOCK_LEN, NUM_KV_HEADS, HEAD_DIM).permute(0, 1, 3, 2, 4)
    for seg, cache, params in ((seg_k, kd, kp), (seg_v, vd, vp)):
        xf = seg.float()
        lo = xf.amin(-1, keepdim=True)
        scale = ((xf.amax(-1, keepdim=True) - lo) / 255.0).clamp(min=1e-6)
        cache.copy_(((xf - lo) / scale).round().clamp(0, 255).to(torch.uint8))
        params[..., 0] = scale.squeeze(-1).bfloat16()
        params[..., 1] = lo.squeeze(-1).bfloat16()
    if kind == 'interleaved':
        # head-major groups of [K_h][V_h] codes and [Kp_h][Vp_h] params
        newpool = torch.zeros_like(pool)
        nk = newpool[:, :2 * SD].view(BSZ, bps, NUM_KV_HEADS, 2, BLOCK_LEN, HEAD_DIM)
        nk[:, :, :, 0] = seg_k
        nk[:, :, :, 1] = seg_v
        np_ = (newpool[:, 2 * SD:].view(torch.bfloat16).view(BSZ, bps, NUM_KV_HEADS, 2, BLOCK_LEN, 2))
        np_[:, :, :, 0] = kp
        np_[:, :, :, 1] = vp
        return newpool
    return pool


def decode(q, pool):
    ptrs = torch.tensor([pool.data_ptr() + i * pool.stride(0) for i in range(pool.shape[0])],
                        dtype=torch.int64, device='cuda')
    bps = KLEN // BLOCK_LEN
    cu = torch.arange(0, (BSZ + 1) * bps, bps, dtype=torch.int32, device='cuda')
    out = torch.empty(BSZ, NUM_HEADS * HEAD_DIM, dtype=q.dtype, device='cuda')
    _tm.attn_decode_bench(
        _tm.from_dlpack(q), _tm.from_dlpack(out),
        ptrs.data_ptr(), cu.data_ptr(),
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        batch_size=BSZ, block_len=BLOCK_LEN, max_k_len=KLEN,
        kv_quant=8, iters=1, warmup=0, stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out.view(BSZ, NUM_HEADS, HEAD_DIM)


def reference(q, k, v):
    group = NUM_HEADS // NUM_KV_HEADS
    kk = k.float().repeat_interleave(group, dim=2)
    vv = v.float().repeat_interleave(group, dim=2)
    scores = torch.einsum('bhd,bkhd->bhk', q.float(), kk) / HEAD_DIM**0.5
    p = scores.softmax(-1)
    return torch.einsum('bhk,bkhd->bhd', p, vv)


def metrics(out, ref):
    diff = (out.float() - ref.float()).abs().max().item()
    p = out.float().softmax(-1).clamp_min(1e-12)
    r = ref.float().softmax(-1).clamp_min(1e-12)
    kl = (r * (r / p).log()).sum(-1).mean().item()
    return diff, kl


def main():
    q, k, v = make_data()
    ref = reference(q, k, v)
    nb = BSZ * (KLEN // BLOCK_LEN)
    stream = torch.cuda.current_stream().cuda_stream
    ctx = _tm.create_device_context(stream)
    with ctx:
        for kind in ('flat', 'interleaved'):
            pool = torch.zeros(nb, 2 * SD + 2 * PSP, dtype=torch.uint8, device='cuda')
            pool = write_pool(pool, kind, k, v)
            out = decode(q, pool)
            ma, kl = metrics(out, ref)
            print(f'{kind:12s} max_abs={ma:.4f} kl={kl:.3e}')


if __name__ == '__main__':
    main()
