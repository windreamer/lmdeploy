# [DEBUG-b304b] what bytes land at corrupted prefill positions?
import sys; sys.path.insert(0, '../../tests/pytorch/kernel')
from test_tm_int8_kv import (_make_data, _allocate_layout, _bridge_fill, _tm_decode,
                             _layer_views, _block_offsets, BATCH, PREFILL_LEN, KLEN,
                             NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_LEN)
import torch
from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm, quant_kv_int8

_, k, v = _make_data()
q, _, _ = _make_data(seed=7)
pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
pool_t = pool.pools[0].tensor
kt = k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
vt = v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
_bridge_fill(kt, vt, [PREFILL_LEN] * BATCH, pool)
kv_now = PREFILL_LEN + 1
fill_kv_cache_tm(k[:, PREFILL_LEN], v[:, PREFILL_LEN], *_layer_views(pool, 0),
                 _cu := torch.tensor([0] * BATCH, dtype=torch.long, device='cuda'),
                 torch.tensor([1] * BATCH, dtype=torch.long, device='cuda'),
                 torch.tensor([kv_now] * BATCH, dtype=torch.long, device='cuda'),
                 _block_offsets())
torch.cuda.synchronize()
views = _layer_views(pool, 0)
after_fill = tuple(x.clone() for x in views)
# fill byte-exactness vs aten quant
codes, scale, lo = quant_kv_int8(k[:, PREFILL_LEN])
kc, ksz = after_fill[0], after_fill[2]
boff = _block_offsets()
blk = int(boff[0, 192 // BLOCK_LEN]); s = 192 % BLOCK_LEN
print('[DEBUG-b304b] fill codes match aten:', (kc[blk, s] == codes.cuda()).all().item(),
      'params:', ksz[blk, s, 0].tolist(), scale[0, 0].item(), lo[0, 0].item())
_out = _tm_decode(q, pool, kv_now, k[:, kv_now - 1], v[:, kv_now - 1])
torch.cuda.synchronize()
d = sum((views[i] != after_fill[i]).sum().item() for i in range(4))
print('[DEBUG-b304b] slots changed by decode:', d)
kcB = views[0]
kcd = (kcB != kc).any(-1).any(-1)  # (blk, s)
changed = []
for blk_i in range(kc.shape[0]):
    for s_i in kcd[blk_i].nonzero().flatten().tolist():
        b = int((boff == blk_i).nonzero()[0, 0]) if (boff == blk_i).any() else -1
        bi = int((boff[b] == blk_i).nonzero()[0, 0]) if b >= 0 else -1
        changed.append((b, bi * BLOCK_LEN + s_i))
print('[DEBUG-b304b] K-code changed (b,pos):', changed[:40], 'total', len(changed))
for b, pos in changed[:6]:
    blk = int(boff[b, pos // BLOCK_LEN]); s = pos % BLOCK_LEN
    tok = codes.cuda()[b]  # decode token codes for this batch
    same_tok = (kcB[blk, s] == tok).all(-1).float().mean().item()
    print(f'[DEBUG-b304b] (b={b},pos={pos}) matches decode-token codes frac={same_tok:.3f}')
