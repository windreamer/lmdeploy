# ruff: noqa: I001, E402, E702
# ruff: noqa: I001, E402
# [DEBUG-b304] Slot-accurate landing map for one mixed-chain step.
# One fill call (T=4, q=[1,1,1,1], kv=[193]*4, block table b*4+[0..3], fp32
# params): snapshot-diff the pool and report EVERY changed byte mapped to
# (region, block, head, kv, slot, pair) plus the value written, then check
# the 4 expected slots.
import torch

from lmdeploy.pytorch.backends.tm_kv import (
    TmInt8BlockCacheLayout,
    fill_kv_cache_tm,
    quant_kv_int8,
)
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

S, H, D, B = 64, 8, 128, 4
NL, NB = 2, 16


def make_alloc(param_dtype):
    kv = [S, H, D]
    specs = (CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv, dtype=torch.uint8)),
             CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv, dtype=torch.uint8)),
             CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv[:-1] + [2], dtype=param_dtype)),
             CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv[:-1] + [2], dtype=param_dtype)))
    return TmInt8BlockCacheLayout(specs, num_layers=NL).allocate(NB, 'cuda')


alloc = make_alloc(torch.float32)
pool = alloc.pools[0].tensor
pool.zero_()
views = tuple(v[0] for v in alloc.tensor_views)
gen = torch.Generator().manual_seed(11)
kt = (torch.randn(B, 4, H, D, generator=gen) * 0.5).cuda().bfloat16()  # 1 token per batch
key = kt.reshape(-1, H, D)
before = pool.clone()
fill_kv_cache_tm(key, -key, *views,
                 torch.arange(B, device='cuda').long(),
                 torch.ones(B, dtype=torch.long, device='cuda'),
                 torch.full((B, ), 193, dtype=torch.long, device='cuda'),
                 (torch.arange(B, device='cuda')[:, None] * 4 + torch.arange(4, device='cuda')[None, :]).int())
torch.cuda.synchronize()

changed = (pool != before).nonzero()  # (M, 3): layer, block, byte
print(f'changed bytes: {changed.shape[0]}')
h2sd = 2 * H * S * D
summary = {}
for _, blk, byte in changed.tolist():
    if byte < h2sd:
        rest = byte
        head, rest = divmod(rest, 2 * S * D)
        kvs, rest = divmod(rest, S * D)
        slot, dim = divmod(rest, D)
        loc = ('code', blk, head, kvs, slot)
    else:
        foff = (byte - h2sd) // 4  # fp32 elem
        head, rest = divmod(foff, 2 * S * 2)
        kvs, rest = divmod(rest, S * 2)
        slot, pair = divmod(rest, 2)
        loc = ('param', blk, head, kvs, slot, pair)
    summary[loc] = summary.get(loc, 0) + 1
agg = {}
for loc, n in summary.items():
    k = (loc[0], loc[1], loc[3] if loc[0] == 'code' else loc[3], loc[4])
    agg[k] = agg.get(k, 0) + n
for k in sorted(agg):
    print('changed:', k, 'x', k and agg[k])

codes, scale, lo = quant_kv_int8(key.float())
print('\nexpected per batch b: K code/min/scale at block 4b+3 slot 0, all 8 heads')
for b in range(B):
    blk = 4 * b + 3
    ok_c = (pool[0, blk, :h2sd].view(H, 2, S, D)[:, 0, 0] == codes[b].cuda()).all().item()
    pr = pool[0, blk, h2sd:].view(torch.float32).view(H, 2, S, 2)[:, 0, 0]
    ok_p = (pr[:, 0].cpu() == scale[b].cpu()).all().item() and (pr[:, 1].cpu() == lo[b].cpu()).all().item()
    print(f'b{b} block {blk}: K codes ok={ok_c} K params ok={ok_p}')
    for other in range(NB):
        if other == blk:
            continue
        pc = pool[0, other, h2sd:].view(torch.float32).view(H, 2, S, 2)[:, 0, 0]
        hit = (pc[:, 1].cpu() == lo[b].cpu())
        if hit.any().item():
            heads = hit.nonzero().flatten().tolist()
            print(f'   !! b{b} min values ALSO in block {other} slot 0, heads {heads}')
