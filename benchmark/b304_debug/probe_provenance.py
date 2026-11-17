# [DEBUG-b304] Provenance search: what token actually sits in slot (block 3,
# slot 0) after the 64-step loop? Compare stored codes/params against quant
# of every candidate token.
import torch

from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout, fill_kv_cache_tm, quant_kv_int8
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

S, H, D, B = 64, 8, 128, 4
NL, NB = 2, 16
PREFILL, STEPS = 192, 64
kv = [S, H, D]
specs = (CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv, dtype=torch.uint8)),
         CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv, dtype=torch.uint8)),
         CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv[:-1] + [2], dtype=torch.float32)),
         CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv[:-1] + [2], dtype=torch.float32)))
alloc = TmInt8BlockCacheLayout(specs, num_layers=NL).allocate(NB, 'cuda')
pool = alloc.pools[0].tensor
views = tuple(v[0] for v in alloc.tensor_views)
gen = torch.Generator().manual_seed(11)
kt = (torch.randn(B, PREFILL + STEPS, H, D, generator=gen) * 0.5).cuda().bfloat16()
vt = (torch.randn(B, PREFILL + STEPS, H, D, generator=gen) * 0.5).cuda().bfloat16()
ktv, vtv = kt.reshape(-1, H, D), vt.reshape(-1, H, D)
starts = torch.arange(B, device='cuda').long()
ones = torch.ones(B, dtype=torch.long, device='cuda')
boff = (torch.arange(B, device='cuda')[:, None] * 4 + torch.arange(4, device='cuda')[None, :]).int()
for step in range(STEPS):
    begin = B * (PREFILL + step)
    fill_kv_cache_tm(ktv[begin:begin + B], vtv[begin:begin + B], *views, starts, ones,
                     torch.full((B, ), PREFILL + step + 1, dtype=torch.long, device='cuda'), boff)
torch.cuda.synchronize()

h2sd = 2 * H * S * D


def slot_content(blk, slot):
    codes = pool[0, blk, :h2sd].view(H, 2, S, D)[:, 0, slot]
    pr = pool[0, blk, h2sd:].view(torch.float32).view(H, 2, S, 2)[:, 0, slot]
    return codes, pr[:, 0], pr[:, 1]


def quant_token(tok):  # (H, D) bf16 -> codes (H, D), scale (H,), lo (H,)
    return quant_kv_int8(tok.float())


gc, gsc, glo = slot_content(3, 0)
print(f'slot (block3, slot0) stored: scale[:3]={gsc[:3].tolist()} min[:3]={glo[:3].tolist()}')

best = []
for b in range(B):
    for p in range(PREFILL, PREFILL + STEPS):
        c, sc, lo = quant_token(kt[b, p])
        cm = (gc.cuda() == c.cuda()).float().mean().item()
        sm = (gsc.cpu() == sc.cpu()).float().mean().item()
        lm = (glo.cpu() == lo.cpu()).float().mean().item()
        best.append((cm + sm + lm, cm, sm, lm, b, p))
best.sort(reverse=True)
for tot, cm, sm, lm, b, p in best[:5]:
    print(f'best match: token (b{b}, p{p}) code_match={cm:.3f} scale_match={sm:.3f} min_match={lm:.3f}')

# also compare against prefill-region tokens (0..192) in case of wild indexing
best2 = []
for b in range(B):
    for p in range(0, PREFILL, 8):
        c, sc, lo = quant_token(kt[b, p])
        cm = (gc.cuda() == c.cuda()).float().mean().item()
        best2.append((cm, b, p))
best2.sort(reverse=True)
print('best prefill-region code matches:', best2[:3])
