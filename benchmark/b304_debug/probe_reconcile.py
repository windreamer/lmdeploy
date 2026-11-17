# [DEBUG-b304] Reconciliation: the 64-step loop, then BOTH readings of the
# pool (per-slot value check vs dequant-layer tail measurement) in one process.
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

h2sd = 2 * H * S * D
starts = torch.arange(B, device='cuda').long()
ones = torch.ones(B, dtype=torch.long, device='cuda')
boff = (torch.arange(B, device='cuda')[:, None] * 4 + torch.arange(4, device='cuda')[None, :]).int()
for step in range(STEPS):
    begin = B * (PREFILL + step)
    fill_kv_cache_tm(ktv[begin:begin + B], vtv[begin:begin + B], *views, starts, ones,
                     torch.full((B, ), PREFILL + step + 1, dtype=torch.long, device='cuda'), boff)
torch.cuda.synchronize()

# (a) per-slot check for every appended (b, pos): K codes/params vs quant of input
bad_slots = 0
for b in range(B):
    for pos in range(PREFILL, PREFILL + STEPS):
        step = pos - PREFILL
        blk = int(boff[b, pos // S])
        tok = kt[b, pos].float()
        codes, scale, lo = quant_kv_int8(tok)
        got_c = pool[0, blk, :h2sd].view(H, 2, S, D)[:, 0, pos % S]
        pr = pool[0, blk, h2sd:].view(torch.float32).view(H, 2, S, 2)[:, 0, pos % S]
        if not (got_c == codes.cuda()).all().item() or \
           not (pr[:, 0].cpu() == scale.cpu()).all().item() or \
           not (pr[:, 1].cpu() == lo.cpu()).all().item():
            bad_slots += 1
            if bad_slots <= 3:
                print(f'[b304] slot mismatch b{b} pos{pos} block {blk} slot {pos % S}')
print(f'[b304] (a) per-slot value check: {bad_slots} bad of {B * STEPS}')

# (b) the mixed_case readback measurement
pool_l = pool[0]
codes_p = pool_l[:, :h2sd].view(NB, H, 2, S, D).float()
params_p = pool_l[:, h2sd:].view(torch.float32).view(NB, H, 2, S, 2).float()
kdeq = codes_p[:, :, 0] * params_p[:, :, 0][..., 0:1] + params_p[:, :, 0][..., 1:2]
rows = kdeq.permute(0, 2, 1, 3).reshape(NB * S, H, D).view(B, 4 * S, H, D)
tail = (rows[:, PREFILL:] - kt[:, PREFILL:].float()).abs().max().item()
print(f'[b304] (b) dequant-layer tail max_abs={tail:.4f}')
err = (rows[:, PREFILL:] - kt[:, PREFILL:].float()).abs().amax(dim=(1, 2, 3))
print(f'[b304] (b) per-batch tail errs: {[round(x, 4) for x in err.tolist()]}')
