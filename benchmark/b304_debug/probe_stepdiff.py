# [DEBUG-b304] Per-step pool diff over the mixed-chain loop: find the FIRST
# step whose write leaves its expected slot set (blocks {4b+3}, slot step%64).
import torch

from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout, fill_kv_cache_tm
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


def changed_slots(a, b):
    out = []
    for _, blk, byte in (a != b).nonzero().tolist():
        if byte < h2sd:
            rest = byte
            head, rest = divmod(rest, 2 * S * D)
            kvs, rest = divmod(rest, S * D)
            slot = rest // D
            out.append((blk, 'code', kvs, slot))
        else:
            foff = (byte - h2sd) // 4
            head, rest = divmod(foff, 2 * S * 2)
            kvs, rest = divmod(rest, S * 2)
            slot = rest // 2
            out.append((blk, 'param', kvs, slot))
    return set(out)


prev = pool.clone()
starts = torch.arange(B, device='cuda').long()
ones = torch.ones(B, dtype=torch.long, device='cuda')
boff = (torch.arange(B, device='cuda')[:, None] * 4 + torch.arange(4, device='cuda')[None, :]).int()
for step in range(STEPS):
    begin = B * (PREFILL + step)
    fill_kv_cache_tm(ktv[begin:begin + B], vtv[begin:begin + B], *views, starts, ones,
                     torch.full((B, ), PREFILL + step + 1, dtype=torch.long, device='cuda'), boff)
    torch.cuda.synchronize()
    now = pool.clone()
    ch = changed_slots(prev, now)
    want_slot = step  # position 192+step -> slot (192+step) % 64
    want = {(4 * b + 3, kind, kvs, want_slot) for b in range(B) for kind in ('code', 'param') for kvs in (0, 1)}
    if ch != want:
        extra, missing = ch - want, want - ch
        print(f'[b304] FIRST VIOLATION at step {step}: extra={sorted(extra)} missing={sorted(missing)}')
        # identify provenance of an extra code slot: compare against all 4 tokens of this step
        blk, kind, kvs, slot = sorted(extra)[0]
        if kind == 'code':
            got = pool[0, blk, blk * 0 + h2sd * 0:][:h2sd].view(H, 2, S, D)[:, kvs, slot]
            for t in range(B):
                tc, _, _ = __import__('lmdeploy.pytorch.backends.tm_kv', fromlist=['x']).quant_kv_int8(
                    ktv[begin:begin + B][t].float())
                match = (got == tc.cuda()).float().mean().item()
                print(f'    extra slot (block {blk}, kv {kvs}, slot {slot}) matches token {t}: {match:.3f}')
        break
    prev = now
else:
    print('[b304] all 64 steps confined to expected slots?!')
