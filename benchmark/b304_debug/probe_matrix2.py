# [DEBUG-b304] Matrix v2: correct-layer readback, tolerance matching, plus an
# exact replica of the pytest mixed-chain ATen decode-append phase.
import torch

from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout, fill_kv_cache_tm
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

S = 64
NL = 2


def make_alloc(H, D, param_dtype, NB):
    kv = [S, H, D]
    specs = (CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv, dtype=torch.uint8)),
             CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv, dtype=torch.uint8)),
             CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv[:-1] + [2], dtype=param_dtype)),
             CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv[:-1] + [2], dtype=param_dtype)))
    layout = TmInt8BlockCacheLayout(specs, num_layers=NL)
    return layout, layout.allocate(NB, 'cuda')


def dequant_layer(alloc, H, D, param_dtype, layer):
    NB = alloc.pools[0].tensor.shape[1]
    pool = alloc.pools[0].tensor[layer]
    h2sd = 2 * H * S * D
    codes = pool[:, :h2sd].view(NB, H, 2, S, D).float()
    params = pool[:, h2sd:].view(param_dtype).view(NB, H, 2, S, 2).float()
    k = codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2]
    return k.permute(0, 2, 1, 3).reshape(NB * S, H, D)  # (NB*S, H, D) pool rows


def const_case(name, H, D, param_dtype, B, P, ncols, layer):
    NB = B * ncols
    layout, alloc = make_alloc(H, D, param_dtype, NB)
    k_cache, v_cache, k_sz, v_sz = tuple(v[layer] for v in alloc.tensor_views)
    T = B * P
    base = (torch.arange(T, dtype=torch.float32) + 1) * 0.01
    key = base[:, None, None].expand(T, H, D).contiguous().cuda()
    tol = 0.02 if param_dtype is torch.bfloat16 else 0.002
    fill_kv_cache_tm(key, -key, k_cache, v_cache, k_sz, v_sz,
                     (torch.arange(B) * P).cuda().long(),
                     torch.full((B, ), P, dtype=torch.long, device='cuda'),
                     torch.full((B, ), P, dtype=torch.long, device='cuda'),
                     (torch.arange(B)[:, None] * ncols + torch.arange(ncols)[None, :]).int().cuda())
    torch.cuda.synchronize()
    rows = dequant_layer(alloc, H, D, param_dtype, layer)[:, 0, 0]
    bad = 0
    for t in range(T):
        b, pos = t // P, t % P
        want = (block := b * ncols + pos // S) * S + pos % S
        got = ((rows - (t + 1) * 0.01).abs() < tol).nonzero().flatten().tolist()
        if got != [want]:
            bad += 1
            if bad <= 4:
                print(f'[{name}] token {t} (b{b},p{pos}) want row {want} got {got}')
    print(f'[{name}] H={H} D={D} pd={str(param_dtype).split(".")[-1]} ncols={ncols} '
          f'layer={layer}: {bad}/{T} misplaced -> {"RED" if bad else "GREEN"}')


def mixed_chain_case(name):
    """Exact replica of pytest mixed-chain ATen phase (bf16 params, randn bf16)."""
    H, D, B, PREFILL, STEPS = 8, 128, 4, 192, 64
    NB = B * ((PREFILL + STEPS) // S)
    layout, alloc = make_alloc(H, D, torch.bfloat16, NB)
    views = tuple(v[0] for v in alloc.tensor_views)
    gen = torch.Generator().manual_seed(11)
    kt = (torch.randn(B, PREFILL + STEPS, H, D, generator=gen) * 0.5).cuda().bfloat16()
    vt = (torch.randn(B, PREFILL + STEPS, H, D, generator=gen) * 0.5).cuda().bfloat16()
    ktv = kt.reshape(-1, H, D)
    vtv = vt.reshape(-1, H, D)
    for step in range(STEPS):
        begin = B * (PREFILL + step)
        fill_kv_cache_tm(ktv[begin:begin + B], vtv[begin:begin + B], *views,
                         torch.nn.functional.pad(torch.cumsum(torch.ones(B, dtype=torch.long), 0), (1, 0))[:-1].cuda(),
                         torch.ones(B, dtype=torch.long, device='cuda'),
                         torch.full((B, ), PREFILL + step + 1, dtype=torch.long, device='cuda'),
                         (torch.arange(B, device='cuda')[:, None] * 4 + torch.arange(4, device='cuda')[None, :]).int())
        torch.cuda.synchronize()
    rows = dequant_layer(alloc, H, D, torch.bfloat16, 0).view(B, 4 * S, H, D)
    tail = (rows[:, PREFILL:] - kt[:, PREFILL:].float()).abs().max().item()
    print(f'[{name}] mixed-chain ATen tail max_abs={tail:.4f} -> {"RED" if tail > 0.05 else "GREEN"}')


const_case('m0', 8, 128, torch.bfloat16, 4, 192, 4, 1)
const_case('m1', 8, 128, torch.bfloat16, 4, 192, 4, 0)
const_case('m2', 8, 128, torch.float32, 4, 192, 4, 1)
const_case('m3', 2, 8, torch.float32, 4, 192, 3, 0)
mixed_chain_case('m4')
