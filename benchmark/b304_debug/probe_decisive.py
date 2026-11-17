# ruff: noqa: I001, E402, E702
# ruff: noqa: I001, E402
# [DEBUG-b304] Decisive cells:
#   d1: mixed-chain replica with FP32 params (isolate dtype from T-per-call/values)
#   d2: mixed-chain replica with BF16 params (control, expect RED)
#   d3: single-token write, pool snapshot-diff -> exact touched slots (bf16 vs fp32)
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
    return TmInt8BlockCacheLayout(specs, num_layers=NL).allocate(NB, 'cuda')


def dequant_layer(alloc, H, D, param_dtype, layer):
    NB = alloc.pools[0].tensor.shape[1]
    pool = alloc.pools[0].tensor[layer]
    h2sd = 2 * H * S * D
    codes = pool[:, :h2sd].view(NB, H, 2, S, D).float()
    params = pool[:, h2sd:].view(param_dtype).view(NB, H, 2, S, 2).float()
    k = codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2]
    return k.permute(0, 2, 1, 3).reshape(NB * S, H, D)


def mixed_case(name, param_dtype):
    H, D, B, PREFILL, STEPS = 8, 128, 4, 192, 64
    NB = B * 4
    alloc = make_alloc(H, D, param_dtype, NB)
    views = tuple(v[0] for v in alloc.tensor_views)
    gen = torch.Generator().manual_seed(11)
    kt = (torch.randn(B, PREFILL + STEPS, H, D, generator=gen) * 0.5).cuda().bfloat16()
    vt = (torch.randn(B, PREFILL + STEPS, H, D, generator=gen) * 0.5).cuda().bfloat16()
    ktv, vtv = kt.reshape(-1, H, D), vt.reshape(-1, H, D)
    starts = torch.nn.functional.pad(torch.cumsum(torch.ones(B, dtype=torch.long), 0), (1, 0))[:-1].cuda()
    boff = (torch.arange(B, device='cuda')[:, None] * 4 + torch.arange(4, device='cuda')[None, :]).int()
    for step in range(STEPS):
        begin = B * (PREFILL + step)
        fill_kv_cache_tm(ktv[begin:begin + B], vtv[begin:begin + B], *views,
                         starts, torch.ones(B, dtype=torch.long, device='cuda'),
                         torch.full((B, ), PREFILL + step + 1, dtype=torch.long, device='cuda'), boff)
    torch.cuda.synchronize()
    rows = dequant_layer(alloc, H, D, param_dtype, 0).view(B, 4 * S, H, D)
    tail = (rows[:, PREFILL:] - kt[:, PREFILL:].float()).abs().max().item()
    print(f'[{name}] pd={str(param_dtype).split(".")[-1]} tail max_abs={tail:.4f} '
          f'-> {"RED" if tail > 0.05 else "GREEN"}')


def single_token_case(name, param_dtype):
    """One token written at (b0, slot 0); snapshot-diff lists every touched
    slot."""
    H, D, B = 8, 128, 4
    NB = 4
    alloc = make_alloc(H, D, param_dtype, NB)
    pool = alloc.pools[0].tensor
    pool.zero_()
    views = tuple(v[0] for v in alloc.tensor_views)
    key = (torch.randn(1, H, D) * 0.5).cuda().bfloat16()
    fill_kv_cache_tm(key, -key, *views,
                     torch.zeros(1, dtype=torch.long, device='cuda'),
                     torch.ones(1, dtype=torch.long, device='cuda'),
                     torch.ones(1, dtype=torch.long, device='cuda'),
                     (torch.arange(B)[:, None] * 1 + torch.arange(1)[None, :]).int().cuda())
    torch.cuda.synchronize()
    h2sd = 2 * H * S * D
    p = pool[0, :, h2sd:].view(param_dtype)
    nz = (p != 0).nonzero().flatten().tolist()
    print(f'[{name}] pd={str(param_dtype).split(".")[-1]} touched param elems (pool row/idx): '
          f'{len(nz)} elems; block starts {sorted(set(i // (2 * H * S * 2 // param_dtype.itemsize) for i in nz))}')
    # expected: all touched elems live in pool block 0 only
    blocks = set(i // (2 * H * S * 2 // param_dtype.itemsize) for i in nz)
    print(f'[{name}] blocks touched: {sorted(blocks)} (expected [0]) -> '
          f'{"RED" if blocks != {0} else "codes/params confined to block 0"}')


mixed_case('d1-mixed-fp32', torch.float32)
mixed_case('d2-mixed-bf16', torch.bfloat16)
single_token_case('d3-single-fp32', torch.float32)
single_token_case('d3-single-bf16', torch.bfloat16)
