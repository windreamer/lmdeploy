# [DEBUG-b304] Geometry/dtype matrix for the fill_kv_cache_tm multi-batch bug.
# Each case writes unique constant tokens (byte-exact readback) and reports
# per-slot placement. Cut one variable at a time from the green probe
# geometry toward the red test geometry to find the load-bearing element.
import torch

from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout, fill_kv_cache_tm
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

NL = 2


def run_case(name, H, D, param_dtype, B, P, ncols, layer, value_mode):
    NB = B * ncols
    kv_shape = [64, H, D]
    specs = (
        CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
        CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
        CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=param_dtype)),
        CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=param_dtype)),
    )
    layout = TmInt8BlockCacheLayout(specs, num_layers=NL)
    alloc = layout.allocate(NB, 'cuda')
    k_cache, v_cache, k_sz, v_sz = tuple(v[layer] for v in alloc.tensor_views)

    T = B * P
    gen = torch.Generator().manual_seed(11)
    if value_mode == 'const':
        base = (torch.arange(T, dtype=torch.float32) + 1) * 0.01
        key = base[:, None, None].expand(T, H, D).contiguous().cuda()
    else:  # random bf16 like the failing test
        key = (torch.randn(T, H, D, generator=gen) * 0.5).cuda().bfloat16()
    val = -key

    q_seqlens = torch.full((B, ), P, dtype=torch.long, device='cuda')
    kv_seqlens = q_seqlens.clone()
    q_start_loc = torch.arange(B, device='cuda') * P
    block_offsets = (torch.arange(B, device='cuda')[:, None] * ncols +
                     torch.arange(ncols, device='cuda')[None, :]).int()

    fill_kv_cache_tm(key, val, k_cache, v_cache, k_sz, v_sz,
                     q_start_loc, q_seqlens, kv_seqlens, block_offsets)
    torch.cuda.synchronize()

    pool = alloc.pools[0].tensor[0]
    h2sd = 2 * H * 64 * D
    codes = pool[:, :h2sd].view(NB, H, 2, 64, D).float()
    params = pool[:, h2sd:].view(param_dtype).view(NB, H, 2, 64, 2).float()
    kdeq = (codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2])
    rows = kdeq.permute(0, 2, 1, 3).reshape(NB * 64, H, D)  # (NB*64, H, D)

    if value_mode == 'const':
        bad = 0
        for t in range(T):
            b, pos = t // P, t % P
            want = block_offsets[b, pos // 64].item() * 64 + pos % 64
            got = (rows[:, 0, 0] == (t + 1) * 0.01).nonzero().flatten().tolist()
            if got != [want]:
                bad += 1
                if bad <= 5:
                    print(f'[{name}] token {t} (b{b},p{pos}) want row {want} got {got}')
        print(f'[{name}] H={H} D={D} pd={param_dtype} B={B} ncols={ncols} '
              f'layer={layer} const: {bad}/{T} misplaced -> {"RED" if bad else "GREEN"}')
        return bad
    # random mode: gate per (b, pos) like the failing test (0.05)
    keyf = key.view(B, P, H, D).float()
    gate = (rows.view(B, ncols * 64, H, D)[:, :P] - keyf).abs().max().item()
    print(f'[{name}] H={H} D={D} pd={param_dtype} B={B} ncols={ncols} '
          f'layer={layer} rand: roundtrip_max_abs={gate:.4f} -> {"RED" if gate > 0.05 else "GREEN"}')
    return int(gate > 0.05)


total = 0
total += run_case('c0-test-exact', 8, 128, torch.bfloat16, 4, 192, 4, 1, 'const')
total += run_case('c1-fp32-params', 8, 128, torch.float32, 4, 192, 4, 1, 'const')
total += run_case('c2-D8', 8, 8, torch.bfloat16, 4, 192, 4, 1, 'const')
total += run_case('c3-H2', 2, 128, torch.bfloat16, 4, 192, 4, 1, 'const')
total += run_case('c4-ncols3', 8, 128, torch.bfloat16, 4, 192, 3, 0, 'const')
total += run_case('c5-layer0', 8, 128, torch.bfloat16, 4, 192, 4, 0, 'const')
total += run_case('c6-rand-probe-geom', 2, 8, torch.float32, 4, 192, 3, 0, 'rand')
total += run_case('c7-rand-test-geom', 8, 128, torch.bfloat16, 4, 192, 4, 1, 'rand')
print('[b304] red cases:', total)
