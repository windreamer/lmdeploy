# [DEBUG-b304] Minimal red probe for the fill_kv_cache_tm multi-batch placement bug.
# Writes unique constant-valued tokens (exact round-trip: code 0, scale 1e-6,
# min = base) and asserts each token lands at its contract slot
# row = block_offsets[b, pos // S] * S + pos % S. Any other outcome is the
# ticket symptom "token written into another batch's slot".
import torch

from lmdeploy.pytorch.backends.tm_kv import TmInt8BlockCacheLayout, fill_kv_cache_tm
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec

S, H, D = 64, 2, 8
NL = 1


def run_case(name, B, q_lens, kv_lens, n_new_tokens, tag):
    P = max(kv_lens)
    NB = B * ((P + S - 1) // S)
    kv_shape = [S, H, D]
    specs = (
        CacheTensorSpec(name='k_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
        CacheTensorSpec(name='v_cache', desc=CacheDesc(shape=kv_shape, dtype=torch.uint8)),
        CacheTensorSpec(name='quant_0', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.float32)),
        CacheTensorSpec(name='quant_1', desc=CacheDesc(shape=kv_shape[:-1] + [2], dtype=torch.float32)),
    )
    layout = TmInt8BlockCacheLayout(specs, num_layers=NL)
    alloc = layout.allocate(NB, 'cuda')
    k_cache, v_cache, k_sz, v_sz = tuple(v[0] for v in alloc.tensor_views)

    T = n_new_tokens
    base = (torch.arange(T, dtype=torch.float32) + 1) * 0.01  # unique per token, never 0
    key = base[:, None, None].expand(T, H, D).contiguous().cuda()
    val = (-base)[:, None, None].expand(T, H, D).contiguous().cuda()

    q_seqlens = torch.tensor(q_lens, dtype=torch.long, device='cuda')
    kv_seqlens = torch.tensor(kv_lens, dtype=torch.long, device='cuda')
    q_start_loc = torch.nn.functional.pad(torch.cumsum(q_seqlens.cpu(), 0), (1, 0))[:-1].cuda().long()
    ncols = (P + S - 1) // S
    block_offsets = (torch.arange(B, device='cuda')[:, None] * ncols +
                     torch.arange(ncols, device='cuda')[None, :]).int()

    fill_kv_cache_tm(key, val, k_cache, v_cache, k_sz, v_sz,
                     q_start_loc, q_seqlens, kv_seqlens, block_offsets)
    torch.cuda.synchronize()

    # Exact readback of K codes/params (constant tokens: dequant == base exactly).
    pool = alloc.pools[0].tensor[0]
    h2sd = 2 * H * S * D
    codes = pool[:, :h2sd].view(NB, H, 2, S, D).float()
    params = pool[:, h2sd:].view(torch.float32).view(NB, H, 2, S, 2)
    kdeq = codes[:, :, 0] * params[:, :, 0][..., 0:1] + params[:, :, 0][..., 1:2]
    rows = kdeq.permute(0, 2, 1, 3).reshape(NB * S, H, D)[:, 0, 0]  # (NB*S,)

    # token -> expected slot
    starts = torch.nn.functional.pad(torch.cumsum(q_seqlens.cpu(), 0), (1, 0)).tolist()
    inv = {}  # pool row -> (b, pos)
    for b in range(B):
        for pos in range(kv_lens[b] - q_lens[b], kv_lens[b]):
            blk = block_offsets[b, pos // S].item()
            inv[blk * S + pos % S] = (b, pos)

    # value -> landed row (unique bases make this exact)
    landed = {}
    for r in range(NB * S):
        v = rows[r].item()
        if v == 0.0:
            continue
        t = int(round(v / 0.01)) - 1
        landed.setdefault(t, []).append(r)

    bad = 0
    for t in range(T):
        # which batch owns token t?
        b = max(i for i in range(B) if starts[i] <= t)
        pos = kv_lens[b] - q_lens[b] + (t - starts[b])
        blk = block_offsets[b, pos // S].item()
        want = blk * S + pos % S
        got = landed.get(t, [])
        if got != [want]:
            bad += 1
            if bad <= 8:
                g = [(rr // S, block_offsets.tolist()[int(rr // S / ncols) if False else 0]) for rr in got]
                print(f'[{tag}] token {t} (batch {b}, pos {pos}) expected row {want}, '
                      f'landed rows {got} (blocks {[rr // S for rr in got]}, '
                      f'in-block {[rr % S for rr in got]})')
    verdict = 'RED' if bad else 'GREEN'
    print(f'[{tag}] case={name}: {bad}/{T} misplaced -> {verdict}')
    return bad


bad = 0
# control: single batch, same shape (ticket says green)
bad += run_case('B1', 1, [192], [192], 192, 'b304')
# reproduction of the failing roundtrip shape: B=4 equal 192-token prefills
bad += run_case('B4-prefill', 4, [192] * 4, [192] * 4, 768, 'b304')
# reproduction of the failing mixed-chain shape: 1-token decode appends
bad += run_case('B4-decode', 4, [1] * 4, [193] * 4, 4, 'b304')
# ragged multi-batch prefill (candidate trigger)
bad += run_case('B4-ragged', 4, [3, 0, 5, 2], [3, 7, 9, 6], 10, 'b304')
print('[b304] total misplaced:', bad)
