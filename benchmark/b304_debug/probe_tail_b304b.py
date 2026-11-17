# [DEBUG-b304b] which slots/bytes does the decode kernel's re-store change?
import sys
sys.path.insert(0, 'tests/pytorch/kernel')
import sys; sys.path.insert(0, 'tests/pytorch/kernel'); sys.path.insert(0, '../../tests/pytorch/kernel')
from test_tm_int8_kv import (_make_data, _allocate_layout, _bridge_fill, _tm_decode, _dequant_pool, _layer_views, _block_offsets, _cu, BATCH, PREFILL_LEN, KLEN, NUM_BLOCKS, DECODE_STEPS, NUM_KV_HEADS, HEAD_DIM)
import torch
BLOCK_LEN = 64
from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm
import test_tm_int8_kv

_, k, v = _make_data()
pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
pool_t = pool.pools[0].tensor
kt = k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
vt = v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
_bridge_fill(kt, vt, [PREFILL_LEN] * BATCH, pool)
torch.cuda.synchronize()

step = 0
kv_now = PREFILL_LEN + step + 1
fill_kv_cache_tm(k[:, PREFILL_LEN + step], v[:, PREFILL_LEN + step],
                 *_layer_views(pool, 0),
                 _cu([1] * BATCH)[:-1].long(),
                 torch.tensor([1] * BATCH, dtype=torch.long, device='cuda'),
                 torch.tensor([kv_now] * BATCH, dtype=torch.long, device='cuda'),
                 _block_offsets())
torch.cuda.synchronize()
after_fill = pool_t.clone()
_out = _tm_decode(torch.Generator(device='cpu').manual_seed(7) and _make_data(seed=7)[0], pool, kv_now,
                  k[:, kv_now - 1], v[:, kv_now - 1])
torch.cuda.synchronize()
diff_mask = (pool_t != after_fill)
print('[DEBUG-b304b] slots changed by decode re-store:', diff_mask.any().item(), diff_mask.sum().item())
# which positions (block, s) changed?
nz = diff_mask.nonzero()
if len(nz):
    k_cache = _layer_views(pool, 0)[0]
    print('[DEBUG-b304b] sample changed slots (block, s, h, x):', nz[:8].tolist())
    # dequantize those slots after decode vs source token
    kq, vq_ = _dequant_pool(pool_t)
    src = k.float()
    pos0 = PREFILL_LEN
    for b in range(BATCH):
        for ti in range(kv_now - 2, min(kv_now + 2, KLEN)):
            blk = int(_block_offsets()[b, ti // BLOCK_LEN])
            h = 0
            err = (kq[b, ti, h] - src[b, ti, h]).abs().max().item()
            print(f'[DEBUG-b304b] b={b} pos={ti} err={err:.4f}')
# scale/min params of slot kv_now-1 after decode vs after fill
kc, vc, k_sz, v_sz = _layer_views(pool, 0)
for b in [0]:
    blk = int(_block_offsets()[b, (kv_now - 1) // BLOCK_LEN])
    s = (kv_now - 1) % BLOCK_LEN
    print('[DEBUG-b304b] after-decode k_sz (scale,min) slot:', k_sz[blk, s, 0].tolist(),
          ' v_sz:', v_sz[blk, s, 0].tolist())
    print('[DEBUG-b304b] after-fill   k_sz (scale,min) slot:', after_fill is not None and
          [x for x in after_fill] is not None and k_sz[blk, s, 0].tolist())
# ATen-quant reference of the same token
from lmdeploy.pytorch.backends.tm_kv import quant_kv_int8
codes, scale, lo = quant_kv_int8(k[:, kv_now - 1])

# precise corruption map: (b,pos) -> max code diff after decode vs after fill
kcA = _layer_views(after_fill, 0)[0]; kcB = _layer_views(pool_t, 0)[0]
boff = _block_offsets()
inv = {}
for b in range(BATCH):
    for bi in range(_block_offsets().shape[1]):
        inv[int(boff[b, bi])] = (b, bi * BLOCK_LEN)
codes_changed = {}
for blk in range(kcA.shape[0]):
    d = (kcB[blk] != kcA[blk]).any(dim=-1).any(dim=-1)  # (S,)
    for s in d.nonzero().flatten().tolist():
        key = inv.get(blk, (None, None))
        codes_changed.setdefault(key, []).append(s)
print('[DEBUG-b304b] changed (b,base_pos): in-block slots ->', {k: v[:8] for k, v in codes_changed.items()})
# what value landed: dequant pos193 vs source token 192
kq, vq_ = _dequant_pool(pool_t)
src = k.float()
for p in (191, 192, 193, 194):
    for tok in (191, 192):
        err = (kq[0, p] - src[0, tok]).abs().max().item()
        print(f'[DEBUG-b304b] b0 pos={p} vs src token {tok}: err={err:.4f}')
