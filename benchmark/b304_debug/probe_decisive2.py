# [DEBUG-b304b] full change map over all 4 planes; determinism; bridge-only repro
import sys; sys.path.insert(0, '../../tests/pytorch/kernel')
from test_tm_int8_kv import (_make_data, _allocate_layout, _bridge_fill, _tm_decode,
                             _layer_views, _block_offsets, BATCH, PREFILL_LEN, KLEN,
                             NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_LEN)
import torch
from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm, quant_kv_int8

def change_map(before, after, label):
    views_b, views_a = _layer_views(before, ), _layer_views(after, )
    boff = _block_offsets()
    for name in range(4):
        vb, va = views_b[name], views_a[name]
        dd = (va != vb).any(-1)
        while dd.dim() > 2:
            dd = dd.any(-1)
        slots = [tuple(map(int, x)) for x in dd.nonzero().tolist()]
        pos = []
        for blk, s in slots:
            hit = (boff == blk).nonzero()
            if len(hit):
                b, bi = hit[0].tolist()
                pos.append((b, bi * BLOCK_LEN + s))
            else:
                pos.append((blk, s))
        print(f'[DEBUG-b304b] {label} plane{name} changed (b,pos):', pos[:16], 'n=', len(pos))

_, k, v = _make_data()
q, _, _ = _make_data(seed=7)
pool = _allocate_layout().allocate(NUM_BLOCKS, 'cuda')
kt = k[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
vt = v[:, :PREFILL_LEN].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
_bridge_fill(kt, vt, [PREFILL_LEN] * BATCH, pool)
torch.cuda.synchronize()
snap0 = tuple(x.clone() for x in _layer_views(pool, ))
# bridge-only decode at ti=192 (the exact test_bridge scenario)
_out = _tm_decode(q, pool, PREFILL_LEN, k[:, PREFILL_LEN - 1], v[:, PREFILL_LEN - 1])
torch.cuda.synchronize()
change_map(snap0, pool, 'bridge-decode')
