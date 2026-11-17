"""E1: EAGER baseline (known green). TM int8 decode single call vs bf16
reference and vs pool-dequant reference. Budget: max_abs <= 0.25."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C_  # noqa: E402

assert torch.cuda.is_available()
torch.manual_seed(0)

pool = C_.allocate_pool()
q, k, v = C_.make_qkv(seed=11, klen=C_.KLEN)
C_.bridge_fill(k[:, :C_.PREFILL_LEN].reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
               v[:, :C_.PREFILL_LEN].reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
               [C_.PREFILL_LEN] * C_.BATCH, pool)
torch.cuda.synchronize()

k_cache = C_.layer_views(pool)[0]
ptrs, cu_base = C_.pointer_tables(k_cache)
q_flat, k_flat, v_flat = C_.q2d(q), C_.rows2d(k[:, C_.PREFILL_LEN - 1]), C_.rows2d(v[:, C_.PREFILL_LEN - 1])
out = torch.empty(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
C_.tm_decode(q_flat, k_flat, v_flat, out, ptrs, cu_base, max_k_len=C_.PREFILL_LEN)
torch.cuda.synchronize()
out = out.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM)

C_.gate('e1', out, C_.reference(q, k, v, C_.PREFILL_LEN), 'eager-vs-bf16-ref')
k_deq, v_deq = C_.dequant_pool(pool.pools[0].tensor)
C_.gate('e1', out, C_.reference(q, k_deq, v_deq, C_.PREFILL_LEN), 'eager-vs-pool-dequant-ref', budget=0.05)
C_.save_results('e1_results.json')
print('E1 PASS')
