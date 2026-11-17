"""Standalone repro of the layout-sensitive captured-ATen-writer mis-scatter.

Verbatim copy of the module's build_capture (else-branch = captured
fill_kv_cache_tm) driven by the phase_contract loop: r0 matches the eager
reference bitwise; r1/r2 mis-scatter (~4KB sprayed into wrong slots of
blocks 7/11, cap_nz 12574 vs ref_nz 8382). Deterministic per process shape;
trivially equivalent shapes pass (hoisted views / different pool allocation
order / loop at module scope). The repo's own
test_aten_writer_fill_is_capture_safe passes in isolation.

Indicates the captured fill's correctness depends on allocation lifetime /
layout, not just the persistent-buffer contract. Flagged for upstream
follow-up; not part of the TM-binding verdict.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C_  # noqa: E402

from lmdeploy.pytorch.backends.tm_kv import fill_kv_cache_tm  # noqa: E402


def build_capture(pool, k_cache, max_k_len, use_binding=True):
    bo = C_.block_offsets()
    q_buf = torch.zeros(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_buf = torch.zeros(C_.BATCH, C_.NUM_KV_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_buf = torch.zeros(C_.BATCH, C_.NUM_KV_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    out_buf = torch.zeros(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_new_buf = torch.zeros(C_.BATCH, C_.NUM_KV_HEADS, C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_new_buf = torch.zeros(C_.BATCH, C_.NUM_KV_HEADS, C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_len = torch.ones(C_.BATCH, dtype=torch.long, device='cuda')
    q_loc = torch.arange(C_.BATCH, dtype=torch.long, device='cuda')

    def step():
        if use_binding:
            ptrs, cu_base = C_.pointer_tables(k_cache)
            C_.tm_decode(q_buf, k_buf, v_buf, out_buf, ptrs, cu_base, max_k_len=max_k_len)
        else:
            kv = torch.full((C_.BATCH, ), max_k_len, dtype=torch.long, device='cuda')
            fill_kv_cache_tm(k_new_buf, v_new_buf, *C_.layer_views(pool), q_loc, q_len, kv, bo)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph, dict(bo=bo, q=q_buf, k=k_buf, v=v_buf, out=out_buf,
                       k_new=k_new_buf, v_new=v_new_buf)


def main():
    pool_cap = C_.allocate_pool()
    pool_ref = C_.allocate_pool()
    graph, bufs = build_capture(pool_cap, C_.layer_views(pool_cap)[0], C_.KLEN, use_binding=False)
    oks = []
    for i in range(3):
        _, kk, vv = C_.make_qkv(seed=300 + i, klen=1)
        bufs['k_new'].copy_(kk[:, 0])
        bufs['v_new'].copy_(vv[:, 0])
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        fill_kv_cache_tm(kk[:, 0], vv[:, 0], *C_.layer_views(pool_ref),
                         torch.arange(C_.BATCH, dtype=torch.long, device='cuda'),
                         torch.ones(C_.BATCH, dtype=torch.long, device='cuda'),
                         torch.full((C_.BATCH,), C_.KLEN, dtype=torch.long, device='cuda'), bufs['bo'])
        torch.cuda.synchronize()
        cap_nz = int((pool_cap.pools[0].tensor != 0).sum().item())
        ref_nz = int((pool_ref.pools[0].tensor != 0).sum().item())
        same = bool(torch.equal(pool_cap.pools[0].tensor, pool_ref.pools[0].tensor))
        oks.append(same)
        print(f'r{i}: equal={same} cap_nz={cap_nz} ref_nz={ref_nz}')
    print('anomaly reproduced' if not all(oks) else 'no anomaly in this process shape')


if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()
