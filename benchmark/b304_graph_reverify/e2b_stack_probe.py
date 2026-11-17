"""E2b-stack-probe: decisive diagnostics for the E2 finding (capture succeeds
but replay != eager by ~0.15).

Probes, in order:
  P0 kernel determinism: eager dispatch x2 (identical buffers) -> bitwise?
  P1 replay determinism: replay x2 (no interference)      -> bitwise?
  P2 host-stack interleaving: between two replays, run the binding EAGERLY
     with max_k_len=256 but against a SINK pool (pool2): device state of the
     captured pool (pool1) is untouched; the only perturbation is the host
     stack frame the binding's internal H2D copies read (BuildDecodePlan
     copies finished/cu_q_len/cu_k_len from pageable stack arrays; under
     capture those source addresses are baked into the graph's memcpy nodes).
     If replay output changes, replays consume volatile host memory.
  P3 same with max_k_len=192 on the sink pool.

Capture geometry: ptrs table baked to pool1 (k_cache of pool1).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C_  # noqa: E402
import torch  # noqa: E402

assert torch.cuda.is_available()


def setup_pool():
    pool = C_.allocate_pool()
    _, k, v = C_.make_qkv(seed=11, klen=C_.KLEN)
    C_.bridge_fill(k[:, :C_.PREFILL_LEN].reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   v[:, :C_.PREFILL_LEN].reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   [C_.PREFILL_LEN] * C_.BATCH, pool)
    torch.cuda.synchronize()
    return pool


def nonzero_bytes(t):
    return int((t != 0).sum().item())


def main():
    pool1 = setup_pool()
    pool2 = setup_pool()  # sink for interference re-stores
    kc1, kc2 = C_.layer_views(pool1)[0], C_.layer_views(pool2)[0]
    q, _, _ = C_.make_qkv(seed=3, klen=1)
    _, k, v = C_.make_qkv(seed=11, klen=C_.KLEN)
    q_flat, k_flat, v_flat = C_.q2d(q), C_.rows2d(k[:, 191]), C_.rows2d(v[:, 191])
    out_flat = torch.empty(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    def eager(ptrs, cu_base, mkl, out=None):
        out = out_flat if out is None else out
        C_.tm_decode(q_flat, k_flat, v_flat, out, ptrs, cu_base, max_k_len=mkl)
        torch.cuda.synchronize()
        return out.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM)

    p1, cb1 = C_.pointer_tables(kc1)
    p2, cb2 = C_.pointer_tables(kc2)

    # ---- P0: eager determinism ------------------------------------------
    e_a = eager(p1, cb1, 192)
    e_b = eager(p1, cb1, 192)
    C_.record('e2b', probe='P0/eager-determinism', bitwise=bool(torch.equal(e_a, e_b)),
              max_abs=round(C_.metrics(e_a, e_b)[0], 8))

    # ---- capture ---------------------------------------------------------
    def step_pool1():
        C_.tm_decode(q_flat, k_flat, v_flat, out_flat, p1, cb1, max_k_len=192)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step_pool1()
        step_pool1()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step_pool1()

    # ---- P1: replay determinism, no interference --------------------------
    out_flat.zero_()
    graph.replay()
    torch.cuda.synchronize()
    A = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    graph.replay()
    torch.cuda.synchronize()
    A2 = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    C_.record('e2b', probe='P1/replay-determinism', bitwise=bool(torch.equal(A, A2)),
              max_abs=round(C_.metrics(A, A2)[0], 8))
    m = C_.metrics(A, e_a)
    C_.record('e2b', probe='P1/replay-vs-eager', max_abs=round(m[0], 6),
              bitwise=bool(torch.equal(A, e_a)))

    # ---- P2: host-stack perturbation via eager@256 on the sink pool ------
    tail1_before = nonzero_bytes(pool1.pools[0].tensor)
    eager(p2, cb2, 256)
    tail1_after = nonzero_bytes(pool1.pools[0].tensor)
    graph.replay()
    torch.cuda.synchronize()
    B = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    C_.record('e2b', probe='P2/stack-interference@256', pool1_tail_unchanged=tail1_before == tail1_after,
              replay_changed_vs_A=bool(not torch.equal(B, A)),
              B_vs_A_max_abs=round(C_.metrics(B, A)[0], 6),
              B_vs_eager192_max_abs=round(C_.metrics(B, e_a)[0], 6))

    # ---- P3: same with max_k_len=192 on the sink pool --------------------
    eager(p2, cb2, 192)
    graph.replay()
    torch.cuda.synchronize()
    Cc = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    C_.record('e2b', probe='P3/stack-interference@192',
              C_vs_A_bitwise=bool(torch.equal(Cc, A)),
              C_vs_A_max_abs=round(C_.metrics(Cc, A)[0], 6),
              C_vs_B_bitwise=bool(torch.equal(Cc, B)),
              C_vs_B_max_abs=round(C_.metrics(Cc, B)[0], 6))

    # ---- P4: interference with mutated CONTENT rows (device probe) -------
    q4, _, _ = C_.make_qkv(seed=4, klen=1)
    _, k4, v4 = C_.make_qkv(seed=12, klen=C_.KLEN)
    q_flat.copy_(C_.q2d(q4))
    k_flat.copy_(C_.rows2d(k4[:, 191]))
    v_flat.copy_(C_.rows2d(v4[:, 191]))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    D = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    e_d = eager(p1, cb1, 192)
    m = C_.metrics(D, e_d)
    C_.record('e2b', probe='P4/content-follow', replay_vs_eager_bitwise=bool(torch.equal(D, e_d)),
              replay_vs_eager_max_abs=round(m[0], 6))

    C_.save_results('e2b_results.json')
    print('E2B DONE')


if __name__ == '__main__':
    main()
