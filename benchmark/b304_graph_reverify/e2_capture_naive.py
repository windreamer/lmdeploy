"""E2: CAPTURE-NAIVE — reproduce the contested path: dispatch the TM int8
decode binding inside `torch.cuda.graph` capture, consuming capture-time
metadata buffers.

Sub-cases:
  e2b-exact:  metadata exact (max_k_len = true uniform len 192).
  e2a-padded: metadata padded (max_k_len = 256 window cap, pool rows beyond
              192 neutral zeros), mirroring production's padded max_kv_seqlen.

Each case: eager control dispatch first (identical args, outside capture),
then warmup + capture + replay; replay vs eager control bitwise, plus a
mutation probe (buffer contents changed device-side between replays).
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C_  # noqa: E402

assert torch.cuda.is_available()


def setup(seed_q=3):
    pool = C_.allocate_pool()
    _, k, v = C_.make_qkv(seed=11, klen=C_.KLEN)
    q, _, _ = C_.make_qkv(seed=seed_q, klen=1)
    C_.bridge_fill(k[:, :C_.PREFILL_LEN].reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   v[:, :C_.PREFILL_LEN].reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   [C_.PREFILL_LEN] * C_.BATCH, pool)
    torch.cuda.synchronize()
    return pool, q, k, v


def tail_nonzero(pool_tensor):
    """Non-zero byte count in each sequence's last block (rows 192..255)."""
    ids = [b * C_.C + (C_.C - 1) for b in range(C_.BATCH)]
    return sum(int((pool_tensor[0, i] != 0).sum().item()) for i in ids)


def dispatch(pool, q_flat, k_flat, v_flat, out_flat, max_k_len):
    k_cache = C_.layer_views(pool)[0]
    ptrs, cu_base = C_.pointer_tables(k_cache)
    C_.tm_decode(q_flat, k_flat, v_flat, out_flat, ptrs, cu_base, max_k_len=max_k_len)


def run_case(label, max_k_len):
    pool, q, k, v = setup()
    pool_tensor = pool.pools[0].tensor
    q_flat = C_.q2d(q)
    k_flat = C_.rows2d(k[:, C_.PREFILL_LEN - 1])
    v_flat = C_.rows2d(v[:, C_.PREFILL_LEN - 1])
    out_flat = torch.empty(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    # -- eager control with identical args --------------------------------
    dispatch(pool, q_flat, k_flat, v_flat, out_flat, max_k_len)
    torch.cuda.synchronize()
    eager_ref = out_flat.clone().view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM)
    C_.record('e2', label=label, note='eager control ok', max_k_len=max_k_len,
              eager_tail_nonzero_bytes=tail_nonzero(pool_tensor))
    if max_k_len > C_.PREFILL_LEN:
        m = C_.metrics(eager_ref, C_.reference(q, k, v, C_.PREFILL_LEN))
        C_.record('e2', label=label, note='eager padded vs TRUE-length bf16 ref '
                  '(semantic gap; report-only, not a capture claim)',
                  max_abs=round(m[0], 6), kl=f'{m[1]:.3e}')

    # -- capture the contested path ---------------------------------------
    def step():
        dispatch(pool, q_flat, k_flat, v_flat, out_flat, max_k_len)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(s):
            step()
            step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    except Exception as e:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        C_.record('e2', label=label, capture='FAILED',
                  exc_type=type(e).__name__, exc=str(e)[:600])
        return

    out_flat.zero_()
    graph.replay()
    torch.cuda.synchronize()
    out_r = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM)
    m = C_.metrics(out_r, eager_ref)
    C_.record('e2', label=label, capture='OK',
              replay_vs_eager_max_abs=round(m[0], 8),
              replay_vs_eager_bitwise=bool(torch.equal(out_r, eager_ref)))

    # -- mutation probe: fresh contents device-side, replay must follow ----
    q2, _, _ = C_.make_qkv(seed=4, klen=1)
    _, k2, v2 = C_.make_qkv(seed=12, klen=C_.KLEN)
    q_flat.copy_(C_.q2d(q2))
    k_flat.copy_(C_.rows2d(k2[:, C_.PREFILL_LEN - 1]))
    v_flat.copy_(C_.rows2d(v2[:, C_.PREFILL_LEN - 1]))
    dispatch(pool, q_flat, k_flat, v_flat, out_flat, max_k_len)
    torch.cuda.synchronize()
    eager2 = out_flat.clone().view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM)

    out_flat.zero_()
    graph.replay()
    torch.cuda.synchronize()
    out_r2 = out_flat.view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM)
    m2 = C_.metrics(out_r2, eager2)
    C_.record('e2', label=label, probe='mutation',
              replay_vs_eager_max_abs=round(m2[0], 8),
              replay_vs_eager_bitwise=bool(torch.equal(out_r2, eager2)))
    C_.gate('e2', out_r2, C_.reference(q2, k2, v2, C_.PREFILL_LEN),
            f'{label}/mutation-vs-bf16-ref',
            budget=C_.MAX_ABS_BUDGET if max_k_len == C_.PREFILL_LEN else 9e9)


if __name__ == '__main__':
    try:
        run_case('e2b-exact', C_.PREFILL_LEN)
    except Exception:
        C_.record('e2', label='e2b-exact', fatal=traceback.format_exc()[-600:])
    try:
        run_case('e2a-padded', C_.KLEN)
    except Exception:
        C_.record('e2', label='e2a-padded', fatal=traceback.format_exc()[-600:])
    C_.save_results('e2_results.json')
    print('E2 DONE')
