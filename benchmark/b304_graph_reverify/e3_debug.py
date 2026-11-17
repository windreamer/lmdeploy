"""e3_debug: pinpoint which captured inputs the TM decode replay actually
consumes, and which of {replay-pre-control, replay-post-control, eager} is
closest to the dequant ground truth.

Steps after one capture (in-capture pointer tables, max_k_len=256, full
window):
  D0 baseline: replay -> O1
  D1 mutate q_buf only       -> replay -> O2   (does replay follow q?)
  D2 mutate k/v rows only    -> replay -> O3   (does replay follow kv rows?)
  D3 zero out_buf, replay    -> O4             (does the kernel node re-run/write?)
  D4 eager control -> E; E vs dequant-ref; O vs dequant-ref per candidate
  D5 pure-host deep activity (no device change): eager control again ->
     replay -> O5 vs O4 (stack sensitivity without device change)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C_  # noqa: E402
import e3_capture_stable as E3  # noqa: E402

assert torch.cuda.is_available()


def main():
    pool = C_.allocate_pool()
    E3.POOL_TENSOR = pool.pools[0].tensor
    _, k, v = C_.make_qkv(seed=11, klen=C_.KLEN)
    C_.bridge_fill(k.reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   v.reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   [C_.KLEN] * C_.BATCH, pool)
    torch.cuda.synchronize()
    k_cache = C_.layer_views(pool)[0]

    graph, bufs = E3.build_capture(pool, k_cache, C_.KLEN)
    print('buf ptrs', {n: bufs[n].data_ptr() for n in ('q', 'k', 'v', 'out')}, flush=True)

    q0, _, _ = C_.make_qkv(seed=3, klen=1)
    E3.set_step_inputs(bufs, q0, k[:, 255], v[:, 255])
    torch.cuda.synchronize()

    def dequant_ref(q):
        kd, vd = C_.dequant_pool(E3.POOL_TENSOR)
        return C_.reference(q, kd, vd, C_.KLEN)

    def cmp(name, a, b):
        C_.record('e3d', probe=name, max_abs=round(C_.metrics(a, b)[0], 8),
                  bitwise=bool(torch.equal(a, b)))

    graph.replay(); torch.cuda.synchronize()
    O1 = bufs['out'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    ref0 = dequant_ref(q0)
    C_.gate('e3d', O1, ref0, 'D0/replay-vs-dequant-ref')
    E = E3.eager_same_args(bufs, k_cache, C_.KLEN)
    cmp('D0/O1-vs-eager', O1, E)

    # D1: q only
    q1, _, _ = C_.make_qkv(seed=99, klen=1)
    bufs['q'].copy_(C_.q2d(q1))
    torch.cuda.synchronize()
    graph.replay(); torch.cuda.synchronize()
    O2 = bufs['out'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    cmp('D1/O2-vs-O1', O2, O1)
    C_.gate('e3d', O2, dequant_ref(q1), 'D1/replay-vs-dequant-ref(new q)')
    E2 = E3.eager_same_args(bufs, k_cache, C_.KLEN)
    cmp('D1/O2-vs-eager', O2, E2)

    # D2: k/v rows only
    _, k2, v2 = C_.make_qkv(seed=98, klen=C_.KLEN)
    bufs['k'].copy_(C_.rows2d(k2[:, 255]))
    bufs['v'].copy_(C_.rows2d(v2[:, 255]))
    torch.cuda.synchronize()
    graph.replay(); torch.cuda.synchronize()
    O3 = bufs['out'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    cmp('D2/O3-vs-O2', O3, O2)
    E3.POOL_TENSOR[0]  # keep alive
    # pool still holds seed-11 rows at slot 255 (no writer call), so the
    # kernel's re-store will overwrite slot 255 with k2/v2 rows; reference
    # must be computed AFTER the re-store to include it.
    ref2 = dequant_ref(q1)
    C_.gate('e3d', O3, ref2, 'D2/replay-vs-dequant-ref(after re-store)')
    E3p = E3.eager_same_args(bufs, k_cache, C_.KLEN)
    cmp('D2/O3-vs-eager', O3, E3p)

    # D3: zero out, replay (does the kernel node write at all?)
    bufs['out'].zero_()
    torch.cuda.synchronize()
    graph.replay(); torch.cuda.synchronize()
    O4 = bufs['out'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    nz = int((O4 != 0).sum().item())
    C_.record('e3d', probe='D3/out-nonzero-after-replay', nonzero=nz)
    cmp('D3/O4-vs-O3', O4, O3)

    # D5: pure host-side perturbation: run the eager binding dispatch (no
    # device-visible state change beyond idempotent re-store) then replay
    E3.eager_same_args(bufs, k_cache, C_.KLEN)
    graph.replay(); torch.cuda.synchronize()
    O5 = bufs['out'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    cmp('D5/O5-vs-O4', O5, O4)

    # D6: a SECOND graph captured with max_k_len=192 (different ko pattern),
    # replayed against the same buffers -> does its replay read its own
    # stack values (i*192)?
    graph2, bufs2 = E3.build_capture(pool, k_cache, 192)
    bufs2['q'].copy_(bufs['q']); bufs2['k'].copy_(bufs['k']); bufs2['v'].copy_(bufs['v'])
    torch.cuda.synchronize()
    graph2.replay(); torch.cuda.synchronize()
    O6 = bufs2['out'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM).clone()
    # true-length is 256; a 192-window consumer attends 192 keys
    kd, vd = C_.dequant_pool(E3.POOL_TENSOR)
    p = C_.reference(bufs2['q'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM), kd, vd, C_.KLEN).softmax(-1)
    p192 = C_.reference(bufs2['q'].view(C_.BATCH, C_.NUM_HEADS, C_.HEAD_DIM), kd, vd, 192).softmax(-1)
    dist_full = (p - O6.float().softmax(-1)).abs().max().item()
    dist_192 = (p192 - O6.float().softmax(-1)).abs().max().item()
    C_.record('e3d', probe='D6/graph192-replay', dist_to_full_window_prob=round(dist_full, 6),
              dist_to_192_window_prob=round(dist_192, 6))

    C_.save_results('e3d_results.json')
    print('E3D DONE')


if __name__ == '__main__':
    main()
