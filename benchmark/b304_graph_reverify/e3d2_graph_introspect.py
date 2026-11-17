"""e3d2: introspect the captured CUDA graphs via cudaGraphGetNodes.

Compares node counts/types for:
  graph_torch: capture of the pointer-table rebuild + a trivial torch write
               (pure-torch control — must capture normally);
  graph_bind:  capture of pointer-table rebuild + the TM decode binding call
               (the contested path).

If graph_bind lacks the binding's memcpy/kernel nodes, the binding's work was
silently dropped at capture (replays are inert by construction).

Also re-checks out_buf state transitions around the first replay.
"""
import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C_  # noqa: E402
import e3_capture_stable as E3  # noqa: E402
import torch  # noqa: E402

assert torch.cuda.is_available()

# cudaGraphNodeType: 0=kernel,1=memcpy,2=memset,3=host,4=graph,5=event_record
# 6=event_wait,7=condition,8=conditional,9=alloc,10=free,11=memcpy_from_symbol
# 12=captures...
TYPES = {0: 'kernel', 1: 'memcpy', 2: 'memset', 3: 'host', 4: 'graph', 5: 'event_record',
         6: 'event_wait', 8: 'conditional', 9: 'alloc', 10: 'free', 11: 'memcpy_from_symbol',
         12: 'memcpy_to_symbol'}


def graph_nodes(raw_handle):
    cudart = ctypes.CDLL('libcudart.so')
    get_nodes = cudart.cudaGraphGetNodes
    get_nodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    n = ctypes.c_size_t(0)
    rc = get_nodes(ctypes.c_void_p(raw_handle), None, ctypes.byref(n))
    assert rc == 0, f'cudaGraphGetNodes size rc={rc}'
    nodes = (ctypes.c_void_p * n.value)()
    rc = get_nodes(ctypes.c_void_p(raw_handle), nodes, ctypes.byref(n))
    assert rc == 0, f'cudaGraphGetNodes rc={rc}'
    get_type = cudart.cudaGraphNodeGetType
    get_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    hist = {}
    for i in range(n.value):
        t = ctypes.c_int(-1)
        rc = get_type(nodes[i], ctypes.byref(t))
        assert rc == 0
        name = TYPES.get(t.value, f'type{t.value}')
        hist[name] = hist.get(name, 0) + 1
    return n.value, hist


def main():
    pool = C_.allocate_pool()
    E3.POOL_TENSOR = pool.pools[0].tensor
    _, k, v = C_.make_qkv(seed=11, klen=C_.KLEN)
    C_.bridge_fill(k.reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   v.reshape(-1, C_.NUM_KV_HEADS, C_.HEAD_DIM),
                   [C_.KLEN] * C_.BATCH, pool)
    torch.cuda.synchronize()
    k_cache = C_.layer_views(pool)[0]

    # ---- graph_torch: tables + trivial torch op --------------------------
    C_.block_offsets()
    sink = torch.zeros(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_const = torch.ones(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    def step_torch():
        ptrs, cu_base = C_.pointer_tables(k_cache)
        sink.copy_(q_const * 2.0 + ptrs.sum() * 0)  # trivial captured writes

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step_torch()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g_torch = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g_torch):
        step_torch()
    try:
        handle_t = g_torch.raw_cuda_graph()
    except AttributeError:
        handle_t = ctypes.cast(g_torch.pool(), ctypes.c_void_p)  # fallback won't be used
    n_t, hist_t = graph_nodes(handle_t)
    C_.record('e3d2', graph='torch-control', nodes=n_t, types=hist_t)

    # ---- graph_bind: tables + TM binding ---------------------------------
    q_buf = torch.zeros(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_buf = torch.zeros(C_.BATCH, C_.NUM_KV_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_buf = torch.zeros(C_.BATCH, C_.NUM_KV_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    out_buf = torch.zeros(C_.BATCH, C_.NUM_HEADS * C_.HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    def step_bind():
        ptrs, cu_base = C_.pointer_tables(k_cache)
        C_.tm_decode(q_buf, k_buf, v_buf, out_buf, ptrs, cu_base, max_k_len=C_.KLEN)

    s2 = torch.cuda.Stream()
    s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        step_bind()
        step_bind()
    torch.cuda.current_stream().wait_stream(s2)
    torch.cuda.synchronize()
    g_bind = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g_bind):
        step_bind()
    bufs = dict(q=q_buf, k=k_buf, v=v_buf, out=out_buf)
    handle_b = g_bind.raw_cuda_graph()
    n_b, hist_b = graph_nodes(handle_b)
    C_.record('e3d2', graph='tm-binding', nodes=n_b, types=hist_b)

    # ---- out_buf transitions around replay --------------------------------
    q0, _, _ = C_.make_qkv(seed=3, klen=1)
    out_after_capture = bufs['out'].clone()
    E3.set_step_inputs(bufs, q0, k[:, 255], v[:, 255])
    torch.cuda.synchronize()
    out_pre_replay = bufs['out'].clone()
    g_bind.replay()
    torch.cuda.synchronize()
    out_post_replay = bufs['out'].clone()
    C_.record('e3d2', probe='out_after_capture_iszero', zero=bool((out_after_capture == 0).all().item()))
    C_.record('e3d2', probe='out_pre_minus_after', changed=not torch.equal(out_pre_replay, out_after_capture),
              zero=bool((out_pre_replay == 0).all().item()))
    C_.record('e3d2', probe='replay_changed_out', changed=not torch.equal(out_post_replay, out_pre_replay),
              post_zero=bool((out_post_replay == 0).all().item()))

    # g_torch replay control: does a pure-torch graph behave?
    g_torch.replay()
    torch.cuda.synchronize()
    C_.record('e3d2', probe='torch_graph_replay_works', sink_all_two=bool((sink == 2.0).all().item()))

    C_.save_results('e3d2_results.json')
    print('E3D2 DONE')


if __name__ == '__main__':
    main()
