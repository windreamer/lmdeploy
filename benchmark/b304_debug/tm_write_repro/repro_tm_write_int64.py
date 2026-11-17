# Copyright (c) OpenMMLab. All rights reserved.
"""[TMWriteBench] Minimal repro for the "q_len=1 illegal access" in
`_tm.tm_process_kv` (ProcessKV_v2, kv_cache_utils_v2.cu:306).

Root cause: the cumulative-length buffers fed to `tm_process_kv` must be
**int32**. The repo harness helpers build them as
`pad(cumsum(int32_tensor), (1,0))`, but `torch.cumsum` promotes int32 to
**int64**; the kernel reads `const int*`, so the int64 array is seen as the
interleaved int32 sequence [v0, 0, v1, 0, ...]. Garbage q_len/k_len gives a
garbage timestep, a block-table index past the table, and a wild
`block_ptrs[block_id]` store (fault site: kv_cache_utils_v2.cu:191 via
block.h:111; the error surfaces at the launch check :306).

Run:  .venv/bin/python benchmark/b304_debug/tm_write_repro/repro_tm_write_int64.py
Expected output: INT64 CRASH (illegal access at :306) then INT32 OK with
byte-correct placement.
"""

import sys

import torch

sys.path.insert(0, "/data/windreamer/codebase/lmdeploy/benchmark/b304_graph_reverify")

from lmdeploy.pytorch.backends.tm_kv import tm_block_pointer_table
from lmdeploy.turbomind import _tm

import common as C_

B, H, D, S = C_.BATCH, C_.NUM_KV_HEADS, C_.HEAD_DIM, C_.BLOCK_LEN
KLEN = 193


def cu_int64(vals):
    """The harness pattern: pad(cumsum(int32)) -- promoted to int64."""
    return torch.nn.functional.pad(torch.cumsum(torch.tensor(vals, dtype=torch.int32), 0), (1, 0)).cuda()


def cu_int32(vals):
    """Correct: keep the whole chain int32."""
    out = torch.zeros(len(vals) + 1, dtype=torch.int32)
    acc = 0
    for i, v in enumerate(vals):
        acc += v
        out[i + 1] = acc
    return out.cuda()


def dispatch(cu_q, cu_k):
    pool = C_.allocate_pool()
    k_cache = C_.layer_views(pool)[0]
    _, k, v = C_.make_qkv(seed=11, klen=KLEN)
    kq = k[:, 192].reshape(-1, H, D).contiguous()
    vq = v[:, 192].reshape(-1, H, D).contiguous()
    ptrs, base = tm_block_pointer_table(C_.block_offsets(), k_cache.data_ptr(), k_cache.stride(0))
    _tm.tm_process_kv(
        _tm.from_dlpack(kq), _tm.from_dlpack(vq),
        ptrs.data_ptr(), cu_q.data_ptr(), cu_k.data_ptr(), base.data_ptr(),
        stride_c=H, stride_s=H, stride_h=1, max_q_len=1, num_kv_heads=H, head_dim=D,
        batch_size=B, block_len=S, kv_quant=8,
        stream_ptr=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return base


def main():
    print(f"cu helper dtype: {cu_int64([1] * B).dtype} (kernel expects int32)")
    # int32 first: the int64 crash poisons the CUDA context
    base = dispatch(cu_int32([1] * B), cu_int32([KLEN] * B))
    print(f"INT32 OK, bases {base.tolist()}")
    try:
        dispatch(cu_int64([1] * B), cu_int64([KLEN] * B))
        print("INT64: no crash (layout luck) -- still miswritten, see bench collateral check")
    except RuntimeError as e:
        print(f"INT64 CRASH: {str(e)[:120]}")


if __name__ == "__main__":
    main()
