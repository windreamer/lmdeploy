<!--
Copyright (c) OpenMMLab. All rights reserved.
-->

# NOTES.md — TMWriteBench 排错记录:q_len=1 非法访存根因(2026-11-17)

结论:**`_tm.tm_process_kv` 的 kernel 与 binding 无缺陷。** 触发条件是调用方喂给
`cu_q_len_addr`/`cu_k_len_addr` 的累积长度数组是 **int64**,而 kernel 形参是
`const int*`。修复 = 构造链路全程 int32(见 `repro_tm_write_int64.py`)。

## 证据链

1. **复现**:decode 步形状(B=4, q_len=1, kv=193, H=8, D=128, S=64)经
   `common.py bridge_fill` 稳定崩溃,报错点 `kv_cache_utils_v2.cu:306`
   (launch 后的 `TM_CUDA_CHECK(cudaGetLastError())`,异步错误在此浮现)。
2. **fault 定位**(compute-sanitizer memcheck):`Invalid __global__ write of size 8 bytes ... misaligned` at `kv_cache_utils_v2.cu:191`(`Store(&k_cache[di])`),
   经 `block.h:111`(`block = block_ptrs[block_id]`),仅 batch 3(block z=3)出错。
3. **kernel 本身无缺陷**:同源码独立编译(driver + cudaMalloc + 阻塞 memcpy,
   `tm_write_repro/kernel_dbg.cu` 的 `main()`),参数全部正确读入
   (q_len=1, k_len=193, hist=192, ti=192, blkidx=3),零错误。
4. **torch 路径二分**:自建 pybind 模块(`mod_dbg.cu`,同一 kernel,无
   ContextGuard/from_dlpack)同样崩溃 ⟹ 与 TM binding 机制无关。
5. **kernel 内打印**(mod_dbg 版):四个 block 读到的指针一致,但内容为
   `[0,0,1,0,2]` / `[0,0,193,0,386]` 而非 `[0,1,2,3,4]` / `[0,193,386,579,772]`;
   device 侧计算的 `cu_block_base`/ptrs 表始终正确。
6. **根因**:`[0,0,1,0,2]` 恰为 int64 `[0,1,2,3,4]` 的 int32 小端视图
   (`torch.cumsum` 对 int32 提升 int64;`F.pad` 保持 int64)。q_len/k_len 变为
   交错垃圾(如 batch2 读出 q_len=-1、batch3 读出 k_len=386)→ ti=384 →
   `blkidx=6` → 表项 `cu_blk[3]+6=18` 越过 16 项表 → `block_ptrs[18]` 读到
   邻近堆字节当指针 → 野写 → IMA。是否崩溃取决于越界读到什么(布局敏感);
   `decode m=1/kv≤128` 等形状不崩溃但静默写错位置。
7. **修复验证**:`cu` 全程 int32(`repro_tm_write_int64.py` 的 `cu_int32`)后,
   同形状经同一 binding 无崩溃且逐字节落位正确;`bench_tm_write.py` 全矩阵
   19/19 绿(decode append m∈{1,8,32} 全通过)。

## 波及面

- 使用 `pad(cumsum(int32))` 模式的现有 helper 全部中招:
  `benchmark/b304_graph_reverify/common.py::cu`、
  `tests/pytorch/kernel/test_tm_int8_kv.py::_cu`、`benchmark/bench_tm_write.py`
  修复前的 `make_cu`。
- 均匀 batch 的 prefill 下,int32 视图的交错值仍可让写入「大体发生」,宽松的
  自洽门(池反量化 ref)挡不住;e1 第一门 argmatch=0.0 即为旁证(输出实为垃圾,
  仅 max_abs≤0.25 预算放行)。已复核:e1 风格 prefill 的原始偏移逐字节比对
  code match 仅 0.67%。
- 修复建议(下游提交,不在本票):各 helper 末尾 `.to(torch.int32)`,或在
  binding 文档标明「缓冲必须 int32 且对 launch 流可见」。

## 文件

- `kernel_dbg.cu` — kv_cache_utils_v2.cu 只读副本 + 参数打印 + 独立 driver(证据 3)。
- `kernel_clean.cu` — 无打印副本(rope 计时用)。
- `mod_dbg.cu` / `mod_dbg.so` / `stubs.cc` — 绕过 TM binding 的最小 pybind 包装(证据 4)。
- `repro_tm_write_int64.py` — 最小复现(INT32 OK → INT64 CRASH)。
- `rope_timing` 构建脚本与结果见 `benchmark/bench_tm_write_report.md` 附录。
