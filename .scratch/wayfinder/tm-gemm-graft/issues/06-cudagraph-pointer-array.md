---
id: 06
title: CUDA Graph 下 per-step 指针数组的稳定性
labels: [wayfinder:research]
status: closed
assignee:
blocked-by: []
---

## Question

TurboMind attention kernel 的设备端指针数组（`BlockIteratorParams{char** block_ptrs}`，`attention_params.h:17-21`）能否在 PyTorch Engine 的 `graph_runner`（`backends/graph_runner/`）capture/replay 下工作？

背景：「固定 buffer 地址 + 每步仅改写 buffer 内容」是 CUDA Graph 重放的标准做法；TM 的块池是 slab 分配，块地址分配后稳定，理论上指针数组只需原地改写。但需确认 PyTorch 侧事实。此票为 attention 第二批次铺路，也是淘汰叙事的技术前提。

1. `graph_runner` 的 capture 边界：哪些输入是捕获期冻结的指针、哪些允许 replay 期改写内容？attention metadata 走哪条路？
2. `block_offsets`（`backends/attention.py:16`）到指针数组的转换在哪层做，能否预分配 pinned + device 常驻缓冲每步原地改写？
3. V4 step context（`attention.py:29,79`）路径是否已有等价的稳定缓冲先例？
4. piecewise / standard / full 三种 graph 模式（`backends/graph_runner/`）对该方案的影响差异。

产出：可行性结论 + 限制清单，写入 `.scratch/research/cudagraph-pointer-array.md`，带 `文件:行` 引用。

## Resolution（2026-11-16）

**可行**。full/piecewise/eager 三条执行路径均无根本障碍（票面"三种模式"已修正：standard 是 PCG 的 prefill 策略，第三条路径是 eager）。

- 推荐实现：注册 `CudaAttentionMetaBuilder`（接缝已存在，`step_metadata.py:72-108`），runner 常驻指针数组缓冲，每步从稳定 `input_buffers['block_offsets']` 做 **device 侧 gather**（`ptr = pool_base + block_id×block_bytes`），与现有 `fill_graph_common_inputs` 同构；host pinned 分级为备选。机制先例齐备：DSAIndexerMetaBuilder 与本方案完全同构。
- full graph 是唯一需按缓冲契约施工的路径；piecewise 下 attention 本就是 eager boundary；eager 无约束。
- 关键限制（全 8 条见研究文件）：地址变更必经 `CUDAGraphRunner.reset()`；池 storage 永不搬移的不变量须与 cache 引擎侧约定并保持；`readonly_block_num`/`cu_q_len`/`cu_k_len` 需同一 builder 稳定化（`cu_seqlens` 稳定缓冲可复用）。
- 详细结论与引用：`.scratch/research/cudagraph-pointer-array.md`
