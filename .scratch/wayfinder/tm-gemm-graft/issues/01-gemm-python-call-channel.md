---
id: 01
title: GEMM 的 python 调用通道盘点
labels: [wayfinder:research]
status: closed
assignee:
blocked-by: []
---

## Question

TurboMind 的混合精度 GEMM 家族（INT4/8×FP16）当前在 C++ 侧如何被调用？从 python 侧驱动它需要新增暴露什么？

背景：`.scratch/research/tm-pt-seam-inventory.md` §2——GEMM 资产在 `src/turbomind/kernels/gemm/`（Family 抽象、registry、dispatch_cache、cublas 兜底、weight_plan）；现有 python 绑定面 `src/turbomind/python/bind.cpp` 只确认了 DLPack Tensor（:698）、Module（:838）、TurboMind（:906），GEMM 相关暴露面未盘点；`kernels/linear_attn/python_bind.cpp` 是一个现成的绑定模式参照。

1. GEMM 的 C++ 调用入口链：`LinearWeight::prepare()` 打包后，谁按 Family 派发 kernel（`linear_weight.h:44`、`weight_plan.h:23`）？shape/stride/scales 参数怎么传？
2. `bind.cpp` 已暴露的符号里，哪些 GEMM 相关能力已经可用？
3. 若在 python 侧（TorchOpsBackend 的 LinearImpl）直接驱动：需要新增哪些绑定（family 选择、plan 查询、launch 入口、dispatch_cache 预热）？`linear_attn/python_bind.cpp` 的模式是否可复制？
4. DLPack 边界上，packed 权重与 activation 的 dtype/stride 约定是什么？

产出：事实清单 + 「需新增暴露面」差距表，写入 `.scratch/research/gemm-python-channel.md`，每条断言带 `文件:行` 引用。

## Resolution（2026-11-16）

**调用通道已端到端存在，无需新建绑定文件。** 票面假设过时：

- 完整链路已在 `src/turbomind/python/linear_bind.cpp` + `lmdeploy/turbomind/linear.py`：family 选择（`get_weight_plan`）、权重打包（`prepare_weight`）、plan 查询（`get_exec_plan`/`tune`）、launch（`forward_dense`/`forward_moe`）、dispatch-cache 持久化（`import/export_records`）全部可用。
- TorchOpsBackend 侧接入先例即 `TurbomindAwqLinearW4A16Impl`（`backends/cuda/turbomind_awq_modules.py`），可作复制模板推广到其它量化格式。
- 零星缺口：`set_measure` 未绑定（建议补）；引擎实例与独立 `LlamaLinear` 的 dispatch cache 不共享（引擎内权重须走 `TurboMind.gemm(index)`）；bias 无 epilogue（python 逐层加）；通信由 python `dist.all_reduce` 承担。
- 主要工作量在 python 侧 `WeightFormat` 子类与 pack 布局核对，不在 C++ 绑定。
- 详细差距表（10 项）与 DLPack dtype/stride 约定：`.scratch/research/gemm-python-channel.md`
