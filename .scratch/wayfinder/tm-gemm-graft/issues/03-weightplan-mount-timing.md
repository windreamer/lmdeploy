---
id: 03
title: WeightPlan 挂载时机与显存开销
labels: [wayfinder:research]
status: open
assignee:
blocked-by: []
---

## Question

`DataFormat→Family→WeightPlan→pack` 权重打包管线挂在 PyTorch Engine 加载流程的哪个位置？显存与时间的代价是什么？

背景：TM 侧 `LinearWeight` 携 `DataFormat` + `WeightPlan`，`prepare()/set_plan()` 在设备上重排（`linear_weight.h:60-61,106`）；python 侧归一化在 `lmdeploy/turbomind/weight_format.py`（六种 WeightFormat，AWQ/GPTQ 的 TM-native 布局注释 :265-300）与 `builders/`。PyTorch 侧挂接点 `agent.py:47-49` 的 `load_model_weights` / `process_weights_after_loading`。论文基线：70B 模型打包 1.47s（§5.3）。

1. pack 是就地转换还是 packed 与原始权重短暂共存？峰值显存倍数是多少（以 W4A16 7B 为例算一遍）？
2. `process_weights_after_loading` 的执行上下文（是否已分 rank 切片、是否有空档做设备端重排）适合放 pack 吗？
3. `update_weights`（热更新）路径走 `update_weights` 还是重走 pack？
4. `weight_format.py` 的归一化（checkpoint → TM-native）目前依赖 TM 加载流程的哪些前提？脱离 C++ runtime 后还成立吗？

产出：挂载点建议 + 显存/时间事实表，写入 `.scratch/research/weightplan-mount.md`，带 `文件:行` 引用。
