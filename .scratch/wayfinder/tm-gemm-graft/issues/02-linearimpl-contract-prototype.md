---
id: 02
title: LinearImpl 契约原型
labels: [wayfinder:prototype]
status: closed
assignee: windreamer (2026-11-16,subagent 构建 + HITL 评审通过)
blocked-by: [01]
---

## Question

TurboMind GEMM 的参数约定如何映射到 PyTorch Engine 的 `LinearImpl` 契约？做一个可评审的 stub，不要求跑通端到端。

背景（2026-11-16 更新，票 01 结论）：调用通道已端到端存在，**从零映射的前提不成立**。`TurbomindAwqLinearW4A16Impl`（`backends/cuda/turbomind_awq_modules.py`）是被验证的模板；本票的实质是把这个模板推广为通用 TM GEMM LinearImpl 时的契约评审：挂载点复用 `process_weights_after_loading` 的 per-module `update_weights` 钩子（票 03 结论）。

1. `forward` 的 x（activation，\[tokens, in\]）到 TM GEMM 入参（padding/alignment、是否需要转置、scales/zeros 的形状与排布）的适配写在哪一层？（DLPack 约定见 `.scratch/research/gemm-python-channel.md` §4）
2. `all_reduce/group/rank/scatter_size` 与 TM 的并行约定如何对应？（先例：python `dist.all_reduce`，`turbomind_awq_modules.py:69-70`）
3. `forward_batched`/`supports_batched` 对 TM GEMM 是否可支持（dispatch_cache 命中率影响）？
4. weight_loader 产物到 `DataFormat` 的桥接放 `process_weights_after_loading` 还是 LinearImpl 构造？（票 03 结论：钩子内）
5. torch 原始权重的释放做成什么形式的 opt-in（热更新落点约束，票 03 §3）？

## Resolution（2026-11-16,HITL 评审通过）

stub 评审通过,asset 留存 `prototype/tm_linear_impl/`(ops_backend_patch.py / tm_linear_impl.py / weight_release.py / README.md)。七项契约决议:

1. **激活量化**:kernel 内 `ConvertInput` 现场量化(T1 不做 python 预量化)。
2. **探询失败降级**:对齐现有 env 三态(`LMDEPLOY_W4A16_GEMM_BACKEND`,`envs.py:187`)——`auto` 态回落 Triton;`turbomind` 强制态显式 RuntimeError(票面 ② 与现有 `awq_modules.py:108-118` 行为一致);跨格式回退的产品策略归票 04。
3. **权重释放**:spill-only 一档**默认开启**——pack 后 GPU torch 权重转 CPU 精确副本并释放显存,热更新从 CPU 副本回灌重 pack;不做 immutable 档(省 CPU RAM 的 0/2 档弃)。
4. **初始加载**:去掉逐模块 stream sync(峰值钉在单层暂存上界);热更新路径保留 sync。
5. **gate/up fuse_weight:进 T1**(用户改推荐;`fuse_weight` 现成实现直接用)。
6. **nn 侧形态**:新 `TmGemmLinear` 模块,继承现有 BaseLinear(per-format 模块惯例,对齐 nn/linear/awq.py)。
7. **MLA 吸收豁免条款**(写入规格,备 GLM 档):被吸收模块(kv_b_proj 等,`deepseek_v2.py:500` 消费 `get_unquantized_weight`)永不参与释放,保留 torch 副本;TM 通道不补去量化导出绑定。

已知边界(带入实现):cudagraph 兼容(`is_graph_compatible`)未验证;独立 `get_linear()` 与引擎 `TurboMind.gemm(index)` dispatch cache 不共享;`set_measure` 未绑定(票 01 差距表 #6)。
