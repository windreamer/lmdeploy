---
id: 09
title: tiny ct 模型与 ct→AutoAWQ repack 通路验证
labels: [wayfinder:task]
status: closed
assignee: windreamer (subagent TinyCtPipeline, 2026-11-16)
blocked-by: []
---

## Question

按票 08 结论与用户拍板（2026-11-16）：在 3090 上先建 tiny 级 compressed-tensors 小模型，把「ct 权重 → repack(AutoAWQ 布局) → pytorch AWQ Triton 路径 / TM GEMM 通道」整条通路走通，为 wtdcode 190.8GB 真模 repack 铺路；全量真模验证移交上游 GitHub CI（H200 + 预下载模型）。

1. tiny ct 模型：以真实几何为基座（Llama-3-8B 或 GLM-5.3-flash config，`tests/turbomind/linear/models.yaml` 已有 glm5 系条目），产出 compressed-tensors W4A16 格式 tiny 变体（真实 ct 打包语义，保形状砍层）。
2. repack 转换器雏形：ct → AutoAWQ 布局（qweight int32 位打包 / scales / qzeros，sym 无 zero_point 按 AutoAWQ 约定补零），流式按分片，tiny 上验证往返。
3. 通路冒烟：repack 产物分别进 pytorch AWQ Triton 路径与 TM 通道（`turbomind_awq_modules` 模板），同权重双路 KL ≈ 0。

## Resolution（2026-11-16,agent 执行,交付全绿）

1. **tiny ct 生成器**（`.scratch/ticket09/gen_tiny_ct.py`）：Llama-3-8B 真实几何砍 2 层,真实 compressed-tensors W4A16 g128 sym pack-quantized 语义,零下载。
2. **ct→AutoAWQ 转换器**（`ct_to_awq.py`）：8 层往返逐位 err=0.0（AWQ 沿 N 打包 nibble 序 \[0,4,1,5,2,6,3,7\]、qzeros=0x88888888 补零约定）。
3. **双路冒烟**（`smoke_dual_results.json`）：AWQ Triton impl vs TM GEMM 通道,同权重同输入,impl 级对照——**argmax 一致率 8/8 = 1.000**;KL 在 K=4096 六层 ≈0,仅 down_proj（K=14336）m≥16 时 1.9e-4/2.2e-5,定位为 kernel 切分累加顺序差（两路对 fp32 参考 relerr 相当,TM 2.75e-4 优于 Triton 4.0e-4）,非 repack 错误。

**带进规格的输入**:

- 票 05 KL 阈值 amendment 的首个数据点:tm-vs-awq 最坏 1.9e-4（m≥16 大 K 层）。
- 票 04 决议的实证:`_turbomind_gemm_capability_supported` 的 sm90 硬门应按票 04 决议由 registry 探询取代——u4 g128 家族在 sm86 实测可用（本票 8 层在 3090 全跑通）。
- C++ 绑定小缺口:`set_measure`、`WeightPlan.family` 查询（票 01 差距表 #6 扩充）。
- 工程陷阱备忘:AWQ nibble 序非对合;`torch.full` int32 0x88888888 字面量溢出。
