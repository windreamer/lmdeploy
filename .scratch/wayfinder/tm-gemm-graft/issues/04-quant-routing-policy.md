---
id: 04
title: 混合精度路由策略
labels: [wayfinder:grilling]
status: closed
assignee: windreamer (2026-11-16)
blocked-by: [02]
---

## Question

哪些 Linear 实例路由到 TurboMind GEMM 后端？路由策略与配置旋钮的形态是什么？

背景：现有 CUDA 后端的量化变体分发（awq/w8a8/blocked-fp8/static-fp8/MoE 各变体）在 `backends/cuda/op_backend.py:33-250`；TM 家族覆盖 INT4/8×FP16（论文 §4.3 的收益域）。开工时调 `grilling` + `domain-modeling`（HITL）。

1. 路由键：按 quant tag（仅 AWQ/GPTQ W4A16？INT8×FP16？）还是显式 per-layer 配置？
2. 不在 TM 覆盖内的量化变体回退到现有路径（awq Triton、blocked-fp8/deep_gemm）的机制——同一个 OpsBackend 内双实现共存还是两个后端互斥？
3. 配置旋钮形态：`--backend pytorch --tm-gemm on|auto|off`？模型级还是全局级？
4. TM GEMM 覆盖矩阵（层数×家族×shape 约束）里，哪些 shape 落 cublas 兜底，兜底时要不要告警？

## Resolution（2026-11-16,HITL 评审通过）

四项决议（全部按推荐）:

1. **路由机制**:保持 `LMDEPLOY_W4A16_GEMM_BACKEND` env 三态不变（`envs.py:187`,默认 auto）,仅把探询实现从硬编码 `_turbomind_support_reason` 换成 registry 探询（`get_weight_plan`,自带 arch/shape 过滤,是 ground truth）。不引入第二套路由。
2. **旋钮节奏**:T1 保持 env-only;CLI/config 旋钮随票 07 兼容层在 /to-spec 一并设计（`--backend turbomind` 别名语境）。
3. **下一批格式优先级**:GPTQ（sym g128,与 AWQ 同 pack 管线,增量 = 一个 WeightFormat 子类）→ w8a8-fp8（激活走 ConvertInput）→ nvfp4/mxfp4（SM90 限定,CI 验）。
4. **auto 探询粒度**:per-module（现状行为）——每个 Linear 独立探询,shape 不满足者单独回落 Triton。

**Amendment(2026-11-16,路线图定稿)**:按能力决策树,w8a8-fp8 与 fp4 系 kernel 仅 SM90、本机不能测 → 顺延至 CI 批(默认保留,留 H200 评估);格式扩展修正为「GPTQ(批 2)→ 其余 CI 批」。见 ROADMAP.md。
