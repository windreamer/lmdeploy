---
id: 05
title: 验收口径
labels: [wayfinder:grilling]
status: closed
assignee: windreamer (2026-11-16)
blocked-by: []
---

## Question

嫁接后的数值与性能验收口径是什么？达标线由谁定义？

背景：论文基线——INT4×FP16 batch 1–16 平均比 FP16×FP16 快 134%（§5.3），对照组是 vLLM+MARLIN / TensorRT-LLM / OmniServe+QServe；数值验证用 perplexity 与 MMLU/GSM8K（§5.2 Table 1）。仓库现状：pytorch 后端已有 awq（W4A16 Triton）、blocked-fp8（deep_gemm）路径可作对照。开工时调 `grilling` + `domain-modeling`（HITL）。

1. 数值参考实现选谁：HF forward？pytorch 后端同量化路径？容差政策（rel 阈值、per-layer vs end-to-end）？
2. 性能基线选谁：pytorch 后端 awq 路径（同引擎对照）、原生 TurboMind（跨引擎对照）、还是外部 vLLM+MARLIN？三者口径不同，达标线不同。
3. 模型 × GPU 矩阵：哪些模型（GLM-5.3 flash？Llama 族？）哪些卡（A100/H800？B200？）进验收集？
4. batch/序列长度档位：decode 小 batch（论文收益域 1–16）为主还是含 prefill 大 batch？

## Round 1 已锁定（2026-11-16）

- **数值锚**：双锚——主锚 pytorch 后端 AWQ W4A16 路径（logit KL + argmax 一致率 + 端到端文本），次锚 HF bf16 forward（语义漂移检查）。硬 KL 阈值不在本票定：T1 首跑测出 tm-vs-awq 分布后按分布定，回填本票 amendment。
- **性能基线**：主对照 = pytorch AWQ Triton W4A16（同引擎 delta 可归因）；上限参照 = 原生 TurboMind。硬门槛唯一：decode 吞吐/延迟非劣于 AWQ Triton；对原生 TM 的比率为报告项（目标 ≥90%，不设门槛）。
- **档位**：decode batch ∈ {1,2,4,8,16,32}（核心 1–16）；序列长两档 ≈2k / ≈32k；prefill/piecewise 档位延后到 attention 批次。
- **Q3（模型×GPU）**：待定——用户提出「本机 3090 + 应变 tiny 减层小模型能否支撑核心任务」，调研 new_engine 的验收模型实践中（agent TinyModelPractice）。

## Round 2 已锁定（2026-11-16）

- **模型×GPU**：双层结构——dev-loop = 本机 RTX 3090 + **Llama-3-8B-AWQ 真模型**（~5.8GB）直接跑；GLM 几何覆盖 = **GLM tiny**（移植 new_engine 式生成器：真 checkpoint 保形状砍层 + seeded 随机权重 + config 截断，`/tmp` 落盘缺失即生成）。**验收数字卡 = H200 单机 8 卡，验收矩阵含 TP=2 档**。外部金样本门 = `awq_ref.json`（AWQ Triton 路径固定 32 prompt 离线转储 first_top20/greedy，TM 路径须对上；铁律：自洽 parity 不证明语义正确）。质量抽测 = Llama perplexity（真权重）。
- **kernel 正确性假设**（accept_scoped）：「TM kernel 内部数值正确」为工作假设，验证责任集中于调用契约。三条边界：①仅覆盖有生产暴露的 family×arch（SM86 路径、nvfp4/mxfp4 除外，遇之按未验证对待）②双锚对照不通过先查边界后查 kernel ③family×arch/量化格式/pack 路径任一变化即回到未验证状态。

## Resolution（2026-11-16）

验收口径定稿：

1. 数值：双锚（AWQ W4A16 主锚 / HF bf16 次锚）+ 外部金样本门（awq_ref.json）；硬 KL 阈值 T1 首跑按分布定后回填 amendment。
2. 性能：主对照 pytorch AWQ Triton W4A16（硬门槛：decode 非劣）；上限参照原生 TM（报告项，目标 ≥90%）；档位 decode batch {1,2,4,8,16,32} × seq {2k,32k}；TP=2 一档进矩阵。
3. 矩阵：Llama-3-8B-AWQ（3090 dev-loop + H200 验收）、GLM tiny（几何覆盖）；GLM 真模 AWQ 基座与 H200 环境为执行前提，开 Task 票「GLM AWQ 基座与 H200 验收环境确认」（08）。
4. 3090 上 TM W4A16 family（SM86）可用性为 T1 dev-loop 前置观察项；若不可用，dev-loop 降级为仅数值对照，性能全在 H200 测。

## Amendment 1（2026-11-16，票 08 结论 + 用户拍板）

- **GLM 真模档延后**：GLM-5.3-flash 为 320B/18B-active MoE 多模态，AutoAWQ 本机量化不可行（仓库归档、不支持该架构、需数百 GB 内存）。真模 AWQ 验证移交上游 GitHub CI 环境（H200 + 预下载模型，参考现有测试布局）。
- **近期替代**：先建 tiny 级 compressed-tensors 小模型验证「ct→repack→双路」整条通路（票 09），通过后再考虑 wtdcode 190.8GB 真模 repack。
- GLM tiny 生成不需下载 328GB：config + safetensors 分片头即可定形状（`tests/turbomind/linear/models.yaml` 已含 glm5 系几何条目）。

详细决议见本票 Round 1/2 两节；调研依据：`.scratch/research/`（论文简报、接缝盘点）与 new_engine 实践（TinyModelPractice 报告，验收模型清单/golden 模式/`gpu`-`slow` 分档）。
