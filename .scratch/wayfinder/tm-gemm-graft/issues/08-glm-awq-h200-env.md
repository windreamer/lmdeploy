---
id: 08
title: GLM AWQ 基座与 H200 验收环境确认
labels: [wayfinder:task]
status: closed
assignee: windreamer (subagent AcceptPrep, 2026-11-16)
blocked-by: []
---

## Question

票 05 定下的验收矩阵有两个执行前提需要落实。本票回答「拿不拿得到、怎么拿」，是执行前提确认，不是实现工作。

1. GLM-5.3-flash 的 AWQ（W4A16）checkpoint：是否已有内部或公开产物？若没有，AutoAWQ 离线量化一条产线的可行性——权重来源（HF/ModelScope）、量化耗时与磁盘占用、量化后 perplexity 抽检流程。产出物是 GLM tiny 生成器的基座，也是 GLM 档验收的真模。
2. H200 单机 8 卡验收环境：访问申请方式与维护人、lmdeploy 构建环境在 H200（sm90a）与 CUDA 13.1 工具链下的兼容性确认、TP=2 档的双卡可达性。

## Resolution

记录：GLM AWQ 产物路径与生成/获取命令、H200 访问方式与环境版本事实。
若 GLM AWQ 不可行或 H200 不可用，回票 05 修矩阵（amendment），不要在本票内自行改口径。

## Resolution（2026-11-16，AFK 部分完成）

**结论:本机 AutoAWQ 量化不可行,改道「tiny-ct 通路先行 + 上游 CI 全量验证」。**

1. GLM-5.3-flash = 320B/18B-active MoE 多模态(`zai-org/GLM-5.3-Flash`,FP8 328GB;45 层 = 3 dense + 42 sparse,288 专家 top-8,34 层 GDN + 11 层 MLA,VLM 壳,需 transformers ≥5.16)。AutoAWQ 三重不可行:仓库 2025-05 归档只读、模型目录无 GLM 系、量化需数百 GB 内存(本机 47Gi)。
2. 公开产物 `wtdcode/GLM-5.3-Flash-AWQ-W4A16`(190.8GB)为 **compressed-tensors** 格式(仅路由专家 INT4 g128,其余 BF16),lmdeploy 三条 AWQ 路径均吃 AutoAWQ 布局 → 需一次性 repack。用户拍板:**先建 tiny-ct 模型验证整条通路**(新票 09),真模 repack 与全量验证移交上游 GitHub CI(H200 + 预下载模型,用户确认上游 CI 具备)。
3. GLM tiny 生成不需下载 328GB:config + 分片头定形状;`tests/turbomind/linear/models.yaml` 已含 glm5 系几何。
4. H200:构建零改动(`CMakeLists.txt:258-260` 对 CUDA≥12 自动追加 `90a-real`;CUDA 13.1 nvcc 含 compute_90;驱动需 ≥R580)。TP=2 显存核算:权重 ~89GiB/卡 + KV ~11.8GB + GDN 态 ~2.3GB/卡 ≈ 单卡 105GiB/141GB,余量充足;scratch ≥500GB;TP=2 选卡要求同 NUMA/NVLink + NCCL 冒烟。完整 checklist 见本票历史(调研报告 B1/B3 节,已由用户转向上游 CI 方案)。
