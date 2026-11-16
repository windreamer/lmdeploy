---
id: 07
title: 淘汰窗口与社区协调
labels: [wayfinder:grilling]
status: closed
assignee: windreamer (2026-11-16)
blocked-by: []
---

## Question

`--backend turbomind` 的淘汰怎么执行：公告形态、迁移路径、窗口长度、分阶段切分？

背景：预置决策（`HISTORY/2026-11.md`）——turbomind 模型面冻结，新特性只进 PyTorch Engine，验证达标后删 C++ runtime，与上游社区协调。上游用户面：`--backend turbomind` CLI、qwenvit/internvit 视觉编码器的 C++ 实现（`src/turbomind/models/qwenvit/`、`internvit/`）、TM 特有并行组合（attn_dp/attn_cp 等，`lmdeploy/turbomind/turbomind.py:60-93`）。开工时调 `grilling` + `domain-modeling`（HITL）。

1. 冻结公告的落点：README Latest News？docs deprecation 章节？单独 migration 文档？
2. 窗口长度：几个 minor release？与 GEMM 嫁接验收达标的先后关系（先公告后达标，还是达标即公告）？
3. qwenvit/internvit 用户归宿：pytorch 侧 VLM 路径（glm4_1v 等）是否视为等效替代？
4. TM 特有并行组合（attn_dp/cp、outer_dp）在 pytorch Engine 有无承接计划？没有的话淘汰公告要不要明示？

## Resolution（2026-11-16）

淘汰协调定稿——**静默冻结 + 无硬窗口 + 隐式兼容**：

1. **公告时点**：静默冻结。过程不公告；删除 C++ runtime 前一次性公告。
2. **窗口**：无硬时限。删除由票 05 验收达标驱动；旧 release（PyPI 冻结点之前的版本）自然留存可装，用户可留在旧版——不承诺向后移植 bugfix/安全修复（已接受的后果）。
3. **兼容层（替代迁移映射表）**：`--backend turbomind` 保留为合法取值，透明路由到 PyTorch Engine（别名）；TurboMind 专有配置键（attn_dp/attn_cp 等）以警告方式忽略，不报错。常用参数（quant policy、tp、session len 等）语义等价直通。**该兼容层进入 /to-spec 的规格内容**，是删除 C++ runtime 的前置条件。
4. **TM 特有并行组合**：不承诺承接，公告明示移除 + issue 反馈渠道。
5. 公告内容（删除时一次性）：C++ runtime 已移除、`--backend turbomind` 的别名行为、被忽略配置键清单、VLM 用户在 pytorch 侧的对应实现（qwen2_5_vl/qwen3_vl/internvl/internvl3_hf/glm4_1v 覆盖已核实）。

依据：pytorch 已是默认主力路径，turbomind 用户配置面窄（多为 `--backend turbomind` + 常规参数）；隐式兼容把迁移成本从用户侧移到我们侧的一个别名层，社区协调成本最小化。
