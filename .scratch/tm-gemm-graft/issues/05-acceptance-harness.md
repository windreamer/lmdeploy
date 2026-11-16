# 05: 验收 harness 转正

**What to build:** 一键执行票 05 验收矩阵的性能对照:decode batch {1,2,4,8,16,32} × seq {≈2k, ≈32k} × TP 档,三口径报告(同引擎 AWQ Triton 硬门槛 / 原生 TurboMind 比率报告项 / fp32 参考),本地 3090 与 H200 CI 复用同一 harness 与同一报告格式。

**Blocked by:** 03(需要引擎端到端与金样本门在位)。

**Status:** ready-for-agent

- [ ] harness 一键执行档位矩阵,产出结构化报告(含硬门槛判定与 TM 比率)
- [ ] TP=2 档支持且本地可演练(H200 CI 复用同一入口)
- [ ] KL 阈值 amendment 模板就位(T1 首跑分布回填)
- [ ] 上游 CI job 定义草稿(复用上游预下载模型与 runner)
