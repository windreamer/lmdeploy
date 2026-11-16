# 03: 引擎接线 + 金样本门 + spill 热更

**What to build:** tiny-ct / Llama-3-8B-AWQ 模型经完整 PyTorch 引擎跑通 TM GEMM 路径:金样本门对 `awq_ref.json`;热更新 API 从 CPU 副本回灌重新打包后金样本仍过;auto 路由在引擎内 per-module 生效。本票完成即 PR₁ 端到端可 demo。

**Blocked by:** 01, 02。

**Status:** ready-for-agent

- [ ] 引擎 e2e(TP=1):tiny-ct 模型金样本门通过(固定 32 prompt 转储对照)
- [ ] Llama-3-8B-AWQ 引擎级冒烟 + 金样本门通过
- [ ] 热更新 API 回灌 → 重新打包 → 金样本门复过(数值保持)
- [ ] 引擎日志可见 per-module 路由选择(观测性)
- [ ] S2/S3 测试进套件,`gpu`/`slow` 分档标记(默认套件零真权重)
