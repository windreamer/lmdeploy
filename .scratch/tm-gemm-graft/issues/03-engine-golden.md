# 03: 引擎接线 + 金样本门 + spill 热更

**What to build:** tiny-ct / Llama-3-8B-AWQ 模型经完整 PyTorch 引擎跑通 TM GEMM 路径:金样本门对 `awq_ref.json`;热更新 API 从 CPU 副本回灌重新打包后金样本仍过;auto 路由在引擎内 per-module 生效;ct 模型经加载时转换(票 04)直接进 TM GEMM 通道。本票完成即 PR₁ 端到端可 demo。

**Blocked by:** 01, 02, 04(均已落地)。

**Status:** done(commit `d7a05121`,2026-11-16)

**票 04 移交的加载时集成前置清单**:

1. ct_repack 输出 shard 键名 `qweight/scales/qzeros` 与 dtype(int32/fp16/int32)精确满足 `AWQFormat.suffix_map` + `AWQFormat.accepts`;
2. 布局即 TM-native:`qweight [K,N/8]` unpack→`[K,N]` 不转置、`scales [G,N]`、qzeros 全 8——`AWQFormat.normalize/dequant` 零改动可吃;`update_weights(qw, sc, qz)` 直接消费;
3. 集成形态二选一:离线 `convert_checkpoint` 出 AutoAWQ 目录走既有 AWQ 加载路,或加载钩子内对 ct 张量逐模块 `convert_shard`(3 B/元素上界);
4. 待办:真模 shard key ← HF 权重名映射规则;bf16 激活路未测(生成器固定 fp16)。

- [x] `build_linear`/quant_method 自动路由接线(票 02 移交:显式构造 → 引擎自动分发;auto 态逐层生效,probe 拒绝层保留 TM 模块类换回 Triton impl;测试 test_w4a16_engine_routing 10 passed)
- [x] 引擎 e2e(TP=1):tiny-ct 模型经加载时转换金样本门通过(8/8 线性全走 TM 模块)
- [x] Llama-3.1-8B-AWQ 引擎级冒烟 + 金样本门通过(ModelScope `LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` 5.4GB;top1/greedy 32/32,最坏截断 KL 3.48e-05)
- [x] 热更新 API 回灌 → 重新打包 → 金样本门复过(A→B→A 往返精确回到原始 greedy ids)
- [x] 引擎日志可见 per-module 路由选择(INFO 逐模块,测试断言日志可见)
- [x] S2/S3 测试进套件,`gpu`/`slow` 分档标记(真权重门标 slow,默认套件零真权重)
- [x] cudagraph 兼容观察项记录(capture/replay 与 eager 位级一致;热更 reset_graph_runner 已由 agent 既有逻辑覆盖)
