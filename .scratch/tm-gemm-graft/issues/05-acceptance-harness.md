# 05: 验收 harness 转正

**What to build:** 一键执行票 05 验收矩阵的性能对照:decode batch {1,2,4,8,16,32} × seq {≈2k, ≈32k} × TP 档,三口径报告(同引擎 AWQ Triton 硬门槛 / 原生 TurboMind 比率报告项 / fp32 参考),本地 3090 与 H200 CI 复用同一 harness 与同一报告格式。

**Blocked by:** 03(需要引擎端到端与金样本门在位)。

**Status:** done(commit `40b940dd`,2026-11-16)

- [x] harness 一键执行档位矩阵,产出结构化报告(`benchmark/w4a16_acceptance.py`;report.json + report.md;公共解码窗口计时,窗口塌缩报错拒绝出数)
- [x] TP=2 档支持且本地可演练(`h200` 预设含 tps=\[1,2\];单 3090 未实测,管道级就绪)
- [x] KL 阈值 amendment 模板就位(`benchmark/w4a16_amendment_template.md`,链金样本门)
- [x] 上游 CI job 定义草稿:复用入口 = `--preset h200` + `--model-path` 参数化(job 定义本身随 CI 批)

**实测(Llama-3.1-8B-AWQ @ 3090,dev 预设)**:

| 档位      | AWQ Triton tok/s | TM graft tok/s | graft/Triton | 硬门槛                        | native TM | graft/native |
| --------- | ---------------- | -------------- | ------------ | ----------------------------- | --------- | ------------ |
| 2k×b1     | 52.1             | 130.9          | 2.51         | pass                          | 132.0     | 0.99         |
| 2k×b2     | 133.3            | 256.3          | 1.92         | pass                          | 252.0     | 1.02         |
| 2k×b4     | 296.8            | 468.1          | 1.58         | pass                          | 461.0     | 1.02         |
| 2k×b8     | 564.1            | 793.4          | 1.41         | pass                          | 768.9     | 1.03         |
| 2k×b16    | 937.6            | 959.4          | 1.02         | pass                          | 1161.2    | 0.83         |
| 2k×b32    | 923.7            | 926.0          | 1.00         | pass                          | 1477.6    | 0.63         |
| 32k×b1    | 42.0             | 82.3           | 1.96         | pass                          | 80.4      | 1.02         |
| 32k×b2    | 83.3             | 118.5          | 1.42         | pass                          | 117.9     | 1.00         |
| 32k×b4–32 | —                | —              | —            | skipped(KV 容量不足,三路同因) | —         | —            |

**硬门槛 PASS 8/8**;graft/native = 0.99–1.03(b≤8 与 32k),2k×b16/32 为 0.83/0.63——缺口定位在 pytorch engine 运行时(triton 与 graft 该档步耗相同),非被嫁接 kernel;H200 TP 档复测重点区间。

**遗留**:① TP=2 未实测(H200);② 上游 CI job 定义随 CI 批;③ native TM 不吃裸 ct tiny(按票 04 repack→awq 目录,未在本次验证)。
