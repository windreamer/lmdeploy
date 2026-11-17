# B3-01: 衔接原型——prefill 写入与 TM int8 decode 读取互通

**What to build:** 验证 prefill 阶段产生的 KV 块能被 TM int8 decode kernel 正确读取,并给出 B3-04 的产品化清单。gate 票:通过才放行 B3-04。

**Blocked by:** 批 2 benchmark(已提供 int8 decode binding)。

**Status:** done(2026-11-17,判定可行;commit 待 B3-02 落库后收口——其并行未暂存改动阻塞了钩子)

- [x] 原型脚本 `benchmark/bench_b301_bridge.py`:PT 契约 prefill 写入方(host-op,与生产 triton fill 逐字节等价)+ decode 写入方(不同舍入,混合出处断言)→ 同一 TM 块池 → 双读者对照 bf16 参考
- [x] 全池与混合链路(前半 prefill 写/后半 decode 写):KL ≤ 9e-4(门限 1e-2);TM max_abs ≤ 0.223(内核 fp16 累加,随 klen 收敛,批 2 容差内)
- [x] 契约发现:TM params 槽 (scale, min) vs PT (scale, zero) 代数等价(min = -zero·scale),codes 字节相同 → B3-04 只需 scatter 适配层(zero→min),量化内核零重写
- [x] B3-04 产品化清单:CacheBackend int8 档布局声明、fill 布局参数化 + scatter 适配、TM fused write 的 (scale,min) 契约、opt-in 模型 checkpoint 导出/迁入需布局转换、门限用 KL+max_abs(argmatch 无判别力)

**环境发现(2026-11-17 独立复验修正:REFUTED)**:原报告的 triton 3.7.1 误编译主张不成立——四项最小用例全部精确通过;「16 块仅 6 块有效」真实可复现,根因是原探针违例调用契约(1-D 块表 stride=1 读 2-D 页表越界,写入确定性落 {0,4,8,12}),非工具链问题。工具链告警撤回;`stride_boff==1` JIT 特化的越契约 IMA 附带发现移交上游参考。复验产物:`benchmark/b301_reverify/`(17/17 PASS)。

**交付**:`benchmark/b3_01_bridge.py` / `b3_01_bridge_results.json` / `b3_01_bridge_report.md`(已暂存,待收口 commit)。
