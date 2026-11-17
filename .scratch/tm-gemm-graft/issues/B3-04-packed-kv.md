# B3-04: packed-KV int8(opt-in CacheBackend 布局 + TM KV 融合写 + TM int8 decode kernel 接入)

**What to build:** int8-KV 模型的 KV 池按 TM 块布局(opt-in)分配,KV 写入与 attention decode 走 TM;图捕获走 Triton fallback。

**Blocked by:** 批 1 全部落地。

**Status:** done(WIP `0e06c40c` → 四缺陷修复 `1559be51` → 图路径 `429b0681` → 记录 `cf7dcf65`;混合链 8/8 绿)

- [x] opt-in CacheBackend int8 布局(plan.py 按 quant_policy INT8 + knob 路由;其余模型零影响)
- [x] 写入器:`fill_kv_cache_tm`(triton-free ATen convert+scatter;布尔掩码 host sync 已修为稠密索引 + where,capture-safe)
- [x] decode:`TmInt8KVAttentionImpl`(eager 单 token 走 TM decode;捕获期走 Triton fallback——GraphReverify CONFIRMED 该设计正确)
- [x] 四缺陷修复(ImplB304Fix):导入路径、decode 签名(k_rows/v_rows)、capture guard、fallback capture-safe;sanitizer memcheck/racecheck/initcheck 0
- [x] S1 + 混合链路金样本断言(混合出处 KL ≤9e-4)
- [ ] 图路径 GraphReverify 判定:CONFIRMED(binding 三缺陷:栈 H2D 烘焙/捕获区内释放/无 device 长度)→ 修复由 ImplGraphFix 落地(`429b0681`,图内 TM decode 位级验证)→ B3-04 完全闭环
- [ ] E1(eager 入口烘焙长度,已修三缺陷的同族残留)+ W2(图状态单槽)修复:待派(见 `benchmark/binding_graph_audit.md` §2/§4)

**遗留**:checkpoint 导出/迁入的布局转换(记录未实现);`stride_boff==1` JIT IMA 与 ATen writer replay 错乱两个上游候选(已记录)。
