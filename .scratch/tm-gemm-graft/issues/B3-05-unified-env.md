# B3-05: 统一大 env LMDEPLOY_TM_KERNELS

**What to build:** 单一 env 承载三类 kernel 旋钮(w4a16/moe_grouped/attn_int8_kv;gdn 类槽位留档——GDN 已回滚),类级 CSV 覆盖全局态,旧 env 向后兼容(未指定类时旧 env 生效,冲突新 env 优先 + warning)。

**Blocked by:** B3-02(moe_grouped 类的路由读点)。

**Status:** done(2026-11-17;源码随 `982f4e13` 落库,测试 `19ac15ee`)

- [x] 统一解析器 `parse_tm_kernels()` + `tm_kernel_state(cls)`(envs.py;语法 off/auto/all/类级 CSV,`;` 分隔,类名精确/态大小写不敏感,未知类跳过,非法类态 warn 后留档)
- [x] 优先级:类条目 > 全局态 > 旧 env > auto;冲突一次性 warn(legacy 显式设置且映射态不同才警)
- [x] 消费点迁移:`_build_linear_w4a16` 读 `tm_kernel_state('w4a16')`(派生属性保持批 1/2 测试 patch 兼容);`tm_moe_route` 读 `tm_kernel_state('moe_grouped')`;attn_int8_kv 类由 tm_kv.py 既有实现承接(一致性核对结论:类条目优先级完全一致;唯一分歧 = 全局 token 不作用于 tm_kv 类,tm_kv 为准,待其对齐)
- [x] 62 用例解析矩阵 + 路由回归全绿;批 1/2 套件零回归

**遗留**:tm_kv.py 全局 token 不生效(以 tm_kv 为准);同类非法条目双源各警一次(无行为影响)。
