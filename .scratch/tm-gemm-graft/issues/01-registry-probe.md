# 01: registry 探询替换硬码门

**What to build:** W4A16 模型在 `auto` 态于 SM86 等非明星架构上真实选中 TM GEMM(不再无谓回落 Triton);`turbomind` 强制态不支持时明确报错;其他值直通 Triton。路由探询以 kernel registry 查询为准(自带 arch/shape 过滤),替换硬编码能力检查。

**Blocked by:** None (can start immediately)。

**Status:** ready-for-agent

- [ ] env 三态单测矩阵:auto(支持→TM/不支持→Triton)、turbomind(支持→TM/不支持→RuntimeError)、其他→Triton
- [ ] 探询拒绝组合表落地(shape/家族×arch 的拒绝理由可读)
- [ ] 3090 上 W4A16 模型 auto 态选择断言(TM 被选中)
- [ ] 既有 AWQ Triton 路径行为不回归(既有单测全绿)
