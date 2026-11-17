# B3-02: MoE grouped W4A16 嫁接

**What to build:** GLM-5.3 类模型(288 专家 top-8)的 MoE 层走 TM grouped GEMM:awq+MoE 组合解除 unsupported,路由表适配,专家交错打包,S1 对照与 tiny ct-MoE e2e。

**Blocked by:** 批 1 全部落地。

**Status:** done(commit `80077f64`,2026-11-17)

- [x] 路由解除:`get_quant_method` moe kind 已返回 awq/ct;新 `FusedMoETmW4A16BuildSpec` → `TurbomindMoeW4A16Impl`,env 三态与 dense 一致
- [x] 路由表适配:`blocked_route` 复用 triton anchor 同一 `_get_sorted_idx` 排序器(天然同路由);skip sentinel(-1)→ 第 E 组 + w2 scatter 清零
- [x] 专家打包:canonical AWQ g128 逐专家 → `prepare_weight(list)` → `LinkLinearExperts`;融合能力以真实 `fuse_weight` 尝试探测(sticky w1/w3 回退,SM86 生效)
- [x] S1:vs 逐元素 fp32 参考 KL ≤5.1e-8、argmax 1.0(m=1/8/32);vs triton fused_moe 同权重 KL≈0
- [x] tiny ct-MoE(g128, Mixtral 8×2top)加载链路通,引擎日志确认 TM 路由;金样本门过(KL 2.27e-3 / top1 0.53,阈值按首跑分布钉定)
- [x] 测试:7+1skip / 2 / 2;批 1 套件回归绿

**遗留**:g32 原始 ct MoE checkpoint 需重量化(非既有库能力);SM90 融合分支已实现待 H200 验收;EP/DeepEP 未接(ep>1 显式 RuntimeError);统一大 env 归票 B3-05。
