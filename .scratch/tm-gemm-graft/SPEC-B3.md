---
labels: [ready-for-agent]
feature: tm-graft-batch3
created: 2026-11-17
source: ROADMAP 批 2 判定(2026-11-16)+ 用户拍板(状态池 adapter / decode-only int8 / PR 时点)
workflow: 本仓库开发分支续用,每阶段一 commit;stack PR 于批 3 落完后统一开
---

# SPEC:批 3 嫁接(MoE grouped / GDN decode / packed-KV int8)

## Problem Statement

批 1 把 TM 的 dense W4A16 GEMM 嫁接进了 PyTorch 引擎并验证达标(decode 非劣成立、小中 batch 反超 1.4–2.5×、小 batch ≈ 原生 TM)。但用户主力模型(GLM-5.3-flash:42 个 sparse 层 + 34 层 GDN + 11 层 MLA)的核心算子——MoE grouped GEMM、GDN 递归状态更新、低比特 KV 读取——仍在 PyTorch 侧以较慢的实现运行;批 2 测速证明这三处 TM kernel 分别快 1.58–7.67×、1.41–3.50×(decode)、1.34–1.86×(生产档)。

## Solution

把三处已证优势的 TM kernel 按统一模式嫁接进 PyTorch 引擎:① MoE grouped W4A16 复用批 1 的 Linear 通道(`forward_moe` + 专家交错打包),MoE builder 解除 unsupported 限制;② GDN recurrent decode 走 TM kernel,状态池保留 PyTorch indexed 标准、由桥层每步重排为 TM 复合块并生成 TMA 描述符(chunked prefill 保留 fla 不动);③ int8 KV 模型启用 opt-in TM CacheBackend 布局,KV 写入与 attention decode 成对走 TM 融合内核。三者共用批 1 已验收的探询/回落/释放/热更模式,验收复用批 1 harness 框架。

## User Stories

1. 作为 GLM-5.3-flash 服务用户,我的 MoE 层走 TM grouped GEMM,以便 sparse 层推理拿到最高 7.67× 的单层加速。
2. 作为 GLM-5.3-flash 服务用户,我的 GDN 层 decode 走 TM kernel,以便 34 层线性注意力的解码步进更快。
3. 作为 int8-KV 用户,我的 KV 池在启用 opt-in 布局后由 TM 融合内核读取,以便 decode 档拿到 1.34–1.86× 的带宽优势。
4. 作为在线服务用户,三种嫁接都遵循 auto 探询语义:不满足条件的层/模型自动回落 PyTorch 原生实现,以便升级无风险。
5. 作为在线服务用户,探询失败在 forced 态得到明确报错,以便快速定位配置问题。
6. 作为服务运维,MoE/GDN/int8-KV 三类启停都有独立旋钮,以便按模型特征逐类开关而非全有全无。
7. 作为 RL 热更新用户,MoE 专家权重的热更路径与 dense 一致(规范化 → 打包 → spill),以便训练侧协议统一。
8. 作为引擎开发者,GDN 状态池保持 PyTorch indexed 标准,以便 checkpoint/PD 迁移/驱逐语义零改动。
9. 作为引擎开发者,int8-KV opt-in 布局只作用于显式启用的模型,以便其余模型的 KV 池零影响。
10. 作为 QA,MoE 嫁接的 impl 级对照与 triton fused_moe 同权重同路由,以便正确性可机械验证。
11. 作为 QA,GDN 嫁接的 decode 输出与 PT fla 生产 kernel 位级/容差对照,以便回归可判定。
12. 作为 QA,int8-KV 读取与 PT int8 写入格式互通(同一池两种写入方),以便混合链路不失配。
13. 作为上游 CI 维护者,三类嫁接的测试默认套件零真权重,以便 CI 快速反馈。
14. 作为 H200 验收执行者,批 1 harness 框架扩展覆盖三类新算子的档位矩阵,以便验收口径统一。
15. 作为 GLM-5.3-flash 用户,三类嫁接叠加后端到端 decode 吞吐显著优于纯 PyTorch 路径,以便整体拥有成本下降。
16. 作为引擎维护者,chunked prefill 保留 fla 不做 TM 替换,以便已证无优势的路径不浪费工程。
17. 作为引擎维护者,MLA 层继续走 PT flash_mla(TM 简化 MLA 能力子集已排除),以便能力不降级。
18. 作为下游量化用户,GPTQ/ct/AWQ 三种来源的 MoE 权重都能进入 TM grouped 路径,以便格式不设门槛(按可转换性)。
19. 作为引擎开发者,统一大 env `LMDEPLOY_TM_KERNELS` 随本批引入并承载三类旋钮,以便过渡期切换面单一。
20. 作为服务用户,启用 int8-KV 后 prefill 路径行为与文档声明一致,以便混合链路可预期。

## Implementation Decisions

- **MoE grouped**:
  - 专家权重经 `LinkLinearExperts` 交错打包(gate/up 交错已支持),`forward_moe` 消费 `indices/offsets`(kBlocked striding);路由表由 PyTorch 侧 moe_utils 产出,接口适配 = 路由转换函数(不引入 TM 侧 moe_utils)。
  - g32 来源权重经既有转换库转 g128(AWQ/CT 转换函数已就位);GPTQ MoE 本批不做(批 2 遗留,见 Out of Scope)。
  - `get_quant_method` 解除 awq+MoE 组合的 unsupported 限制,映射到 TM grouped 路径;MoE fused-silu SM90-only 的约束同批 1(SM86 独立 w1/w3)。
  - nn 侧:MergedTmGemmLinear/QKVTmGemmLinear 先例之上新增 MoE 对位模块;TP 切片沿 ep/moe_tp 既有语义。
- **GDN decode(adapter 方案,已拍板)**:
  - 状态池保留 PyTorch indexed 标准;桥层在 decode 步前把 PT 池张量重排为 TM `(layer_group, head_group)` 复合块并生成 TMA 描述符,decode 步后回写 PT 池。
  - 热路径优化目标:重排/回写均为设备端操作且随层数线性;若实测桥层开销吃掉 kernel 收益,回退方案 = 状态常驻 TM 复合格式、checkpoint/迁移按需转换(写入规格 amendment,不预先实现)。
  - chunked prefill 保留 fla(批 2 实测 fla 反超);TM GatedDeltaRule 仅接 recurrent decode 档。
  - 硬约束沿 TM 侧:head_dim=128、state dtype f32 或 = 输入 dtype、k/v 头数整除 attn_tp×cp。
- **packed-KV int8(decode-only,已拍板)**:
  - opt-in 范围:模型级启用(quant_policy INT8 + TM attention 探询通过);KV 池经 `CacheBackend.build_block_layout` 按 TM 块布局(per-token scale 槽)分配。
  - KV 写入与 attention decode 成对走 TM 融合内核(rope+量化+分块写);bf16 attention 与 chunked prefill 保持 PyTorch 侧。
  - **写入侧工具链回退(2026-11-17 拍板)**:若 triton 在本机不可用(误编译主张待独立复验),prefill 写入/布局转换改由 **TM 侧桥接 CUDA kernel** 承担——复用既有 `invokeProcessKV_v2`(rope+量化+分块写)或经 binding 暴露专用 convert+scatter kernel,不依赖 triton 工具链;该回退同时作为长期形态候选(删除对 triton 版本的敏感性)。
  - 混合链路衔接:int8 块布局下 prefill 写路径的可行性是**批 3 第一票(衔接原型)**——原型验证 prefill 写入的块能被 TM int8 decode 读取;不可行则 int8-KV 范围收缩(decode-only 模型或放弃,amendment 记录)。
- **统一大 env**:引入 `LMDEPLOY_TM_KERNELS`(单变量,类级 CSV 覆盖;类 = w4a16/moe_grouped/gdn/attn_int8_kv;态 = auto/on/off);既有 `LMDEPLOY_W4A16_GEMM_BACKEND` 向后兼容,新 env 未指定该类时旧 env 生效,冲突时新 env 优先 + warning。
- **验收框架**:复用批 1 harness(`benchmark/w4a16_acceptance.py` 框架),新增三类 rung(MoE/GDN/int8-KV)与各自对照锚;金样本门模式不变(外部转储判据)。
- **MLA 不变**:继续走 PT flash_mla(TM 简化 MLA 能力子集,已排除)。

## Testing Decisions

- 好测试 = 断言外部行为(routed 输出数值、路由选择、显存回落、报错语义),不断言内部调用序列;度量统一 KL/argmax/max_abs,禁止手写度量公式之外的实现细节断言。
- 四条接缝(全部复用既有):
  - S1 impl 级:MoE routed 输出 vs triton fused_moe(同权重同路由);GDN decode 输出 vs PT fla 生产 kernel;int8-KV decode vs PT int8 读取路。阈值沿批 1 同档(KL ≤1e-2、argmax ≥0.99,首跑分布回紧)。
  - S2 端到端金样本门:GLM 几何 tiny(MoE/GDN 覆盖)+ Llama AWQ 回归门。
  - S3 热更:MoE 专家权重与 GDN 状态经热更 API 往返数值保持。
  - S4 路由:统一大 env 三态 × 三类的选择矩阵(unit,无 GPU 依赖部分单列)。
  - S5 格式:GPTQ/ct→AWQ 的 MoE 权重转换位级往返。
- 先例:票 09/批 1 harness 与 tiny 生成器模式(随机权重承担自洽断言,金样本承担语义断言);`tests/turbomind/linear/models.yaml` 几何条目(glm52 含 MoE/MLA/indexer 形状)。

## Out of Scope

- attention bf16 嫁接(批 2 判定出局;FA3 基线 H200 CI 复测)。
- MLA 嫁接(能力子集,用户确认排除)。
- 多机通信(DeepEP a2a 等,原则 1)。
- GPTQ MoE、asym GPTQ 真实 checkpoint 验证(批 2 遗留,随批 3 后续 spec)。
- GDN chunked prefill 的 TM 替换(fla 反超,保留)。
- C++ runtime 删除(判据不变:全部有价值 kernel 嫁接达标 + 兼容层合入;本规格不含)。
- 兼容层 `--backend turbomind` 别名(时机 = 超集达成;本批继续 env 承载)。

## Further Notes

- 三项嫁接的优先顺序即交付顺序:① MoE grouped(接缝最低、收益主场)→ ② GDN decode(adapter)→ ③ int8-KV(衔接原型先行,原型通过才进入实现)。
- H200 验收:批 1 harness 预设扩展后统一复测(含 TP=2 与大 batch 缺口区间);上游 CI job 定义随本批落。
- 决策台账:HISTORY `2026-11-16 批 2 测速判定` 与本规格互为引用;本规格三决策(GDN adapter / decode-only int8 / PR 时点)已于 2026-11-17 用户拍板。
- 术语见 `GLOSSARY.md`;规范见 `AGENTS.md`/`WRITING_STYLE.md`。
