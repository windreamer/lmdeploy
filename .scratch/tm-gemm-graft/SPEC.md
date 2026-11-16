---
labels: [ready-for-agent]
feature: tm-gemm-graft
created: 2026-11-16
source: wayfinder map tm-gemm-graft(9 张决议票全闭合)
workflow: 本仓库开发分支 + 每阶段一个 commit + 向上游 stack PR
---

# SPEC：TurboMind 混合精度 GEMM 嫁接进 PyTorch Engine

## Problem Statement

我是 LMDeploy 的重度用户。TurboMind 后端在我的混合精度(W4A16)模型上速度快、资源省,但它是一个独立的 C++ runtime:投机解码、PD 分离、视觉模型、CUDA Graph 等新能力都只进 PyTorch 引擎,TurboMind 侧重复实现吃力且滞后。我想要 TurboMind 的 kernel 性能,但不想要第二套引擎的维护成本和功能孤岛。

## Solution

把 TurboMind 的混合精度 GEMM kernel 变成 PyTorch 引擎里 Linear 层的一个自动选中的实现:W4A16 模型默认走 TM GEMM(不满足条件的层逐层回落 Triton,行为可预期);权重打包/释放/热更新在现有加载钩子上无缝工作;`compressed-tensors` 权重用随附工具一键转成 AutoAWQ 布局;语义正确性由金样本门背书。`--backend turbomind` 永远可用——淘汰后它透明路由到新引擎。C++ runtime 在验收达标后删除,重复维护就此终结。

## User Stories

01. 作为在线服务用户,我部署 W4A16 AWQ 模型时 PyTorch 引擎自动选中 TM GEMM kernel,以便不加配置获得更高 decode 吞吐。
02. 作为在线服务用户,我不满足 shape 的层自动回落 Triton 实现,以便服务永不因 kernel 覆盖缺口而中断。
03. 作为在线服务用户,我用 `LMDEPLOY_W4A16_GEMM_BACKEND=turbomind` 强制选择时,不支持的情形得到明确报错,以便排查而不是静默变慢。
04. 作为在线服务用户,我把该 env 设为其他值时行为与历史版本完全一致,以便升级零风险。
05. 作为显存紧张的服务用户,权重打包后 GPU 上的 torch 原始权重自动溢出到 CPU 精确副本并释放显存,以便同卡塞下更大模型或更长上下文。
06. 作为 RL/热更新训练工程师,我从 CPU 副本回灌权重重新打包后数值保持正确,以便热更流程无感切换。
07. 作为 RL/热更新训练工程师,热更新路径保留同步语义,以便并发 forward 流上不出现竞态。
08. 作为 MoE/dense 模型用户,gate/up 两个投影融合为一次 TM GEMM 调用,以便拿到 fused 路径的收益。
09. 作为在线服务用户,我对同一份权重跑金样本门(`awq_ref.json` 固定 prompt 集),以便确信 TM 路径语义与 AWQ Triton 路径一致。
10. 作为引擎开发者,我在 3090 上用 tiny-ct 模型零下载跑通数值与性能对照,以便开发迭代不被下载和真卡依赖阻塞。
11. 作为持有 compressed-tensors 权重的用户,我用 repack CLI 一键转换 AutoAWQ 布局,以便进入本引擎的量化生态。
12. 作为引擎开发者,repack 转换有位级往返测试(err=0),以便布局陷阱(nibble 序、补零约定)被机械拦截。
13. 作为引擎维护者,路由探询基于 kernel registry 而非硬编码架构表,以便 SM86 等"非明星"架构上的可用家族不再被无谓回退。
14. 作为引擎维护者,`auto` 探询按 per-module 粒度工作,以便一个模型的各层各取最优实现。
15. 作为上游 CI 维护者,验收矩阵(H200 + 预下载模型)复用与本地完全相同的 harness,以便本地绿即 CI 绿。
16. 作为 GLM 团队服务开发者,GLM-5.3-flash 的几何以 tiny 变体进测试矩阵,以便后续 MLA/GDN 批次的契约在真几何上验证。
17. 作为 TurboMind 存量用户,淘汰后 `--backend turbomind` 仍被接受并透明路由到 PyTorch 引擎,以便零改动续用。
18. 作为 TurboMind 存量用户,我手填的 TM 专有配置键被警告忽略而不是报错,以便旧启动脚本照跑。
19. 作为上游评审者,每个 stack PR 聚焦一个阶段、独立可读可回退,以便评审成本可控。
20. 作为新贡献者,我从 AGENTS.md/HISTORY 读到全部决议的取舍与依据,以便不重开已死的争论。
21. 作为服务方,KL 容差按实测分布设定(tm-vs-awq 双路对照数据),以便阈值有依据而非拍脑袋。
22. 作为引擎开发者,后续接入 GPTQ 格式只需新增一个 WeightFormat 子类,以便量化格式扩展保持恒定成本。
23. 作为 QA,`gpu`/`slow` 分档标记让默认测试套件零真权重依赖,以便 CI 快速反馈。
24. 作为引擎维护者,C++ runtime 删除作为独立 PR 在验收达标后进行,以便删除永不先于替代路径被证明。

## Implementation Decisions

- **路由**:保持现有 env 三态语义(`auto` = registry 探询通过走 TM,否则逐层回落 Triton;`turbomind` 强制,不支持即 RuntimeError;其他值 = 全 Triton)。探询实现从硬编码架构表替换为 kernel registry 查询(`get_weight_plan` 是 ground truth,自带 arch/shape 过滤)。per-module 粒度。
- **接入形态**:新 `TmGemmLinear` 模块继承现有 Linear 基类(per-format 模块惯例,对齐 AWQ 先例);CUDA 后端的 BuildSpec 分发表新增 TM GEMM 分支。原型 `prototype/tm_linear_impl/` 编码了该决策(patch 形态与 impl 骨架),实现时以其为蓝本。
- **激活量化**:kernel 内 `ConvertInput` 现场量化(fp16/bf16→fp8+group scales);不做 python 预量化。
- **gate/up 融合**:进第一实现(`fuse_weight` 现成)。
- **权重释放**:spill-only 单档、默认开启——打包完成后 GPU 侧 torch 原始权重转 CPU 精确副本并释放显存;热更新从 CPU 副本回灌重新打包。不做 immutable 档。MLA 吸收豁免条款:被吸收模块(kv_b_proj 类,消费 `get_unquantized_weight`)永不参与释放,保留 torch 副本。
- **挂载点**:权重桥接挂现有加载钩子(`process_weights_after_loading` 第一遍 per-module 更新回调);初始加载去掉逐模块同步(瞬时峰值钉在单层暂存上界,≈稳态×1.06),热更新路径保留同步。
- **通信**:全部由 python 分布式层承接(all_reduce 与 reduce_scatter 两式),TM 通道不感知通信。
- **batched**:不支持 batched 前向(`supports_batched=False`),显式拒绝而非错误复用。
- **bias**:kernel 无 epilogue-bias,python 逐层相加。
- **compressed-tensors 加载时转换**:W4A16 模型含 ct 键名时,weight_loader 检测并内存内规范化为 canonical AWQ(复用位级转换函数:AWQ 沿 N 打包 nibble 序 \[0,4,1,5,2,6,3,7\]、qzeros 补零约定 `0x88888888`);流式 per-module,无离线 CLI、无磁盘双份权重。tiny-ct 生成器为测试资产。
- **兼容层(延后)**:建立时机 = PyTorch 引擎能力(含已嫁接 TM kernel)成为 TurboMind 后端**超集**之后。届时 `--backend turbomind` 保留为合法取值并透明路由 PyTorch 引擎,TurboMind 专有配置键警告忽略;该层是删除 C++ runtime 的前置条件。过渡期 `--backend turbomind` 行为不变(指向 C++ 引擎),近期能力切换由 env 旋钮承载,不做 CLI 级别名。
- **kernel 正确性假设**(带三边界):TM kernel 内部数值正确为工作假设,验证集中于调用契约;①仅覆盖有生产暴露的 family×arch(SM86 路径与 fp4 系除外);②对照不通过先查边界后查 kernel;③family×arch/量化格式/pack 路径任一变化即回到未验证。
- **验收矩阵**:Llama-3-8B-AWQ(3090 dev-loop + H200 验收)、GLM tiny(几何覆盖)、GLM 真模档(上游 CI);档位 decode batch {1,2,4,8,16,32} × seq {≈2k, ≈32k},含 TP=2;性能硬门槛唯一 = decode 非劣 AWQ Triton,对原生 TurboMind 比率 ≥90% 为报告项;KL 阈值按 T1 首跑分布定后回填本规格(已有数据点:最坏 1.9e-4)。

## Testing Decisions

- 好测试 = 断言外部行为(选择结果、数值一致性、显存回落、报错语义),不断言内部调用序列。
- 五条接缝(全部复用现有类/API,零新增代码缝):
  - S1 impl 级数值对照:同权重同输入,TmGemmLinear vs AWQ Triton 实现直接实例化,KL + argmax 一致率;票 09 harness 转正为公共测试工具。
  - S2 端到端金样本门:固定 32 prompt 离线转储 `awq_ref.json`,引擎加载 tiny-ct/Llama-AWQ 模型对门。
  - S3 加载/热更:spill 生效(显存回落可测)、CPU 副本回灌重打包后数值保持;走现有热更新 API。
  - S4 路由:env 三态 × 探询拒绝组合,unit 级断言选择结果与报错语义。
  - S5 repack:ct→AutoAWQ 位级往返 err=0。
- 先例:仓库既有集成测试组织方式;`awq_ref.json` 金样本与 `gpu`/`slow` 分档借 new_engine 验收实践(金样本门是唯一不依赖本仓库自身实现的外部判据;随机权重只承担等价/自洽类断言)。
- 性能对照纳入验收 harness(非单测):同引擎 AWQ Triton 为主对照,原生 TurboMind 为上限参照,档位见矩阵。

## Out of Scope

- attention/packed-KV 嫁接、MLA/MoE/GDN 家族嫁接(后续 effort,各建新图;CUDA Graph 技术前提已铺平)。
- BF16/常规精度路径的 kernel 嫁接(论文证明无优势)。
- GPTQ / w8a8-fp8 / fp4 系格式(顺序已定:GPTQ → fp8 → fp4,各自小增量,不在本规格展开)。
- CLI 旋钮与 `--backend turbomind` 别名兼容层:建立时机 = PyTorch 能力为 TurboMind 超集之后(attention/GDN/MLA/MoE 批次落地);过渡期 `--backend turbomind` 保持现状指向 C++ 引擎,切换由 env 旋钮承载。
- H200 环境建设细节(上游 CI 已具备,复用)。
- **C++ runtime 删除执行**:入口判据 = **全部有价值 kernel 完成嫁接并逐批验收达标**(GEMM → attention/packed-KV → MLA/MoE/GDN 家族 → GPTQ/fp8/fp4)+ 兼容层合入。仅 GEMM 批次达标不构成删除条件。本规格只锁判据方向,不含删除步骤;删除 PR 在后续批次 effort 完成后另立规划。

## Further Notes

- 工作流:本仓库开发分支,每阶段一个 commit(格式 `类型(范围): 摘要`),向上游开 stack PR(PR₁ 嫁接核心 → PR₂ repack+验收 harness;兼容层与删除 PR 均延后,见 Out of Scope 的时机条款)。远程 push 逐次审批。
- 决议正文:wayfinder 地图 `Decisions so far`(9 条,全部闭合)与本仓库 HISTORY 分册同源;本规格是其收束。
- 票 09 的实证:SM86 上 u4 g128 家族可用(3090 dev-loop 全量可跑);tm-vs-awq KL 最坏 1.9e-4。
- 术语见根目录 `GLOSSARY.md`;文字与提交流程见 `AGENTS.md`/`WRITING_STYLE.md`。
