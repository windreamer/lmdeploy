---
labels: [wayfinder:map]
effort: tm-gemm-graft
created: 2026-11-16
---

# MAP：TurboMind 混合精度 GEMM 嫁接进 PyTorch Engine

## Destination

产出一份可执行的嫁接规格：PyTorch Engine 内由 TurboMind 混合精度 GEMM 家族（INT4/8×FP16）驱动的 Linear 路径——路由策略、LinearImpl 契约、WeightPlan 挂载、验收口径——决策全部锁定；TurboMind C++ runtime 的淘汰契约与窗口确定。图终不写实现代码；实现归 `/to-spec` → `/to-tickets` → `/implement` 主流程。

## Notes

- 领域术语见根目录 `GLOSSARY.md`；仓库规范见 `AGENTS.md`；文字风格见 `WRITING_STYLE.md`。
- 预置决策（本图开工前已定，正文见 `HISTORY/2026-11.md`）：
  - [TurboMind kernel 嫁接的价值锚点定为混合精度 GEMM](../../HISTORY/2026-11.md)：GEMM 优先，attention 第二批。
  - [Turbomind C++ runtime 计划淘汰](../../HISTORY/2026-11.md)：模型面冻结，验证达标后删除，与上游社区协调窗口。
- 背景材料：`.scratch/research/eurosys-turbomind-paper.md`（论文四项贡献均在 kernel 层；BF16 无优势）、`.scratch/research/tm-pt-seam-inventory.md`（两侧接缝与七条硬耦合点）。
- Tracker：本地 markdown，本目录。claim = ticket front-matter 的 `assignee`；blocking = `blocked-by` 字段（本地 tracker 无原生依赖关系，用字段约定）。开票类型对应的 skill：`research`（AFK 交子代理）、`prototype`/`grilling`（HITL，开工时调 `grilling` + `domain-modeling`）。
- 一次 session 只解一张票（research 票除外）。
- 实施模式（2026-11-16 拍板，HISTORY 有正文）：本仓库开发分支 + 每阶段一个 commit + 向上游 stack PR。/to-spec 的 ticket 切分须与 commit/PR 边界对齐。

## Decisions so far

- [CUDA Graph 下 per-step 指针数组的稳定性](issues/06-cudagraph-pointer-array.md)：可行——三路径均可落地，推荐 device gather 改写，走既有 `CudaAttentionMetaBuilder` 缝；两条 cache 引擎不变量待约定（池 storage 不搬移、地址变更必经 graph reset）。
- [WeightPlan 挂载时机与显存开销](issues/03-weightplan-mount-timing.md)：挂载缝已被 `TurbomindAwqLinearW4A16Impl` 走通（`process_weights_after_loading` 的 per-module `update_weights` 钩子），复用即可；TM 原生 pack 峰值 ≈稳态×1.06，现状 torch 权重共存 ≈×1.9——释放 torch 原始权重须 opt-in（热更新落点）；热更新全量重 pack 是 RL 场景真实成本。
- [GEMM 的 python 调用通道盘点](issues/01-gemm-python-call-channel.md)：通道已端到端存在（`linear_bind.cpp` + `linear.py`），`TurbomindAwqLinearW4A16Impl` 即复制模板；零星缺口 `set_measure` 绑定、双 `Gemm` 实例 cache 不共享、bias 走 python；主工作量在 `WeightFormat` 子类而非 C++ 绑定。
- [验收口径](issues/05-acceptance-criteria.md)：双锚数值（AWQ W4A16 主锚 / HF bf16 次锚）+ `awq_ref.json` 外部金样本门；性能硬门槛唯一 = decode 非劣 AWQ Triton，对原生 TM ≥90% 为报告项；矩阵 = Llama-3-8B-AWQ + GLM tiny，验收卡 H200×8 含 TP=2；kernel 正确性按带三条款的工作假设处理（边界查错优先、family×arch 变化即重验）。
- [淘汰窗口与社区协调](issues/07-deprecation-window.md)：静默冻结 + 无硬窗口 + 隐式兼容——删除由验收驱动，旧 PyPI release 自然留存；`--backend turbomind` 作别名透明路由新引擎，TM 专有配置键警告式忽略；兼容层规格进入 /to-spec，是删除的前置条件；TM 特有并行组合不承诺承接。
- [GLM AWQ 基座与 H200 验收环境确认](issues/08-glm-awq-h200-env.md)：本机 AutoAWQ 量化不可行（GLM-5.3-flash 为 320B MoE，AutoAWQ 已归档且不支持该架构）；公开 W4A16 产物为 compressed-tensors 格式需 repack。拍板：tiny-ct 通路先行（票 09），真模全量验证走上游 GitHub CI（H200 + 预下载模型）；H200 构建零改动（90a-real 已在 CMake），TP=2 显存核算余量充足。
- [LinearImpl 契约原型](issues/02-linearimpl-contract-prototype.md)：评审通过，asset 在 `prototype/tm_linear_impl/`。七项决议：kernel 内激活量化、env 三态降级（强制态报错）、spill-only 默认开启的权重释放、初始加载去逐模块 sync、gate/up 融合进 T1、新 TmGemmLinear 继承 BaseLinear、MLA 吸收豁免条款。探询改用 registry（`get_weight_plan`）替代硬编码。
- [混合精度路由策略](issues/04-quant-routing-policy.md)：保持 env 三态（auto/forced/回落）只换 registry 探询；T1 env-only，CLI 旋钮随兼容层进 /to-spec；格式优先级 GPTQ → w8a8-fp8 → fp4 系；auto 探询 per-module。
- [tiny ct 模型与 ct→AutoAWQ repack 通路验证](issues/09-tiny-ct-repack-pipeline.md)：交付全绿——转换器逐位往返 err=0；双路冒烟 argmax 一致率 8/8=1.000,最坏 KL 1.9e-4（大 K 层 m≥16,kernel 累加顺序差,TM 精度反优）。SM86 上 u4 g128 家族实测可用,佐证票 04 的 registry 探询决议;KL 数据点供票 05 阈值 amendment。

## Not yet specified

（空——目的地范围内的雾已全部毕业或收束）

## Out of scope

- BF16/常规精度路径的 kernel 嫁接：论文证明无优势（W16A16KV16 比 vLLM 慢 2.1%/5.2%），超出本图目的地。
- attention/packed-KV 嫁接批次（`AttentionMetadata` 增补、builder 缝设计）：超出本图目的地（GEMM 路径规格）。技术前提已由[CUDA Graph 指针数组稳定性](issues/06-cudagraph-pointer-array.md)结论铺平,作为下一个 effort 建新图。
- 第二模型族批次（MLA / MoE / GDN kernel 嫁接取舍）：同上,后续 effort。
- 退役执行排序(冻结 → 维护 → 删除的阶段划分):[淘汰窗口与社区协调](issues/07-deprecation-window.md)已锁契约与兼容层要求。**删除入口判据 = 全部有价值 kernel 完成嫁接并逐批验收达标 + 兼容层合入**(仅 GEMM 批次达标不构成删除条件);删除 PR 在后续批次 effort 完成后另立规划。
