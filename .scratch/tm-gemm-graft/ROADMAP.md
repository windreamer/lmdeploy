<!--
Copyright (c) OpenMMLab. All rights reserved.
-->

# ROADMAP:TurboMind kernel 组合迁移路线图

来源:KernelPortfolio / ConfigCliSurface 盘点 + 用户拍板(2026-11-16)。此文件是组合级路线图;每批启动时另立 spec(复用 GEMM 批流程)。决策正文同步 HISTORY。

## 三条原则

1. **单机为界**:TM 嫁接范围 = 单机 kernel 性能;多机通信能力(DeepEP a2a 等)不迁移,通信面归 PyTorch 侧。
2. **KV Cache 默认 PyTorch 标准**:标准布局为默认;TM 布局(含 per-token scale 槽的 packed-KV)为 benchmark 驱动的 opt-in——经 `CacheBackend` 布局抽象实现,仅当 kernel 级 benchmark 证明显著收益才引入。
3. **能力决策树**:PyTorch 无对应实现 → 嫁接;有对应且本机(3090)能测 → tiny 测速,TM 快则嫁接、不快则保留;本机不能测 → 默认保留,留 H200 CI 评估。

## 逐类判定

| 类别                                    | 判定                                 | 依据                                                                                                                                                                                                                                                                                                                                                            | 批次                                     |
| --------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| GEMM u4 g128(W4A16/AWQ/GPTQ/CT)         | **嫁接**                             | 本机实测可用,TM 精度反优;GPTQ 为同管线增量                                                                                                                                                                                                                                                                                                                      | 批 1(W4A16)+ 批 2(GPTQ)                  |
| GEMM MoE grouped u4                     | **嫁接**(批 2 实测胜出)              | GLM-5.3 sparse 几何(288 专家 top-8),m=1/8/32 下 TM = Triton 的 1.58×/3.87×/7.67×;正确性门禁全过;SM86 无 gated-silu 融合走独立 w1/w3(与 TM C++ 生产一致)                                                                                                                                                                                                         | 批 3(spec 已立,SPEC-B3)                  |
| GDN/linear-attn                         | **两段式决策框架**(2026-11-17 拍板)  | kernel 本体快(decode 1.41–3.50×);adapter 桥层净负已回滚。**第一段:标量 g 形态**(qwen3_5/glm5_next 系)——零拷贝直读 benchmark 通过(`385468de`:净 vs fla 2.77×/1.79×/1.56×,转置损失 1.31×/1.06×/0.90×)后默认启用 TM GDN。**第二段:channelwise/KDA 形态**(Kimi 系等)——出作用域评估票(recurrent 逐行衰减 = 低成本;chunked 融合逐通道 cumsum = 成本大头),可行则全面化 | 第一段 benchmark 已完成;第二段评估票待立 |
| attention GQA/MHA(bf16,PyTorch 标准 KV) | **不嫁接**(批 2 实测,低预期兑现)     | 本机可测(SM80 全矩阵);吞吐档(B=128/klen=4096)仅 1.04×,中档 1.1–1.5×;FA3 基线留 H200 CI 复测;前置指针数组 builder 已由票 06 研究证可行(留作未来参考)                                                                                                                                                                                                             | —                                        |
| packed-KV 低比特(kv int4/int8)          | **立项路径 B**(批 2 实测 delta 显著) | 生产档 TM = PT 的 1.34–1.86×,小 batch 2.9–7.5×;归因实证:TM int8-vs-bf16 = 1.7–1.9× 而 PT 仅 1.2–1.3×,三级流水 dequant delta 真实;decode-only int8 + opt-in CacheBackend 布局,衔接原型已证互通(B3-01)                                                                                                                                                            | 批 3(spec 已立,SPEC-B3)                  |
| TM KV 写入(rope+量化融合写)             | 随 attention 立项才成对迁移,单独排除 | 绑死 TM KV 池布局                                                                                                                                                                                                                                                                                                                                               | —                                        |
| GEMM dense bf16                         | **不嫁接**                           | TM dense family 本就路由 cuBLAS;论文证无优势                                                                                                                                                                                                                                                                                                                    | —                                        |
| MLA                                     | **不嫁接**(用户确认)                 | TM 为简化 MLA(bf16-only、无 sparse、无 fp8),能力子集;GLM/DeepSeek 系 MLA 一律走 PT flash_mla                                                                                                                                                                                                                                                                    | —                                        |
| MoE DeepEP a2a                          | **不嫁接**                           | 多机通信(原则 1);且 TM C++ dispatcher bf16-only,弱于 PT deep_ep                                                                                                                                                                                                                                                                                                 | —                                        |
| GEMM fp8 w8a8(block-128)                | 保留,留 CI                           | kernel 仅 SM90,本机不能测                                                                                                                                                                                                                                                                                                                                       | CI 批                                    |
| GEMM mxfp4 / nvfp4 / mxfp4_fp8          | 保留,留 CI                           | 仅 SM90;nvfp4 系还需补 WeightFormat/converter 分支(CI 上做)                                                                                                                                                                                                                                                                                                     | CI 批                                    |
| fused allreduce+RMSNorm                 | 保留,留 CI                           | TM 独有加分资产(单机多卡 NVLink 融合),单卡不可测                                                                                                                                                                                                                                                                                                                | CI 批                                    |
| rms_norm / 量化小 kernel / rotary       | 保留现状                             | rms_norm PT 已有 triton;量化小 kernel 已随批 1 通道可用;rotary 不可单独嫁接                                                                                                                                                                                                                                                                                     | —                                        |

## 批 2 · 判定(2026-11-16 实测,`benchmark/batch2_benchmarks_report.md`)

- **GPTQ 格式**:✅ 落地(commit `174a975a`)——转换库 + 加载时转换 + TM GEMM 路由,S1 双参照全绿。
- **MoE grouped W4A16**:✅ **嫁接胜出**——288 专家 top-8 GLM-5.3 几何,m=1/8/32 下 TM = Triton 的 1.58×/3.87×/7.67×;SM86 上 u4 grouped 无 gated-silu 融合,走独立 w1/w3(与 TM C++ 生产一致)→ **批 3 嫁接 spec 首项**。
- **GDN**:✅ **分裂判定**——recurrent decode TM 1.41–3.50×(B=1/8/32)→ 嫁接 decode;chunked prefill PT fla 反超(T=1024/B=8 时 TM 仅 0.61–0.65×)→ prefill 保留 fla,双轨。状态池语义(复合块 vs indexed pool)为嫁接 spec 的核心决策项。
- **attention bf16**:❌ **不嫁接**(低预期兑现)——吞吐档(B=128/klen=4096)仅 1.04×,中档 1.1–1.5×;FA3 基线留 H200 CI 复测。
- **packed-KV int8**:✅ **立项路径 B(opt-in TM CacheBackend 布局抽象)**——生产档 TM = PT 的 1.34–1.86×,小 batch 2.9–7.5×;归因实证:TM int8-vs-bf16 = 1.7–1.9×,PT int8-vs-bf16 仅 1.2–1.3×,三级流水 dequant 的 delta 真实存在。与 TM int8 attention decode kernel 成对立项,成对迁移 KV 融合写。

**批 3 组成(由批 2 判定生成)**:① MoE grouped W4A16 嫁接 spec;② GDN decode 嫁接 spec(chunked 保留 fla);③ packed-KV int8 + TM attention decode kernel 整层立项 spec(含 opt-in CacheBackend 布局)。

## 批次序列

- **批 1(进行中)**:GEMM dense W4A16 嫁接 = `.scratch/tm-gemm-graft/issues/` 现有 6 票(01 probe、02 TmGemmLinear、03 引擎+金样本、04 repack CLI、05 验收 harness;06 兼容层已延后)。
- **批 2(测速定夺批)**:五项 benchmark/增量,赢了各立嫁接 spec——① GPTQ 格式(WeightFormat 子类增量);② MoE grouped W4A16 测速;③ GDN kernel 级 benchmark;④ attention bf16 kernel benchmark(低预期,按数据说话);⑤ packed-KV kernel 级 benchmark(TM int8/int4 attention 读取管线 vs PT int8 KV,张量直喂;delta 显著 → 立项路径 B:opt-in TM CacheBackend 布局抽象)。
- **CI 批(默认保留,留 H200 CI 评估)**:fp8 w8a8、mxfp4、nvfp4/mxfp4_fp8(补 WeightFormat 后)、fused AR+RMSNorm。
- **不嫁接(终局)**:dense bf16、MLA、DeepEP a2a;TM KV 写入仅在 attention/packed-KV 立项时成对迁移。

## 统一大 env(过渡期切换面)

```
LMDEPLOY_TM_KERNELS=off | auto | all | <类>=<态>[,...]
# 类:w4a16 / moe_grouped / gdn / attention / ...(随批次扩展)
# 态:auto(能力探询,默认)| on(强制,不支持报错)| off(显式关)
```

- 默认 `auto`;类级 CSV 覆盖全局值。
- 向后兼容:`LMDEPLOY_W4A16_GEMM_BACKEND` 继续有效;新 env 未指定 `w4a16` 类时旧 env 生效,两者冲突时新 env 优先 + warning。
- 三态语义与票 02/04 决议一致(auto=探询/显式=强制失败报错/off=原生);CLI 大旋钮按拍板延后(超集后随兼容层)。
- 落地节奏:批 1 先以既有 `LMDEPLOY_W4A16_GEMM_BACKEND` 运行,统一大 env 在批 2 立项时引入(那时才有第二类 kernel 可选)。

## 与既有决议的衔接

- 票 04 决议修正:格式扩展顺序「GPTQ → w8a8-fp8 → fp4 系」中,fp8/fp4 顺延至 CI 批(本机不能测);GPTQ 留批 2。
- 票 05 观察项解除:SM86 上 u4 g128 实测可用,3090 dev-loop 全量可跑。
- attention 批次的 CUDA Graph 前提(指针数组 builder 可行性)已由票 06 研究铺平,但按原则 2 其范围缩至 bf16 + PyTorch 标准 KV。
