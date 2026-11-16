# EuroSys 论文调研：LMDeploy/TurboMind

来源：arXiv:2508.15601（*LMDeploy Accelerates Mixed-Precision LLM Inference with TurboMind*，EuroSys 2027 已接收）。调研执行：EuroSysPaper agent（2026-11-16）。此文件是研究记录，非设计文档。

## 四项贡献全部在 kernel 层

| 技术                      | 机制                                                                              | 差异化声称                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 硬件感知权重打包（§4.1）  | 完全离线：借 FP16 memory-to-register 管线，swizzle 离线烘进权重布局（Appendix B） | 运行时仅 `cp.async + LDS` 两条指令装载，零 swizzle、零 bank conflict、天然 coalesced；对比 MARLIN 需按架构手调 |
| 自适应 head 对齐（§4.2）  | 每 decode 步在 kernel 内对 Q 轻量重排（Appendix C）                               | 对比 PyTorch/TRT/vLLM 需先把 KV 反量化回 FP16；开销 \<1% attention 延迟                                        |
| 指令级并行（§4.3）        | tensor core 算 tile k ∥ ALU 处理 tile k+1 ∥ cp.async 预取 tile k+2                | INT4×FP16 batch 1–16 平均比 FP16×FP16 快 134%（最高 220.3%），MARLIN 同场景劣化 20.3%                          |
| KV 内存装载流水线（§4.4） | 三级并行：算 slice i ∥ on-the-fly 反量化 slice i+1 ∥ 预取 tile i+1                | HBM 带宽利用率：16-bit KV 91%/95%，8-bit KV 86%/93%                                                            |

端到端：延迟最高降 61%（均值 30%），吞吐升 156%（均值 58%），对比 vLLM+MARLIN / TensorRT-LLM / OmniServe+QServe。

## 关键否定性发现

- W16A16KV16 时 LMDeploy 比 vLLM **慢 2.1%/5.2%**，论文自述 "gains originate from mixed-precision optimization rather than framework differences"（§5.2）——引擎/调度层无优势自证，全部优势在 kernel。
- 论文未把 persistent batch / blocked KV / 调度列为贡献；PagedAttention/continuous batching 归为 vLLM 已有工作（§2）。调度与显存联合优化列为未来工作（§6）。

## 若把 kernel 嫁接到其他引擎，必须带走的耦合资产

1. **离线 repacker 及其权重格式**（32×8 布局，swizzle 烘干；运行时依赖 `cp.async + LDS/ldmatrix` lane 重排语义）。
2. **低比特 packed KV cache 布局约定**（kernel 装载时反量化；外部引擎保留 FP16 KV 则拿不到 attention 收益）。
3. 固定精度配置：运行时无在线精度决策（§6）。

调度器本身无耦合；Q 重排完全在 kernel 内完成，不需引擎侧配合。
