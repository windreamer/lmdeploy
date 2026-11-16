<!--
Copyright (c) OpenMMLab. All rights reserved.
-->

# GLOSSARY.md

LMDeploy 的推理引擎领域词汇。本文件只定义概念,不承载实现细节。

## Language

**Backend(后端)**:
用户可选的推理执行路径;LMDeploy 当前有 `pytorch` 与 `turbomind` 两个。CLI `--backend` 即此义。
_Avoid_: engine(指用户可见概念时;`lmdeploy/pytorch/engine` 包名另计)

**TurboMind**:
C++/CUDA 实现的高性能推理后端,资源占用低、单机速度快;核心价值在高性能 Kernel 与特殊权重布局。

**PyTorch Engine**:
基于 eager PyTorch + Triton/TileLang Kernel 的推理后端,功能支持最全的主力路径。

**Kernel 层**:
后端中负责单算子/单层计算的部分(attention、GEMM、rotary 等),不含调度与状态管理。
_Avoid_: 算子层

**Runtime 层**:
后端中负责调度、KV cache 管理、请求生命周期、通信的部分。
_Avoid_: 引擎框架

**Weight Layout(权重布局)**:
Kernel 期望的权重排布方式(为特定 GEMM/attention Kernel 预处理后的格式),与 HuggingFace 原生排布不同,加载时需转换。

**DeepEP**:
Expert-Parallel 通信库,MoE 分发/聚合用;FetchContent 依赖,要求 NCCL ≥ 2.30.4。

**NCCL Stub**:
`libturbomind_nccl_stub.so`,为 DeepEP 提供 NCCL device API 的 dlopen 垫层,随包安装到 `lib/` 子目录。

**PD(Prefill/Decode 分离)**:
将 prefill 与 decode 部署到不同实例的架构;TurboMind 以 C++ 实现代价高,是重构动机之一。
