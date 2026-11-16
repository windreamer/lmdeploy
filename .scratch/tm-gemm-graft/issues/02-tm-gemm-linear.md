# 02: TmGemmLinear W4A16 实现

**What to build:** 用户的 Linear 层经 TM GEMM 通道计算:gate/up 融合单次调用、打包后 GPU torch 权重 spill 至 CPU 精确副本并释放显存、bias 逐层相加、通信走 python 分布式。以 `prototype/tm_linear_impl/` 为蓝本落成正式模块(继承现有 Linear 基类,BuildSpec 分发新增分支)。

**Blocked by:** 01(registry 探询;实现本身不依赖,但 S1 对照与选择断言依赖其行为)。

**Status:** ready-for-agent

- [ ] S1 对照:同权重同输入,impl 级 KL + argmax 一致率(对照 AWQ Triton,阈值参考票 09 数据点:最坏 1.9e-4)
- [ ] fuse 开/关数值一致(gate/up 融合与直通对照)
- [ ] spill 生效断言:打包后显存回落可测;CPU 副本回灌重新打包后数值保持
- [ ] supports_batched 显式拒绝;bias 逐层加正确
- [ ] 初始加载无逐模块 sync;全部新增代码有单测且默认套件零真权重依赖
