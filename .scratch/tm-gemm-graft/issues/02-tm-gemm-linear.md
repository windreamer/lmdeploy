# 02: TmGemmLinear W4A16 实现

**What to build:** 用户的 Linear 层经 TM GEMM 通道计算:gate/up 融合单次调用、打包后 GPU torch 权重 spill 至 CPU 精确副本并释放显存、bias 逐层相加、通信走 python 分布式。以 `prototype/tm_linear_impl/` 为蓝本落成正式模块(继承现有 Linear 基类,BuildSpec 分发新增分支)。

**Blocked by:** 01(registry 探询;实现本身不依赖,但 S1 对照与选择断言依赖其行为)。

**Status:** done(commit `bf4f338a`,2026-11-16)

- [x] S1 对照:同权重同输入,impl 级 KL + argmax 一致率(最坏 KL 5.6e-4,argmax 全形状 1.0000)
- [x] fuse 开/关数值一致(fuse_weight 路径;SM90-only,SM86 跳过并注明)
- [x] spill 生效断言:显存精确回落(qweight+scales+qzeros 零误差);CPU 副本回灌重 pack 后数值位级保持
- [x] supports_batched 显式拒绝;bias 逐层加正确
- [x] 新增代码全有单测(15 passed, 1 skipped);默认套件零真权重依赖

**实现偏离(相对 prototype,均已在实现中修正)**:

1. 释放挂钩移至 nn 侧 `update_weights`(pack 后直接替换 `param.data`,免重注册);prototype `restore()` 只分配不回填——实为 bug,已修正。
2. 释放档位收敛为 spill-only 单档默认开启(无 env 开关);豁免 = 构造时 `release_torch_weights=False`。
3. bias 不参与释放(forward 每步读取,释放会崩),每层仅 N×2 字节驻留。
4. SiLU 融合仅 SM90 family 有 gated-silu epilogue;3090 上 skip,H200 验收硬件上运行。
5. sync 语义:`_published` 标志区分首次加载(免 sync)与热更(保留 sync)。

**带入后续**(票 03/后续票):`build_linear`/quant_method 自动路由未接(当前显式构造);fused nn 侧 merged 模块未建;cudagraph 兼容未验证。
