# Ticket 09 报告：tiny ct → AutoAWQ repack → 双路冒烟

日期：2026-11-16。机器：RTX 3090 24GB（sm_86），torch 2.12.1+cu130，venv 3.13。

## 文件

| 文件             | 交付                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gen_tiny_ct.py` | ① tiny ct 生成器：Llama-3-8B 真实几何（K=4096/14336，N=4096/14336/28672/4096(qkv 融合 4096)），砍到 2 层 8 个线性层；真实 compressed-tensors W4A16 g128 sym pack-quantized 语义（scale=amax/7，q∈\[-8,7\]，zp=8，8 code/int32 LSB-first 沿 K，`weight_packed [N,K/8]` + `weight_scale [N,G]` fp16 + `weight_zero_point [N,G]` int8）。产物 `tiny_ct_model/`（每层 .pt + config.json）。零下载，纯 config+randn。 |
| `ct_to_awq.py`   | ② ct→AutoAWQ 转换器：`ct_to_awq()` 产出 qweight int32 `[K, N/8]`（沿 N 打包，nibble 序 \[0,4,1,5,2,6,3,7\]）、scales fp16 `[G, N]`、qzeros int32 `[G, N/8]`（sym 无 zero_point 按 AutoAWQ 约定填 0x88888888）。`awq_dequant()` 精确复刻 lmdeploy `_unpack_awq_gemm` 做往返校验。**8 层全部 roundtrip err = 0.0（逐位精确）**。                                                                                   |
| `smoke_dual.py`  | ③ 双路冒烟：同权重同输入，A 路 = `AwqLinearW4A16Impl`（pytorch AWQ Triton `awq_linear`），B 路 = `TurbomindAwqLinearW4A16Impl`（TM GEMM 通道）。结果落 `smoke_dual_results.json`。                                                                                                                                                                                                                               |

## 双路冒烟结果（m=4）

- argmax 一致率：**8/8 层全部 1.000**。
- KL(A‖B)：K=4096 的 6 层 ≈ 0（1e-21 ~ 1e-35，浮点噪声级）；`down_proj`（K=14336）2 层为 1.9e-4 / 2.2e-5。
- 两路对 fp32 dequant 参考的 relerr 均在 2e-5 ~ 4e-4（fp16 GEMM 正常范围）。

### down_proj 差异定位（已定位，非 repack 错误）

随 m 扫描（K=14336 层）：m=1 → KL=0，m=4 → 6.8e-18，m=16 → 2.5e-3，m=64 → 5.4e-4。
即小 batch（单 kernel 实例、同累加顺序）两路逐位一致；差异只在较大 m 触发不同 kernel/切分（split-k / 不同 tile 的累加顺序）时出现，且两路对参考的 relerr 相当（A≈4.0e-4，B≈2.75e-4，B 略优）。结论：**累加顺序级数值差异，非打包/布局错误**。

## 口径说明

按票面降级方案采用 **impl 级直接实例化对照**（不经引擎/OpsBackend）：两路用完全相同的 qweight/scales/qzeros 张量与 bias、同 x。后端选择器（`_build_linear_w4a16`）未经过，其中 `_turbomind_support_reason` 把 TM 通道硬门在 sm90 —— 本机 sm86 是刻意绕过 impl 级验证的。

## 发现的问题清单

1. **sm90 硬门与实际能力不符**：`cuda/awq_modules.py:_is_turbomind_gemm_capability_supported` 只放行 (9,0)，但 TM registry 的 u4 g128 家族（sm80_16816_4.cu 等）在 sm86 上实际可用——本票全部 8 层在 3090 上双路跑通且精度正常。3090 上被无谓回退到 Triton 路。概念性问题，建议回地图开票。
2. **python `WeightPlan` 包装丢失 `family`**：C++ `WeightPlan` 绑定含 family（linear_bind.cpp），但 `lmdeploy/turbomind/linear.py` 的包装对象只暴露 `shape_constraints`，无法在 python 侧查"这次是哪个 family 在跑"（本票只能旁路推断）。诊断性缺口。
3. **AWQ nibble 序陷阱（已踩）**：AWQ 沿 **输出维 N** 打包（qweight `[K, N/8]`，每 int32 装 8 个连续输出通道），与 ct 沿 **输入维 K** 打包（`[N, K/8]`）正交；且 int32 内 nibble 序为 \[0,4,1,5,2,6,3,7\] 置换——顺序号 j 落在 nibble 位 `order[j]`（不是 `order.index(j)`），两种映射不等价（order 非对合）。写反会静默产生错误权重（数值上看仍是合法 u4）。
4. **int32 字面量陷阱**：`torch.full(..., 0x88888888, dtype=torch.int32)` 溢出报错，需 int64 中转再 cast。
5. **精度意外（轻微）**：TM 通道输出与 fp32 参考的 relerr 稳定优于 AWQ Triton 路（2.75e-4 vs 4.0e-4 @ K=14336）；Triton `awq_linear` 疑似 fp16 累加。非阻塞，供真模 repack 时设容忍度参考。
6. **ct 语义确认**：compressed-tensors pack-quantized W4A16：8 元素/int32 无废位、LSB-first、sym 零点恒为 8 不打包（`weight_zero_point` 为未打包 int 张量）；`compressed_tensors` python 包本环境未安装（也不允许装），语义按官方文档实现并由双向逐位往返验证。

## 复现

```bash
.venv/bin/python .scratch/ticket09/gen_tiny_ct.py     # 生成 tiny_ct_model/
.venv/bin/python .scratch/ticket09/ct_to_awq.py       # 转换 + 逐位往返校验
.venv/bin/python .scratch/ticket09/smoke_dual.py      # 双路 KL/argmax 冒烟
```
