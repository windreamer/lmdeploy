# Copyright (c) OpenMMLab. All rights reserved.
"""CudaOpsBackend 分发树上新增「TM GEMM 通用 Linear」分支的 patch 形态。

本文件是 throwaway 原型:以 diff 风格注释展示改动,不修改 lmdeploy/ 源码树,
不要求可运行。所有行号引用当前工作副本。

背景
----
现状:W4A16 一条量化线已挂入分发表 —— ``LinearW4A16BuildSpec`` 在
``lmdeploy/pytorch/backends/cuda/op_backend.py:166-169`` 被路由到
``_build_linear_w4a16``(``backends/cuda/awq_modules.py:88-118``),后者按
``_envs.w4a16_gemm_backend`` 在 ``TurbomindAwqLinearW4A16Impl`` 与 AWQ triton
kernel 之间选择。本票把 TurboMind 通道从「AWQ W4A16 特例」推广为「按
WeightFormat 参数化的通用 TM GEMM LinearImpl」,分发树需要一个新 spec 类型:
``LinearTmGemmBuildSpec``(定义见 tm_linear_impl.py,连同 ``TurbomindLinearImpl``)。

设计取舍
--------
* 新增独立 BuildSpec 类型而非复用 ``LinearW4A16BuildSpec``:BuildSpec 是
  「immutable requirements」(backends/base.py:17-19 注释),携带
  weight_format/input_dtype/output_dtype/释放策略等通用字段;塞进 W4A16 spec
  会把 AWQ 语义泄漏进通用契约。isinstance 分发表(op_backend.py:83-248)天然
  支持并列新类型,两条量化线互不影响。
* 分支插入点选在 ``LinearW4A16BuildSpec`` 分支之前之后均可(类型不相交);
  diff 里放在其后,保持「先特例后通用」的阅读顺序。
* 可行性判定不再硬编码 w_bit==4 / group_size==128 /
  K%128==0 / N%32==0(``_turbomind_support_reason``,
  backends/cuda/awq_modules.py:22-43),而是直接探询 registry:
  ``Linear.get_weight_plan`` 是 ground truth(内部遍历全部 family 跑
  ``Family::supports``,family.cc:63-90;registry 已按 arch/shared-mem 过滤,
  registry.cu:23-41),plan 的 ``shape_constraints`` 给出 (K,N) 的
  min/align(gemm 侧 WeightQuery→WeightPlan,weight_plan.h:15-22)。
  详见 tm_linear_impl.py 中 ``_turbomind_gemm_support_reason``。
"""

# ============================================================================
# diff 形态(lmdeploy/pytorch/backends/cuda/op_backend.py)
# ============================================================================

DIFF = r"""
--- a/lmdeploy/pytorch/backends/cuda/op_backend.py
+++ b/lmdeploy/pytorch/backends/cuda/op_backend.py
@@ class CudaOpsBackend(DefaultOpsBackend): build_op(...)  # :81 起
         from ..rejection_sampling import RejectionSamplingBuildSpec
         from ..static_fp8_modules import LinearStaticF8BuildSpec
+        from ..tm_gemm_modules import LinearTmGemmBuildSpec
         if isinstance(spec, SiluAndMulBuildSpec):
             ...
@@ build_op 分发表尾部(紧跟 LinearW4A16BuildSpec 分支之后, :166-169)
         if isinstance(spec, LinearW4A16BuildSpec):
             from .awq_modules import _build_linear_w4a16
             return cast(ImplT, _build_linear_w4a16(spec))
+        if isinstance(spec, LinearTmGemmBuildSpec):
+            from .tm_gemm_modules import _build_linear_tm_gemm
+            return cast(ImplT, _build_linear_tm_gemm(spec))
         if isinstance(spec, LinearW8A8BuildSpec):
             ...
"""

# ============================================================================
# 新文件 lmdeploy/pytorch/backends/tm_gemm_modules.py 的契约面(摘要)
# (完整 impl 体在 tm_linear_impl.py;此处仅列 BuildSpec,因为它才是
#  OpsBackend 分发契约的一部分)
# ============================================================================

BUILDSPEC_SNIPPET = r'''
# lmdeploy/pytorch/backends/tm_gemm_modules.py (新文件, 摘要)
from dataclasses import dataclass

import torch

from lmdeploy.turbomind.weight_format import WeightFormat

from .base import BuildSpec
from .linear import LinearImpl


@dataclass(frozen=True)
class LinearTmGemmBuildSpec(BuildSpec[LinearImpl]):
    """通用 TM GEMM linear 的不可变构建请求。

    weight_format : WeightFormat 实例或 (block_in 等参数已定) 的工厂产物,
                    决定 DataFormat(AWQ/GPTQ/Trivial/FP8/MXFP4/CT,
                    weight_format.py:96-165 抽象 + :169 起六个实现)。
                    TM 侧 family 匹配只消费 make_data_format() 的
                    DataFormat{weight_dtype, [block_in, block_out],
                    scales_dtype, zeros_dtype}(weight_format.py:148-153;
                    linear_weight.cc:26-40)。
    input_dtype   : 期望激活 dtype;None = 与 weight scales dtype 一致
                    (对齐 get_weight_plan 的 query.data_type 缺省,
                    linear.py:196-199)。
    output_dtype  : 硬要求项 —— WeightQuery.output_dtype 在 C++ 侧是
                    硬约束(weight_plan.h:15),选不出 family 直接
                    NotImplementedError(linear.py:213-220)。
    release_torch_weights : 票 03 §3 的 opt-in 释放(见 weight_release.py);
                    放在 BuildSpec 让模型构建方显式声明,而不是依赖环境变量
                    之外的黑盒。两者可叠加:env 为进程级开关,此字段为
                    模型级开关。
    """

    in_features: int          # 已按 TP 切片后的本地 K
    out_features: int         # 已按 TP 切片后的本地 N
    bias: bool
    weight_format: WeightFormat
    input_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    release_torch_weights: bool = False
    fusion_type: str | None = None   # 'silu' 预留(gate_up 融合, linear.py:207-211)
'''

# ============================================================================
# 新文件 lmdeploy/pytorch/backends/cuda/tm_gemm_modules.py 的工厂(摘要)
# ============================================================================

FACTORY_SNIPPET = r'''
# lmdeploy/pytorch/backends/cuda/tm_gemm_modules.py (新文件, 摘要)
def _turbomind_gemm_support_reason(spec: LinearTmGemmBuildSpec) -> str | None:
    """以 registry 探询代替硬编码门控(取代 awq_modules.py:22-43 的
    w4/g128/K%128/N%32 检查)。失败原因字符串与该先例同构。"""
    # 1) torch.cuda.is_available() / capability 门控:registry Add 已按
    #    arch 过滤(registry.cu:23-41),探询即隐式覆盖;
    # 2) turbomind.is_available() 门控(与 awq_modules.py:40-43 相同);
    # 3) linear = get_linear();  plan = linear.get_weight_plan(
    #        weight_format=spec.weight_format,
    #        dtype=<scales dtype 或 fallback>,
    #        input_dtype=spec.input_dtype, output_dtype=spec.output_dtype)
    #    NotImplementedError → 'no GEMM family accepts ...'(linear.py:213-220);
    # 4) (min_k, min_n), (align_k, align_n) = plan.shape_constraints,
    #    校验 K>=min_k, K%align_k==0, N>=min_n, N%align_n==0
    #    (commit 侧同款检查见 builders/_base.py:330-341;
    #     注意 silu 融合时 min_n 按 (min_n+1)//2 折半, linear.py:210-212)。


def _build_linear_tm_gemm(spec: LinearTmGemmBuildSpec) -> LinearImpl:
    """Build the generic TurboMind GEMM linear implementation."""
    reason = _turbomind_gemm_support_reason(spec)
    if reason is not None:
        # 提供降级:此处可以像 awq_modules 那样回落到格式对应的 torch 后端;
        # stub 不展开,README 记为分歧点(是否要求调用方显式选择回落)。
        raise RuntimeError(
            f'TurboMind GEMM linear was requested but is unavailable: {reason}.'
        )
    return TurbomindLinearImpl(
        in_features=spec.in_features,
        out_features=spec.out_features,
        weight_format=spec.weight_format,
        bias=spec.bias,
        input_dtype=spec.input_dtype,
        output_dtype=spec.output_dtype,
        release_torch_weights=spec.release_torch_weights,
        fusion_type=spec.fusion_type,
    )
'''

if __name__ == '__main__':  # throwaway 自检:仅打印 patch 形态
    print(DIFF)
    print(BUILDSPEC_SNIPPET)
    print(FACTORY_SNIPPET)
