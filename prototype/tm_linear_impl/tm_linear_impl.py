# Copyright (c) OpenMMLab. All rights reserved.
"""通用 TurbomindLinearImpl —— 契约原型(throwaway,不进源码树)。

模板与推广点
------------
模板 = ``TurbomindAwqLinearW4A16Impl``
(``lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py``,全文 75 行)。
它只服务 AWQ W4A16(g128、sm90、fp16);本文件把五个硬编码点推广为契约参数:

1. 量化格式参数化:``AWQFormat(block_in=group_size)`` 硬编码 → 构造入参
   ``weight_format: WeightFormat``(weight_format.py 六个实现,AWQ/GPTQ/
   Trivial/FP8/CT/MXFP4;TM 侧只消费其 ``make_data_format()`` 的
   DataFormat,weight_format.py:148-153)。
2. forward_batched / supports_batched:模板无 → 本文件明确实现为
   ``supports_batched = False``(论证见 README 问题 3;TM 通道无
   「一份输入 × 多份独立权重」入口,``forward_moe`` 是 offsets 分 token
   语义,不等于 bmm)。
3. all_reduce 承接:模板只处理 ``all_reduce``+``group``
   (turbomind_awq_modules.py:69-70)→ 本文件补足通用 LinearImpl 契约的
   ``rank``/``scatter_size`` 分支(backends/linear.py:27-39),先例
   ``DefaultLinearImpl.forward``(backends/default/linear.py:32-36):
   scatter_size 非空走 ``reduce_scatter_by_tp_sizes``。
4. bias 逐层加:与模板一致(out + bias,turbomind_awq_modules.py:66-67);
   C++ forward 无 bias 形参、cublas family 拒绝一切 epilogue
   (linear_bind.cpp:209-262;cublas.cu:139;研究差距表 #9),epilogue-bias
   不存在,必须 python 逐层加。
5. torch 原始权重 opt-in 释放:模板 ``update_weights`` 恒返回原张量
   (turbomind_awq_modules.py:57)→ 本文件接入 weight_release.py 的
   opt-in 机制(票 03 §3 约束:torch 参数是热更新 load_weights 的落点,
   不可无条件断)。

假 kernel 边界标注
------------------
每个直接触碰 TM 通道(``get_linear`` / ``prepare_weight`` / ``get_exec_plan``
/ ``linear(...)``)的调用点都以 ``[TM CHANNEL]`` 标注 —— 这些是真实通道
(lmdeploy/turbomind/linear.py,票 01 端到端验证),不是 mock;「假」的部分
只有:适配逻辑未经量化格式矩阵回归、未做 cudagraph 兼容性验证、降级路径
未实现。
"""

from __future__ import annotations

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.backends.linear import LinearImpl
from lmdeploy.turbomind.linear import ExecPlan, Linear, Weight, get_linear
from lmdeploy.turbomind.weight_format import WeightFormat

from .weight_release import TorchWeightRelease, release_enabled


class TurbomindLinearImpl(LinearImpl):
    """Drive the TurboMind Linear API for any supported WeightFormat.

    契约对应:
    - 构造:由 ``_build_linear_tm_gemm`` 从 ``LinearTmGemmBuildSpec`` 构建
      (见 ops_backend_patch.py);构造期**只**缓存 plan 探询结果,不 pack ——
      权重桥接在 update_weights 钩子内(README 问题 4,票 03 结论)。
    - forward 契约继承通用 ``LinearImpl``
      (backends/linear.py:13-39:``supports_batched``、
      ``forward(x, weight, bias, all_reduce, group, rank, scatter_size)``)。
    """

    # 见 README 问题 3:TM 通道不支持 bmm 型批量投影。
    supports_batched = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_format: WeightFormat,
        bias: bool,
        input_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
        release_torch_weights: bool = False,
        fusion_type: str | None = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_format = weight_format
        self.bias_enabled = bias
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.fusion_type = fusion_type  # 'silu' 预留(gate_up 融合)

        self._linear: Linear | None = None
        self._weight: Weight | None = None
        self._plan = None  # WeightPlan,update_weights 时惰性解析一次

        # 释放开关:模型级(BuildSpec)或进程级(env)任一为真即启用。
        # 票 03 §3:热更新 load_weights 以 torch 参数为落点,故必须 opt-in。
        self._release_torch_weights = bool(release_torch_weights) or release_enabled()
        self._release_helper = TorchWeightRelease()

    # ------------------------------------------------------------------
    # 权重生命周期:process_weights_after_loading 的 per-module
    # update_weights() 钩子(nn/linear/awq.py:196-200 调用形态;
    # model_weight_loader.py:196-198 遍历形态)。挂载点结论见 README 问题 4。
    # ------------------------------------------------------------------
    def update_weights(self, weight, scales=None, zeros=None, bias=None):
        """Bridge weight_loader products into TM DataFormat and pack.

        与模板的差异:模板签名是 AWQ 专用的
        ``(qweight, scales, qzeros, bias)``(turbomind_awq_modules.py:44);
        通用实现用 ``(weight, scales, zeros, bias)``,由 weight_format 负责各
        checkpoint 后缀到 TM-native 布局的归一化(AWQ 无 .t(),Trivial/FP8/
        CT/MXFP4 需 .t();weight_format.py:243-257、:293-303 与各 normalize)。

        返回值按 nn 侧 ``register_all_parameters`` 的期望原样回传
        (nn/linear/awq.py:198-201);release 启用时回传的是「瘦身后的参数
        占位」,见 weight_release.py。
        """
        self._release_pack_state()

        linear = get_linear()  # [TM CHANNEL] 按设备单例(linear.py:577-584)

        if self._plan is None:
            # family 选择是确定性查询(family.cc:63-90),pack 格式不变则
            # plan 不变;缓存避免热更新重复遍历 registry。
            # [TM CHANNEL] WeightQuery → registry → WeightPlan
            self._plan = linear.get_weight_plan(
                weight_format=self.weight_format,
                dtype=scales.dtype if scales is not None else weight.dtype,
                input_dtype=self.input_dtype,
                output_dtype=self.output_dtype,
                fusion_type=self.fusion_type,
            )

        if self.fusion_type == 'silu':
            # [TM CHANNEL] gate/up 交错打包(linear.py:353-429)。
            # MergedAwqLinear 现状是 gate/up 两个独立模块(nn/linear/awq.py:215 起);
            # 融合是可选路径,契约上要求调用方传 (gate, up) 二元组。
            gate_w, up_w = weight
            self._weight = linear.fuse_weight(
                (gate_w, up_w),
                plan=self._plan,
                scales=scales,
                zeros=zeros,
            )
        else:
            # [TM CHANNEL] normalize(_normalize_params → WeightFormat.normalize
            # + _apply_bridge scales/zeros 扩块,linear.py:251-259、:48-68)
            # → _tm.LinearWeight.alloc/copy_from → impl.prepare() 设备端 pack
            # (linear.py:240-273;pack 语义 = family_->Pack,weight_plan.cc:23-31)。
            self._weight = linear.prepare_weight(weight, plan=self._plan, scales=scales, zeros=zeros)

        # 模板为热更新引擎流并发 forward 保留逐模块 sync
        # (turbomind_awq_modules.py:54-57);初始加载可省(weightplan-mount.md
        # 挂载点建议 3),stub 保留 sync 语义、区分场景的开关记为分歧点。
        torch.cuda.current_stream().synchronize()

        if self._release_torch_weights:
            # opt-in:瘦身 torch 原始权重,消除 packed 与原始的 ≈×1.9 共存
            # (weightplan-mount.md §1;机制与热更新自愈见 weight_release.py)。
            # 注意:trivial 16-bit 格式收益≈0(TM 持等尺寸副本)且 pack 仅元数据
            # (cublas.cu:199-206),release_helper 会拒绝并原样返回。
            weight, scales, zeros, bias = self._release_helper.release(
                weight, scales, zeros, bias, tm_weight=self._weight, weight_format=self.weight_format
            )
        return weight, scales, zeros, bias

    def _release_pack_state(self):
        """Re-pack 前释放旧 Weight(模板 _release
        语义,turbomind_awq_modules.py:30-40)。"""
        weight, self._weight = self._weight, None
        if weight is None:
            self._linear = None
            return
        # forward 异步于任意引擎流;待全部设备工作完成后释放。
        torch.cuda.synchronize()
        weight.close()
        self._linear = None

    def __del__(self):
        try:
            if self._weight is None:
                return
            linear = self._linear
            with torch.cuda.device(linear.device if linear is not None else torch.cuda.current_device()):
                self._release_pack_state()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # forward:x(activation)→ TM GEMM 入参的适配层(README 问题 1)。
    # ------------------------------------------------------------------
    def forward(
        self,
        x,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        all_reduce: bool = False,
        group: torch.distributed.ProcessGroup | None = None,
        rank: int = 0,
        scatter_size: list[int] | None = None,
    ):
        linear = self._linear
        weight_handle = self._weight
        if weight_handle is None:
            raise RuntimeError('update_weights() must run before forward (TM packed weight missing)')

        # --- 适配 1:形状。TM 消费逻辑 [m, k] RowMajor,内部自行
        # view({-1,k})(LlamaLinear.cu:95、:56);3D 输入在此摊平,
        # 输出按调用方约定回形。
        in_shape = x.shape
        x2 = x.flatten(0, -2)

        # --- 适配 2:连续性。Tensor.__dlpack__ 导出 strides=nullptr,消费端
        # 按 C 连续行主解释(bind.cpp:172-177);非连续 torch 张量会被静默
        # 误读,必须显式保连续。release 启用时 weight 形参可能已瘦身,
        # forward 用的是 TM handle,此形参仅作契约占位(与模板一致,
        # turbomind_awq_modules.py:59-73 同样不再读 qweight)。
        if not x2.is_contiguous():
            x2 = x2.contiguous()

        # --- 适配 3:量化变换不在 python 侧。fp16/bf16 激活由 family 的
        # ConvertInput→QuantizeSymm 现场量化(family.cc:129-133);已量化激活
        # 走 input_scales 槽(linear.py:476-498 支持,此 stub 不展开,
        # README 问题 1 备选)。
        # [TM CHANNEL] ExecPlan 查询 = dispatch cache (kReuse) 命中路径
        # (gemm.cu:194-217);miss 时启发式选核并 Insert。
        exec_plan: ExecPlan = linear.get_exec_plan(x2, weight_handle)
        # [TM CHANNEL] launch:forward_dense → LlamaLinear::Forward → gemm_.Run
        # (linear.py:476-498 ← linear_bind.cpp:209-231 ← LlamaLinear.cu:258-277)。
        out, out_scales = linear(x2, weight_handle, exec_plan=exec_plan)

        # --- 适配 4:bias 逐层加(C++ 无 epilogue-bias,研究差距表 #9;
        # 先例 turbomind_awq_modules.py:66-67)。fp8 输出家族的 out_scales
        # 需先反量化到浮点才能加 bias(family.h:18-22 output_spec;
        # stub 仅标注,fp8-output 家族接入时补 DequantizeSymm 调用)。
        if bias is not None:
            out = out + bias

        out = out.reshape(in_shape[:-1] + (self.out_features, ))
        if out.ndim == 2:
            out = out.unsqueeze(0)

        # --- 适配 5:通信承接(README 问题 2)。TM 通道不管通信
        # (linear_bind.cpp 无 comm 面,研究差距表 #10);先例
        # turbomind_awq_modules.py:69-70 + DefaultLinearImpl
        # (backends/default/linear.py:32-36)。
        if all_reduce:
            if scatter_size is not None:
                from lmdeploy.pytorch.distributed import reduce_scatter_by_tp_sizes

                out = reduce_scatter_by_tp_sizes(out, rank, scatter_size, group=group)
            else:
                dist.all_reduce(out, group=group)
        return out

    # ------------------------------------------------------------------
    # forward_batched(README 问题 3):显式不支持。
    # ------------------------------------------------------------------
    def forward_batched(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'TurboMind linear channel has no one-input-many-weights entry '
            '(forward_moe partitions tokens, not weights); '
            'BatchedLinear gate must keep supports_batched=False.'
        )

    # ------------------------------------------------------------------
    # get_unquantized_weight:MLA kv_b_proj 吸收路径
    # (models/deepseek_v2.py:500;nn/linear/base.py:189-199 默认用单位矩阵
    # 过一遍 forward,对 TM 路径即「去量化」)。专用实现避免 m=K 的巨型
    # 单位矩阵 GEMM,直接用 WeightFormat.dequant(weight_format.py:226-241
    # AWQ 先例)。
    # ------------------------------------------------------------------
    def get_unquantized_weight(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'stub: needs the torch-side raw (pre-release) tensors or a '
            'TM-side dequant bind; see README 问题 5 / weight_release.py.'
        )
