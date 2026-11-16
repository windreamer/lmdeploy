# Copyright (c) OpenMMLab. All rights reserved.
"""Torch 原始权重 opt-in 释放:最小机制草稿(throwaway,不进源码树)。

票据约束(票 03 §3,详见 .scratch/research/weightplan-mount.md)
---------------------------------------------------------------
* pack 完成后释放 torch 侧原始权重可消除 packed/torch 的 ≈×1.9 显存共存
  (weightplan-mount.md §1:W4A16 7B 峰值 7.30 GB → 稳态 3.81 GB 量级),
  forward 已全走 TM handle,torch 参数在 forward 中不再被读
  (turbomind_awq_modules.py:59-73)。
* **不可无条件断**:torch 参数是热更新三条路径的落点 ——
  ``_load_updated_weight_bucket → target.load_weights(iter(renamed))``
  (agent.py:1786-1800)逐参数写 torch 张量,``default_weight_loader`` 断言
  ``param.size() == loaded_weight.size()`` 后 ``copy_``
  (model_weight_loader.py:24-31);释放(清零 storage)后任何热更新都会在
  断言处崩溃。所以释放必须 opt-in,且必须给出热更新自愈路径。

机制(本草稿)
--------------
两级 opt-in + 「溢出-回灌」自愈:

1. 开关层:
   - 进程级:env ``TM_GEMM_RELEASE_TORCH_WEIGHTS``(与 TM 侧
     ``TM_GEMM_WEIGHT_PACK`` 门控先例同构,family.cc:79-84 / WeightPackEnv);
   - 模型级:``LinearTmGemmBuildSpec.release_torch_weights``
     (ops_backend_patch.py),叠加语义取或。
   - ``=1`` spill 模式:GPU 原始权重 → CPU 常驻副本(内容精确),
     GPU 显存省下、热更新可无损回灌;``=2`` immutable 模式:不保留副本,
     显存与主机内存都省,但被释放模块自此拒绝热更新(带清晰报错)。

2. 释放层 ``release()``:update_weights 钩子内、pack 发布之后执行
   (时序安全区见 weightplan-mount.md §2:KV pool 分配之前,无并发工作)。
   每个张量:精确拷贝到 CPU(spill 模式)→ 参数 storage 换成 0 元素张量
   (nn 侧 ``register_all_parameters`` 按返回值注册,awq.py:180-195)。

3. 自愈层 ``arm()/restore()``:包装每个参数的 ``weight_loader`` 属性 ——
   首次 loader 调用先按记录重新分配 GPU storage 并从 CPU 副本回灌,再委托
   原 loader 写入新数据。未触及模块的参数内容来自 CPU 精确副本,后续
   update_weights 重 pack 不会读到垃圾(**这是选 CPU 溢出而非
   「从 TM handle 反量化」的原因**:u4 payload 打包在 family 布局里,
   python 绑定无 unpack 入口,且去量化→重量化不保证 bit 级一致)。

   **安装时序约束**:`register_all_parameters → setup_loaders` 每次都会把
   ``param.weight_loader`` 重置为模块方法(nn/linear/awq.py:60-67、:194),
   因此 ``arm()`` 必须在 nn 侧 ``update_weights`` 完成
   ``register_all_parameters`` **之后**调用 —— 这是本机制对 nn 层的唯一
   侵入点(一行),不能在 impl 内部完成。

4. 拒绝释放的情形(保守门控):
   - trivial 16-bit:pack 仅元数据(cublas.cu:199-206),TM 持等尺寸副本,
     释放零收益、徒增回灌复杂度 → 拒绝;
   - 无 TM handle(pack 未完成)→ 拒绝。
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

_RELEASE_ENV = 'TM_GEMM_RELEASE_TORCH_WEIGHTS'

# 释放级别
SPILL = 1    # GPU→CPU 溢出,热更新可回灌(默认语义)
IMMUTABLE = 2  # 不保留副本,被释放模块拒绝热更新


def release_enabled() -> bool:
    """进程级 opt-in(与 BuildSpec 字段取或)。"""
    return os.environ.get(_RELEASE_ENV, '0') not in ('', '0')


def release_level() -> int:
    """返回释放级别;未启用返回 0。"""
    raw = os.environ.get(_RELEASE_ENV, '0')
    return int(raw) if raw.isdigit() else 0


class _ReleaseSpec(NamedTuple):
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    spill: torch.Tensor | None  # CPU 精确副本;immutable 模式为 None


class TorchWeightRelease:
    """Per-impl 释放状态机:release → (nn 注册) → arm → (热更新时) restore。

    使用方(TmGemmLinear / AwqLinear 形态的 nn 模块)::

        # nn.update_weights():
        tensors = self.impl.update_weights(self.qweight, self.scales,
                                           self.qzeros, self.bias)
        self.register_all_parameters(*tensors)          # awq.py:196-200
        self.impl.post_register_release(self)            # arm();仅此一行侵入
    """

    def __init__(self):
        self._specs: list[_ReleaseSpec] = []
        self._armed = False

    # ------------------------------------------------------------------
    # 阶段 1:释放(update_weights 钩子内、pack 发布之后)
    # ------------------------------------------------------------------
    def release(
        self,
        weight: torch.Tensor | tuple,
        scales: torch.Tensor | None,
        zeros: torch.Tensor | None,
        bias: torch.Tensor | None,
        *,
        tm_weight,
        weight_format,
    ):
        """返回可能被瘦身的 (weight, scales, zeros, bias) 供 nn 重新注册。

        ``weight`` 为 tuple(gate/up 融合路径)时整体跳过释放(stub 简化,
        融合模块的参数布局不同,记 README 待办)。
        """
        if not self._eligible(weight, tm_weight, weight_format):
            return weight, scales, zeros, bias

        level = release_level()
        names = ('weight', 'scales', 'zeros', 'bias')
        tensors = (weight, scales, zeros, bias)
        slimmed = []
        for name, tensor in zip(names, tensors):
            if tensor is None:
                slimmed.append(None)
                continue
            spill = None
            if level != IMMUTABLE:
                # D2H 精确副本。此处已过 pack 发布 sync(turbomind_awq_modules.py:
                # 54-57 同位置),流内有序,无需额外 sync。
                spill = tensor.detach().to('cpu', copy=True)
            self._specs.append(_ReleaseSpec(name, tuple(tensor.shape), tensor.dtype, tensor.device, spill))
            # 0 元素同 dtype/device 占位:register_all_parameters 会包成
            # 空参数,storage 即刻可回收(引用仅剩 Parameter 对象壳)。
            slimmed.append(tensor.new_empty(0))
        return tuple(slimmed)

    def _eligible(self, weight, tm_weight, weight_format) -> bool:
        if isinstance(weight, tuple):
            return False  # 融合路径 stub 不处理
        if tm_weight is None:
            return False  # pack 未完成,没有可承载语义的 handle
        if weight_format.name == 'trivial':
            return False  # 零收益(cublas.cu:199-206),拒绝
        return release_enabled()

    # ------------------------------------------------------------------
    # 阶段 2:布防(nn 注册之后;见模块 docstring 的时序约束)
    # ------------------------------------------------------------------
    def arm(self, module):
        """包装本 impl 释放过的模块参数的 weight_loader,回灌先行。

        一次调用覆盖模块全部已释放参数。

        任一 loader 首次被调用时,先整模块回灌:未触及参数从 CPU 副本恢复 精确内容,再执行原 loader。
        """
        if self._armed or not self._specs:
            return
        param_names = {spec.name for spec in self._specs}
        first_names = [n for n in param_names if hasattr(module, n)]
        original_loaders = {n: getattr(module, n).weight_loader for n in first_names}
        helper = self

        def _make_wrapper(name, original):
            def wrapper(param, loaded_weight, **kwargs):
                if helper._specs:  # 首次触达:整模块回灌
                    helper.restore()
                return original(param, loaded_weight, **kwargs)

            return wrapper

        for name in first_names:
            param = getattr(module, name)
            if original_loaders[name] is None:
                def _raising_loader(param, loaded_weight, _n=name):
                    raise RuntimeError(
                        f'hot update into released-immutable parameter '
                        f'{_n} is unsupported (TM_GEMM_RELEASE_TORCH_WEIGHTS=2)'
                    )
                param.weight_loader = _raising_loader
            else:
                param.weight_loader = _make_wrapper(name, original_loaders[name])
        self._armed = True

    # ------------------------------------------------------------------
    # 阶段 3:回灌(arm 的 wrapper 内部触发;也可显式调用)
    # ------------------------------------------------------------------
    def restore(self):
        """按记录重新分配 GPU storage 并回灌 CPU 副本内容。"""
        specs, self._specs = self._specs, []
        self._armed = False
        for spec in specs:
            if spec.spill is None:
                # immutable 模式没有副本:上层应在 arm 阶段拦截,这里兜底。
                raise RuntimeError(f'released-immutable parameter {spec.name} cannot be restored')
            tensor = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            tensor.copy_(spec.spill)
            spec.spill = None  # 释放主机内存

    @property
    def resident_bytes(self) -> int:
        """当前溢出到 CPU 的字节数(监控用)。"""
        return sum(s.spill.numel() * s.spill.element_size() for s in self._specs if s.spill is not None)
