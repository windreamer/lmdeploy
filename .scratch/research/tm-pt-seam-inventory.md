# 双后端接缝盘点（PyTorch Engine ↔ TurboMind kernels）

来源：SeamInventory agent（2026-11-16，只读盘点）。此文件是研究记录，非设计文档。引用为 `文件:行`。

## 1. PyTorch 引擎：kernel 调用接缝

- 设备级后端分发：`lmdeploy/pytorch/backends/selector.py:7-46`（Cuda/Ascend/Maca/Camb OpsBackend）。
- 后端抽象：`backends/base.py:17`（`BuildSpec[T]`）、`base.py:21`（`OpsBackend`：`get_name`/`build_op`/`get_attention_metadata_cls`/`get_k_block_shape`/`get_v_block_shape`/`get_cache_backend`/`update_step_context`/`model_build_context`/`build_communicator`/`build_graph_runner`）。
- CUDA 后端 `build_op` 为 ~30 种 BuildSpec 分发表：`backends/cuda/op_backend.py:33-250`。
- Attention 接口：`AttentionMetadata`（`block_offsets`、`q_start_loc/q_seqlens/kv_seqlens`、`cu_seqlens_q/k`、`quant_policy`）`backends/attention.py:15`；`AttentionImpl.forward(query,key,value,k_cache,v_cache,attn_metadata,...)` `attention.py:132,206`；`PagedAttentionBuildSpec` `attention.py:225`；V4 稀疏契约 `attention.py:29,79,101,120`。
- 模型层从不直接触碰 kernel：`nn/attention.py:23` 的 `Attention` 模块经 `build_op(PagedAttentionBuildSpec)` 拿 impl 并转发（:46-61,127-175）。
- attention 实现选择：FlashMLA→FA3→Triton 兜底：`backends/cuda/attention/__init__.py:88-140`。
- KV cache 形状由后端声明：`op_backend.py:252-274`（CUDA `(block_size,num_heads,head_size)`）；cache backend 抽象 `backends/cache.py:32`（`build_block_layout`/`build_state_layout`/`build_block_copy`）；`CacheEngine` `engine/cache_engine/engine.py:48`；物理布局 `CachePool(entry_axis)` `engine/cache_engine/layout.py:20,45`。
- Rotary：`backends/rotary_embedding.py:67,84`（含 ApplyRotaryEmb 融合写 cache）。
- Linear/MoE：`backends/linear.py:9-39`（`LinearImpl.forward(x,weight,bias,all_reduce,group,rank,scatter_size)`）；MoE 量化变体 `backends/cuda/moe/*`，EP 路径 `moe/ep.py`。
- 执行入口：`engine/model_agent/agent.py:226` `model_forward()`；cudagraph 由 `backends/graph_runner/` 承担。
- 权重：HF checkpoint 逐参数 `param.weight_loader`（`weight_loader/model_weight_loader.py:17,26`）；agent 挂 `process_weights_after_loading`（agent.py:47-49）。布局假设常规 `[out,in]`。

## 2. TurboMind：可保留的 kernel 资产

- Attention：`src/turbomind/kernels/attention/`（`dispatchAttention`、`attention_sm{70,75,80}_{64..576}.cu`、split-K `reduce.cu`、运行时注册表 `registry.h:12`；KV 写入 `kv_cache_utils_v2.h:8,43`；设备端 RoPE `rotary_embedding.h:46`）。
- GEMM：`src/turbomind/kernels/gemm/`（SM90 `gemm_universal_sm90*`，bf16/fp8/mixed/mxfp4/nvfp4；计划缓存 `dispatch_cache.cu`；权重打包 `weight_plan.h:23` `WeightPlan::pack`）。
- 线性注意力/GDN：`kernels/linear_attn/`（delta_rule + sm_90/sm_120/pre_sm90）。
- 小 kernel：`kernels/norm/rms_norm.cu`、`kernels/quantization.cu`、`attention/quantization.h`。
- MoE a2a（EP）：`comm/nccl/deepep/moe_a2a_utils.cu` + `token_dispatcher`。

## 3. 硬耦合点（TM kernel ↔ C++ runtime）

1. **block table = 设备端指针数组**：host pinned 里写 `CacheBlock::base(0)+offset` 成 `void*` 数组再 H2D（`unified_attention_layer.cc:289-304`）；kernel 参数 `BlockIteratorParams{char** block_ptrs; const int* cu_block_nums; ...}`（`attention_params.h:17-21`），设备端解引用（`block_iterator.h:27`）。PyTorch 整型 `block_offsets` 不能直接喂。
2. **KV 块物理布局由 kernel 侧 `block::Layout` 定义**：`[block_len,H,D]`，MLA `HeadDim==576` 特化（`block_iterator.h:55-99`）。
3. **调度器私有标量作 kernel 入参**：`readonly_block_num`（`attention_params.h:29`，被 `invokeProcessKV_v2` 消费）、`cu_q_len/cu_k_len`、`cache_block_offset`。GDN 递归状态与 checkpoint 同 slot 复合（`GatedDeltaNetLayer.cc:149-157`）。
4. **Attn-DP/CP 内建 kernel**：全局 `token_mask`（`attention_params.h:34-36`）、`cp_rank/cp_size + cp_post_fn`（:82-86）。
5. **GEMM 权重布局管线**：`LinearWeight` 携 `DataFormat` + `WeightPlan`，`prepare()` 设备端重排（`linear_weight.h:60-61`）；python 侧只做 checkpoint→TM-native 归一化（`lmdeploy/turbomind/weight_format.py` 六种 WeightFormat）。保留 GEMM kernel 必须保留 `DataFormat→Family→WeightPlan→pack` 管线。
6. **模型层↔引擎 BatchOp 契约**：`Run(BatchOp,phase,env)` + phase 索引设备缓冲（`language_model.cc:99`、`unified_attention_layer.cc:175-197`）。
7. **持久 batch 语义**在 C++ runtime：`seq_len/resume_len/filled_len/inflight_*`（engine/README.md invariants；kernel 的 `finished/autoregres/generating` 输入由此推导）。

## 4. 功能差异（事实清单）

- PyTorch 有、TM 无：投机解码（`spec_decode/` 全套 + FA3 硬依赖）、PD 分离（`disagg/` + mooncake kv_connector）、CUDA Graph（`graph_runner/`）、V4 稀疏注意力、纯 torch 模型动物园（glm4_1v、deepseek_v32 等）。
- TM 有、PyTorch 无对应：多 phase 异步流水线（`BatchData` 槽）、复合 cache-object 注册表、对象化 slab 分配与驱逐回放、generation checkpoint partial-sibling 发布、双 python/C++ 权重加载、细粒度并行组合（attn_dp/attn_tp/attn_cp/mlp_dp/mlp_tp/outer_dp）、GDN/KDA C++ kernel。
- 两侧都有：EP MoE（DeepEP）、前缀缓存 trie、MLA。
