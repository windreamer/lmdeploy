# WeightPlan 挂载时机与显存开销

来源:Ticket 03(`.scratch/wayfinder/tm-gemm-graft/issues/03-weightplan-mount-timing.md`,wayfinder:research,2026-11-16,只读盘点)。
一句话结论:TM 原生管线 pack 是**就地覆写**(仅 u16 暂存,峰值 ≈ 稳态 ×1.06,W4A16 7B 例算),而 PyTorch 引擎现存的唯一挂载点(`process_weights_after_loading` 的 `update_weights` 钩子,已被 `TurbomindAwqLinearW4A16Impl` 走通)目前让 packed 与 torch 原始权重**全程共存**(≈2×),且会被热更新路径全量重走;该钩子就是正确的挂载点,不需要另开接缝。

引用均为当前工作副本 `文件:行`。票面所写 `agent.py:47-49` 对应本副本 `agent.py:43-44`(import)与调用点 `agent.py:1809`/`agent.py:1977`;本文一律用实际行号。

## 1. pack 就地还是共存?峰值显存倍数(W4A16 7B 算例)

### TM 原生流程:就地覆写

- `LinearWeight::prepare()` 调 `plan_->pack(*this, stream)`(`src/turbomind/models/linear_weight.cc:61-68`),`WeightPlan::pack` 转 `family_->Pack` 并回填 family/format 元数据(`src/turbomind/kernels/gemm/weight_plan.cc:23-31`)。
- 设备端 `PackWeight`(u4/量化路径,第二个重载)的步骤:
  1. 分配 `tmp` uint16 `[input_dim, output_dim]`(每元素 2 B),把 u4/u8 权重扩展进去(`src/turbomind/kernels/gemm/convert_v3.cu:85-92`);
  2. 分配 `trans` uint16 `[output_dim, input_dim]`,转置(`convert_v3.cu:90-93`);
  3. **`cudaMemsetAsync` 清零原始 `linear.weight` 缓冲,再由 convert kernel 把打包结果写回同一缓冲**(`convert_v3.cu:94-97`)。
     即 packed 与"原始权重"不共存:原始数据被就地销毁,只有 tmp+trans 两份 u16 暂存。第一个重载(含 16-bit 权重路径)同样 memset+回写(`convert_v3.cu:36-59`)。
- SM90 W4A16(AWQ g=128 → mixed fp16/u4 family)每模块的完整序列 = `PackWeight` + qparams 融合打包(旧 scales/zeros move 出、新 u8 `packed_q` 写回后旧的释放,`src/turbomind/kernels/gemm/kernel/sm90_64n16_4.cu:22-56`;qparams 重载 `convert_v3.cu:133-152`)。
- 对照:trivial fp16/bf16(cuBLAS family)的 `pack` 只改元数据、零设备工作(`src/turbomind/kernels/gemm/cublas.cu:199-206`)。

**算式**(LLaMA-7B 形状:hidden 4096,32 层,inter 11008,g=128;数字为按代码验证的字节数推导 \[推导\],未实测):

- 4-bit 线性层参数量 = 32×(4×4096² + 3×4096×11008) = 32×202.4M ≈ **6.48B 元素**;
- 稳态 packed = qweight 6.48B×0.5 B = **3.24 GB** + scales(u8 融合后)6.48B/128×1 B ≈ 0.05 GB;非量化部分(embed+lm_head+norm,bf16)≈ 0.52 GB → 稳态 ≈ **3.81 GB**;
- pack 瞬时(单模块,K=4096,N=11008,45.1M 元素):weight 0.5 + tmp 2 + trans 2 = **4.5 B/元素 = 203 MB**,即该层 packed 尺寸(22.6 MB)的 **9 倍**;
- 模型级峰值 = 3.81 GB + 0.20 GB ≈ 4.06 GB → **≈ 稳态 ×1.06**(逐模块串行,暂存随做随释放,`Module::prepare` 树遍历 `src/turbomind/core/module.cc:58-63`)。

### PyTorch 挂载现状:结构性地共存

`TurbomindAwqLinearW4A16Impl.update_weights`(`lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py:44-57`)走 `prepare_weight` → `_prepare_one`:`_tm.LinearWeight` 新开参数缓冲 `param().alloc` + `copy_from` torch 张量(`lmdeploy/turbomind/linear.py:240-269`),再 `impl.prepare()` 就地 pack。torch 侧 `qweight/scales/qzeros` 原样返回并被重新注册(`lmdeploy/pytorch/nn/linear/awq.py:198-201`;impl 返回原张量 `turbomind_awq_modules.py:57`),**永不释放** → torch int32 qweight(0.5 B/元素)与 TM packed Weight(0.5 B/元素)全程共存:

- 峰值 ≈ 3.24×2 + 0.10 + 0.52 + 0.20 ≈ 7.30 GB → **≈ TM 稳态 ×1.9**(\[推导\])。

## 2. `process_weights_after_loading` 的执行上下文适合放 pack 吗?

**适合**,三个前提都满足:

- **已分 rank 切片**:TP 切片发生在 `load_weights` 逐参数 `weight_loader`(`chunk_aligned(...)[rank]`,`awq.py:110-134`;`default_weight_loader` `lmdeploy/pytorch/weight_loader/model_weight_loader.py:25-31`);模型本身直接建在目标设备上(`build_patched_model(..., device=device)`,`agent.py:1512`;量化参数在设备上预分配 `awq.py:156-166`)。`process_weights_after_loading`(`model_weight_loader.py:193-207`)运行时参数已 per-rank、已在 GPU。
- **有空档**:该钩子在 `build_model` 内(`agent.py:1522-1533` → `load_model_weights` 末尾 `model_weight_loader.py:210-215`),位于 KV cache pool 分配(**先** `build_model` `executor/base.py:575`、**后** `build_cache_engine` `executor/base.py:589`)与 cudagraph warmup 之前,加载期 GPU 别无并发工作,瞬时峰值不挤占 cache 预算。
- 两遍结构(先所有模块 `update_weights()`,再 `process_weights_after_loading()` 钩子,`model_weight_loader.py:196-207`)给"pack 一次 + 模型层吸收"留了正确顺序。
- 一个瑕疵:现实现每模块一次显式 `torch.cuda.current_stream().synchronize()`(`turbomind_awq_modules.py:54-57`),注释说明是为热更新时引擎流并发 forward 所设;初始加载期本可省去(流内顺序分配可复用暂存,见 §1 的 9× 上界)。

## 3. `update_weights` 热更新路径是否重走 pack?

**重走,且是全量**。三条热更新路径(`update_weights_from_ipc` `agent.py:1863-1911`、`update_params` `agent.py:1941-1991`、`update_weights_from_distributed` `agent.py:2020-2056`)的共同序列:

1. `_load_updated_weight_bucket` → `target.load_weights(iter(renamed))`(`agent.py:1786-1800`):只覆写 torch 参数,**不 pack**;
2. 结束后 `_finalize_updated_weights` / `process_weights_after_loading(m)`(`agent.py:1802-1814`、`1975-1981`、`2055-2056`):对**每一个**含 `update_weights` 属性的模块调钩子(`model_weight_loader.py:196-198`)→ `AwqLinear.update_weights` 无参重跑 `impl.update_weights`(`awq.py:198-201`)→ `_release()` 旧 Weight + `prepare_weight` 重归一化 + 设备端重 pack(`turbomind_awq_modules.py:47-52`)。
   即:不论 bucket 是否触及该模块,所有 W4A16 线性层每次热更新都重新 pack;随后 `torch.cuda.synchronize()` + `reset_graph_runner()`(`agent.py:1810-1816`)。

## 4. `weight_format.py` 归一化依赖 TM 加载流程的哪些前提?

归一化本体(normalize/pack_u4_row/bridge)是纯 torch 操作,仅依赖 `_tm.DataFormat/DataType` 描述符(`lmdeploy/turbomind/weight_format.py:20-22,161-165`),不触碰 C++ runtime。其**语义**依赖以下前提:

1. **TM-native 目标布局 = \[K=in_dim, N=out_dim\]**,由 pack kernel 的读入约定定义(`linear.weight` 按 `[input_dim, output_dim]` u4/u8 读,`convert_v3.cu:85-92`;`LinearConfig.input_dim = shape[0]`,builders 提交链 `lmdeploy/turbomind/builders/_base.py:322`、`linear.py:249-250`)。AWQ/GPTQ checkpoint 恰好原生就是这个顺序(注释 `weight_format.py:243-250,293-303`),trivial/FP8/CT/MXFP4 需 `.t()`(`weight_format.py:209-213,337-345,402-408,451-457`)。
2. **DataFormat 块语义 \[block_in=K 组, block_out=N 块\]** → `MakeQuantDesc` 取 block_sizes\[0\] 为 K 向组(`linear_weight.cc:26-40`),family 匹配走 `WeightQuery`+`Family::supports`(`weight_plan.h:15-22`、`family.cc:81-106`);要求 scales 形 \[K/g, N\]。
3. **bridge 可扩展性**:scales/zeros 须能 `repeat_interleave` 到 kernel blocking(`linear.py:62-84`,提交前 `_bridge_linear` `_base.py:13-20`;sm90 pack 对冗余行列的容忍注释 `sm90_64n16_4.cu:32-42`)。
4. **提交字节守恒不变式**:`dst.byte_size == shard.nbytes`(`_base.py:156-159`),由 `pack()` 用 alloc_shape/alloc_dtype 把 uint8 重标 UINT4 满足(`weight_format.py:48-52,252-257`)。
5. **TP 切片在 pack 之前完成、受 shape_constraints 约束**:K≥min_k、K%align_k==0、N≥min_n、N%align_n==0(`_base.py:330-341`),N 侧切片须保持 int32 行(`pack_u4_row` 每 8 元素一 int32,`weight_format.py:75-97`)。
6. **两阶段时序**:normalize 在 commit 时,设备端 pack 延迟到全部权重提交后的单次 `ModelRoot::prepare()`(`turbomind.py:168-176` → `turbomind.cc:93-99` → `module.cc:58-63`),引擎创建之前。
7. **Family 选择依赖 per-device Gemm registry 与架构门控**(`gemm.get_weight_plan` `_base.py:308-312`;`TM_GEMM_WEIGHT_PACK` 环境门控:`WeightPackEnv()` `convert_v3.cu:24-32`,`family.cc:100-106` —— env=0 时 requires_packing 的 family 全部不可选)。

**脱离 C++ runtime 后**:前提 1/2/3/5/7 仍成立——它们由 TM GEMM kernel 与 registry 定义,不依赖 runtime 调度器;前提 4/6 的载体(C++ Module 参数分配 + prepare 树遍历)不存在于 PyTorch 引擎,但已被证明可绕过:`TurbomindAwqLinearW4A16Impl` 用 standalone `_tm.LinearWeight` + `set_plan` + `impl.prepare()`(`turbomind_awq_modules.py:50-56` → `linear.py:244-269`)替代了整棵 C++ 模型树,只保留 `_tm.LinearWeight`/`Gemm::get_weight_plan` 这条最小 C++ 依赖。结论:归一化可整体脱离 C++ 加载流程,但 pack 的**目标布局无法纯 torch 复刻**(布局由 kernel 消费端定义);另注意前提 5 在 PyTorch 引擎中方向相反——先按 rank 切再 pack,等价于 TM 的 per-GPU commit 之后 prepare,`_turbomind_support_reason` 已按 per-rank 尺寸做 K%128/N%32 检查(`cuda/awq_modules.py:20-43`)。

## 挂载点建议

1. **唯一挂载缝:`process_weights_after_loading` 第一遍的 per-module `update_weights()` 钩子**(`model_weight_loader.py:196-198`),与现有 `TurbomindAwqLinearW4A16Impl` 相同;不要再造第二条入口——热更新三路径已自动复用(§3),TM 侧 `ModelRoot::prepare` 语义由 `impl.prepare()` 等价承载。
2. **显存优先**:pack 完成后释放 torch 侧 qweight(消除 §1 的 1.9× 共存);forward 已全走 impl(`turbomind_awq_modules.py:59-73`),但释放须做成 opt-in——原始 torch 参数是热更新 `load_weights` 的落点(`agent.py:1800`),不可无条件断。
3. **时延**:初始加载可去掉逐模块 sync;热更新保留(引擎流并发 forward 所需)。不要全模型批量 enqueue 后一次 sync 之外再做逐层等待——同一流上顺序分配即可把峰值钉在单层 203 MB 的上界(§1)。

## 显存/时间事实表

| 项目                            | 事实                                                                                  | 引用                           |
| ------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------ |
| pack 就地性(TM 原生)            | memset 原缓冲 + convert 回写,packed 与原始不共存                                      | convert_v3.cu:94-97            |
| 单层暂存                        | tmp u16 \[K,N\] + trans u16 \[N,K\] = 4 B/元素 = 该层 packed 的 9×                    | convert_v3.cu:85-93            |
| qparams 暂存                    | 旧 scales/zeros → 新 u8 packed_q,旧的 move 后释放                                     | sm90_64n16_4.cu:47-56          |
| trivial family pack             | 仅元数据,零设备工作/显存                                                              | cublas.cu:199-206              |
| W4A16 7B 稳态                   | ≈3.81 GB(packed 3.24 + scales 0.05 + bf16 非量化 0.52)\[推导\]                        | —                              |
| W4A16 7B 峰值(TM 原生,逐层串行) | ≈4.06 GB ≈ 稳态 ×1.06 \[推导\]                                                        | §1 算式                        |
| W4A16 7B 峰值(PyTorch 挂载现状) | ≈7.30 GB ≈ TM 稳态 ×1.9,torch 原始权重常驻共存 \[推导\]                               | awq.py:198-201                 |
| pack 时长基线                   | 70B 全模型 1.47s(论文 §5.3,票面给定,repo 不可验证);7B 按参数量线性外推 ≈0.15s(未实测) | 票面                           |
| 每模块 kernel 数(u4)            | memset + extend + transpose + pack + qparams fuse,全部异步入队当前流                  | sm90_64n16_4.cu:29,47-56       |
| TM 原生同步点                   | 整树一次遍历,无逐层 sync,仅尾 `cudaGetLastError`                                      | turbomind.cc:93-99             |
| PyTorch 挂载同步点              | 每模块一次显式 stream sync(热更新必需,初始加载可省)                                   | turbomind_awq_modules.py:54-57 |
| 热更新代价                      | 每次热更后所有 W4A16 模块重走归一化+pack,随后 sync+图失效                             | agent.py:1802-1816,1975-1981   |
| 挂载时序安全区                  | 钩子位于 KV pool 分配(executor/base.py:575→589)与 cudagraph warmup 之前               | §2                             |

## 对后续决策的影响

- 接缝已存在且被验证(`TurbomindAwqLinearW4A16Impl`),GEMM-graft 应复用它而非新开挂载点;要做的增量是"释放 torch 原始权重"与"初始加载免逐层 sync"两处显存/时延优化。
- 显存预算按 TM 原生 ×1.06 报;若保留 torch 共存则按 ×1.9 报,勿混用。
- 热更新路径的全量重 pack(§3)是 RL 场景的真实成本,设计热更协议时应知晓。
