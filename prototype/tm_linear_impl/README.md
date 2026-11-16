# LinearImpl 契约原型(stub)—— 票 02 设计笔记

> throwaway 原型,不进 `lmdeploy/` 源码树,不要求端到端跑通。
> 模板:`TurbomindAwqLinearW4A16Impl`(`lmdeploy/pytorch/backends/cuda/turbomind_awq_modules.py`,票 01 验证的端到端通道消费者)。
> 研究输入:`.scratch/research/gemm-python-channel.md`(下称「通道盘点」)、`weightplan-mount.md`(下称「挂载盘点」)、`tm-pt-seam-inventory.md`(下称「接缝盘点」)。
> 本目录文件:`ops_backend_patch.py`(OpsBackend 分发 patch 形态)、`tm_linear_impl.py`(通用 impl)、`weight_release.py`(释放机制)、本笔记。

## stub 结构

```
prototype/tm_linear_impl/
├── ops_backend_patch.py   # diff 风格:CudaOpsBackend.build_op 分发树上新增
│                          #   LinearTmGemmBuildSpec → _build_linear_tm_gemm 分支
│                          #   (插入点:op_backend.py:166-169 W4A16 分支之后);
│                          #   附 BuildSpec/工厂摘要,可行性判定改为 registry 探询
├── tm_linear_impl.py      # 通用 TurbomindLinearImpl(LinearImpl 子类):
│                          #   格式参数化 / forward 适配层 / all_reduce+scatter 承接 /
│                          #   bias 逐层加 / supports_batched=False / opt-in 释放挂接;
│                          #   [TM CHANNEL] 注释 = 假 kernel 边界上的真通道调用点标注
├── weight_release.py      # opt-in 释放两级开关(spill/immutable)+ CPU 溢出 +
│                          #   weight_loader 包装回灌的热更新自愈
└── README.md              # 本笔记
```

调用链(patch 后):模型 nn 层 → `build_op(LinearTmGemmBuildSpec)` → `TurbomindLinearImpl` → `update_weights()` 钩子(pack)→ `forward()`(`get_exec_plan` + `linear(x, weight)`)。与 AWQ 现状(`awq.py:180-203` → `awq_modules.py:88-118` → `turbomind_awq_modules.py`)逐环对应。

______________________________________________________________________

## 问题 1:x(activation)→ TM GEMM 入参的适配写在哪一层?

**推荐:适配集中在 `TurbomindLinearImpl.forward`(impl 层),且只做形状/连续性两件事;量化变换不在 python 侧做;权重侧排布完全推到 update_weights 的 WeightFormat 归一化。**

依据:

- **形状**:TM 消费逻辑 `[m,k]` RowMajor,`LlamaLinear::GetArguments` 先 `view({-1,k})` 再用 `A.stride(0)` 允许行主 padded view(LlamaLinear.cu:95、:56;通道盘点 §4)。impl 只需 `x.flatten(0,-2)`,**永远不需要转置激活**;m 维自由,kernels 接受任意 m(exec plan 逐调用查询)。
- **连续性**:`Tensor.__dlpack__` 导出 `strides=nullptr`,消费端按 C 连续行主解释(bind.cpp:172-177);`from_dlpack` 才接受显式步长(bind.cpp:257-276)。impl 必须保证 `x.contiguous()`(tm_linear_impl.py forward 适配 2),否则静默误读 —— 这是模板没有显式做的隐含前提,通用化时应显式化。
- **量化变换**:`ConvertInput` 在 forward 内现场把 fp16/bf16 激活 `QuantizeSymm` 成 fp8+group scales(family.cc:129-133);已量化激活自带 `input_scales` 直接作 U(LlamaLinear.h:36-40)。python 侧不需要预变换 → 适配层零量化逻辑。
- **scales/zeros 排布**:不是 forward 期问题。pack 时 `_normalize_params` 做 `WeightFormat.normalize` + `_apply_bridge`(按 `WeightBridge.replicate_scales` repeat_interleave 并裁边,linear.py:251-259、:48-68;family.h:40-47),TM-native `[K,N]`/`[K/g,N]` 布局由各 WeightFormat 定义(AWQ 免转置,GPTQ +1 offset,Trivial/FP8/CT/MXFP4 需 `.t()`;weight_format.py:243-257、:293-303 及各 normalize)。q_desc/V 槽在 `GetArguments` 内组装(LlamaLinear.cu:29-36、:118)。
- **padding/alignment**:K/N 对齐是**权重侧构建期**问题 —— build 期检查(先例 `_turbomind_support_reason` 的 K%128/N%32,cuda/awq_modules.py:27-30;通用化后用 `plan.shape_constraints`,linear.py:213-225;commit 侧同款 `_base.py:330-341`)。激活侧无需 padding。
- 输出:`ExecPlan.output` 的 shape/stride/dtype 由 `family->output_spec` 给出(family.h:95-98;output_spec.h:7-12),python 侧 `torch.empty_strided` 复刻(linear.py:93-107、:452-464)—— 输出适配已被通道吃掉,impl 只回形 reshape + 2D→3D(模板先例 turbomind_awq_modules.py:68-69)。

备选:

- (a) 适配下沉到 nn 层 —— 否:nn 必须保持 impl 无关(`LinearBase._forward_default` 只传 x/all_reduce/tp_sizes,nn/linear/base.py:200-203)。
- (b) fp8 w8a8 场景在 python 侧预量化激活(用已暴露的模块级 `QuantizeSymm/QuantizeGroupwise`,linear_bind.cpp:306-406)以避免 forward 内 `ConvertInput` 每次重算 —— 保留为后续优化;`Family::ConvertInput` 无独立绑定(通道盘点 §3.3),且 `input_scales` 入参已通(`linear(x, weight, input_scales=...)`,linear.py:476-497)。
- (c) 在 `Linear._activate()` 之外做 DLPack 转换预检查 —— 过度设计;通道已封装(`_tm.from_dlpack`,linear.py:475)。

**需要拍板**:fp8-input 家族(w8a8)若纳入第一梯队,激活量化走 kernel 内 `ConvertInput` 还是 python 预量化 + `input_scales`?后者省重复量化但引入 scales 生命周期管理。stub 当前选 kernel 内(最简、与模板同构)。

______________________________________________________________________

## 问题 2:`all_reduce/group/rank/scatter_size` 与 TM 并行约定的对应?

**推荐:TM 通道完全不承接通信;四个契约字段全部由 impl.forward 内的 python dist 完成 —— `all_reduce`+`group` 用 `dist.all_reduce`(模板先例 turbomind_awq_modules.py:69-70),`scatter_size` 非空时用 `reduce_scatter_by_tp_sizes(out, rank, scatter_size, group)`(`DefaultLinearImpl` 先例,backends/default/linear.py:32-36)。**

依据:

- TM GEMM 通道无任何 comm 暴露面:`linear_bind.cpp`(:52-494)只有 plan/pack/launch/tune,无 NCCL/collective;TP 在 TM 侧是引擎内建(细粒度并行组合,接缝盘点 §4),但 python 直驱通道不经过引擎调度器 → 通信天然留在 PyTorch 侧(通道盘点差距表 #10)。
- 契约字段由调用方填充:nn 层把 `self.tp_group` 传入(`AwqLinear._forward_default`,awq.py:201-203);`rank/scatter_size` 仅 DP_TP/TP 分散场景使用,同样由 nn 层先算好(`LinearBase._forward_lora` 的 TPMode.DP_TP 分支在 nn 层做 reduce_scatter,nn/linear/base.py:206-214;DP_TP 整体由 `LinearForwardDPTP` 在 nn 层编排,base.py:24-76)。impl 只需实现契约的最后一公里。
- 精度语义:TM GEMM 输出已是 `output_spec` 指定 dtype(all_reduce 在其上做,与模板/默认实现一致);`tp_reduce_dtype` 的升精度归约发生在 nn 层(base.py:208-213),impl 无需感知。

备选:

- (a) 让 TM 输出直接写进 `reduce_scatter` 的 out 缓冲(`linear(x, weight, out=...)` 支持,linear.py:476-486)以省一次拷贝 —— 可行的后续优化,契约不变,stub 未做。
- (b) 使用 TM 引擎侧 NCCL(comm/nccl/,接缝盘点 §2)—— 否:那是 C++ runtime 的资产,python 通道无绑定,且双引擎通信域混用是另一张票的事。

**需要拍板**:无(通信归 python dist 是两边先例的一致结论;唯一自由度是 out 复用优化,属实现细节)。

______________________________________________________________________

## 问题 3:`forward_batched`/`supports_batched` 对 TM GEMM 是否可支持?

**推荐:`supports_batched = False`,显式抛错;BatchedLinear 走它既有的逐投影回退分支。dispatch_cache 命中率影响:可忽略。**

依据:

- `forward_batched` 的语义是「一份输入 × B 份独立权重」的 bmm(`BatchedLinear.forward`:`inputs.transpose(0,1)` 后 `impl.forward_batched(inputs, self._weight)`,nn/linear/default.py:44-53;`DefaultLinearImpl` 即 `torch.bmm`,backends/default/linear.py:13-19)。
- TM 通道两个入口都不匹配:`forward_dense` 是单 Weight 单 GEMM(linear.py:476-498);`forward_moe` 是 grouped 语义 —— **offsets 把 token 分给专家**,每个 token 恰好进一个专家(LlamaLinear.cu:61-64 offsets→blocked striding;linear.py:500-526),不是「每个 token 进所有 B 份权重」。硬用 grouped 表达 bmm 需要把输入复制 B 份 + offsets 全 0,内存与语义双输。
- 触发面分析:`BatchedLinear` 的启用门是「colwise + 非 all_reduce + 非 DP_TP + **无 bias** + **bf16 trivial 权重** + 全部投影 `supports_batched`」(nn/linear/default.py:37-45)。TM dense 16-bit family 的 priority 刻意低于原生、使 dense 权重路由 cuBLAS(cublas.cu:210-215 注释;family 全集通道盘点 §1.1)—— 即 TM 路径上不会出现 bf16 trivial 线性层,门条件天然不满足,`supports_batched=False` 没有实际损失。
- dispatch_cache 命中率:缓存按 `GemmDesc`(含 batch 维)lower-bound 复用(gemm.cu:194-217;dispatch_cache.cu:224-251、:319-327)。不支持 batched 不会稀释缓存 —— 反而避免引入新的 desc 形状族;若未来做 loop-of-dense 伪批量,B 次 (m,k,n) 相同查询全部命中同一记录,缓存也无压力。

备选:

- (a) `forward_batched` 实现为 B 次 `forward_dense` 循环 —— 可写但无收益:门条件已排除 TM 参与;拒绝实现并抛清晰错误(stub 取此)能防止未来误开。
- (b) 给 `linear_bind.cpp` 加 bmm/多 Weight 绑定 —— 超出本票;C++ 侧无现成「共享输入多权重」kernel 入口,属新家族/新 epilogue 级工作。

**需要拍板**:无强分歧。若未来引入「量化权重的 BatchedLinear」需求(目前门条件排除了),再评估 loop-of-dense。

______________________________________________________________________

## 问题 4:weight_loader 产物 → `DataFormat` 的桥接放哪?

**推荐:`process_weights_after_loading` 第一遍的 per-module `update_weights()` 钩子(票 03 结论;impl 的 `update_weights` 承接,nn 侧 `mod.update_weights()` 无参重入)。LinearImpl 构造期不做任何桥接。**

依据(全部承自票 03 / 挂载盘点,引用列全):

- 时序正确:钩子运行时权重已 per-rank 切片(chunk_aligned 切片在 weight_loader 内,awq.py:110-137)且已在 GPU(`build_patched_model(..., device=device)`,agent.py:1512);构造期参数是 `create_weights` 的空张量(awq.py:146-166),无从桥接。
- 位置安全:钩子在 KV cache pool 分配之前(`build_model` executor/base.py:575 → `build_cache_engine` :589)、cudagraph warmup 之前 —— 加载期 GPU 无并发工作,pack 的瞬时暂存(单层 203 MB,挂载盘点 §1)不挤占 cache 预算。
- 热更新自动复用:三条热更新路径结束都以 `process_weights_after_loading` 收尾(agent.py:1802-1814、1975-1981、2055-2056),对每个含 `update_weights` 的模块重跑钩子(model_weight_loader.py:196-198)→ 新数据重归一化重 pack,无需第二条入口。挂载盘点「挂载点建议 1」明确:**不要再造第二条入口**。
- TM 侧语义等价:引擎的 `ModelRoot::prepare()` 树遍历(turbomind.cc:93-99 → module.cc:58-63)由 `impl.prepare()`(经 `prepare_weight`,linear.py:240-273)+ 钩子时序等价承载(挂载盘点 §4 结论)。
- 构造期唯一该做的事是 plan 探询的**类型级**缓存(格式→family 映射与数据无关,`get_weight_plan` 查 registry,family.cc:63-90);stub 的 `TurbomindLinearImpl.__init__` 只存参,plan 在首次 `update_weights` 解析并缓存(tm_linear_impl.py),因为 `dtype=scales.dtype` 依赖加载后的真实 dtype。

备选:

- (a) LinearImpl 构造 —— 否:无数据、无设备上下文(见上)。
- (b) `process_weights_after_loading` 的第二遍模块级钩子(`process_weights_after_loading` 方法,MLA 吸收用,model_weight_loader.py:199-207)—— 否:那是「模型层吸收」语义(kv_b_proj→kc/vc,deepseek_v2.py:500),与 pack 无关;且第二遍在第一遍之后,若 pack 放第二遍会破坏 MLA 对第一遍产物的依赖顺序。

**需要拍板**:无(票 03 已结论,票面问题 4 本身标注「票 03 结论:钩子内」)。stub 遵循。

______________________________________________________________________

## 问题 5:torch 原始权重的 opt-in 释放做成什么形式?

**推荐:两级 opt-in(进程 env + BuildSpec 字段取或)× 两档语义(spill / immutable);释放动作在 `update_weights` 钩子内 pack 发布之后;热更新自愈用 `weight_loader` 包装回灌。实现见 `weight_release.py`,挂接点在 `TurbomindLinearImpl.update_weights` 返回值(tm_linear_impl.py)。**

依据:

- **必须 opt-in**:torch 参数是热更新 `target.load_weights` 的写入落点(agent.py:1786-1800),`default_weight_loader` 断言尺寸后 `copy_`(model_weight_loader.py:24-31)—— 无条件释放会让任何热更新在断言处崩溃(票 03 §3 的约束原文:挂载盘点「挂载点建议 2」)。
- **收益**:释放消除 packed/torch ≈×1.9 的结构性共存(挂载盘点 §1:W4A16 7B 峰值 7.30 GB vs TM 稳态 3.81 GB);forward 全走 TM handle,torch 参数已不被读(turbomind_awq_modules.py:59-73)。
- **opt-in 形态选项**(stub 取 1+2 叠加):
  1. env `TM_GEMM_RELEASE_TORCH_WEIGHTS`(进程级,先例 `TM_GEMM_WEIGHT_PACK` 的 `WeightPackEnv` 门控,family.cc:79-84)—— 零接口改动,实验友好;
  2. `LinearTmGemmBuildSpec.release_torch_weights`(模型级)—— 显式、可被推理服务层按部署形态决定(RL 热更新训练场景可关,纯推理可开);
  3. 纯显式 API(`engine.release_torch_weights()` post-load)—— 被否:失去钩子时序(pack 刚发布、KV pool 未分配,挂载盘点 §2 的安全区),且热更新路径无法自动维持。
- **两档语义**:
  - spill(默认):GPU 原始权重 → CPU 精确副本,storage 清零;显存省下、热更新可无损回灌。代价 = 等量主机 RAM(7B W4A16 ≈3.2 GB)。
  - immutable:不留副本;被释放模块自此拒绝热更新(`arm` 阶段装上抛错 loader)。适合确认无热更新的部署。
- **热更新自愈**:`arm()` 包装参数的 `weight_loader` —— 首次 loader 调用先整模块回灌(未触及参数从 CPU 副本恢复精确内容),再委托原 loader。选「CPU 溢出」而非「从 TM handle 反量化恢复」:u4 payload 在 family 布局(transpose/interleave)中,python 绑定无 unpack 入口(通道盘点 §4:4-bit 无 DLPack 表示,仅 int32 行打包入口);去量化→重量化不保证 bit 级一致,重 pack 出来的 packed 会与热更新前不同 —— 对 RL 权重同步语义不可接受。
- **保守门控**:trivial 16-bit 拒绝释放(pack 仅元数据 cublas.cu:199-206,TM 持等尺寸副本,零收益);gate/up 融合路径 stub 暂不处理。
- **对 nn 层的唯一侵入**:`register_all_parameters → setup_loaders` 每次重置 `param.weight_loader`(awq.py:60-67、:194),故 `arm()` 必须在 nn 侧 `register_all_parameters` 之后调用 —— 体现为 nn `update_weights` 尾部一行 `impl.post_register_release(self)`(weight_release.py docstring 有完整调用样例)。

备选:

- (a) 只留 scales/zeros(≈1% 字节)、仅释放 qweight —— 回灌时未触及模块仍缺 qweight 内容,问题不变,收益打折;不值得。
- (b) 释放即置 `param = None`(摘除参数)—— 破坏 `named_modules` 钩子遍历对参数的依赖(state_dict/热更改名映射),否。
- (c) 惰性 restore 按「单参数」粒度 —— 否:bucket 可能只触及一个模块的部分参数(如只 qweight),单参回灌会让其余参数保持空,随后 update_weights 重 pack 读到垃圾;必须整模块回灌(stub 的 arm 语义)。

**需要拍板**:

1. spill 模式的 CPU RAM 换显存是否可接受(还是只提供 immutable 档)?
2. `TM_GEMM_RELEASE_TORCH_WEIGHTS` 的档位编码(0/1/2)是否照此进 envs?
3. 热更新场景默认值:RL 训练侧是否统一 `=0`(不释放)由部署方显式开启?

______________________________________________________________________

## 需要用户拍板的分歧点清单(汇总)

| #   | 分歧点                                                                                | stub 立场                                                    | 备选                                                                |
| --- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- |
| 1   | fp8-input(w8a8)激活量化:kernel 内 `ConvertInput` vs python 预量化 + `input_scales`    | kernel 内(与模板同构,最简)                                   | python 预量化(省每步重复量化;需管理 scales 生命周期)                |
| 2   | 降级策略:`_build_linear_tm_gemm` 探询失败时                                           | 直接 RuntimeError(显式选择)                                  | 像 `_build_linear_w4a16` 那样静默回落格式对应 torch 后端(auto 语义) |
| 3   | 释放档位:是否保留 spill(CPU RAM 换热更新安全)档;env 编码 0/1/2                        | 保留两档,env 叠加 BuildSpec 字段                             | 仅 immutable;或仅 API 显式                                          |
| 4   | 逐模块 sync(热更新必需)在初始加载期是否去掉                                           | stub 保留 sync,优化记为后续                                  | 初始加载免 sync(挂载盘点建议 3,需 impl 感知「加载 vs 热更」上下文)  |
| 5   | gate/up 融合(`fuse_weight` 路径)是否进第一梯队                                        | 预留参数 + 抛 NotImplementedError                            | W4A16 merged 层直接融合可再省 gate/up 两份 handle 的元数据开销      |
| 6   | 通用 spec 的 nn 侧模块(`TmGemmLinear` 形态)与既有 `AwqLinear`/`BaseLinear` 的关系     | 未在 stub 范围(票面只要求 OpsBackend 分支 + LinearImpl 变体) | 新 nn 模块 vs 扩展 BaseLinear.update_weights 签名                   |
| 7   | `get_unquantized_weight`(MLA 吸收路径):torch 侧原始张量(与释放冲突)vs TM 侧去量化绑定 | 抛 NotImplementedError,标注两难                              | 释放机制上豁免被 MLA 吸收的模块;或补 TM→torch 去量化导出绑定        |

## 已知边界(不要求本票解决)

- cudagraph 兼容:`Weight.is_graph_compatible`(linear.py:103-106)已暴露;TM family 的 `is_graph_compatible` 位(family.h:118-129)与 graph_runner 的协同未验证 —— 记为接跑通票的验证项。
- 独立 `get_linear()` 与引擎 `TurboMind.gemm(index)` 的 dispatch cache 不共享(通道盘点 §3.4):PyTorch 引擎场景只用前者,预热记录 export/import 需自管。
- `set_measure` 未绑定(通道盘点差距表 #6):python 无法显式切 kAppend/kReuse,只能 `tune`+`import_records` 间接达成 —— 若要控制首包时延,顺手补绑。
