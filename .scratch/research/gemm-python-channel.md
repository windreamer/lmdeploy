# GEMM 的 python 调用通道盘点

出处:tm-gemm-graft/issues/01(wayfinder:research)。日期:2026-11-16。只读盘点,引用为 `文件:行`,全部取自源码。

一句话结论:票面假设的"GEMM 暴露面未盘点"已过时——family 选择、权重打包、exec-plan 查询、launch、dispatch-cache 持久化在 `src/turbomind/python/linear_bind.cpp` + `lmdeploy/turbomind/linear.py` 已端到端存在,且有 TorchOpsBackend 侧 LinearImpl 接入先例(AWQ W4A16);真正缺口见文末差距表,均为零星项。

## 0. 票面背景修正

- `bind.cpp` 除 DLPack Tensor(:635-697)、Module(:838)、TurboMind(:906)外,在模块尾部挂载 `bind_delta_rule`(bind.cpp:970)、`bind_linear`(bind.cpp:971)、`bind_moe_gate_v2`(bind.cpp:972);GEMM 通道即 `bind_linear`,实现在 `src/turbomind/python/linear_bind.cpp`(:52-494)。
- python 封装 `lmdeploy/turbomind/linear.py`(:1-584)把绑定包装成 torch 接口:`Linear`/`Weight`/`WeightPlan`/`ExecPlan`/`get_linear`(linear.py:14)。
- Torch 后端已有消费者:`TurbomindAwqLinearW4A16Impl`(backends/cuda/turbomind_awq_modules.py:13),经 backends/cuda/awq_modules.py:113-114 挂入 W4A16 后端选择。

## 1. C++ 调用入口链

### 1.1 Family / registry / cublas 兜底

- **Family**:`src/turbomind/kernels/gemm/family.h:51-137`。字段与函数指针:id、priority、input_format、output_format、align_k/n、min_k/n、requires_packing、grouped、supports、pack、gate_up_block、fused_output、is_graph_compatible、output_spec(ctor family.h:118-129)。
- **supports 判定**(family.cc:63-90):grouped 门控(:74-76);`WeightPackEnv` 环境门控——`TM_GEMM_WEIGHT_PACK` 0/1 决定只接受"要求打包/不要求打包"的 family(family.cc:79-84;convert_v3.cu:16-31);非量化 input dtype 匹配与 output dtype 匹配(family.cc:85-89)。
- **ConvertInput**(family.h:105-110;family.cc:102-135):非量化时直通并 CHECK 输入 dtype(:121-127);输入已是 family dtype(fp8)时要求自带 input_scales 作 U(:128-132);否则 `QuantizeSymm` 现场量化(:129-133)。
- **registry**:各 kernel .cu 以静态 `Registrar` 写入 `gKernelFactories()`(registrar.h:43-53);`Registry` ctor 逐 factory 跑 Collector 并 Add(registry.cu:14-20);Add 按 arch 与 `sharedMemPerBlockOptin` 过滤(registry.cu:23-41),每个 kernel 再注册一个 TransposedKernel 镜像(registry.cu:47-49;kernel.cu 的 `transpose()` 定义)。
- **family 全集**(id,priority 举例):dense_f16/dense_bf16=100/101、priority 260,注释明示高于原生 dense 家族(190/200/250),使 dense 权重在选 plan 时路由 cuBLAS(cublas.cu:210-215);f16/bf16 mma 16816(190/200,sm80_16816_16.cu:25-26);u4 g32/g128 dedup/group(139-190,sm90_16816_4.cu:43-46);e4m3 w8a8(sm90_16816_8.cu:25);sm90 GMMA bf16(250,sm90_64n16_16.cu:48)、u4 mixed id29/35(250,sm90_64n16_4.cu:69-86)、fp8 id32(250,sm90_64n16_e4m3.cu:82)、mxfp4 id30(301,sm90_64n16_mxfp4.cu:40)、nvfp4 id31(250,sm90_64n16_nvfp4.cu:48);fp8 w8a8 id28(300,输入 `DataFormat{kFloat8_e4m3,{128,1},kFloat}`、输出 bf16、`fp8_output_spec`,sm90_64n32_8.cu:66-79);mxfp4_fp8 folded/unfolded(sm90_64n32_mxfp4_fp8_folded.cu:76、…unfolded.cu:35)。
- **cublas 兜底**:CublasKernel 以 dense_f16/dense_bf16/f16_f32/bf16_f32 四 family 注册(cublas.cu:216-222);`is_feasible` 限 flat striding、零 pack、无 epilogue(:139)、num≤1(:142)、无量化(:145)、order_c=kColMajor(:151)、fp16/bf16/f32(:154);Launch 即 `cublasGemmEx`(:118-135);dense pack 仅设 `k_desc={Dtype,kRowMajor,K,N,ld=N}`(:196-208)。bf16 GMMA family 的 collector 里也 add_cublas 作后备 kernel(sm90_64n16_16.cu:90)。

### 1.2 prepare() 后谁派发

- **选 plan**:`Gemm::GetWeightPlan(WeightQuery)`(gemm.h:51;gemm.cu:338-378)遍历 registry families 调 `supports`(:343);`input_dtype` 仅偏好(:346;weight_plan.h:14),`output_dtype` 硬要求(weight_plan.h:15);取 priority 最高,平级判 ambiguous 返回 nullopt(gemm.cu:347-360)。WeightQuery 字段见 weight_plan.h:12-19。
- **打包**:`LinearWeight::prepare()` 调 `plan_->pack(*this, stream)` 并 `EnsureFloatDtype(bias)`(linear_weight.cc:58-65);`WeightPlan::pack` → `family_->Pack` 并回填 family/input_format/epilogue/output_format(weight_plan.cc:23-31)。plan 由调用方经 `set_plan` 注入(linear_weight.cc:41-44)。
- **派发**:`LlamaLinear::Forward(input,input_scales,weight,indices,offsets,out,out_scales)`(LlamaLinear.h:63-71;LlamaLinear.cu:258-277):
  1. `GetOutputSpec` → `family->output_spec`(LlamaLinear.cu:84-99);
  2. `GetArguments`(:101-182):A/U 经 `family->ConvertInput`(:54);`desc_A={dtype,kRowMajor,m,k,A.stride(0)}`(:56);U 的 desc_U 为 ColMajor(:58-60);offsets → `num=k_desc.num` + offsets 指针(blocked striding,:61-64);indices → `idxs`(indexed,:65-67);B/V 取 `weight.weight/scales` + `k_desc/q_desc`(:29-36,:118);global_scale 独立槽(:121-126);输出 D/W 缺省按 output_spec 分配(:128-150);stream 取 `Context::stream()`(:180);
  3. policy 为 kMeasure 时 `Tune` 否则 `GetExecPlan`(:269);
  4. `gemm_.Run(plan,args)`(:271-276)。
- **Gemm::GetExecPlan**(gemm.cu:396-411):`Context::Init` 由 Arguments 构 `GemmDesc`(desc.h:25-52)→ `Impl::Dispatch`(gemm.cu:194-217):kReuse 先 `cache_.LowerBound`(batch 维 lower-bound 语义,dispatch_cache.cu:224-251、:319-327),再 `cache_.Find`(exact,dispatch_cache.cu:317-323);miss 走启发式 `Find`(cluster+populate+mio/mma 估计,gemm.cu:208-215、:228-303)并 `Insert`。kernel 级可行性过滤 `Kernel::is_feasible`(kernel.cu:96-193;striding accept 矩阵 :59-93)。
- **Run**:`Impl::Launch` → `kernel->Launch(operation,alpha,A/Adesc,U/Udesc,B/Bdesc,V/Vdesc,global_scale,beta,C/D/Ddesc,W/Wdesc,swizzle,splits,workspace,stream)`(gemm.cu:431-439、:166-191)。Workspace 为 per-GEMM scratch(barriers 1MiB/partials 32MiB/tensormaps,types.h:243-263;构造与释放 gemm.cu:74-130)。
- **参数传递汇总**:shape/stride 全在 `MatrixLayout{type,order,rows,cols,ld,pack,num,offsets,idxs}`(types.h:208-216;`get_mode`:idxs→indexed、ld==0/offsets→blocked,types.h:227-235);输入 scales 走 U 槽,权重 scales/zeros 走 V 槽(q_desc),输出 scales 走 W 槽,global_scale 独立(LlamaLinear.cu:118-150)。
- **C++ 运行时持有者**:`Context` 独占一个 `LlamaLinear`(models/llama/context.h:39,52),各层持引用执行:Ffn(models/llama/LlamaFfnLayer.h:43)、attention(unified_attention_layer.h:87)、MoE(moe_ffn_layer.cc:43)、GDN(GatedDeltaNetLayer.h:50)、language_model(language_model.cc:41)、model_executor(engine/model_executor.cc:28)。
- **dispatch cache 持久化/预热**:`DispatchCache{LowerBound,Find,Insert,Export,Import}`(dispatch_cache.h:10-21);文件格式 magic "TMGEMM2"/version 6(dispatch_cache.cu:56-60),Record={GemmDesc,KernelDesc,swizzle,splits}(:63-68),Import 按 KernelDesc 匹配已注册 kernel(:118-166);链路 `Gemm::Export/Import`(gemm.cu:441-448)→ `LlamaLinear::Export/Import`(Import 后 `dispatch_policy_=kReuse`,LlamaLinear.cu:381-399)。

## 2. bind.cpp / linear_bind.cpp 已暴露的 GEMM 相关能力

`_turbomind` 模块内(bind.cpp:397 起)与 GEMM 通道直接相关的已暴露面:

**bind.cpp**:

- Tensor + DLPack:`Tensor.__dlpack__/__dlpack_device__/copy_from(含 stream 重载)/t/reinterpret`(bind.cpp:635-697)、`from_dlpack`(:698)、`generic_copy_on_stream`(:699-708)。
- `DataFormat`/`QuantParamDesc` 绑定(dtype、block_sizes、scales、zeros、is_quantized、rank,bind.cpp:593-612);`DataType` 枚举含 `TYPE_FP8_E4M3/TYPE_FP4_E2M1/TYPE_UINT4`(:615-618)。
- `LinearConfig`(bind_config,bind.cpp:629)、`Param.alloc/get/set`(:824-836)、`Module`(get/param/prepare/create_child/add_child_raw/set_meta_pack,:838-876)、`create_module`(:878-892)。
- `ContextGuard` + `create_device_context(stream_ptr)`——把 torch 的 CUDA stream 借给 TM context(bind.cpp:771-822)。
- `TurboMind.gemm(index)` 返回引擎内 `gemm::Gemm&`(bind.cpp:948;turbomind.h:39;turbomind.cc:107)。

**linear_bind.cpp(bind_linear)**:

- `MatrixLayout`(type/rows/cols/ld/num/offsets 指针,:60-77)、`Epilogue`/`ActivationType` 枚举(:73-82)。
- `Family`(input_format/align_k/align_n/min_k/min_n/grouped/data_type/supported_epilogues/output_format,:83-92)。
- `WeightQuery`(:94-100)、`WeightPlan`(family/shape_constraints/bridge/gate_up/pack,:102-126)、`_GemmOutputSpec`(:127-135)、`GemmExecPlan`(output shape/stride/dtype/scales,:137-148)。
- `Gemm`(独立实例化 + get_weight_plan/data_types,:150-153);`LinearWeight`(init/prepare/set_plan/is_graph_compatible + 六个可读写属性含 k_desc/q_desc,:155-168);`GemmWorkspace`(按 stream 指针构造,:170-172)。
- `LlamaLinear`(:174-304):`get_weight_plan`(:177)、`get_exec_plan`(:181)、`_get_output_spec`(:198)、`forward_dense`(:209)、`forward_moe`(:233)、`tune`(:263)、`import_records/export_records`(:291-304)。
- 量化助手模块函数:`QuantizeSymm/DequantizeSymm/QuantizeSymmBlock/DequantizeSymmBlock/QuantizeGroupwise`(:306-406);`LinkLinearExperts`(MoE 专家链接成 grouped 权重,:408-416);`invokeMoeDispatch/invokeMoeCombine`(:419-494)。

**python 封装(已可用)**:`get_weight_plan`(linear.py:184-228)、`prepare_weight/fuse_weight`(含 gate/up 交错,linear.py:303-345、:353-429)、`get_exec_plan`(:431-450)、`__call__`/`forward_dense`(:476-498)、`forward_moe`(:500-526)、`tune`(:528-555)、`import_records/export_records`(:565-569)、`get_linear()` 按设备单例(:577-584)、per-stream `GemmWorkspace` 缓存(:172-180)、`_activate()` 经 `create_device_context` 对齐 torch stream(:150-160)。

## 3. python 侧直接驱动:还差什么绑定

**票面四项逐一对照(均已存在)**:

- family 选择 → `_tm.LlamaLinear.get_weight_plan(WeightQuery)`(linear_bind.cpp:177-180)→ `Linear.get_weight_plan`(linear.py:184-228);
- plan 查询 → `get_exec_plan`/`tune`(linear_bind.cpp:181-197、:263-290;linear.py:431-450、:528-555);
- launch 入口 → `forward_dense`/`forward_moe`/`__call__`(linear_bind.cpp:209-262;linear.py:476-526);
- dispatch_cache 预热 → `tune`(测量并 Insert,gemm.cu:305-345、:413-429)+ `import_records/export_records`(linear_bind.cpp:291-304;C++ 侧 LlamaLinear.cu:381-399)。

**linear_attn/python_bind.cpp 模式可否复制**:模式本身(模块级 `plan→prepare→run` 三函数 + py::capsule 持 `Plan*`,python_bind.cpp:398-436、:459-476;bind.cpp:970 挂载)可复制,但对 GEMM 是冗余——linear_bind.cpp 的类绑定已覆盖同构需求且更强(plan 是持久对象而非 dict+capsule)。若需扩展,应改 linear_bind.cpp 而非另起新文件。

**真实缺口(绑定层面)**:

1. `LlamaLinear::set_measure(bool)` 未绑定(LlamaLinear.h:96;LlamaLinear.cu:376-378)——python 无法显式切换 kAppend(记录)/kReuse 策略,现只能靠 `tune`+`export_records`/`import_records` 间接达成。
2. `LlamaLinear::GetTuningSeq`/`Gemm::GetTuningSeq` 未绑定(LlamaLinear.h:105;gemm.h:56)。
3. `Family::ConvertInput` 无独立入口(仅嵌在 forward 内,family.h:105-110)——无法从 python 单独做"fp16→fp8 + scales"预变换。
4. 两套 `Gemm` 实例的 dispatch cache 不共享:引擎走 `TurboMind.gemm(index)`(bind.cpp:948),python 独立驱动走自建 `LlamaLinear`(linear.py:140-146)持有自己的 `gemm_`(LlamaLinear.cu:207)——独立通道的预热记录必须自行 import/export,不会惠及引擎实例(反之亦然)。

## 4. DLPack 边界的 dtype/stride 约定

- **dtype 映射**:python 封装只认 6 种 torch dtype:`uint8/int32/float16/bfloat16/float32/float8_e4m3fn`(linear.py:22-29)。DLPack 出口:fp8_e4m3 导出为不透明 `kDLUInt/8bit`(bind.cpp:130-134);入口接受 `kDLFloat8_e4m3fn`(bind.cpp:287-292),`lanes!=1` 拒绝(:211-213)。4-bit(`TYPE_UINT4/FP4_E2M1`,bind.cpp:616-618)无 DLPack 表示:python 侧以 int32 行打包携带(`pack_u4_row`,weight_format.py:72-84),借 `PackedTensor.alloc_shape/alloc_dtype`(weight_format.py:56-60)声明逻辑形状,经 `copy_from` 拷入——拷贝按 byte_size 相等校验(bind.cpp:645-650)。
- **stride**:`Tensor.__dlpack__` 向 DLPack 传 `strides=nullptr`(bind.cpp:172-177)→ 消费端按 C 连续行主解释,传出的 TM tensor 必须实际连续;`from_dlpack` 接受显式元素步长,缺省按行主补全(bind.cpp:257-276),`byte_offset` 并入数据指针(:278-279),非正维报错(:248-256)。
- **activation(A/U)**:逻辑 `[m,k]` RowMajor;flat 查询用 `ld=k`(LlamaLinear.cu:319),真实 launch 用 `A.stride(0)` 允许行主 padded view(先 `view({-1,k})`,:95、:56)。fp16/bf16 输入在 forward 内被 `ConvertInput`→`QuantizeSymm` 动态量化成 fp8+group scales(family.cc:129-133);已量化输入需自带 `input_scales` 直接作 U(LlamaLinear.h:36-40)。
- **packed 权重(B/V)**:B 槽 = `weight.weight` + `k_desc`,V 槽 = `weight.scales` + `q_desc`(LlamaLinear.cu:29-36、:118)。三种打包后布局:dense `[K,N]` RowMajor ld=N(cublas.cu:196-208);fp8 w8a8 转置为 `[N,K]` ColMajor、scales `[N,G]` ColMajor(sm90_64n32_8.cu:30-63);sm90 u4 mixed 权重交错 + scales/zeros 压成 1D uint8 qparams,`q_desc={kUint8,kColMajor,N,G,ld,pack=kSm90MixedQParamPack}` 再 transpose,同时改写 `weight_format={kUint4,{G,1},Dtype,kUint4}`(sm90_64n16_4.cu:23-53)。
- **scales/zeros 输入约定**:须先按 `WeightBridge.replicate_scales{replicate_k,replicate_n}` 扩到"每 K 组一行 × 每列"(family.h:40-47;python `_apply_bridge` repeat_interleave 并裁到 output_dim,linear.py:48-68);pack 只读前导 `group_count × output_dim`,允许更多行/更宽 stride(sm90_64n16_4.cu:26-33)。
- **输出(D/W)**:布局由 `family->output_spec` 决定(family.h:95-98;OutputSpec{layout,dtype,scales_layout,scales_dtype},output_spec.h:7-12),经 `GemmExecPlan` 暴露 shape/stride/dtype/scales(linear_bind.cpp:137-148);python 用 `torch.empty_strided` 复刻(linear.py:93-107、:452-464)。gated-SiLU epilogue 把末维减半(family.h:18-22);fp8 输出要求配套输出 scales(group-128,写 W 槽,LlamaLinear.h:41-44;LlamaLinear.cu:139-150)。另有 `MakePaddedOutput` 约定输出行 stride pad 到 16B 供 conv1d 等 16B 向量读(linear_weight.h:101-108)。

## 5. 需新增暴露面差距表

| #   | 能力                  | 现状                                                                                                                        | 缺口 / 建议                                                                              |
| --- | --------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 1   | family 选择           | 已有:`Linear.get_weight_plan`(linear.py:184-228 ← linear_bind.cpp:177)                                                      | 无                                                                                       |
| 2   | 权重打包              | 已有:`prepare_weight`/`fuse_weight`(linear.py:303-429 ← `set_plan`+`prepare`,linear_bind.cpp:156-168)                       | 无                                                                                       |
| 3   | plan 查询             | 已有:`get_exec_plan`/`_get_output_spec`/`tune`(linear_bind.cpp:181-290)                                                     | 无                                                                                       |
| 4   | launch 入口           | 已有:`forward_dense`/`forward_moe`/`__call__`(linear_bind.cpp:209-262)                                                      | 无                                                                                       |
| 5   | dispatch cache 预热   | 已有:`tune` 隐式 Insert + `import/export_records`(linear_bind.cpp:263-304)                                                  | 无;但引擎实例与独立实例缓存不共享(§3.4)                                                  |
| 6   | record/reuse 策略切换 | C++ 有 `set_measure`(LlamaLinear.cu:376-378)                                                                                | 未绑定;建议在 linear_bind.cpp 补 `set_measure`                                           |
| 7   | tuning 序列查询       | C++ 有 `GetTuningSeq`(LlamaLinear.h:105)                                                                                    | 未绑定;低优先级                                                                          |
| 8   | 独立激活量化入口      | 模块级 `QuantizeSymm/QuantizeGroupwise` 已暴露(linear_bind.cpp:306-406)                                                     | `Family::ConvertInput` 未单独绑定(一般不需要)                                            |
| 9   | bias 融合             | `LinearWeight.bias` 参数槽存在(linear_weight.h:89-95)但 forward 无 bias 形参,cublas kernel 拒绝一切 epilogue(cublas.cu:139) | python 逐层 `out + bias`(turbomind_awq_modules.py:66-67);C++ epilogue-bias 不存在        |
| 10  | TP all_reduce/scatter | `LinearImpl.forward` 契约含 all_reduce/group/rank/scatter_size(backends/linear.py:11-39)                                    | TM 通道不管通信;AWQ 先例由 python `dist.all_reduce` 承担(turbomind_awq_modules.py:69-70) |

## 对后续决策的影响

- TorchOpsBackend 的 LinearImpl 直驱 GEMM 不需要新绑定文件:`get_linear()`(linear.py:577)已提供全流程,AWQ W4A16(turbomind_awq_modules.py)可作为复制模板向其它量化格式推广。
- 若要接入 TM 引擎已加载的权重,应走 `TurboMind.gemm(index)` 而非独立 `LlamaLinear`,否则 dispatch cache 与引擎不共享。
- 接入更多量化格式的主要工作量在 python 侧 `WeightFormat` 子类(weight_format.py)与 pack 布局核对,而非 C++ 绑定;仅 `set_measure` 值得顺手补绑。
