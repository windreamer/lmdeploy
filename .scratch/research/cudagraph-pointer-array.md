# CUDA Graph 下 per-step 指针数组的稳定性

来源：票 06（tm-gemm-graft，`.scratch/wayfinder/tm-gemm-graft/issues/06-cudagraph-pointer-array.md`），2026-11-16，只读研究。此文件是研究记录，非设计文档。引用为 `文件:行`。
一句话结论：**可行**——三种 graph 路径均无根本障碍；eager/PCG 下 attention 元数据本来就是逐 step 活对象，full graph 下指针数组须走既有 `CudaAttentionMetaBuilder` 稳定缓冲契约做 device 侧原地改写，pinned host 分级缓冲为可选方案。

## 0. 事实修正：路径与"三种模式"的实指

- 票面 `backends/graph_runner/` 实为两处：基类 `backends/graph_runner.py:58`（`GraphRunner`，`__call__` 直通 model，graph_runner.py:61-63）；CUDA 实现 `backends/cuda/graph_runner/`，由 `CUDAGraphRunner`（runner.py:118）三路分发。
- 实际执行路径是 **full / piecewise / eager** 三条（`backends/cuda/graph_runner/README.md:5-11`）。`standard.py` 不是第三种执行模式，而是 piecewise 的"标准 decoder prefill 策略"（token bucket、描述符、frame inputs，standard.py:31-41；README 文件表）。分发逻辑：decode 且 model 支持 → full（runner.py:192-194,263-284）；支持的 prefill 且有 plan → piecewise replay/prepare（runner.py:270-277）；否则 eager（runner.py:279-284）。

## 1. capture 边界：哪些冻结指针、哪些改写内容

### full graph（仅 decode）

- **捕获期冻结（runner 生命周期内地址不变）**：`make_buffers_cudagraph` 一次性分配的全部 `input_buffers`——`input_ids`/`position_ids`/`block_offsets`/`qkv_lens`(q_start,q,kv)/`cu_seqlens`/`local_adapter_ids`/`dcp_local_kv_seqlens`（models/utils/cudagraph.py:101-152）、`state_ids`（cudagraph.py:294-299）、`mrope_position_ids`（cudagraph.py:186-188）；step-meta 派生缓冲（FA3 `scheduler_metadata`、FlashMLA `tile_scheduler_metadata`/`num_splits`、DSA indexer `expanded_block_offsets`/`schedule`，cudagraph.py:166-190 + backends/cuda/step_metadata.py:273-306）；`output_buffers`（full_graph.py:155-168）；`past_key_values` 缓存池张量按引用穿过（cudagraph.py:318-320）——池为 engine 常驻单一张量（engine/cache_engine/layout.py:45-58）。
- **replay 期改写内容**：每步 `forward()` 先 `_bind_inputs`：device 端 Triton 融合核 `fill_graph_common_inputs` 把活输入原地写入上述缓冲（full_graph.py:188-202；models/utils/cudagraph.py:203-241；kernels/cuda/step_metadata/fill_graph_common_inputs.py:8,155——非 H2D，纯 device kernel），随后 `graph.replay()`（full_graph.py:176-182）。即**指针全部捕获期冻结、内容每步改写**——正是票面所述标准做法，且现有代码全部用 device 端填充实现。
- **attention metadata 走哪条**：python dataclass 对象每步新建（host 侧，graph 不感知），但其全部 tensor 字段在每步填充后被**重绑**到稳定缓冲（cudagraph.py:243-250,338-360）；派生 kernel 元数据经 `CudaStepMetaPlan.fill_cudagraph_buffers` 写 runner-owned 缓冲并回挂（step_metadata.py:282-294）。
- **时序保证**：replay 是每步 host 驱动（full_graph.py:176-182），填充与任何 pre-replay H2D 在同一 forward stream 上按流序先于 graph kernel（README "Capture and replay both run on the engine's main forward stream"）。

### piecewise（PCG，prefill）

- graph piece 只包住捕获安全的张量代数；**attention 是 `@eager_boundary`**（backends/cuda/attention/default.py:175-218），`attn_metadata` 经 `_FrameValueRef('attn_metadata')` 逐 replay 从活请求 frame 换入（piecewise.py:299-304,647-661；standard.py:36-41,186-196）。**attn_metadata 完全不被捕获**——每步活指针，无冻结。
- 进 graph 的输入只有 `input_ids`/`position_ids`/`mrope_position_ids`（standard.py:33-37,180-196）。

### eager

- 无任何冻结（runner.py:228-231，直接 model forward）。

## 2. `block_offsets` → 指针数组的转换层级；常驻缓冲可否原地改写

- **现状：pytorch 侧不存在指针数组层。** `block_offsets`（int32 device `[bsz,num_blocks]`，backends/attention.py:16）直接作 block table 消费：Triton `flash_attn_with_kvcache`、FA3 `page_table`、FlashMLA `block_table` 均在 kernel 内做 `cache_base + id×stride` 推导；dispatch 前仅统一 int32（runner.py:184-190）。唯一的表变换是 `_map_to_kernel_block_offsets`（管理块→kernel 块展开，engine/inputs_maker.py:953-970）。
- TM kernel 期望 `BlockIteratorParams{char** block_ptrs; const int* cu_block_nums}`（src/turbomind/kernels/attention/attention_params.h:17-21），设备端 `block_ptrs_[block_id_]` 解引用、按 `cu_block_nums_[batch_idx]` 偏移取 per-seq 段（block_iterator.h:20,31,39,75）。
- **TM 侧先例即票面方案**：host `kCPUpinned` 分级缓冲（unified_attention_layer.cc:175-176，容量 `max_batch×max_seq_blocks`）+ 设备常驻 `d.block_ptrs{max_blocks+16}`（:195-196）；每步 `Setup()` host 填 `CacheBlock::base(0)+offset` 再 `BatchCopy` 异步 H2D（:289-304）。
- **pytorch 侧推荐转换层级**：grafted impl 注册 `CudaAttentionMetaBuilder`（接缝已存在，step_metadata.py:72-108）——`make_cudagraph_buffer` 分配 runner 常驻 `uint64[N]` 指针数组 + `int[max_batches+1]` cu；`fill_cudagraph_buffer` 每步从已稳定的 `input_buffers['block_offsets']`（cudagraph.py:119,243）做 **device gather**：`ptr[i] = pool_base + block_id×block_bytes`。纯 device、无 host 参与，与 `fill_graph_common_inputs` 同构。
- **pinned+device 常驻每步原地改写：成立**，两个前提均满足：
  1. **指针值稳定**：块池是单一张量且 engine 常驻（layout.py:45-58，packed/row/contiguous 布局的池基址均不迁移），`ptr = base + id×block_bytes` 是 id 的纯函数；驱逐复用不搬池。
  2. **改写时序**：full graph 每步 replay 前在 `_bind_inputs` 位点（full_graph.py:188-202）改写即可（device gather 或同流 H2D 均按流序生效）。若选 host pinned 写入，还可把 H2D memcpy 节点捕进 graph——pinned src 地址捕获后冻结，而 pinned 缓冲本身常驻、host 在 replay 间改写内容，恰好自洽。host 侧块 id 来源存在（`indice_cpu` numpy，model_inputs.py:210-217；`_tensorlize_block_offsets`，inputs_maker.py:53-60），但会给 forward 路径引入 host 依赖；**device gather 无此代价，为首选**。

## 3. V4 step context 有无等价稳定缓冲先例

- `V4AttentionMetadata.from_step_context`（backends/attention.py:55-81）与 CUDA 子类 `CudaV4AttentionMetadata.from_step_context`（backends/cuda/attention/v4.py:125-135）**每步构造全新 python 对象与新 tensor**（decode 窗口 indices、ratio 压缩索引、index-score meta、prefill shared meta，v4.py:137-289），在模型 forward 内逐 step 调用（models/deepseek_v4.py:783-796）。V4 层面**没有** runner-owned 稳定缓冲；full graph 下这些新 tensor 捕获期落入 graph 私有池、replay 期以相同 kernel 序列重算到同地址——只要派生计算只依赖稳定输入缓冲即正确，但属"graph 池内重算"，非"常驻缓冲改写"。
- **等价先例存在于 step-metadata 层**：`CudaAttentionMetaBuilder` 三件套（eager `build()` 活对象 ↔ graph `make/fill_cudagraph_buffer` 稳定缓冲，step_metadata.py:72-108）。实例：
  - FA3AttentionMetaBuilder：`scheduler_metadata` 稳定缓冲每步 `copy_`（fa3.py:203-242）；
  - FlashMLAAttentionMetaBuilder：`tile_scheduler_metadata`/`num_splits` 每步 `copy_`，另有 `prepare_cudagraph_capture` 在捕获前更换 scheduler 对象（mla.py:149-206）；
  - DSAIndexerMetaBuilder：`expanded_block_offsets`/`schedule`/`indexer_kv_seqlens` 常驻缓冲 + 每步 device 填充（nsa.py:220-370）——与"指针数组常驻 + 原地改写"**完全同构**。
- 结论：机制先例齐备；TM 指针数组若走 builder 缝，即第三个 DSA 式 builder。

## 4. 三种模式对该方案的差异影响

- **full graph（decode）**：约束最强。attention kernel 被捕获 ⇒ 指针数组必须 runner 常驻 + 每步 replay 前原地改写；新增输入必须并入 `make/fill_buffers_cudagraph` 或 builder；生命周期受 reset 顺序约束——`CUDAGraphRunner.reset()` 先清全部 graph/plan 再毁 DeepEP buffer（runner.py:298-303；README "Address-changing lifecycle operations must pass through this complete reset"）。per-key runner（`(batch_size,is_decoding,enable_microbatch,query_len,extra)`，runner.py:167-180,131）各持独立缓冲集。
- **piecewise**：attention 处于 eager boundary，指针数组逐 rebuild 即可，无冻结问题；但 graft impl 必须满足 PCG 契约——`supports_piecewise_cuda_graph` + `eager_boundary` 包装 + token-axis padded 输出适配（default.py:175-222；plan 门控需全部被选 impl 支持，step_metadata.py:148-166 + runner.py:63-115）。descriptor 拒绝动态 extra forward 参数（standard.py:113-121）。
- **standard**：非独立模式，是 PCG 的标准 decoder prefill 策略（512 token 步进 bucket、frame inputs 命名表、`_StandardGraphDescriptor`，standard.py:31-41,80-121）。真正的第三路径是 eager 兜底（runner.py:279-284），无任何约束。
- 对方案总体：eager/PCG 天然兼容；**full graph 是唯一需要按缓冲契约施工的路径**，且机制与 FlashMLA/FA3/DSA builder 同构，不引入新 graph 机制。

## 可行性结论

**可行**（full graph 分支"有条件"，条件均可由既有机制满足）。pinned+device 常驻 + 每步原地改写成立；推荐 device 侧 gather 改写（池基址 × `block_offsets`），host pinned 分级为备选。

限制清单：

1. 池基址/权重/缓冲的任何地址变更必须先过 `CUDAGraphRunner.reset()`（runner.py:298-303）；指针数组缓冲须由 builder/runner 持有，graph 存活期不得释放。
2. full graph 缓冲形状捕获期定死：`num_blocks`/`max_batches` 来自 `cache_config`（full_graph.py:142；cudagraph.py:119）；cache 扩容令 graph 失效，块 id 必须 \< `num_blocks`。
3. per-key runner 各持独立缓冲（runner.py:131,167-180），指针数组按 key 重复分配（量级为指针表，可接受）。
4. TM kernel 期望 per-seq 拼接 + `cu_block_nums`（block_iterator.h:75）；fill kernel 须生成拼接布局，padded 尾项须指向合法哑块或以 cu 截断。
5. host pinned 路径的每步写入须在 replay 前同流 H2D（或捕获 memcpy 节点）；现行 per-step H2D 均在 graph 外、源为非 pinned numpy（model_inputs.py:228-246；agent.py:1345-1362，event 保活 agent.py:1330-1341），引入 pinned 分级需自管生命周期。
6. 指针值稳定依赖"池 storage 永不搬移"（layout.py:45-58），该前提须在 cache 引擎演进中被保持（对应 TM slab 语义）。
7. PCG 下 graft impl 必须实现 `eager_boundary` + 输出适配契约；任一被选 impl 不支持则整 plan 回退 eager（step_metadata.py:148-166）。
8. TM 附加调度标量 `readonly_block_num`/`cu_q_len`/`cu_k_len`（attention_params.h:29-32）需同一 builder 一并稳定化；`cu_seqlens_q/k`、`qkv_lens` 稳定缓冲已存在可复用（cudagraph.py:146-152）。

对后续决策的影响：

- attention 第二批次可按 `CudaAttentionMetaBuilder` 缝设计 TM `BlockIteratorParams` 稳定缓冲，无需新 graph 机制；
- 淘汰叙事的技术前提成立：TM 设备端指针数组方案在 PyTorch graph 全部三路径下可落地；
- 需与 cache 引擎侧约定两条不变量：池 storage 永不搬移、地址变更必经 graph reset。
