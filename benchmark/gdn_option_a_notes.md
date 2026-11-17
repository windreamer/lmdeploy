# GDN 方案 A 调研笔记:状态常驻 TM 复合格式(cache_engine 域)

日期:2026-11-17 | 性质:只读调研,未改任何代码 | 与在途 kernel 侧 benchmark(`GdnZeroCopyBench`,测转置变体)互补,两份结论合议构成 GDN 最终决策依据。
断言均带 `文件:行`;`[INFERENCE]` 标注推断项。

## 0. 结论速览

1. **PT 池现状约定 = V-first**(`[slots, L, H, V, K]`),TM 复合格式 = K-first per-seq slab `[H, K, V]` + layer_group 分组——两者逐元素转置关系,不是"重排"而是"换轴"。
2. **方案 A 有一处硬阻断**:GLM-5.3 的线性层是 KDA(逐 key 通道衰减),decode 走 `fused_recurrent_gated_delta_rule` 融合门控,硬依赖 V-first 布局;而 TM `linear_attn` kernel 全目录**无 channelwise/dt_bias/a_log 支持**(仅 per-head 标量 g,`ComputeBetaG` per-(token,head) 标量,`gated_delta_net_kernels.cu:19-50`)。TM kernel 不补通道衰减,方案 A(和零拷贝)对 GLM-5.3 都无法上 decode 热路径。
3. 批 2 已测过 TM kernel 读 **vk(V-first)转置变体**:惩罚仅 0.90–1.31×(B=32 反而更快),仍全面快于 fla(`benchmark/gdn_layout_results.json`)。若在目标几何+通道衰减形态下复证,**零拷贝直读是方案 A 的严格下位替代**:拿到"桥归零"的全部收益而存储零改动;仅当转置变体惩罚不可接受时,方案 A 才值得立项,且应优先"混合形态"。
4. 方案 A 的真实工程大头不在布局类本身,而在:复合 cache-object 语义在 PT 侧的重建、`state_ptrs`/CUDA-graph 输入管线、spec-decode 多 destination、compact schema 指纹变更。
5. 回滚成本:以 opt-in 布局类落地则为低-中(单元语义/Store 字节区间接口全部不变);全域替换默认布局为中。

## 1. 状态域现状(调研问题 1:触碰点清单)

### 1.1 分配(indexed compact 主线,GLM-5.3/Qwen3.5 CUDA 唯一状态实现)

| 环节       | 位置                                                                                     | 内容                                                                                                                                                                                                               |
| ---------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 声明       | `lmdeploy/pytorch/configurations/glm5_next.py:278-307`                                   | 4 个 spec:`GLM5_KDA_CONV_STATE` `(L, C, d_conv-1)` bf16、`GLM5_KDA_RECURRENT_STATE` `(L, H, K, V)` fp32、kpool tail×2(`store_cacheable=False, prefix_checkpoint=False`);`:285` `state_cache_is_indexed`(CUDA only) |
| 声明       | `lmdeploy/pytorch/configurations/qwen3_5.py:69-91`                                       | 环形(非 indexed)`(L, 1+spec, H, **V_dim, K_dim**)` vs indexed `(L, H, V, K)`——**V-first 写进形状声明**                                                                                                             |
| 规格       | `lmdeploy/pytorch/engine/cache_engine/schema.py:318-342`                                 | `build_state_cache_tensor_specs` → `CacheTensorSpec`(+`LayerRowMap`)                                                                                                                                               |
| 容量       | `lmdeploy/pytorch/config.py:500-534`                                                     | `SSMStateCapacity`:A=M−B−S;`num_units = 1 + R×W + B + S`(W=bank width,AR=1/MTP=1+k,`:517-519`)                                                                                                                     |
| 计划       | `lmdeploy/pytorch/engine/cache_engine/state.py:48-130`                                   | `SSMStatePlan`:单复合 uint8 unit 池 + 请求私有 auxiliary 池;`PackedStateCacheLayout` 的 `transfer_ranges` 合并(`:108-127`);`StateCacheEngine.__init__`(`:104-155`)compact 路径 checkpoint_pools=unit 池            |
| 物理       | `lmdeploy/pytorch/engine/cache_engine/layout.py:180-213`                                 | `PackedStateCacheLayout`:池 `(num_caches, pool_bytes)` uint8,`entry_axis=0`(slot),view `[slots, *spec.shape]`;`:216-241` `ContiguousStateCacheLayout`                                                              |
| 布局选择缝 | `lmdeploy/pytorch/backends/default/cache.py:111-113`(cuda 继承 `backends/cuda/cache.py`) | `build_state_layout` → 默认 `PackedStateCacheLayout`;**方案 A 布局类的插入点**                                                                                                                                     |

### 1.2 decode 读写

- 绑定:`lmdeploy/pytorch/engine/model_agent/agent.py:266`(`context.named_state_caches`);模型取用 `lmdeploy/pytorch/models/glm5_next.py:1817-1845`(`transpose(0,1)` → layer-major,逐层 `(conv, recurrent)` 元组进层)。
- 单元表:`lmdeploy/pytorch/compact_state.py:9-52` `SSMStateMetadata`(`input_state_ids`/`output_state_ids [B,W]`,不可变提交);cursor bind/advance/rebind `:69-113`;advance 在 `strategies/ar/model_inputs.py:30-50`、`agent.py:881-886`。
- kernel:indexed decode `lmdeploy/pytorch/backends/cuda/gated_delta_rule.py:455-470`(`input_state_ids` 读 / `output_state_ids` 写,`transpose_state_layout=True`);KDA decode `lmdeploy/pytorch/backends/cuda/kda.py:354-361`(`state_v_first=True`,A_log/dt_bias 融合门控);conv `backends/cuda/causal_conv1d.py:193-204`。legacy ring 路径(非 indexed/非 CUDA):`_state_select/_state_scatter` triton `backends/cuda/gated_delta_rule.py:173-330` + `spec_state_offsets` `backends/gated_delta_rule.py:50-59`。
- PT kernel 存储约定:`lmdeploy/pytorch/kernels/cuda/gated_delta_rule.py:705`(`transpose_state_layout=True` = `[V,K]`),`:738-747` **channelwise g 与融合 KDA 门控硬依赖 transposed 布局**;indexed prefill 侧 `state_v_first=True`(`backends/cuda/gated_delta_rule.py:483`、`kda.py:300,319,368`)。

### 1.3 checkpoint(decode_checkpoint)

- 保存:计划/掩码 `agent.py:1002-1025`(`_save_decode_checkpoints`;`plan[2]←outputs[rows,offsets]` 回填 `:1018-1021`;TP 一致 mask 广播);设备端掩码 `backends/cuda/cache.py:42-49`;冻结拷贝 `copy_decode_slots` `backends/cuda/cache.py:51-61` → `copy_cache_blocks` 带 mask 的**同格式单元→单元字节拷贝**(`state.py:294-297`)。
- 发布/完成时序:`lmdeploy/pytorch/engine/engine_loop.py:527-581`(`finish_forward_dispatch → publish_forward → acknowledge → complete_forward`;`decode_checkpoints.finish :581`)。
- compact 语义:checkpoint **共享单元、零拷贝**——`state.py:245-246` 显式禁止 `copy_slots`;`reserve_checkpoint`="在可写 bank 之外预留不可变输出"(`lmdeploy/pytorch/paging/compact_state.py:283-303`);prefill 在 kernel 内直接写 checkpoint 单元(`checkpoint_ids/offsets`,`backends/cuda/gated_delta_rule.py:483-487`;64-token/chunk 边界规则 `compact_state.py:170-186`)。
- 恢复:`lmdeploy/pytorch/engine/inputs_maker.py:1059-1111`(checkpoint 驱动 decode、`prepare_decode :1088`);`paging/prefill_scheduler.py:384-434` pin/unpin_restore;`paging/decode_checkpoint.py:155-186`(legacy 恢复+释放)。legacy 保存/恢复拷贝:`agent.py:212,224`(`copy_slots(checkpoint_only=True)`,`state.py:259-292`)。

### 1.4 驱逐/恢复

- 驱逐:`paging/eviction_helper/recompute_eviction_helper.py:73-106`(KV 块容量与 state 容量联动,`make_runtime_state_available` 换入 checkpoint 单元顶替 runtime)。
- 回收:`SSMStateManager.free/release_bank` `paging/compact_state.py:174-272`(submitted holds 延迟物理复用);legacy `zero_slots` `state.py:198-214`。
- 恢复流:`prefill_scheduler.py:576-585`、`paging/kv_load_coordinator.py:413`。

### 1.5 PD 迁移 / Mooncake Store

- **compact 不支持 PD**:`engine/cache_engine/README.md:270`;也不支持 CPU KV 分页(`:272-274` 附近)。legacy 混合模型的 PD:`state.py:193-197` `connector_state_cache_pools` → `agent.py:1735` 注册 → `migration.py:13-101`(describe/validate/assignments,按 `entry_axis`+payload 字节段,不感知池内布局)、`engine/cache_engine/engine.py:296-333`(MR 注册/p2p)。
- Store(compact):整个 unit 池按字节注册,校验 `transfer_ranges` 有序不相交(`kv_connector/mooncake/store/worker.py:313-350, 415-429`);快照槽分配 `mooncake/store/scheduler.py:411-430`;**schema 指纹含 dtype/shape/layers/order/spec/tp**(`tests/pytorch/kv_connector/test_compact_store_wire.py:44-56`),参与 store key → 格式变更=新命名空间,旧条目失效(README:`compact fingerprint isolates legacy/compact formats and incompatible retained state/KV layouts`)。

### 1.6 spec-decode 状态共享

- bank width W=1+k(`config.py:517-519`);每 token 私有 destination(`output_state_ids [B,W]`),decode kernel 的输入共享/输出私占契约见 `kernels/cuda/gated_delta_rule.py:684-692` docstring;ar_spec 侧 `strategies/ar_spec/step_inputs.py:92`、rebind `strategies/base/step_inputs.py:45-50`;acceptance→checkpoint 偏移换算 `agent.py:1018-1021`。legacy ring 版:`kda.py:98-133`(按 accepted history 落环槽)。

## 2. 两侧格式约定对照(方案 A 的核心事实)

|                | PT indexed 池(现状)                                                                                    | TM 复合格式(方案 A 目标)                                                                                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 物理           | 每序列一行 packed uint8:`[conv][recurrent][kpool…]`,256B 对齐(`schema.py:20-29`)                       | per-(layer_group×head_group) block `[L_b×H_b×K×V]`,checkpoint 类复合 ObjectAllocator 注册(`src/turbomind/models/llama/GatedDeltaNetLayer.cc:149` 注册、`:176` `linear_state_offset`、`:332,342` part 解析;通道语义 `src/turbomind/engine/README.md:369-372`) |
| recurrent 轴序 | **V-first** `[H, V, K]`(声明 `qwen3_5.py:78-83`;kernel `kernels/cuda/gated_delta_rule.py:705,738-747`) | **K-first** `[H, K, V]`,经 `state_ptrs[seq*ng + h/H_b]` 寻址(`engine/README.md:386`;`pre_sm90/recurrent.cu:26-58`)                                                                                                                                           |
| 寻址           | 单元 ID(`input/output_state_ids`)索引池行                                                              | 设备端 `int64 state_ptrs [1,B,1]` 直指 slab(`benchmark/bench_gdn.py:49-52` 即此接口)                                                                                                                                                                         |
| gate 形态      | KDA:逐 key 通道 `g [B,T,HV,K]` + A_log/dt_bias 融合(`kda.py:325-328,354-361`)                          | per-head 标量 g(`delta_rule.h:69-99` gate/beta stride;`bench_gdn.py:31-37` 同形态);**linear_attn 全目录无 channelwise/dt_bias/a_log**(grep 零命中)                                                                                                           |

**要点**:GLM-5.3 线性层 = KDA(`models/glm5_next.py:724,733` `Kda()`,`:1331` `is_glm5_kda_layer`;模型几何 34 GDN + 11 MLA)。批 2 的 1.41–3.50×(`benchmark/gdn_results.json`)测的是标量 g 的 GDN recurrent,**不能直接外推到 KDA 通道衰减 decode**——这是两份 benchmark 结论都必须显式声明的前提缺口。

## 3. 方案 A 影响面逐点(调研问题 2)

若 recurrent 状态改为常驻 TM 复合(K-first slab + layer_group blocks):

1. **分配**:新增 state layout 类,插入 `build_state_layout` 缝(`default/cache.py:111-113`);unit 池 `(num_units, pool_bytes)` 结构可保留,复合=unit 内按 `[H,K,V]` 排布,layer_group 块=unit 内子 slab。**无需**引入 TM 全套 ObjectAllocator/驱逐回放(PT 侧没有这套设施;用简化 slab 即可)——复合到什么程度是首要设计决策。conv/kpool spec 不转(无轴序冲突),`transfer_ranges` 仍按字节区间成立。
2. **decode kernel(硬阻断)**:TM pre_sm90 需补 channelwise decay + 融合 KDA 门控;KDA decode 还要求"每 accepted token 一个 destination"的多候选写(W>1),TM kernel 现无 per-call 多 destination 概念(`delta_rule.h` args 仅单一 `state_ptrs`)。
3. **prefill**:留 fla 时,lmdeploy 自有 indexed_chunk 系 kernel 本身支持 K-first(`state_v_first=False` 为默认;FLA 原生 `chunk_gated_delta_rule/chunk_kda` 消费 `[K,V]` 初始态——prefill 侧适配量小 `[INFERENCE:chunk_kda 的初始态轴序未逐行核对]`。
4. **checkpoint 序列化**:格式进 schema 指纹 → 换格式=新 key namespace,旧 Store 条目全部作废(冷启动,非正确性问题);decode freeze `copy_decode_slots` 是同格式字节拷贝,复合格式下**不变**。
5. **PD 字节序**:compact 无 PD → 零增量;legacy 混合模型若不动则不变。
6. **驱逐拷贝**:compact 禁 `copy_slots/zero_slots` → 无新增拷贝。
7. **spec-decode**:bank 展开单元已按 `[B,W]` 私有 destination 运转;TM recurrent 是单 token/call,W>1 需 W 次调用或扩 kernel。
8. **CUDA graph**:piecewise 边界现以 `recurrent_state` 张量+ID 表进 eager 段;`state_ptrs` 是每步变化的数据依赖输入,需常驻设备缓冲+每步生成,graph 管线兼容性需验证。

## 4. 混合形态可行性(调研问题 3)

**热格式=TM K-first、持久化=PT indexed V-first、转换只在 checkpoint/迁移边界**:

- compact 架构里 checkpoint **不是拷贝而是共享单元**,不存在天然的"checkpoint 边界转换"位置;若 checkpoint 单元与 runtime 单元异格式,每步 decode freeze 变成跨格式转置拷贝(频率=decode checkpoint 间隔),但引入双格式单元类型,reserve/restore/Store 全要感知 → 复杂度接近全域方案,**不推荐**。
- 可行的混合变体:**runtime+checkpoint 单元全部 K-first(即全域热格式),仅 Store/PD 导出与恢复边界转换**。转换频率=Store save/load(低频);工程量≈一个单元级转置算子 + schema 双记录;与 B3-01 的"opt-in KV 池非 PT 标准 → 导出时转 PT 标准格式"先例同构。
- 与全域方案差量:仅导出/恢复两个算子 + 指纹策略;decode/prefill 影响面完全相同。结论:**混合形态(持久化边界转换版)工程量 ≈ 全域方案的 90%+,不构成独立的低成本档**;真正的低成本档是零拷贝直读。

## 5. 回滚成本(调研问题 4)

- **opt-in 布局类落地**(推荐形态):回滚=删布局类、恢复默认 `PackedStateCacheLayout` + kernel 入参复位(`transpose_state_layout=True`/`state_v_first=True`)+ schema 指纹恢复。单元语义(ID 表/cursor/checkpoint 共享/Store 字节区间注册)完全不变 → **低-中**(周级 `[INFERENCE]`);已落盘 Store 条目按旧 namespace 失效,冷启动。
- **全域替换默认布局**:revert 面横跨布局+kernel+schema+graph → 中。
- 混合形态(持久化边界转换):回滚面最窄(删导出/恢复转换算子)→ 低。

## 6. 三方案四维对比(调研问题 5)

| 维度             | A. 常驻 TM 复合格式                                                                                                                         | 零拷贝直读(TM vk 变体 + state_ptrs 指进 PT 池)                                                                                          | adapter 桥层(已回滚)                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **热路径开销**   | decode 0 桥(TM kernel K-first 原生);prefill 需 K-first 适配;MTP W>1 与 graph `state_ptrs` 为新增机制。**前提:TM kernel 补 channelwise/KDA** | 0 桥 0 格式变更;vk 转置变体惩罚已测 **0.90–1.31×**(B=32 反而 0.90),对 PT 净胜 1.41–3.62×;**同一 kernel 前提**;每步生成 ptr 数组(小算子) | 每步全量重排 + TMA 描述符,净 **−83%~−148%**(B=1→32),拷贝随 batch×层线性增长 |
| **存储标准纯度** | 终态单一 TM 标准;迁移期双标准;PT kernel 生态(tilelang recurrent/KDA 融合门控)全部需要 K-first 变体,实际是"换标准"而非"存标准"               | 池保持 PT indexed V-first 标准,checkpoint/PD/Store/spec/驱逐语义**零改动**                                                              | 池不变,运行时格式双轨                                                       |
| **工程量**       | 大:布局类+复合寻址、TM kernel 补通道衰减+多 destination、ptr 数组管线、schema 指纹、graph 适配(数周-月级 `[INFERENCE]`)                     | 小:kernel 出状态访问转置变体(swap 两索引项)+ ptr 数组生成 + 旋钮(数天级 `[INFERENCE]`)                                                  | 中(已实现、已回滚)                                                          |

## 7. 决策耦合点(供与 `GdnZeroCopyBench` 合议)

1. **gate 形态缺口是两案共同前置**:GLM-5.3=KDA 通道衰减,TM `linear_attn` 仅标量 g。benchmark 结论必须按"标量 g 的标准 GDN"如实标注适用范围,或在补齐通道衰减+融合门控后按目标几何复测;否则 1.41–3.50× 不能用于 GLM-5.3 决策。
2. **vk 惩罚数据已存在且有利**:`gdn_layout_results.json` 显示转置变体代价可忽略。若带通道衰减的 vk 变体复证通过,零拷贝以近零工程量获得方案 A 的全部 decode 收益,方案 A 的立项理由只剩"未来 TM 侧复合 cache-object 生态复用"(属于 TM 引擎域,非 PyTorch cache_engine 域)。
3. 若仍立项方案 A:锁 opt-in 布局类形态(§5 回滚成本)、持久化边界转换版混合形态(§4)、先补 kernel 通道衰减与多 destination(§3.2),graph `state_ptrs` 管线单列验证项(§3.8)。
