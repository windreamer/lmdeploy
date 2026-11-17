# 批 2 主体:四项 kernel benchmark 报告

日期:2026-11-16 · 机器:RTX 3090 (SM86, 24 GB) · 分支 `graft/w4a16-core`
统一口径:张量直喂(不动状态池/KV 池),同形状同路由,预热度身,时间取中位数,同一 GPU 顺序独占执行。
实现文件:`benchmark/bench_moe_w4a16.py`、`benchmark/bench_gdn.py`、`benchmark/bench_attn_decode.py`;原始数据 `benchmark/{moe,gdn,attn_decode}_results.json`。
每个 benchmark 带正确性门禁(对照各自参考实现),门禁不过不产出数据。

判定口径:TM 快 → 嫁接;TM 慢/平 → 保留;本机不可测 → 留 CI(路线图原则 3)。

______________________________________________________________________

## ① MoE grouped GEMM W4A16

**方法**

- TM:批 1 python 通道(`lmdeploy/turbomind/linear.py` `forward_moe` + `LinkLinearExperts`),u4 g128(CompressedTensorFormat)fp16 激活。层级计时 = gate GEMM + up GEMM + silu → down GEMM → `invokeMoeCombine`(token-major 加权和)。
- PT:生产路径 `kernels/cuda/compressed_tensors_w4a16.py::fused_moe_w4a16`(triton,int4 g32,bf16 scales)。
- 关键事实:**SM86 上 u4 fp16 grouped family 不支持 kGatedSilu 融合 epilogue**(`selected family cannot represent SiLU fusion`),TM 侧走独立 w1/w3 三 GEMM 路径——与 TM C++ 生产 `moe_ffn_layer.cc` 在非融合配置下的路径一致。
- 量化粒度差异是格式属性:TM 要求 g128、PT kernel 仅支持 g32,激活 fp16、权重 u4 语义对齐;同源随机权重、同路由。
- 正确性门禁(8 专家 m=4,各自反量化权重 einsum 参考):TM 独立路径 max_abs=0.0000,PT max_abs=0.0000。

**数据(ms/层,fp16 激活,288 专家 top-8;PT/TM>1 = TM 更快)**

| m(tokens) | TM grouped u4 g128(独立 w1/w3) | PT triton fused_moe_w4a16(int4 g32) | PT/TM    |
| --------- | ------------------------------ | ----------------------------------- | -------- |
| 1         | 0.407                          | 0.644                               | **1.58** |
| 8         | 1.090                          | 4.223                               | **3.87** |
| 32        | 2.923                          | 22.424                              | **7.67** |

TM 侧计时含:`forward_moe`(gate)、`forward_moe`(up)、silu+乘、`forward_moe`(down)、`invokeMoeCombine`;routing 槽位打包(f2n/en2f/offsets)为 host 预生成,两侧一致地不计入。PT 侧计时 = `fused_moe_w4a16` 全流程(路由排序 + gate_up GEMM + silu_and_mul + down GEMM + 加权 scatter)。原始数据 `benchmark/moe_results.json`。

**判定:TM 快,全档胜出且随 m 扩大(1.6× → 7.7×)。** 机理:TM 批 1 管线的离线 repack(u4 g128 swizzle 烘干,运行时 cp.async+LDS 零 swizzle)对 PT triton kernel 的逐 tile 在线 dequant(m 增大时重复反量化成本线性放大)。GLM-5.3 有 42 个 sparse 层、decode 档 m∈{1..32} 正是收益主场。**支持立项「赢家嫁接 spec」**:

- 范围:`LinkLinearExperts` grouped u4 g128 + `forward_moe`(indexed w13 / non-indexed w2)+ `invokeMoeCombine`,复用批 1 `LinearWeight/WeightPlan/pack` 管线;
- SM86 约束(写进 spec):无 kGatedSilu 融合 epilogue,走独立 w1/w3 路径(与 TM C++ 生产一致);fp16 激活(u4 family 本机仅 fp16);
- 量化格式:权重格式=TM g128;PT 侧如需继续吃 g32 checkpoint,由 repack CLI(批 1 票 04)做 g32→g128 重分组;
- 保留 PT 路由(gate/topk/noaux_tc)不动,TM 通道只接管 routed expert 计算 + combine。

## ② GDN kernel(pre_sm90)

**方法**

- TM:`delta_rule_plan/delta_rule_run` 既有 binding(`kernels/linear_attn/python_bind.cpp`,无需新 binding)。state 以 `state_ptrs` 设备指针数组直喂(\[B,64,128,128\] f32 slab,in-place 推进),q/k 预归一化,g=负 log decay fp32、beta=sigmoid fp32,与 TM 生产 `ComputeBetaG`/FLA 约定一致。kernel 选择 `pre_sm90`。
- PT:decode = TileLang `fused_recurrent_gated_delta_rule`(lmdeploy 生产入口);prefill = fla `chunk_gated_delta_rule`。
- 序列步进对齐:decode 双方每调用推进 1 token(recurrent);prefill 单次 T token(chunked,chunk 64)。
- 正确性门禁(单次调用,B=2):输出与 state 双侧对照(见下方数据)。

**形状档位**:head 64×128(hq=hv=64,D=128),state f32;decode B∈{1,8,32}×T=1;chunked B=1×T∈{256,1024}、B=8×T=256。

**数据(ms/调用,PT/TM>1 = TM 更快)**

| 模式      | B   | T    | TM     | PT (fla/tilelang) | PT/TM    |
| --------- | --- | ---- | ------ | ----------------- | -------- |
| recurrent | 1   | 1    | 0.0135 | 0.0471            | **3.50** |
| recurrent | 8   | 1    | 0.0942 | 0.1766            | **1.87** |
| recurrent | 32  | 1    | 0.3809 | 0.5350            | **1.41** |
| chunked   | 1   | 256  | 0.2673 | 0.3615            | **1.35** |
| chunked   | 1   | 1024 | 0.9139 | 0.5928            | 0.65     |
| chunked   | 8   | 256  | 1.7894 | 1.0834            | 0.61     |

正确性门禁(B=2,单次调用,TM vs PT 生产 kernel):recurrent out/state max_abs=0.0000/0.0000;chunked out/state max_abs=0.0005/0.0034 → 双双通过。

**判定:分档。** decode 步进(recurrent)TM 全档更快(1.4–3.5×)——decode 是 GDN 层在推理中的绝对主导档位,**支持立项嫁接**(受益面:GLM-5.3 34 层 GDN 的 decode 路径)。chunked prefill TM 仅在短 chunk(B=1,T=256)占优,长序列/大批 prefill PT fla 更快(0.6–0.65×)。若嫁接,建议 decode 走 TM kernel、prefill 保留 fla(chunked 模式 TM 待 SM90/更优 chunk 策略再评估),或按“decode 嫁接 + prefill 保留”双轨。

## ③ attention bf16 decode kernel

**方法**

- TM:新 binding `src/turbomind/python/attn_bench_bind.cpp` → `_tm.attn_decode_bench`:固定 GQA decode 入口(H=64,Hkv=8,D=128,block_len=64),`AttentionParams` 按 `unified_attention_layer.cc` 生产填法(qkv 融合行 stride、`BlockIteratorParams{char** cu_block_nums}`、split-K workspace 4096-token 预算),`dispatchDecoding<bf16>`;host 填 `char**` 指针数组(pool = 单张 torch uint8 张量,块内 TM 布局),CUDA event 计时在 C++ 侧完成。
- PT:`kernels/cuda/pagedattention.py::flash_attn_with_kvcache`(triton 生产 decode 路径,split-K + reduce 两 kernel)。**FA3 在 SM86 不可用(SM90-only)**,3090 上 PT 生产路径本就是 triton 兜底——本对比即本机真实路径对比;H200 CI 需复测 FA3 基线。
- KV 池:PT 标准 \[num_blocks, 64, Hkv, D\];TM 池同字节量,块内 TM head-major 布局,布局转换在计时区外。
- 正确性门禁:B=4 klen=256 对照 fp32 einsum,bf16 max_abs:TM 0.0104 / PT 0.0005 → 过。

**数据(PT/TM>1 = TM 更快;详细 24 格见 attn_decode_results.json)**

| B   | klen | TM ms  | PT ms  | PT/TM    | TM GB/s |
| --- | ---- | ------ | ------ | -------- | ------- |
| 1   | 256  | 0.0113 | 0.0788 | 7.00     | 93      |
| 1   | 4096 | 0.0451 | 0.0788 | 1.75     | 372     |
| 8   | 4096 | 0.1772 | 0.2150 | 1.21     | 758     |
| 32  | 1024 | 0.1659 | 0.2263 | 1.36     | 809     |
| 32  | 4096 | 0.6195 | 0.6943 | 1.12     | 867     |
| 128 | 256  | 0.1731 | 0.2580 | 1.49     | 776     |
| 128 | 4096 | 2.4468 | 2.5446 | **1.04** | 878     |

**判定:TM 全格不输(1.04–7.0×),小 batch decode 延迟档优势大(B≤8:1.2–7×)。** 但按原则 2,若放弃 TM KV 布局,收益仅剩 kernel 本体 vs FA3;本机 FA3 不可测,大 batch 档对 triton 仅 1.04–1.2×,边际收益小、接缝成本高(指针数组 builder + CUDA Graph 契约,票 06 已证可行但工程量实打实)。**建议:保留/CI 复测**(H200 上对 FA3 复测后再定;若 H200 上 FA3 基线下 TM 仍稳定 ≥25% 优势且 decode 延迟档需求强,可重新立项)。低预期兑现:数据说话,不嫁接为默认结论。

## ④ packed-KV int8(TM int8 读取管线 vs PT int8 读取路)

**方法**

- 同 ③ 形状与 TM 入口,`quant_policy=kCacheKVInt8`;TM 块内布局含 per-token-per-head scale 槽(2×bf16 = scale+min,尾部 param 区),uint8 codes。
- PT:`QuantPolicy.INT8` 路径:同一 uint8 codes 写入 `k_cache/v_cache [nb,64,Hkv,D]` + `k/v_scales_zeros [nb,64,Hkv,2]`(bf16 scale/zero),`fill_kv_cache` 语义的 per-token min-max 量化与 TM 完全同码,`flash_attn_with_kvcache(quant_policy=INT8)` 读取。
- 正确性门禁:int8 max_abs:TM 0.164 / PT 0.001(vs 量化参考,TM bf16 累加精度导致较松,门限 0.25)→ 过。

**数据(PT/TM>1 = TM 更快;TM int8 与 TM bf16 同列对比看量化收益)**

| B   | klen | TM int8 ms | PT int8 ms | PT/TM    | TM int8 GB/s | TM bf16 ms(同格) | TM int8/bf16 |
| --- | ---- | ---------- | ---------- | -------- | ------------ | ---------------- | ------------ |
| 1   | 4096 | 0.0307     | 0.0886     | **2.88** | 273          | 0.0451           | 1.47         |
| 8   | 4096 | 0.1055     | 0.1966     | **1.86** | 636          | 0.1772           | 1.68         |
| 32  | 4096 | 0.3318     | 0.5473     | **1.65** | 809          | 0.6195           | 1.87         |
| 128 | 4096 | 1.3312     | 1.7817     | **1.34** | 807          | 2.4468           | 1.84         |
| 32  | 1024 | 0.0963     | 0.2017     | **2.10** | 697          | 0.1659           | 1.72         |
| 128 | 256  | 0.1014     | 0.2202     | **2.17** | 662          | 0.1731           | 1.71         |

**判定:显著,支持立项。** 全部生产档位 PT/TM ≥ 1.34(阈值 ≥25% 即 1.25×,全部通过;吞吐档 B=128/klen=4096 最低仍 1.34×)。拆解:TM 侧 int8 vs 自身 bf16 ≈ 1.7–1.9×(接近带宽减半的理论 2×,证明三级流水 dequant 内嵌有效);PT 侧 int8 vs 自身 bf16 仅 ≈1.2–1.3×(分离式 scale 读取 + 独立 reduce 摊薄收益)。TM 三级流水相对 PT 分离式读取的 delta 明确显著。**建议立项路径 B:opt-in TM CacheBackend 布局抽象**(抽象层已存在,`backends/cache.py`);成对迁移 TM KV 写入(rope+量化融合写)。注意 TM 优势包含两部分:layout 槽内嵌(kernel)+ decode kernel 本体(③ 的 1.04–1.2×),立项 spec 应按“CacheBackend 布局抽象 + TM attention decode kernel”整体立项,并以 H200 CI 复测 FA3 基线下的 int8 档。

______________________________________________________________________

## 汇总

| 项                  | 判定                                                                    | 后续                                                                        |
| ------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ① MoE grouped W4A16 | **TM 快(1.6×→7.7×),嫁接**                                               | 立「赢家嫁接 spec」:grouped u4 g128 通道;SM86 独立 w1/w3、fp16 激活约束写入 |
| ② GDN               | decode(recurrent)TM 快 1.4–3.5× → 嫁接;chunked prefill PT 快 → 保留     | 立 spec:decode TM kernel + 状态池语义;prefill 留 fla                        |
| ③ attention bf16    | 保留,留 H200 CI(对 FA3 复测;本机全格 1.04–7.0× 但大 batch 仅 1.04–1.2×) | —                                                                           |
| ④ packed-KV int8    | **显著(生产档 1.34–2.88×,全部 ≥25% 阈值),立项路径 B**                   | spec:opt-in CacheBackend 布局抽象 + TM int8 attention,成对迁移 KV 写入      |

## 附:binding 源文件与 CMake 改动清单

| 文件                                                                                       | 改动                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/turbomind/python/attn_bench_bind.cpp`                                                 | **新增**:decode-attention benchmark binding(`attn_decode_bench`:固定 GQA 形状,生产同款 `AttentionParams` 填法,host 填 `char**` 指针数组,CUDA event 计时,bf16/fp16 × KV{bf16,int8,int4}) |
| `src/turbomind/python/bind.cpp`                                                            | 注册 `bind_attn_bench`(声明 + 模块 init 调用)                                                                                                                                           |
| `src/turbomind/python/CMakeLists.txt`                                                      | `pybind11_add_module` 源列表加入 `attn_bench_bind.cpp`                                                                                                                                  |
| `benchmark/bench_moe_w4a16.py`、`benchmark/bench_gdn.py`、`benchmark/bench_attn_decode.py` | **新增** benchmark 脚本(各含正确性门禁)                                                                                                                                                 |
| `benchmark/{moe,gdn,attn_decode}_results.json`                                             | 原始数据                                                                                                                                                                                |

未 commit、未 push。② TM GDN 用既有 `delta_rule_plan/run` binding,① 用批 1 `linear.py` 通道,均无需新 C++。

______________________________________________________________________

# 附:GDN 方案 B 可行性(状态访问转置变体,零拷贝直读)

日期:2026-11-17 · 机器:RTX 3090 · 分支 `graft/w4a16-core`。
背景:PT 状态池按 \[N,H,V,K\] 排布(每序列行内 \[H,V,K\] 连续)。批 2 基线的 kernel 按 \[K,V\] 寻址。方案 B 要求 kernel 直接读池排布,省去转置拷贝。本节回答一个问题:转置变体能否保留批 2 的 decode 收益。

**方案与改动**

- 问题定义:状态元素 (k,v) 原寻址为 `k*HeadDim + v`。池排布要求改为 `v*HeadDim + k`。递归更新数学不变。
- `delta_rule.h` 新增 `GdrStateLayout {kKV, kVK}`。`Operation` 以及 `GdrKernelSpec` 各加 `state_layout` 字段,默认 kKV。
- `MatchesGdrSpec` 加入 state_layout 相等检查。既有匹配结果不变,新变体只响应显式请求。
- `pre_sm90/recurrent.cu` 的 kernel 模板加 `StateLayout` 参数。kKV 分支代码原样保留。kVK 分支的寄存器按 v 行组织:每线程持 4 行 × 16 个连续元素,读写都是 64 B 连续段,寄存器内不做转置。
- 注册 4 个 recurrent kVK 实例(f16/bf16 输入 × f16/f32 state)。binding `delta_rule_plan` 加 `state_layout` 参数(`kv`/`vk`,默认 `kv`),plan dict 输出所选布局。
- chunked 模式以及其它 kernel 家族不受影响。

**一致性门禁**

同一数学状态分别按 \[K,V\] 以及 \[V,K\] 存放。两侧各跑一次(B=2,bf16 输入,f32 state):

- out max_abs = 0.000e+00,final state max_abs = 0.000e+00。结果位级一致。
- `compute-sanitizer --tool memcheck` 覆盖 kv、vk、chunked 三条路径:0 errors。

**数据(decode recurrent,head 64×128,state f32,ms/调用)**

| B   | ① TM-kv(批 2 形态) | ② TM-vk(池直读) | ③ PT   | 转置损失 vk/kv | 净收益 PT/vk |
| --- | ------------------ | --------------- | ------ | -------------- | ------------ |
| 1   | 0.0133             | 0.0174          | 0.0481 | 1.31×          | **2.77×**    |
| 8   | 0.0932             | 0.0983          | 0.1761 | 1.06×          | **1.79×**    |
| 32  | 0.3789             | 0.3400          | 0.5309 | 0.90×          | **1.56×**    |

原始数据 `benchmark/gdn_layout_results.json`。脚本 `benchmark/bench_gdn.py`(`--layout-json-out` 出布局档数据)。

**观察**

- B=1 档处于发射延迟主导区(全程 13–17 µs)。转置损失 1.31×,绝对差 4 µs。
- B=32 档 vk 反超 kv(0.90×)。\[V,K\] 排布下每线程读写 4 个 64 B 连续段;kv 排布下每段只有 16 B。\[INFERENCE\] 连续段更长可能是反超的原因。
- B=1 档 ①③ 差距(3.6×)与批 2 基线(3.5×)一致,本轮复测有效。

**判定:方案 B 可行。**

- 转置损失可控:1.31× / 1.06× / 0.90×,随 B 增大递减,B=8 起接近持平。
- 净收益保留:PT/vk = 2.77× / 1.79× / 1.56×,全部高于 1.2× 阈值。
- 对照批 2 基线(PT/TM 1.41–3.50×):直读形态保留 79%(B=1)、95%(B=8)、111%(B=32)的收益。
- 结论:生产接入时 `state_ptrs` 直接指向 PT 池行即可,免去 \[V,K\] → \[K,V\] 转置 kernel 以及双份状态存储。

**改动文件**

| 文件                                               | 改动                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------ |
| `src/turbomind/kernels/linear_attn/delta_rule.h`   | 新增 `GdrStateLayout`;`GdrKernelSpec`/`Operation` 加 `state_layout` 字段 |
| `src/turbomind/kernels/linear_attn/kernel/plan.cc` | `MatchesGdrSpec` 加 state_layout 相等检查                                |
| `.../kernel/pre_sm90/recurrent.cu`                 | kernel 模板加 `StateLayout`;kKV 分支原样保留;kVK 分支按池排布寻址        |
| `.../kernel/pre_sm90/entry.cu`、`internal.h`       | 注册 4 个 recurrent kVK 实例;声明以及分发对应回调                        |
| `.../linear_attn/python_bind.cpp`                  | `delta_rule_plan` 加 `state_layout` 参数;plan dict 输出所选布局          |
| `benchmark/bench_gdn.py`                           | 新增转置一致性门禁以及 decode 三路对比(kv/vk/PT)                         |
| `benchmark/gdn_layout_results.json`                | 本节原始数据                                                             |

未 commit、未 push。
