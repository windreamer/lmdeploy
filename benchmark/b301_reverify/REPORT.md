# 独立复验报告:triton 3.7.1 "误编译"主张

日期:2026-11-17 | 复验人:独立 agent(与原 agent 无共享代码) | 分支:`graft/w4a16-core`
产物:`benchmark/b301_reverify/`(t1–t5 脚本、t\*.log、t\*\_results.json、env_fingerprint.txt)

## 总判定:**REFUTED**

四项"本机 triton 3.7.1 误编译"主张全部不成立:

- triton 原语(大偏移 store、program_id、整数 div/mod)在本机 3090/SM86 上**逐字节/逐元素精确正确**;
- 生产 `fill_kv_cache`(QuantPolicy.INT8)**按生产契约调用时全部 16/64/512 物理块写入有效、内容与独立 golden 一致**,scale 逐位精确、code 差异仅 ±1(量化边界舍入);
- "16 块仅 6 块有效"**可复现,但根因是原探针违例了 fill_kv_cache 的两个调用契约**,不是编译器错误——用一个可预测的违例变体即可把写入位置精确预测为 {0,4,8,12},与 flat-index 语义完全吻合。

## 1. 环境指纹(实测)

| 项         | 值                                                                                      |
| ---------- | --------------------------------------------------------------------------------------- |
| GPU / 驱动 | RTX 3090 24GB,驱动 580.178.04,CUDA 13.0                                                 |
| SM 架构    | (8, 6)                                                                                  |
| triton     | 3.7.1(site-packages wheel,cp313 manylinux,非 editable/本地构建;INSTALLER 无 DIRECT_URL) |
| torch      | 2.12.1+cu130                                                                            |
| python     | 3.13.12(.venv)                                                                          |

## 2. 方法论

按本机 kernel skills(write-kernel-test-plan / handle-boundary-conditions / debug-quantized-kernel-accuracy)先出最小验证计划再动手。每个用例:从零手写 kernel、独立 golden(纯 torch/python)、精确断言(torch.equal / max_abs / neq 计数)。原 agent 的 kernel 代码未复用为验证器(`.scratch/tm-gemm-graft/` 下无任何 kernel 源码,`bench_b301_bridge.py` 内亦无 `@triton.jit`;其探针本体已不存在,仅最后按允许以"对照项"原样运行其 `check_production_fill_anomaly` harness)。调试中修正过两个**我方 harness bug**(golden 线性化顺序、bool.sum 的 18GiB 具象化),均不影响下述结论方向。

## 3. 逐主张结论

### C1 "自写 kernel 大偏移 store 大面积丢失"(128 程序仅 ~16–32 可见)→ **REFUTED**

`t1_large_offset_store.py`(1D grid,每程序写已知字节模式到 TM 块几何 135KB×128 块,sentinel 0xEE 预填,逐字节断言):

- A(int64 算术,满块向量 store):**128/128 块全部可见,mismatch 0 字节**
- B(launcher 预计算 int64 偏移张量):精确
- C(偏移跨 2^31 边界,2.25GiB 池,pid 255/256 骑线):精确
- D(int32 乘法回绕):按 C 语义回绕,值正确(证明"丢失"不是回绕)
- E(全真 mask 向量 store @ 大偏移):精确

### C2 "tl.program_id(1/2) 返回垃圾" → **REFUTED**

`t2_program_id.py`(每程序写自己的 (pid0,pid1,pid2) 三元组,golden=纯 python 网格索引):

- 1D×128:128/128 store 可见
- 3D (256,16,4)=16384 程序:**0 个垃圾槽**
- 3D (4,5,8)(生产 grid 形状):精确
- 3D (65535,2,1)=131070 程序(大 x 应力):精确

### C3 "kernel 内整数 div/mod 链结果错误" → **REFUTED(语义差异最可能是原探针的假阳性来源)**

`t3_divmod.py`(向量混合符号 4099 元素、15 对标量、生产式 `h//BLOCK+bid` 链+guard):

- 结果**与 C truncation 语义逐元素精确一致**(`torch.div(rounding_mode='trunc')`/`torch.fmod`);
- 与 Python floor 语义在负操作数上不同(4099 元素中 3031 个为负):`7//-2`:triton=-3(python=-4)。**若原探针用 Python `//`/`%` 作 golden,会把正确的 C 语义误判为"div/mod 错误"**。

### C4 "生产 fill_kv_cache(INT8)16 块仅 6 块有效" → **异常可复现,REFUTED 为误编译;根因=探针违例调用契约**

生产契约(读 `lmdeploy/pytorch/engine/inputs_maker.py` + kernel 源;与 upstream/main 一致,非本仓私改):

1. `k_states/v_states` 为 **packed** `[total_tokens, H, D]`,`q_start_loc` 为累计偏移(kernel 只用 `q_offs=q_startloc+…` 索引状态,无 batch 维);
2. `block_offsets` 为 **2-D 逐序列页表** `[bsz, npages]`(绝对 kernel 块号);kernel 以 `tl.load(BlockOffsets + batch_id*stride_boff + kv_block_id)` 平铺索引。

原探针(`check_production_fill_anomaly` 及 `prefill_write_production`)传入:

- **batched** `[B,klen,H,D]` + `q_start_loc=0` → 每个序列实际量化的是 seq0 的 token;
- **1-D** `arange(bsz)*bps`(长度 bsz、stride=1)→ kernel 读 `table[b*1+kvb]`:平铺下标 ≥4 越界读(b+kvb≥4 共 6 处),界内读把 (b,j) 页写到绝对块 `(b+kvb)*bps`,即 **{0,4,8,12}** —— 与"仅前几个块有效"吻合。

**证明(全部可复现,脚本在目录内):**

- `t5_original_control.py`:原 harness 逐字重跑 → **6/16 复现**(越界读的 2 处恰落回有效块号)。
- `t4_...py` misuse-padded 变体(同样 1-D 值、padding 为块 0 保证全界内):写入块**精确 = {0,4,8,12}**,其余 12 块全零 —— 平铺索引语义的可预测必然结果,而非随机"丢失"。
- `t4_...py` correct 变体(生产契约:packed + 累计 q_start_loc + 2-D 单位页表):**16/16、64/64、512/512 块全部有效**,位置精确 `kc[4b+j]`;对照独立 fp32-exact golden:scales **逐位精确(0 差异)**、zeros ≤2 fp32 ulp、codes 仅 ±1(0.08%,FMA/除法边界舍入),**kernel 反量化误差与理想 golden 相同(0.00836,0.5·scale 界内)**。
- compute-sanitizer memcheck 对 klen1024 correct 用例:**0 errors**。

### 附带发现(不属于原主张,移交参考)

`stride_boff==1`(1-D 块表,即越契约调用)的 JIT 特化在**同一进程先跑过 klen≥1024 的 correct 用例后**启动时,**确定性** `illegal memory access`(CUDA_LAUNCH_BLOCKING 下定位在该 launch;隔离运行/先运行则正常;sanitizer 下不复现,0 errors)。仅可达于契约外调用路径;生产路径(2-D 页表)不受影响。复现:

```bash
python benchmark/b301_reverify/t4_production_fill_int8.py   # 注释掉 main 首行 misuse 再跑(排在 klen>=1024 之后)
# 或 /tmp/t4dbg8.py 1(临时调试脚本思路见 REPORT 注)
```

## 4. 复现命令

```bash
.venv/bin/python benchmark/b301_reverify/t1_large_offset_store.py
.venv/bin/python benchmark/b301_reverify/t2_program_id.py
.venv/bin/python benchmark/b301_reverify/t3_divmod.py
.venv/bin/python benchmark/b301_reverify/t4_production_fill_int8.py
.venv/bin/python benchmark/b301_reverify/t5_original_control.py   # 退出码 1 = 原异常复现(契约违例所致)
```

## 5. 对原报告(b3_01_bridge_report.md §4)的影响

- §4"工具链告警"**应撤回**:triton 3.7.1 未发现误编译证据;生产 fill 完好(见 §3/C4)。
- 原型因此改道 host-op 写入方的理由不成立;PT triton fill → TM 池的端到端用例**可以**在本机可信执行(按生产契约构造输入)。
- 移交 B3-04 的"健康工具链复跑"前置条件解除;仅建议 B3-04 写 fill 适配层时:传 2-D 页表 + packed 状态 + 累计 q_start_loc,并以 `t4_production_fill_int8.py` 的 correct 变体作回归模板。
