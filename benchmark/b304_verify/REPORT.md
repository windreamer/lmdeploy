# B3-04 复验报告:decode "末位 KV 零写" 的归属判定与 bench-binding 修复原型

日期:2026-11-17 | 执行:KernelOwnerVerify | 分支:`graft/w4a16-core`
GPU:RTX 3090 24GB,驱动 580.178.04,CUDA 13.0,SM (8,6);`.venv/bin/python`,torch 2.12.1+cu130

## 总判定

**Bench-binding 契约违例,非内核缺陷;生产无影响。** 修复原型已落地并全绿。

- `attention_universal.h` 的 decode 内联 KV 写回(`kProcessKV`)是**设计行为**:decode 时内核把"当前 token"的 K/V(rope 后、量化后)写回池中 `ti = history_len` 位置——即新 token 的 KV append。生产路径传入真实 K/V,写回幂等;不存在"抹掉最新 token"。
- 旧 binding 把 `params.k/v` 指向只填了 Q 的融合 qkv workspace(K/V 段未初始化),每次 dispatch 都把这段垃圾量化后写进池内每序列最后一个有效位置。**缺陷复现**:全部 1024 个 code 字节 + 32 个 param 值被清零/改写,中间位置不动(与 DebugB304Writer 报告一致;allocator 新页恰好为零 → 表现为"零写")。
- `readonly_block_num = nullptr` 是第二处契约违例,但**效果上无害**(nullptr 与"全 0 缓冲"产生相同行为;生产 decode 本来就是 0)。

## 1. 生产约定事实(源码行号)

本分支与 `upstream/main` 在下列文件上**逐字一致**(经 `git fetch upstream && git show upstream/main:…` 对照;`attn_bench_bind.cpp` 上游不存在,系本分支批 2 新增):

| 约定 | 位置 | 内容 |
| --- | --- | --- |
| `params.k/v` 指向真实 K/V | `src/turbomind/models/llama/unified_attention_layer.cc:477,483` | `params.k = params.q + local_head_num * size_per_head;` / `params.v = params.k + local_kv_head_num * size_per_head;` — 指入当前 batch 的融合 qkv GEMM 输出([Q\|K\|V] 行,stride 含 gate 段),K/V 段是**真实值** |
| 三个序列级缓冲恒为有效引擎缓冲 | 同上 `:533-535` | `cu_q_len = d.q_offsets.data() + offset`、`cu_k_len = d.k_offsets.data() + offset`、`readonly_block_num = d.readonly_block_num.data() + offset` — 永不为 null |
| `readonly_block_num` 语义 | `unified_attention_layer.cc:110,263`;`src/turbomind/engine/scheduler.cc:666`(decode 步置 0,注释"decode writes only the new token");`:551-554`(resume 时按 fully-valid whole block 计数) | 只守卫 KV store(`ti < readonly_len` 跳过),decode 常规步为 0 |
| 内核写回机制 | `src/turbomind/kernels/attention/attention_universal.h:59`(`kProcessKV`,DecodingCtaMap 恒真)、`:218,220`(Prologue 从 `params.k/v` Ldg)、`:262`(`ti = history_len`)、`:267`(`readonly_len`)、`:296`(`if (ti >= readonly_len)`)、`:306,314`(`Store(&k_cache[di], out_K)` + `StoreQuantParam`) | decode 每次调度必然读 `params.k/v`、做 rope、量化、写回池内 `ti = history_len` 位置(codes + per-token-per-head {scale,min} 参数) |
| 块内 int8 布局 | `src/turbomind/kernels/attention/block.h:126-232`(`Layout`:data 段 per-head [K\|V],param 段在全 data 之后 per-head [Kp\|Vp]);`block_iterator.h:91-94`;PT 侧同款 `lmdeploy/pytorch/backends/tm_kv.py::TmInt8BlockCacheLayout.allocate` | `codes [nb,H,2,S,D] u8 + params [nb,H,2,S,2] bf16`,生产桥与本报告验证脚本均按此布局 |

结论:生产 decode 的 `params.k/v` 是真实数据 ⇒ 内核写回 = 把新 token 的 KV 写到它该在的位置,幂等(同值重写);`readonly_block_num` 恒有效。**生产不会触发零写。**

## 2. 生产影响定性(复现/排除,实测)

用修复后 binding(生产约定填法)实测,**排除抹写**:

- **bf16 池 + 真实 k/v**:decode 后池**逐字节不变**(bit-exact 幂等;`ConvertKvCache<T,T>` 恒等),输出 vs fp32 参考 max_abs = 0.00051;
- **int8 池 + 真实 k/v**:末位 codes 与预量化往返差 ≤2/255(kernel `warp_stats`+`cvt.rni` vs torch fp32 round 的舍入链差),{scale,min} 参数相等,输出 vs 反量化参考 max_abs = 0.00056;
- **负对照(证明写回活跃且源是调用方 k/v)**:传 `2*k/2*v` → 末位 scale/min 精确 ×2(max_abs_diff 0.00012/0.00000)、codes == quant(2k)±2;
- **零写仅在契约违例下复现**:传显式零 k/v → 末位 codes+params 全零(即旧 binding 的行为);传真实值则零写消失。

脚本:`benchmark/b304_verify/verify_b304_fix.py`(13/13 PASS,exit 0)。

缺陷本体在**未修复构建**上复现:`benchmark/b304_verify/repro_b304_defect.py` → `DEFECT REPRODUCED`(末位 1024/1024 code 字节 + 32/32 param 值被改写并清零,中位 ti=50 不动,exit 0)。

## 3. 修复原型(可独立提交的变更集)

机制:decode 内核必然解引用 `params.k/v`,binding 必须按生产约定供给真实 K/V;`readonly_block_num` 供给合法零缓冲(镜像 `Scheduler::PlanContinue`)。

**文件清单(一个 commit):**

1. `src/turbomind/python/attn_bench_bind.cpp`
   - `BuildDecodePlan` 新增 `k`/`v` 形参;融合 qkv workspace 的 K/V 段按生产布局用 `cudaMemcpy2DAsync` 填入真实值(原来只填 Q,K/V 段未初始化);
   - 新增 `DecodePlan::readonly_block_num` [B] int 零缓冲,`params.readonly_block_num` 指向它(不再 nullptr);
   - `attn_decode_bench` / `tm_attn_decode` 增加 **必填** `k`/`v` 参数([B, Hkv*D],dtype 同 q;形状/类型 TM_CHECK);修正原注释中"K/V pointers … never dereferenced"的错误论断(该论断正是本缺陷根源);
   - pybind 签名:`"q"_a, "k"_a, "v"_a, …`。
2. `benchmark/bench_attn_decode.py`
   - 新增 `last_token_kv()`:从预量化源池提取每序列最新 token 的 K/V 行;
   - `run_tm` 传入 `k`/`v`;
   - **附带修复** `build_tm_pool_int8` 块内布局:由错误的 `[K all][V all][Kp all][Vp all]` 改为 `block::Layout` 的 per-head `[K|V]` codes + per-head `[Kp|Vp]` params(与 `block.h` 及生产桥 `TmInt8BlockCacheLayout` 对齐)。
3. `benchmark/b304_debug/sanitizer_entry_b304.py`
   - reader-1 改传真实 `k[:,KLEN-1]`/`v[:,KLEN-1]`,更新注释(原"kernel-side last-slot zero-write"表述作废)。

建议 commit message:

```
fix(bench): feed real K/V to decode binding; int8 pool layout per block::Layout

The decoding kernel (attention_universal.h kProcessKV) always re-stores the
caller's K/V rows (rope'd, quantized) into the block pool at ti = history_len.
The bench binding pointed params.k/v at a Q-only fused-qkv workspace, so every
decode dispatch quantized uninitialized memory into the last valid KV slot of
each sequence (the B3-04 "zero-write"). Pass k/v [B, Hkv*D] explicitly and
mirror the production decode convention: real K/V values + a valid
readonly_block_num buffer (zeros; Scheduler::PlanContinue).

Also align benchmark pool construction for int8 with block::Layout (per-head
[K|V] codes then per-head [Kp|Vp] params) — the previous [K|V|Kp|Vp] ordering
made the TM kernel read a garbled pool (batch2 int8 gate "passed" at 0.164/0.25;
correct layout yields 0.0005).
```

附录(建议不入 fix commit,单独归档):`benchmark/b304_verify/`(本报告、两个脚本、`initcheck_postfix.log`、`memcheck_postfix.log`)。

## 4. 验证记录(已执行)

| 项 | 修复前 | 修复后 |
| --- | --- | --- |
| `repro_b304_defect.py`(旧 API) | **DEFECT REPRODUCED**(exit 0) | 预期:调用处因缺 k/v 失败(契约生效) |
| `verify_b304_fix.py`(新 API) | — | **13/13 PASS,exit 0**(幂等 + 输出正确 + 负对照) |
| `bench_attn_decode.py` 自带门禁(B=4,klen=256) | int8 TM max_abs **0.164**(错布局) | kv0 **0.0005** ok;kv8 **0.0005** ok(PT 0.0010) |
| compute-sanitizer initcheck(`sanitizer_entry_b304.py`) | **100 处** Uninitialized(`Prologue Ldg params.k/v`,经 TmAttnDecode) | **ERROR SUMMARY: 0 errors**(exit 0) |
| compute-sanitizer memcheck(同入口) | 4 errors | 4 errors,**逐条相同**(均为 `cudaErrorInvalidValue` @ `cudaFuncSetAttribute`/`cudaGetLastError`,属入口内其他内核路径的既有 API 告警,与本缺陷及修复无关,非访存错误) |

日志:`benchmark/b304_verify/{initcheck,memcheck}_postfix.log`;修复前基线:`benchmark/b304_debug/{initcheck,memcheck}.log`。

## 5. 边界判定与上游评估

- **非内核缺陷**:生产约定下零写不复现(§2),内核数学未改动。无需上游内核修复。
- **无可上游上报项**:缺陷主体(`attn_bench_bind.cpp`)与 bench 脚本均为本分支私有,上游不存在;内核与生产路径上游逐字一致且在生产约定下正确。
- B301 报告的附带发现(`stride_boff==1` JIT 特化越契约 IMA)仍是唯一已知的上游可报候选,不属本票,维持 B301 结论。
- **遗留提示(供主会话/后续票)**:批 2 报告 ④ 的 int8 正确性门禁"TM 0.164 / 门限 0.25 → 过"是在错误池布局下测得的(garbage-in,门禁过松放行);时序数字不受影响(读取流量相同),但该 0.164 不应再被引用,正确布局下为 **0.0005**。`.scratch/tm-gemm-graft/` 下旧探针仍用旧 API 签名,属临时产物,未随本修复更新。

## 6. 盲态复核操作单(供独立 agent 执行,勿读本报告结论作前提)

前置:3090 空闲;`build/` 为已配置构建树;模块经 `cmake --build build --target _turbomind -j$(nproc)` 后自动落位 `lmdeploy/turbomind/_turbomind.cpython-313-*.so`(无需手动拷贝)。

```bash
# T1 修复面验证(固定树,预期全 PASS、exit 0)
.venv/bin/python benchmark/b304_verify/verify_b304_fix.py; echo "expect exit=0"

# T2 bench 门禁(预期两行 ok:tm_max_abs=0.0005 / 0.0005)
.venv/bin/python benchmark/bench_attn_decode.py --kv-quants 0 8 --bsz 4 --klen 256 --iters 20 --warmup 5

# T3 缺陷再现(在修复前的 binding 上)
git stash push -- src/turbomind/python/attn_bench_bind.cpp benchmark/bench_attn_decode.py benchmark/b304_debug/sanitizer_entry_b304.py
cmake --build build --target _turbomind -j$(nproc)
.venv/bin/python benchmark/b304_verify/repro_b304_defect.py   # 预期 DEFECT REPRODUCED, exit=0
git stash pop
cmake --build build --target _turbomind -j$(nproc)

# T4 sanitizer(各 ~数分钟,预期两者均 "ERROR SUMMARY: 0 errors"、exit 0;
#  T3 之后需先重建再跑)
compute-sanitizer --tool initcheck --error-exitcode 9 \
  .venv/bin/python benchmark/b304_debug/sanitizer_entry_b304.py
compute-sanitizer --tool memcheck --error-exitcode 9 \
  .venv/bin/python benchmark/b304_debug/sanitizer_entry_b304.py   # 注意:4 条既有 cudaFuncSetAttribute API 告警为基线,非访存错误
```

预期判定:T1–T4 全绿 ⇒ B3-04 关闭,定性 = bench-binding 契约违例,生产无影响,本变更集为正解;若 T2/T3 在生产约定填法下仍见末位被改写(超出 ±2 codes 舍入)⇒ 升级为内核缺陷,产出最小复现并上报(本报告中 §1 行号即定位起点)。
