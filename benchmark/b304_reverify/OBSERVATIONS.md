# B3-04 盲态复核观测记录(BlindReverifyB304)

日期:2026-11-17 | HEAD:`982f4e13`(含 `45e9322c`) | GPU:RTX 3090(空闲,0% util) | 驱动 580.178.04 | torch 2.12.1+cu130 / CUDA 13.0
执行口径:仅记录观测值,不与任何预期比对;原始输出见本目录 `T1_*` / `T2_*` / `T3_*` / `T4_*` 文件。

## T1 修复面验证(`benchmark/b304_verify/verify_b304_fix.py`)

exit=0;输出末行 "ALL CASES PASSED";逐条 PASS 行 13 条,观测值:

```
[PASS] bf16: pool byte-identical after decode
[PASS] bf16: output vs fp32 reference  (max_abs=0.00051)
[PASS] int8: last-slot K codes within +-2  (max_code_delta=2)
[PASS] int8: last-slot V codes within +-2  (max_code_delta=2)
[PASS] int8: last-slot params ~equal (scale,min)
[PASS] int8: mid slot untouched
[PASS] int8: output vs dequant reference  (max_abs=0.00056)
[PASS] int8: pool changed under scaled k/v
[PASS] int8: last-slot codes == quant(2*k_last)  (max_code_delta=2)
[PASS] int8: last-slot scale ~= 2x original  (max_abs_diff=0.00012)
[PASS] int8: last-slot min ~= 2x original  (max_abs_diff=0.00000)
[PASS] zero k/v: last slot zeroed (contract violation only)
[PASS] bench: bf16 pool byte-identical after 13 dispatches
```

原始输出:`T1_verify_b304_fix.out`。

## T2 bench 门禁(`benchmark/bench_attn_decode.py --kv-quants 0 8 --bsz 4 --klen 256 --iters 20 --warmup 5`)

exit=0。观测行:

```
[check] kv_quant=0 tm_max_abs=0.0005 pt_max_abs=0.0005 -> ok
[check] kv_quant=8 tm_max_abs=0.0005 pt_max_abs=0.0010 -> ok
[0] B=   4 klen=  256  TM   0.0145 ms (  289.7 GB/s)  PT   0.0861 ms (   48.7 GB/s)  PT/TM x5.948
[8] B=   4 klen=  256  TM   0.0154 ms (  136.5 GB/s)  PT   0.0940 ms (   22.3 GB/s)  PT/TM x6.122
```

(附带观测:启动日志有 `fast_hadamard_transform not installed, falling back to matmul-based Hadamard transform` 的 INFO,与检查无关。)
原始输出:`T2_bench_gate.out`。

## T3 缺陷再现(修复前 binding)

**执行障碍与替代步骤(如实记录)**:操作单 prescribed 的
`git stash push -- src/turbomind/python/attn_bench_bind.cpp benchmark/bench_attn_decode.py benchmark/b304_debug/sanitizer_entry_b304.py`
失败:`error: could not write index`——共享树的 git index 含 14 个 `AA`(both added)未合并条目,全部位于 `lmdeploy/serve/parsers/*`(他人进行中的 cherry-pick/rebase,非本票文件;无 MERGE_HEAD)。失败未产生任何 stash 条目、未改动任何文件(事后 `git diff --stat` 三文件为空确认)。

**替代执行**(等效且不触碰共享 git 状态):
1. `git show 45e9322c^:src/turbomind/python/attn_bench_bind.cpp` 落盘为 `attn_bench_bind.prefix.cpp`,其 sha256(前16位 `78ac4d41a2ef2f16`)与 `45e9322c^` 原文件逐字节一致;
2. 手动将该内容替换进 `src/turbomind/python/attn_bench_bind.cpp`(HEAD 版份备份于 `attn_bench_bind.head.cpp`;T3 仅运行 repro 脚本,故只回退 binding 一个文件,操作单 stash 的另两个脚本不参与 T3);
3. `cmake --build build --target _turbomind -j$(nproc)` → exit 0(19s,1 个 TU 重编 + 链接);
4. 运行 `benchmark/b304_verify/repro_b304_defect.py`,观测:

```
last slot  (ti=99) : code bytes changed = 1024/1024, param values changed = 32/32
mid slot   (ti=50) : code bytes changed = 0/1024, param values changed = 0/32
last slot fully zeroed after call: True
DEFECT REPRODUCED
EXIT=0
```

5. 立即还原 HEAD 版 binding 并重建(exit 0,19s);还原有效性双重确认:`git diff --stat` 空 + 重跑 T1 全 PASS。

**T3 过程异常记录**:末位(ti=99)1024/1024 code 字节与 32/32 param 值全部被改写且**全零化**;中位(ti=50)0 改动。原始输出:`T3_repro_prefix.out`。

## T4 sanitizer(`benchmark/b304_debug/sanitizer_entry_b304.py`,重建后运行)

| 工具 | ERROR SUMMARY | exit | 明细 |
| --- | --- | --- | --- |
| initcheck | **0 errors** | 0 | 无 uninitialized 观测;脚本尾行 `SANITIZER ENTRY B304 DONE` |
| memcheck | **4 errors** | 9(`--error-exitcode 9` 触发) | 4 条均为 **CUDA API 层** `cudaErrorInvalidValue`("invalid argument"):2× `cudaFuncSetAttribute` + 2× `cudaGetLastError`;调用栈同源:`registry.cu:29`(Registry 构造内的 lambda)← `Registry::instance()`(registry.cu:107)← `dispatchDecoding<__nv_bfloat16>`(decoding.cu:33)← `TmAttnDecode`(attn_bench_bind.cpp:421)。**无任何 device 侧访存错误**(无 invalid read/write、无 OOB) |

原始输出:`T4_initcheck.out`(全文)、`T4_memcheck.out`(含完整栈)。

## 汇总(纯观测)

- T1:13/13 PASS,exit 0
- T2:kv0/kv8 两门禁 `-> ok`(tm_max_abs 均 0.0005),exit 0
- T3:旧 binding 下末位 1024/1024 code + 32/32 param 被改写且全零化,中位不动,`DEFECT REPRODUCED`,exit 0;新 binding 恢复后 T1 复跑全绿
- T4:initcheck 0 errors / exit 0;memcheck 4 errors / exit 9,全部为 `cudaFuncSetAttribute`/`cudaGetLastError` 的 `cudaErrorInvalidValue` API 告警,无访存类错误
- 执行偏差 1 处:T3 的 stash 步骤因他人未合并 index 无法执行,改用字节等效的手动文件替换(已留痕并完整还原)
- 树收尾:HEAD 三个涉票文件与 `982f4e13` 逐字节一致;`_turbomind` 模块为修复版;本目录外无新增文件;未 commit / 未 push
