# B3-04 盲态复核观测记录(只记观测,不含判定)

执行者: BlindReverifyH6 · 2026-11-17 · 仅依据 `benchmark/b304_verify/REPORT.md` §6 操作单,未读 §1–§5 结论。

## 环境观测(执行前)

- HEAD = `5c26c52a` ("test(kv): fix mixed-chain cu-table stride mismatch; un-xfail after binding fix");被验证 commit `45e9322c` 为 HEAD 祖先(`git merge-base --is-ancestor` 通过)。
- 目标三文件(`src/turbomind/python/attn_bench_bind.cpp`、`benchmark/bench_attn_decode.py`、`benchmark/b304_debug/sanitizer_entry_b304.py`)在执行前与 HEAD 一致,无未提交改动。
- 工作树存在**与本任务无关**的他人改动:`lmdeploy/serve/parsers/**` 13 个文件处于 unmerged (AA) 状态、`lmdeploy/pytorch/backends/cuda/op_backend.py` staged、`.scratch/tm-gemm-graft/ROADMAP.md` 修改。全程未触碰。
- GPU: RTX 3090;执行期间 `nvidia-smi --query-compute-apps` 可见 4 个外部进程(17/75/374/23 MiB),`memory.used ≈ 1093–1106 MiB, util ≈ 16–17%`。存在他人负载,非独占。
- `.so` 位置:`lmdeploy/turbomind/_turbomind.cpython-313-x86_64-linux-gnu.so`(构建由 ninja 直接落位,未手动拷贝)。注意 build 日志中的链接绝对路径为 `/home/windreamer/codebase/lmdeploy/...`(cwd 为 `/data/...`,同一棵树)。

## T1 修复面验证

命令:

```
.venv/bin/python benchmark/b304_verify/verify_b304_fix.py; echo "expect exit=0"
```

原始输出(exit=0,Wall time 2.18 s,完整 13 行 PASS):

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
ALL CASES PASSED
```

观测值汇总:exit=0;bf16 max_abs=0.00051;int8 max_abs=0.00056;int8 code_delta 上限观测 2(K、V、scaled 重量化三处均 =2);scale 重标定 diff=0.00012,min 重标定 diff=0.00000;bf16 pool 逐字节一致。

## T2 bench 门禁

命令:

```
.venv/bin/python benchmark/bench_attn_decode.py --kv-quants 0 8 --bsz 4 --klen 256 --iters 20 --warmup 5
```

原始输出(exit=0,Wall time 3.53 s;首行 `fast_hadamard_transform not installed, falling back to matmul-based Hadamard transform` 为环境提示):

```
[check] kv_quant=0 tm_max_abs=0.0005 pt_max_abs=0.0005 -> ok
[check] kv_quant=8 tm_max_abs=0.0005 pt_max_abs=0.0010 -> ok
[0] B=   4 klen=  256  TM   0.0144 ms (  291.3 GB/s)  PT   0.0977 ms (   42.9 GB/s)  PT/TM x6.788
[8] B=   4 klen=  256  TM   0.0149 ms (  140.9 GB/s)  PT   0.0911 ms (   23.0 GB/s)  PT/TM x6.125
```

观测值汇总:tm_max_abs = 0.0005(kv_quant=0)、0.0005(kv_quant=8);两行 check 输出 `ok`。

## T3 缺陷再现(修复前 binding)

**操作单偏差记录(重要)**:操作单的 `git stash push -- <三文件>` 在本树不可执行,原因有二,均如实观测:

1. 三文件与 HEAD 一致且修复已提交(`45e9322c`),工作树无 diff 可 stash(即使 index 可写,stash 也会报 "no local changes to save");
2. index 被他人 unmerged 文件阻塞:`git stash push` 实际报 `error: could not write index` + 13 条 `needs merge`(exit=1)。

采用的等价替代(仅动 worktree,不触碰 index):`git restore --source=45e9322c^ --worktree -- <三文件>`;复现后以 `git restore --source=HEAD --worktree -- <三文件>` 还原。两步 restore 均 exit=0,还原后 `git status --porcelain -- <三文件>` 为空。

`git diff --stat 45e9322c^ 45e9322c -- <三文件>`:attn_bench_bind.cpp +133 行级变更、bench_attn_decode.py 48、sanitizer_entry_b304.py 19(即修复前版本确有实质差异)。

重建(修复前源):

```
cmake --build build --target _turbomind -j$(nproc)   # exit=0, 5.65 s, "Linking CXX shared module .../_turbomind...so"
```

.so 时间戳刷新 15:50:25,size 903,640,000 B。

复现运行:

```
.venv/bin/python benchmark/b304_verify/repro_b304_defect.py   # exit=0, Wall time 2.07 s
```

原始输出(末 4 行):

```
last slot  (ti=99) : code bytes changed = 1024/1024, param values changed = 32/32
mid slot   (ti=50) : code bytes changed = 0/1024, param values changed = 0/32
last slot fully zeroed after call: True
DEFECT REPRODUCED
```

观测值汇总:exit=0;日志含 `DEFECT REPRODUCED`;末位槽(ti=99)1024/1024 code 字节与 32/32 param 值被改写且"fully zeroed = True";中间槽(ti=50)0/1024、0/32 未动。

还原 + 重建(修复后源):restore exit=0、三文件 status 干净;重建 exit=0,5.74 s,.so 时间戳 15:51:00,size 903,715,040 B(与修复前 .so 尺寸不同,确认换回修复实现)。

## T4 sanitizer(修复后 binding,构建于 T3 还原之后)

initcheck:

```
compute-sanitizer --tool initcheck --error-exitcode 9 .venv/bin/python benchmark/b304_debug/sanitizer_entry_b304.py
```

观测:exit=0;末 4 行含 `SANITIZER ENTRY B304 DONE`、`========= ERROR SUMMARY: 0 errors`;日志中 `========= ` 前缀行仅 2 条(banner + summary),即 0 条 hit;无 leak 行。Wall time 4.91 s。

memcheck:

```
compute-sanitizer --tool memcheck --error-exitcode 9 .venv/bin/python benchmark/b304_debug/sanitizer_entry_b304.py
```

观测:**exit=9**(`--error-exitcode` 生效);`========= ERROR SUMMARY: 4 errors`;`Program hit cudaErrorInvalidValue (error 1) due to "invalid argument"` 共 4 条:

- 2× `CUDA API call to cudaFuncSetAttribute`
- 2× `CUDA API call to cudaGetLastError`(两条 FuncSetAttribute 后各伴随一条,符合 sticky-error 传播形态)

分类计数:Invalid global/local/shared 访存行 = **0**;leak 行 = **0**;日志中除上述 4 条 API hit 外无其它 hit。Wall time 5.60 s。

最小定位(只读 grep,非断言):`cudaFuncSetAttribute` 调用点位于 `src/turbomind/kernels/attention/kernel_impl.h:73`、`src/turbomind/kernels/gemm/kernel_impl.h:108`、`kernel_impl_sm90*.h`(均设置 `cudaFuncAttributeMaxDynamicSharedMemorySize`,sm90 变体另有 `NonPortableClusterSizeAllowed`)。本次 entry 路径只命中 2 次,与 sm90 文件无关(sm90 内核不会在 3090 上实例化启动);属内核注册期 opt-in 属性调用返回 invalid argument 的 API 层告警,无任何访存违例行。操作单注记中提及存在 4 条既有 cudaFuncSetAttribute 基线告警;本次实际命中 4 条总错误(2+2 形态),与"API 告警、非访存错误"的类别吻合,数值/形态记录如上供比对。

## 内存/GPU 状态(执行后)

`nvidia-smi`: memory.used 1106 MiB, util 17%(与执行前同量级,均含外部进程);无本任务遗留进程。

## 偏差与异常清单

1. T3 字面 stash 命令不可执行(已提交修复 + 他人 unmerged index),改用 worktree-only restore,内容等价(源 `45e9322c^` ↔ stash 语义目标一致),起点/终点树状态均验证干净。
2. memcheck exit=9(ERROR SUMMARY: 4 errors,全部为 cudaFuncSetAttribute/cudaGetLastError API invalid-argument hit,0 访存违例,0 leak)。
3. 其余无异常:无零写(T3 中间槽 0 改写为观测到的隔离性,T1 zero k/v 用例中末位清零为脚本标注的 "contract violation only" 路径)、无垃圾值、无崩溃、无断言失败。

## 产物

- `benchmark/b304_reverify_h6/T1_verify.log`、`T1_exit.txt`
- `benchmark/b304_reverify_h6/T2_bench.log`
- `benchmark/b304_reverify_h6/T3_build_prefix.log`、`T3_repro.log`、`T3_build_postfix.log`
- `benchmark/b304_reverify_h6/T4_initcheck.log`、`T4_memcheck.log`

未 commit、未 push;树已还原至执行前状态(三文件 clean)。
