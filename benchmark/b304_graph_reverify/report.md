# B3-04 图捕获争议 — TM int8 decode binding 独立验证报告

**分支**: `graft/w4a16-core` · **GPU**: RTX 3090(独占)· **环境**: torch 2.12.1+cu130, CUDA driver 13.0, compute-sanitizer (CUDA 13)
**目录**: `benchmark/b304_graph_reverify/`(全部脚本 + `*_results.json` + sanitizer 日志;`lmdeploy/pytorch/` 未改动,未 commit)
**日期**: 2026-11-17

______________________________________________________________________

## 判定:CONFIRMED(结论成立,机制修正)

**主张结论「图内必须回落 Triton」成立**:在当前 `tm_attn_decode` binding 下,任何 Python 层的缓冲契约(含 NSA 式常驻缓冲 + 每步 device 端改写)都无法使图内 TM decode 正确工作。

**主张机制「无法安全消费 capture metadata (padded buffers)」不准确**,实测失败点在 binding 内部(见 §3 机制),且 padded-metadata 的语义危害在 **eager 下同样存在**(§2 E2a),并非 capture 特有。`int8_kv.py::_can_tm_decode` 的 capture 回落判断(`is_current_stream_capturing`)应保留,但其 docstring 给出的理由("padded capture buffers the binding cannot consume safely")只覆盖次要因素;真正的主因见 §3。

**关键反直觉事实:capture 不会失败**。`torch.cuda.CUDAGraph` 捕获 TM decode binding 在本栈上**成功**(无 `StreamCaptureInvalidated`),产出的是**静默错误**的图——这比捕获失败更危险,也意味着另一 agent 的"图捕获失败"经历若指 capture 报错,则与本机行为不同(可能为旧驱动/torch 或不同路径);若指回落实测,则与本验证一致。

______________________________________________________________________

## 1. E1 EAGER 基线 — 绿(复现已知状态)

```
命令: python e1_eager_baseline.py
```

| 对照                       | max_abs  | KL      | 判定       |
| -------------------------- | -------- | ------- | ---------- |
| vs bf16 参考(192 keys/seq) | 0.2324   | 1.02e-3 | ✅ ≤0.25   |
| vs pool-dequant 参考       | 0.000502 | 8.46e-9 | ✅(近位级) |

prefill(`tm_process_kv`)→ pool → `_tm.tm_attn_decode`(新签名,k/v rows)链路正确。

## 2. E2 CAPTURE-NAIVE(争议路径复现)— capture 成功,replay 不健全

```
命令: python e2_capture_naive.py
```

**e2b-exact(精确 metadata, max_k_len=192, 所有序列真长 192)**:

- capture: **OK**(两次 warmup + `torch.cuda.graph` 捕获均无异常)
- replay vs 同参数 eager control(输入位级相同): **max_abs=0.152,非位级** — 图回放与直接执行同一调用结果不同
- mutation probe(仅改 q/k/v rows 内容后 replay): replay vs 新内容 eager: max_abs=0.149;vs 新内容 bf16 参考 0.168(≤0.25,**假绿**——预算门无法识别陈旧,见 §4)

**e2a-padded(padded max_k_len=256,真长 192,镜像生产 padded max_kv_seqlen)**:

- eager padded dispatch vs 真长参考: max_abs=0.185、KL 8.8e-4 — **eager 下即超 0.25 预算(语义危害非 capture 特有)**
- **pool 尾部写破坏**: padded dispatch 把最新 token rows re-store 到 `ti = max_k_len-1 = 255`(超出真长),pool 尾块出现 8384 个非零字节(`eager_tail_nonzero_bytes: 8384`)——padded metadata 经此 binding 会**写坏 pool**,同样与 capture 无关
- capture: OK;replay vs eager: 0.116(同样不健全)

## 3. 失败机制(三项,均已实证)

### 3.1 binding 内部 metadata 是"页栈备份",replay 重读已死栈(因果证明)

`BuildDecodePlan`(`src/turbomind/python/attn_bench_bind.cpp:135-160`)每次调用从**栈上数组** `f/qo/ko` 做 3 次 H2D `cudaMemcpyAsync`。捕获时这些节点被固化为指向栈地址的 memcpy 节点(图节点直方图:8× `memcpy_from_symbol`,见 §3.3)。

```
命令: python e2b_stack_probe.py
P0 kernel 确定性:      eager×2 同输入 → bitwise True
P1 replay 确定性:      replay×2 无干扰  → bitwise True
P2 干扰@256:           两次 replay 之间仅插入一次 eager binding 调用(max_k_len=256,
                       写入 sink pool,pool1 尾部字节零变化=设备状态未动),
                       下一次 replay 输出改变: B_vs_A_max_abs = 0.1157
P3 干扰@192:           再插一次 eager@192 → replay 再变: C_vs_A=0.152, C_vs_B=0.037
```

纯 host 栈扰动(设备状态位级不变)即改变 replay 输出 → **replay 在消费易变宿主栈**。

### 3.2 plan 缓冲在捕获区内被 cudaFreeAsync 释放 → replay 指向悬空/回收分配

图节点直方图(`python e3d2_graph_introspect.py`):

```
tm-binding 图: 36 节点 {kernel:12, free:8, memcpy:6, memset:2, memcpy_from_symbol:8, alloc:0}
torch 控制图:  15 节点 {kernel:14, memcpy:1},replay 正确(sink_all_two=True)
```

`TmAttnDecode` 返回时 8 个 plan 张量(qkv/finished/cu_q/cu_k/readonly/split_cnt/partial_O/partial_ML)经 `cudaFreeAsync` 记录为 **8 个 free 节点,而分配节点为 0** → 每次 replay 结束即释放、且无重分配节点;replay 间任何 eager TM 调用会回收复用这些地址。这解释了 E3 的"replay 惰性"(§3 之下):捕获后的 replay 从不写 out 缓冲、从不跟随新内容。

### 3.3 binding 无 device 端长度接口 → 前缀增长不可表达

`TmAttnDecode` 硬编码 `cu_q_len_arg/cu_k_len_arg = nullptr` → 内部回退 `ko[i]=i*max_k_len`(host 常量,捕获时烘焙)。内核本身支持 per-seq 长度(`attention_universal.h:398` 运行时读 device `cu_k_len`),但该 binding 不暴露。因此**即使调用方全部满足常驻缓冲契约,每步 seqlens 也无法进入图**;唯一随 replay 更新的是 q/k_rows/v_rows/pool 内容(D2D memcpy2D 节点),长度恒为烘焙的 max_k_len(所有序列统一)。

### 3.4 E3 CAPTURE-STABLE(核心判定实验)— replay 惰性实锤

```
命令: python e3_capture_stable.py --phase all --rounds 6
```

常驻缓冲契约完全满足(q/k/v rows、out、block table、pool 全部捕获前分配,指针表在捕获区内 device 重建):

- **每轮 replay 输出位级不变**(`round_over_round_max_abs=0.0`,`content_followed=False`,mutation 4 轮 + growth 6 轮全部如此)— 新 q/k/v 内容完全不进入输出
- D3 探针(`python e3_debug.py`):`out_buf.zero_()` 后 replay,out 保持全零(`nonzero=0`)— **捕获内核节点不写输出缓冲**
- D1/D2 探针:仅改 q / 仅改 k/v rows 后 replay,输出位级不变(O2==O1、O3==O2)
- replay 输出 ≈ warmup 遗留物(陈旧但"接近正确",vs dequant 参考 0.030-0.040 恒定)
- growth 语义观测(报告性):冻结窗口(256)vs 真长参考偏差 0.147-0.182 — 统一 k_len 语义的固有缺口
- **方法论警示:所有轮次的 0.25 int8 预算门全部"通过"** — 预算门无法区分陈旧 replay;图内验证必须用位级 replay-vs-eager + content-follow 探针

## 4. 正确 redo 要点(供 Main 决策;`lmdeploy/pytorch/` 未动)

图内回落 Triton **当前是必要且正确的**。若未来要让 TM decode 进图,需按序满足:

1. **binding C++ 改造**(前置,不可绕过):
   - 暴露 device `cu_q_len/cu_k_len` 参数,删除栈上 `f/qo/ko` H2D(finished/readonly 改 memset 或常驻零缓冲);
   - plan workspace(qkv/split_cnt/partial_O/partial_ML 等)改为调用方/上下文持有,捕获区内零 `cudaFreeAsync`(消除 8 free/0 alloc 形态);
   - 或提供 capture-plan API(捕获前一次 plan,捕获区只 launch)。
2. **metadata 契约**(Python,照 NSA builder 模式):常驻 block_offsets/seqlens 缓冲捕获前分配、每步 device 端填写;指针表捕获区内 device 重建(现路径已证明 sound,见 D0 前 pointer-table 部分与 torch 控制图)。
3. **padded max_k_len 禁止直达现 binding**(eager 同样):必须先有 per-seq masking(device cu_k_len),否则 padding keys 进 softmax + `ti=k_len-1` 写坏 pool 尾(8384B 实测)。
4. **k/v rows re-store 通路可用**:captured memcpy2D 对常驻 rows 缓冲内容更新有效(独立于上述缺陷),`_tm_last_new_kv` stash 模式保留,但 stash 必须是图的持久输入缓冲。
5. **验证门( redo 验收)**:①replay-vs-eager 位级;②每轮 content-follow delta ≥0.005;③host 干扰不变性(无关 eager dispatch 前后 replay 位级不变);④pool 尾完整性;⑤不得以 0.25 int8 预算作为图内正确性门。
6. **关联发现(非本判定驱动,建议上游跟进)**:纯 torch 的 ATen masked-scatter writer 捕获图在特定进程分配形态下 replay 位置错乱(见 `e3_contract_anomaly.py`,可复现:r0 位级正确,r1+ 喷洒 12574/18862 vs 应 8382/8384,或陈旧内容);孤立形态与仓库自带 `test_aten_writer_fill_is_capture_safe` 均绿。生产"图 replay + eager 步交错"场景若依赖图内 ATen fill,需先复核此隐患。

## 5. E4 SANITIZER

```
命令: compute-sanitizer --tool memcheck --error-exitcode 9 python e3_capture_stable.py --phase all --rounds 3
结果: ERROR SUMMARY: 4 errors — 全部为 eager 路径固有的 cudaFuncSetAttribute invalid-arg
      (及配对 cudaGetLastError 吞没),无任何内存错误/越界/非法访问;capture+replay 全程干净

命令: compute-sanitizer --tool racecheck --error-exitcode 9 python e3_capture_stable.py --phase all --rounds 2
结果: RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) — mis-scatter 非数据竞争

命令: compute-sanitizer --tool initcheck python e1_eager_baseline.py
结果: ERROR SUMMARY: 0 errors
```

## 6. 证据文件

| 文件                                                      | 内容                                                 |
| --------------------------------------------------------- | ---------------------------------------------------- |
| `e1_eager_baseline.py` / `e1_results.json`                | EAGER 基线                                           |
| `e2_capture_naive.py` / `e2_results.json`                 | 争议路径复现(exact/padded × capture/replay/mutation) |
| `e2b_stack_probe.py` / `e2b_results.json`                 | P0-P4 栈干扰因果探针                                 |
| `e3_capture_stable.py` / `e3_results.json`                | 常驻缓冲契约(mutation/growth/contract 三相)          |
| `e3_debug.py`, `e3d2_graph_introspect.py` / `e3d*.json`   | 惰性/写失效探针 + 图节点直方图                       |
| `e3_contract_anomaly.py`                                  | ATen writer 布局敏感回放异常独立复现                 |
| `memcheck_e3.log`, `racecheck_e3.log`, `initcheck_e1.log` | sanitizer 原始日志                                   |
