# B3-01 衔接原型判定报告:prefill 写入与 TM int8 decode 读取互通

日期:2026-11-17 | 分支:`graft/w4a16-core` | 设备:3090 (24 GB)
脚本:`benchmark/bench_b301_bridge.py` | 数据:`benchmark/b3_01_bridge_results.json`
状态:**可行(语义/布局层面)**,附一条工具链告警(见 §4)。

## 1. 结论

**prefill 阶段产生的 KV 块可以被 TM int8 decode kernel 正确读取。判定:可行。**
批 3 packed-KV int8 方案(SPEC-B3,decode-only + opt-in CacheBackend)不受此票阻塞。

判据(全部通过):

| 用例         | B×klen        | TM KL(vs bf16 参考) | PT KL  | TM max_abs | PT vs 池反量化 |
| ------------ | ------------- | ------------------- | ------ | ---------- | -------------- |
| prefill-full | 4×256         | 9.0e-4              | 2.0e-6 | 0.223      | 0.0108         |
| prefill-full | 4×1024        | 2.4e-4              | 1.3e-7 | 0.104      | 0.0027         |
| prefill-full | 8×4096        | 5.7e-5              | 8.5e-9 | 0.045      | 0.0007         |
| mixed-chain  | 4×256 (半池)  | 9.0e-4              | 2.0e-6 | 0.222      | 0.0107         |
| mixed-chain  | 4×1024 (半池) | 2.4e-4              | 1.3e-7 | 0.106      | 0.0028         |
| mixed-chain  | 8×4096 (半池) | 5.7e-5              | 9.2e-9 | 0.044      | 0.0007         |

- 门限沿批 1 S1 档:KL ≤ 1e-2,无 NaN。TM 侧 max_abs ≤ 0.223,在批 2 bench 同档容差(0.25)内,为 TM kernel 内部 fp16 累加精度,随 klen 增大收敛(0.223→0.045),不是布局错位。
- `max_abs_vs_dequant`(读者输出 vs 池字节直接反量化参考)PT 侧 ≤ 0.011:PT 读取路把池字节理解为与写入侧完全一致的布局;TM 侧 0.045–0.219 即上述 kernel 精度。
- **混合链路核心断言成立**:同一池前半由 prefill 写入方、后半由 decode 写入方(不同舍入模式,`second_half_writer_verified` 已断言两半字节确实来自不同写入方),两个读者全程正确。

## 2. 互通契约(写入/读取双方必须遵守)

TM 块布局(`block::Layout` L(H2SDQ+H2S2T),int8 档,每物理块 64 token × 8 kv head × 128):

```
[K codes  H*S*D u8][V codes  H*S*D u8][K params H*S*2 bf16][V params H*S*2 bf16]
```

- codes:per-token-per-head 非对称 uint8,`scale=(max-min)/255`,`code=clamp(round((x-min)/scale),0,255)`,fp32 计算。
- params:**每 token 每 head 两个 bf16 = (scale, min)**,TM 反量化 `code*scale+min`。
- 与 PT 现行约定 `(scale, zero)`,反量化 `(code-zero)*scale` **在代数上等价**(min = -zero·scale),codes 字节相同;仅 param 槽第二元素语义不同。prefill 侧只需把 zero 转成 min 再落 TM 池(纯 scatter + 一次乘法,`prefill_write_production` 已示范)。

## 3. B3-04 产品化清单

1. **CacheBackend 声明**:`CacheBackend.build_block_layout` 增加 int8 档(上述几何);块池 `[nb, pool_bytes] uint8`,块指针数组 `char**` 语义沿 `unified_attention_layer.cc Setup()`。
2. **fill 布局参数化**:prefill 保留 PT triton fill(`fill_kv_cache` QuantPolicy.INT8)产 codes+(scale,zero),新增 scatter 适配层把 codes 原样拷入 TM 块池、params 转 `(scale, min=-zero*scale)`。不重写量化 kernel。
3. **rope+量化写入融合**(decode 侧):未来 TM fused write kernel 的输出契约 = §2 布局;舍入模式可与 prefill 不同(实测 ±1 code 差异对两读者无害),但 (scale,min) 槽语义必须一致。
4. **迁移/检查点影响面**:opt-in 模型的 KV 池不再是 PT `[n,S,H,D]+[n,S,H,2]` 标准 → checkpoint 导出/换入需按布局转换(或在导出时反量化为 PT 标准格式);PD 分裂无影响(prefill 写 + decode 读同池已验证);非 opt-in 模型零影响。
5. **门限**:接受批 1 S1 档 KL ≤ 1e-2;TM 侧 max_abs 预算 0.25(×0.5 幅度档)。勿用「输出 argmax 一致率」作 int8-KV 门限——输出各行近并列,argmatch 波动大且无判别力。

## 4. 工具链告警(重要,移交 B3-04 前置确认)

本机 triton 3.7.1(该 venv 自带 build)存在**误编译**,已在原型中取证:

- 自写 fill kernel(1D grid + launcher 预计算索引)大偏移向量 store 大面积丢失;`tl.program_id(1/2)` 返回垃圾;kernel 内整数 div/mod 链结果错误(probe:128 程序仅 ~16–32 个的 store 可见)。
- **生产 kernel 同样中招**:`fill_kv_cache`(QuantPolicy.INT8,3D grid)在 B4/klen256 下 16 个物理块仅 6 个写入有效 params,其余全零(`production-fill-anomaly` 行,见 results json)。
- 含义:本票的「PT 真实 triton fill → TM 池」端到端用例在本机**无法可信执行**,判定改由 host-op 写入方(同一量化契约逐字节等价实现)承载;契约层面的互通结论不受影响。
- 移交项:B3-04 实现在健康工具链(与批 2 bench 相同环境)上复跑 `--check` 类用例,确认生产 fill kernel 完好后再接 scatter 适配层。

## 5. 复现

```bash
python benchmark/bench_b301_bridge.py            # 全部用例 + 异常探针
# 输出:benchmark/b3_01_bridge_results.json
```

零下载、不动引擎源码、张量直喂。未 commit。
