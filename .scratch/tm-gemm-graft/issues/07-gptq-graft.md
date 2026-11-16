# 07: GPTQ 格式嫁接(批 2)

**What to build:** GPTQ(sym/asym,g64/g128)checkpoint 经加载时转换(零用户步骤)进 TM GEMM 通道;与 AWQ 共用全部下游家族(TmGemmLinear/merged/QKV/TP splitter),零新增 nn 模块。

**Blocked by:** 批 1 全部落地(1e44bb98/bf4f338a/92278976/d7a05121)。

**Status:** done(commit `174a975a`,2026-11-16)

- [x] 转换库 `lmdeploy/turbomind/gptq_repack.py`:打包轴转置(K 沿 vs N 沿)、nibble 序 shuffle(AWQ \[0,4,1,5,2,6,3,7\])、零点偏移 +1 烤入、溢出拒绝、sym 缺 qzeros 合成 0x88888888
- [x] `quant_method='gptq'` → awq family 映射(config.py,先例同 compressed-tensors);fp16 强制
- [x] `normalize_gptq_weights` 接入加载路径与热更 bucket 路径(形状关系判别 GPTQ/AWQ 打包几何)
- [x] S1 双参照(逐元素反量化 matmul + AWQ Triton):argmax ≥0.99、KL ≤1e-2
- [x] 测试:16 CPU(repack,含 pack-order trap)+ 8(S1/路由);回归 80 passed/4 skipped

**遗留**:asym 真实 checkpoint 零点分布未验证(只测合成);GPTQ MoE 不在范围(moe builder 对 gptq 映射后的 awq 走既有 unsupported 报错);真实 checkpoint 的严格 S1 归票 05 harness;统一大 env `LMDEPLOY_TM_KERNELS` 为批 2 立项项,gptq 暂搭 w4a16 语义。
