# 04: compressed-tensors 加载时转换 + 转换库

**What to build:** 用户加载 compressed-tensors W4A16 权重的模型时,零额外步骤直接工作:weight_loader 检测 ct 键名(`weight_packed`/`weight_scale`/`weight_zero_point`)→ 内存内规范化为 canonical AWQ(复用票 09 位级函数)→ 走现有 AWQ loader → TM GEMM 通道。**不做 CLI**(用户拍板 2026-11-16:加载时转换为默认,离线 CLI 不需要)。

**Blocked by:** 03 的加载路径部分依赖 02(TmGemmLinear)落地;转换库本体与测试无阻塞。

**Status:** done(commit `92278976`,2026-11-16;加载路径集成项**移交票 03**——02 已落地,集成属引擎接线范畴)

- [x] 转换库:`lmdeploy/turbomind/ct_repack.py`(ct→AutoAWQ 可导入模块,流式分片单 shard 常驻,位级往返 err=0 测试进套件,含 nibble 序防御用例;10 passed,CPU-only)
- [x] tiny-ct 生成器:`tests/turbomind/tiny_ct.py`(llama3_8b/glm4_9b/glm52 preset,glm52 全几何 26 shard 实测往返 err=0)
- [ ] 加载路径集成:ct 模型经 pytorch 引擎加载 → TM GEMM 通道金样本门通过 → **移交票 03**(前置清单见票 03 文件)
- [x] 加载期成本声明:流式 per-module 转换,单模块内存上界注明(~3 B/元素 verify 关)
