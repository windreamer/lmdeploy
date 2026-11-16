# 04: repack CLI + tiny-ct 生成器转正

**What to build:** 持有 compressed-tensors W4A16 权重的用户用一条命令转出 AutoAWQ 布局;开发者零下载生成 tiny-ct 测试模型。票 09 的生成器与转换器按正式工具标准转正(进仓库工具/测试工具位),位级往返测试进套件。

**Blocked by:** None (can start immediately)。

**Status:** ready-for-agent

- [ ] ct→AutoAWQ 转换 CLI:位级往返 err=0 测试进套件(sym 补零约定、nibble 序陷阱有防御性测试)
- [ ] tiny-ct 生成器:真实几何(含 glm5 系条目)可配置,零下载,生成 \< 分钟级
- [ ] CLI 端到端冒烟:生成→转换→双路 S1 冒烟一条命令跑通
- [ ] 流式按分片处理声明(单分片内存上界注明),为 190.8GB 真模 repack 预留
