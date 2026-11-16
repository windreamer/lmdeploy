# 06: `--backend turbomind` 别名兼容层

**What to build:**（延后票,本轮不执行)TurboMind 存量用户零改动续用:`--backend turbomind` 被接受并透明路由 PyTorch 引擎;TM 专有配置键(attn_dp/attn_cp 等)以警告忽略而非报错;deprecation 文档(en/zh_cn)与一次性公告草稿落库。

**Blocked by:** 无原生阻塞票;**重开条件 = PyTorch 引擎能力(含已嫁接 TM kernel)成为 TurboMind 后端超集**(attention/GDN/MLA/MoE 批次落地后)。过渡期 `--backend turbomind` 保持现状指向 C++ 引擎,能力切换由 env 旋钮承载(票 01/02)。

**Status:** deferred(2026-11-16 用户拍板:超集前建别名会把能力回归伪装成兼容)

- [ ] 重开时:别名路由行为断言进套件
- [ ] 重开时:TM 专有配置键警告忽略清单 + 逐键断言
- [ ] 重开时:常用参数语义等价直通断言
- [ ] 重开时:deprecation 文档(en/zh_cn)与公告草稿(按票 07 决议第 5 条)
