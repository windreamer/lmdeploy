<!--
Copyright (c) OpenMMLab. All rights reserved.
-->

# AGENTS.md — 项目开发与写作规范（强制）

本规范约束 LMDeploy 仓库内的本地开发实践：构建、文档、代码注释。作者以及助手必须遵守。
中文文档与代码注释的文字风格规范独立成文：`WRITING_STYLE.md`（取材 ASD-STE100，按中文改写）。

## 适用范围

- 全部 Python 与 CUDA/C++ 源码的注释。
- 中文文档（仓库根、`docs/`、`autotest/`、`scripts/` 下全部中文 `.md`）适用 `WRITING_STYLE.md`。
- 上游英文文档与社区贡献流程不受本规范约束。

## 一、文档与注释写作规范

文字风格规范独立维护于 `WRITING_STYLE.md`，内容覆盖：同义词、句式与段落、术语、项目内黑话禁用、内容边界、自检。本文件其余各节提到「第一节」或文字风格时，均指该文件。

## 二、Python 环境

1. 全部 Python 命令使用仓库根的 `.venv`（Python 3.13）：`.venv/bin/python`。
2. 依赖安装优先使用 `uv pip install`（作用于 `.venv`）；`uv` 不可用时回退 `.venv/bin/pip`。

## 三、构建与扩展产物

1. 配置：cwd 必须是 `build/`，运行 `bash ../generate.sh`。脚本硬编码 `cmake ..`，从仓库根运行会配置到父目录。追加 CMake 变量写在脚本参数之后。
2. 增量构建：`ninja -C build`。本地开发没有安装步骤。
3. 扩展产物直写源码树（`BUILD_PY_FFI=ON`）：`lmdeploy/turbomind/_turbomind.cpython-*.so` 与 `lmdeploy/turbomind/lib/libturbomind_nccl_stub.so`。仓库根 `import lmdeploy` 天然 editable。禁止为本地开发执行 `pip install .`。
4. 冒烟验证：`.venv/bin/python -c "import lmdeploy; from lmdeploy.turbomind import _turbomind"`。

## 四、注释规范

1. 注释一律使用英文，用语符合 `WRITING_STYLE.md` 第四节。
2. 每个模块文件头写模块职责与几何约定。
3. 非常规选择必须写「为什么」。
4. 禁止注释复述代码。注释写意图、约束、陷阱。
5. 临时调试代码不落库。

## 五、版权与开源合规

1. 新建源码文件必须在首行添加版权头：`Copyright (c) OpenMMLab. All rights reserved.`。Python 使用 `#` 前缀，CUDA/C++ 使用 `//` 前缀。
2. 移植或改编第三方代码时，必须同时满足三条：文件头保留原项目的版权声明与许可证名称；文件头标注改动内容；在 `NOTICE` 登记来源项目、许可证、版权人、对应文件。本仓库暂无 `NOTICE`，首次移植时创建。
3. 仅参考语义、未复制代码的第三方项目，在 `NOTICE` 标注「语义参考」。
4. 本仓库许可证为 Apache-2.0（见 `LICENSE`）。
5. 规范类文档（`AGENTS.md`、`WRITING_STYLE.md`、`GLOSSARY.md`、`HISTORY.md`）建议添加版权头。

## 六、提交流程

1. 涉及 CUDA kernel / triton kernel / 显存读写路径的改动:对**新增或修改的路径**运行 `compute-sanitizer`——memcheck 必跑,双写入方场景加 racecheck,分配初始化相关加 initcheck。限定在改动路径的小形状用例(sanitizer 慢,不跑全量),命令形如 `compute-sanitizer --tool memcheck .venv/bin/python -m pytest <相关测试>`。
2. 按第三节与第四节核对本次改动是否引入新问题(构建、注释)。
3. 运行机械自检(`WRITING_STYLE.md` 第七节)。
4. `ninja -C build` 通过,且第三节第 4 条冒烟验证通过。
5. 提交信息格式:`类型(范围): 摘要`,与上游近期提交一致(示例:`fix(api): ...`、`feat(ascend): ...`)。正文列出改动、验证与结果。

## 七、文档体系

文档与决策记录的分工固定为：文档描述现状，HISTORY 记录决策，注释就地说明意图。

1. 术语唯一来源是 `GLOSSARY.md`。新概念先补入术语表，再写入文档。
2. 决策只进 HISTORY。不设 ADR 目录，不设 ADR 编号。
3. `HISTORY.md` 是索引：读法、分册表、条目写法。决策正文按 `HISTORY/<YYYY-MM>.md` 分册存放。
4. 文档与代码引用决策写 `HISTORY <日期> <标题>`。禁止写 ADR 编号。

## 八、代码修改规范

edit 按内容定位修改点。文本过期时，改动会落错位置。

1. 使用 edit 前先读目标区域。禁止凭记忆修改。
2. 每次只修改一块。一次 edit 只表达一个意图。
3. `old_string` 必须唯一定位。无法唯一时，先补上下文。
4. 修改后重读该区域，确认改动落在目标位置。
