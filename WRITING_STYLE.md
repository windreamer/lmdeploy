<!--
Copyright (c) OpenMMLab. All rights reserved.
-->

# WRITING_STYLE.md — 文字风格规范

本规范约束本仓库全部中文文档与代码注释的文字风格，写作助手与开发者必须遵守。规范取材 ASD-STE100（Simplified Technical English），按中文改写。

## 适用范围

- 仓库根、`docs/`、`autotest/`、`scripts/` 下全部中文 `.md` 文件。

上游英文文档不受本规范约束。代码注释一律使用英文（见 `AGENTS.md` 第四节），不受本规范约束。

## 一、同义词

同义词只使用指定词：

| 指定词 | 禁用词     |
| ------ | ---------- |
| 使用   | 采用、运用 |
| 检查   | 检验、核查 |
| 如果   | 若、假如   |
| 以及   | 及、与     |

- 「若干」改为「数个」。
- 「与」「及」作介词或固定搭配（如「与…相比」）时保留。
- 术语只用 `GLOSSARY.md` 的写法。新概念先补入 `GLOSSARY.md`，再写入文档。

## 二、句式与段落

1. 每句一个意思。只用主动语态，主语必须明确。
2. 单句不超过 40 字。汉字每字计 1，连续英文或数字串计 1，标点不计。代码块、行内代码、链接不计句长。
3. 同一主语的相邻短句可以连成一句。连句必须保持基本文意一致。连成的长句豁免长度限制。
4. 段落必须连贯。一个段落只讲一个话题，段落内不得随意换行，句子按顺序承接。每段不超过 6 句。
5. 禁止反问句、双重否定、成语、比喻。
6. 操作步骤用祈使句，每步只含一个动作。
7. 列表项与表格单元格各自成句，不受段落连贯约束，仍守本节其余规则。

### 示例

✗ 若构建出现异常，应立即予以检查，以免导致较为严重的后果。

✓ 如果构建出现异常，开发者应立即检查 CMake 输出。

✗ 这堪称整个系统中最隐蔽的头号大坑。

✓ 这个缺陷最难定位。缺陷藏在 NCCL 探测顺序里。

## 三、术语

1. 英文术语保留原文，不强行翻译。示例：engine、kernel、CUDA、tokenizer、RUNPATH。
2. 新概念先给出中文定义，再在正文使用。
3. 概念名用常用中文词，不用生僻词。

## 四、英文用语（注释）

代码注释一律英文，用语符合 ASD-STE100。规则手册与机械检查工具在本机 skill：`~/.agents/skills/asd-ste100-skill/`（`references/writing-rules.md` 与 `scripts/ste-lint.py`）。注释场景裁剪如下：

01. 一词一义。一个英文词只表达一个概念，同一概念换文件不得换词。
02. 动词表动作。不用动词派生名词，不造短语动词。用 `start`，不用 `kick off`；用 `read the offset`，不用 `do the offset math`。
03. 只用简单时态与祈使式。禁止完成时。`-ing` 只作技术名词。
04. 主动语态。被动式仅限执行者无关的描述文字。
05. 单句不超过 20 词，一句一个意思。段落不超过 4 句。
06. 写完整句子，不省略主语、动词、冠词。禁止电报式省略（如 `fix bug here`）。
07. 禁止分号。一句话拆成两句。
08. 名词簇不超过 3 个词。
09. 禁止成语、比喻、俚语、营销形容词、新造缩写。缩写首次出现写全称。
10. 技术名词保留原文，按 NVIDIA 或上游文档拼写。示例：Tensor、Tensor Core、CUDA Graph。

### 示例

✗ `// The kernel has been reading one block per iteration; do the offset math and kick off the copy.`

✓ `// The kernel reads one block per iteration. Pack weights offline so the loader issues two instructions.`

## 五、禁止项目内黑话

1. 不使用设计代号、绰号指代模块、方案或机制。
2. 不使用拟物代称。禁用示例：接缝（中文用「边界」）、前厅、厨房、骨架、脊柱、门（验收门、静态门）、卡（准入卡、价值卡）。
3. 引用既有设计时，用一句平实描述说明它是什么。
4. 机制首次出现时，先定义，后使用。

## 六、内容边界

1. 设计文档只描述现状与目标形态。现状部分的读者无需其他材料即可理解。
2. 变更叙事、调查过程、决策讨论不入正文。
3. 文档不写日期批注。拍板、追记、修复记录等变更叙事一律进 HISTORY 对应日期条目。

## 七、自检

提交前执行机械检查。命令自组文件清单，不硬编码具体路径。

```bash
DOCS=$(find docs autotest -name '*.md' 2>/dev/null; echo AGENTS.md WRITING_STYLE.md GLOSSARY.md HISTORY.md)

# 1. 禁用词与黑话。GLOSSARY.md、WRITING_STYLE.md 自身是定义处，
#    HISTORY 分册是记录处，三者排除。
grep -nE '采用|运用|检验|核查|假如|若干|接缝|前厅|厨房|骨架|脊柱|验收门|静态门|准入卡|价值卡' \
  $(echo "$DOCS" | grep -vE '^HISTORY/|GLOSSARY\.md|WRITING_STYLE\.md')

# 2. 变更叙事：日期批注、旧版对照、拍板记录。HISTORY 分册是记录处，
#    规范文件是规则定义处，两者排除。
grep -nE '已拍板|拍板|现状注|此前|旧版|原先|曾经|已迁移|迁移中|旧名|旧称' \
  $(echo "$DOCS" | grep -vE '^HISTORY/|GLOSSARY\.md|WRITING_STYLE\.md|AGENTS\.md')

# 3. 注释密度：单文件注释行占比上限 8%（注释一律英文，只查密度）。
for f in $(find lmdeploy src scripts autotest -name '*.py' 2>/dev/null); do
  n=$(wc -l < "$f"); c=$(grep -cE '^\s*#' "$f")
  [ "$n" -gt 60 ] && [ $((c*100/n)) -gt 8 ] && echo "$f: $((c*100/n))%"
done
for f in $(find src -name '*.cc' -o -name '*.cu' -o -name '*.h' 2>/dev/null); do
  n=$(wc -l < "$f"); c=$(grep -cE '^\s*//' "$f")
  [ "$n" -gt 60 ] && [ $((c*100/n)) -gt 8 ] && echo "$f: $((c*100/n))%"
done

# 4. 注释 STE 检查：只查本次改动的文件（上游注释未全量合规，改到即合规）。
STE=~/.agents/skills/asd-ste100-skill/scripts/ste-lint.py
CHANGED=$(git diff --name-only HEAD 2>/dev/null; git ls-files -o --exclude-standard 2>/dev/null)
: > /tmp/ste_comments.txt
for f in $CHANGED; do
  case "$f" in
    *.py) grep -hE '^\s*#' "$f" >> /tmp/ste_comments.txt ;;
    *.cc|*.cu|*.h|*.cpp) grep -hE '^\s*//' "$f" >> /tmp/ste_comments.txt ;;
  esac
done
python3 "$STE" /tmp/ste_comments.txt
```

「若」字单独出现在代码标识符（如 `slot`）时不计。人工检查段落连贯、句长、语态、主语、比喻，以及文档内相对链接与反引号路径是否可解析。

已有文档尚未全部合规。作者修改某文档时，应同时把该文档改为合规写法。不要求一次性重写。
