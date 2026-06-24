## 题目描述

# NS-2026-06 标注净化师：噪声标签清洗与鲁棒分类

## 赛题任务

本题编号为 **NS-2026-06**。

### 任务介绍

一套老旧的校园问答归档系统积累了大量英文讨论样本。历史标注来自弱规则、旧模型和不同标注组，速度很快，但不可避免地留下了系统性错标：有些来源组更容易混淆，有些类别样本偏少，有些弱标签置信度并不可靠。

你是一名标注净化师，需要在不接触原始文本的前提下，利用公开的匿名特征、弱标签和一小部分可信验证集，训练一个鲁棒分类器，为测试样本预测真实类别。

本题定位为 **优化型噪声标签清洗题**。公开数据不提供原始文本，只发布 96 维匿名数值特征。训练集只提供 `weak_label`，该标签包含约 15% 到 25% 的非均匀噪声；可信验证集提供人工校验标签；测试集只提供特征，隐藏标签用于评分。

你的任务是：

- 分析 `weak_label`、`weak_confidence`、`source_group` 和 `annotator_group` 暴露出的噪声结构。
- 使用 `trusted_valid.csv` 设计可靠的验证策略，而不是只相信训练集弱标签。
- 通过样本过滤、重加权、噪声鲁棒训练、类别校准或集成方法，预测 `test.csv` 中每个样本的真实类别。
- 输出符合格式的 `results.csv`。

### 赛题背景

现实中的数据标注常来自历史系统、众包流程、弱规则、旧模型或多来源合并，标签错误并不是独立同分布噪声。高分路线不是简单套一个分类器，而是识别哪些来源、标注器和类别更容易错标，并把小规模可信验证集用作噪声校准依据。

本题重点考察：

- 噪声标签学习与 confident learning 思路。
- 可信验证集驱动的样本重加权、过滤和校准。
- 类别不平衡下的 Macro F1 优化。
- 高优先级类别召回保护。
- 对公开数据来源、特征泄漏和榜单过拟合风险的控制。

## 赛题数据

在本页下方“相关资源”处下载数据包，或在 https://modelscope.cn/datasets/SteamedFresh/NS-2026-AA-data 处获取。

```text
NS-2026-06/
├── train.csv
├── trusted_valid.csv
├── test.csv
├── label_map.json
├── noise_profile_public.json
├── DATA_NOTICE.md
├── example_submission/
│   └── results.csv
└── tools/
    ├── baseline.py
    └── check_format.py
```

### 数据规模

| 文件                | 行数 | 标签状态 | 说明                                     |
| ------------------- | ---: | -------- | ---------------------------------------- |
| `train.csv`         | 6015 | 弱标签   | 可用于训练，但 `weak_label` 含非均匀噪声 |
| `trusted_valid.csv` | 1031 | 可信标签 | 用于本地验证、校准和噪声分析             |
| `test.csv`          | 1547 | 无标签   | 需要提交预测                             |

### 字段说明

`train.csv`：

| 字段              | 类型   | 含义                                  |
| ----------------- | ------ | ------------------------------------- |
| `id`              | string | 匿名样本 ID，不含原始来源和标签       |
| `f000` … `f095`   | float  | 96 维匿名数值特征                     |
| `weak_label`      | string | 带噪声训练标签                        |
| `weak_confidence` | float  | 弱标注器置信度，范围约为 0.36 到 0.98 |
| `source_group`    | string | 匿名来源组，仅表示来源域差异          |
| `annotator_group` | string | 匿名标注器组，仅训练集提供            |

`trusted_valid.csv`：

| 字段            | 类型   | 含义                     |
| --------------- | ------ | ------------------------ |
| `id`            | string | 匿名样本 ID              |
| `f000` … `f095` | float  | 与训练集同空间的匿名特征 |
| `label`         | string | 可信标签                 |
| `source_group`  | string | 匿名来源组               |

`test.csv`：

| 字段            | 类型   | 含义            |
| --------------- | ------ | --------------- |
| `id`            | string | 匿名测试样本 ID |
| `f000` … `f095` | float  | 匿名特征        |

### 标签说明

合法标签在 `label_map.json` 中给出：

| 标签                | 中文说明             | 是否高优先级 |
| ------------------- | -------------------- | ------------ |
| `computer_hardware` | 计算机硬件与外设问题 | 是           |
| `system_software`   | 系统软件与桌面环境   | 否           |
| `mobility_vehicle`  | 交通与车辆出行       | 否           |
| `sports_recreation` | 体育与社群活动       | 否           |
| `medical_health`    | 医疗健康咨询         | 是           |
| `space_science`     | 空间科学与科研动态   | 是           |

高优先级类别会额外计入召回分。

### 补充说明

`noise_profile_public.json` 只提供粗粒度噪声和来源组统计，不包含真实纠错表、原始类别映射或测试标签。

### 本地工具

格式检查：

```bash
python tools/check_format.py example_submission/results.csv --test-csv test.csv --label-map label_map.json
```

公开 baseline：

```bash
python tools/baseline.py --data-dir . --out results.csv
python tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json
```

`baseline.py` 使用弱标签训练一个逻辑回归模型，再按弱标签一致性过滤部分训练样本，并把可信验证集作为高权重样本加入最终训练。该 baseline 只作为入门参考，不代表最优方案。

### 注意事项

- 允许使用公开机器学习库和通用预训练方法。
- 训练弱标签并不等于真实标签，应使用可信验证集检查噪声模式。

## 评测说明

### 题目定位

本题主标签为 **优化题**。本题按固定评分公式连续计分并排名。1200 分是理论上限，组委会不承诺公开数据边界下存在可复现的 exact 1200 / 1200 标准路线。

高分应来自更好的噪声标签清洗、可信验证集校准、类别不平衡处理、鲁棒分类器和集成策略，而不是逐样本恢复隐藏标签。

### 提交说明

提交文件必须压缩为 `NS-2026-06-answer.zip`。

压缩包内必须只包含：

```text
results.csv
```

不要在压缩包中包含文件夹、模型、权重、日志或其他文件。自动评分只读取 `results.csv`。

### 提交格式

`results.csv` 必须包含 1547 条测试样本预测，表头固定为：

```
id,label
tst_00000_xxxxxxxx,computer_hardware
```

要求：

- `id` 必须完整覆盖 `test.csv` 中所有测试样本。
- `id` 不能缺失、重复或出现未知样本。
- `label` 必须属于 `label_map.json` 中列出的 6 个合法标签。
- 行顺序不影响评分。

### 得分计算

总分 1200：

| 项目             | 分值 |
| ---------------- | ---: |
| Macro F1         |  900 |
| 高优先级类别召回 |  200 |
| 格式合法性       |  100 |

```text
score = 900 * macro_f1 + 200 * priority_recall + 100
```

`macro_f1` 按 6 个类别计算宏平均 F1，鼓励各类别均衡表现。`priority_recall` 只统计 `computer_hardware`、`medical_health`、`space_science` 三个高优先级类别的整体召回率。

格式合法时获得 100 分格式分；格式非法时总分为 0。

### 无效提交

以下情况会得到 0 分：

- 压缩包内没有 `results.csv`。
- `results.csv` 表头不是 `id,label`，或包含额外列。
- 存在重复 ID、未知 ID、缺失 ID 或空 ID。
- 标签为空或不属于合法标签集合。
- 文件无法按 UTF-8/UTF-8-SIG CSV 解析。
- 评分脚本无法正常读取提交文件。

### 本地检查

提交前建议运行：

```bash
python tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json
```

该脚本只检查格式和 ID 覆盖，不会给出隐藏测试分数。

## 条款

### 数据使用与外部资源

允许使用公开开源机器学习库、数据清洗库、通用预训练模型、公开论文和教程。

### AI 协助规则

本题 AI 协助等级为 **A1**。

### Writeup 补充要求

获奖候选队伍必须提交 Writeup 和可复现实验材料。Writeup 必须包含方法概述、任务理解、关键改进、验证与复现、AI 使用声明、证据截图和代码包说明。A1 规则下，如果 AI 接触测试输入并生成最终答案，或 Agent 托管实验并生成提交，原则上视为违反等级规则。

本题还必须包括：

- 训练/验证策略，以及为何可信验证集能代表隐藏测试目标。
- 弱标签噪声分析，包括 `weak_label`、`weak_confidence`、`source_group`、`annotator_group` 的使用方式。
- 样本过滤、重加权、标签校准、模型选择或集成策略。
- 本地 Macro F1、高优先级召回、每类 Precision/Recall/F1 和混淆矩阵。
- 至少 5 个疑似错标训练样本的 ID、弱标签、模型判断和处理方式。
- 训练与生成提交的命令、日志、随机种子、依赖版本。
- AI 使用声明，说明 AI 工具、模型、用途、输入边界，以及 AI 是否接触测试输入。

如果 Writeup 无法证明最终提交来自可复现流程，或 AI 使用声明与 A1 规则冲突，组委会可取消该题评奖资格或该题成绩。

Writeup需包含最终提交压缩包文件 SHA256，指平台上传的 NS-2026-06-answer.zip 的 SHA256。