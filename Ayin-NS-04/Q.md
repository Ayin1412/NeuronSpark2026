## 题目描述

# NS-2026-04 分子炼金术：小分子性质预测

## 赛题任务

本题编号为 **NS-2026-04**。

### 任务介绍

你是一名分子性质预测员，需要根据小分子的 SMILES 表示预测其物理化学性质或活性指标。参赛者需要从分子字符串中构造描述符、fingerprint 或图结构，并输出每个测试分子的连续预测值。

你的任务是：

- 解析训练集 SMILES 和目标值。
- 构建能跨 scaffold 泛化的分子表示和模型。
- 对测试分子输出连续预测值。

### 赛题背景

AI for Science 正在成为机器学习的重要方向。小分子性质预测可用于药物发现、材料筛选和实验优先级排序。与普通表格回归不同，分子任务需要处理图结构、官能团、化学骨架和分布外 scaffold。

高分方案通常来自合理的分子表示、scaffold split 验证、特征集成和对异常 SMILES 的稳健处理。

## 赛题数据

在本页下方“相关资源”处下载数据包，或在 https://modelscope.cn/datasets/SteamedFresh/NS-2026-A-data 处获取。

```text
NS-2026-04/
├── train.csv
├── test.csv
├── scaffold_split.json
├── features/
│   ├── hashed_smiles_train.npz
│   └── hashed_smiles_test.npz
├── example_submission/
│   └── results.csv
└── tools/
    ├── check_format.py
    └── make_features.py
```

### 数据说明

`train.csv` 字段：

| 字段     | 类型   | 含义        |
| -------- | ------ | ----------- |
| `id`     | string | 分子 ID     |
| `smiles` | string | 分子 SMILES |
| `target` | float  | 目标性质    |

`test.csv` 不含 `target`。`features/` 提供预提取 hashed SMILES token fingerprint，降低 RDKit 安装门槛。`scaffold_split.json` 给出推荐本地训练/验证划分。

## 正式数据规模

| 文件                      |        规模 |
| ------------------------- | ----------: |
| `train.csv`               |     8500 条 |
| `test.csv`                |     3000 条 |
| `hashed_smiles_train.npz` | 8500 x 2048 |
| `hashed_smiles_test.npz`  | 3000 x 2048 |

公开特征使用 2048 维 hashed SMILES token fingerprint，可直接用于线性模型、树模型或其它表格回归模型。选手也可以自行安装 RDKit、构建 Morgan fingerprint、分子图或使用符合规则的公开预训练分子模型。

本题鼓励使用 scaffold-aware validation。公开包中的 `scaffold_split.json` 提供一个推荐本地验证划分，选手也可以自行构造验证方案。

### 注意事项

- 使用外部预训练模型或外部数据时，必须说明来源、许可证、是否公开和是否含测试标签风险。

## 评测说明

### 提交说明

提交文件必须压缩为 `NS-2026-04-answer.zip`，压缩包内只包含 `results.csv`。

### 提交格式

```
id,prediction
mol_001,0.732
```

格式要求：

- 必须包含且仅包含 `id,prediction` 两列。
- 必须覆盖全部测试分子。
- `prediction` 必须为有限数值。
- 不允许重复 ID、缺失 ID 或测试集外 ID。

### 得分计算

本题为单目标回归任务，总分 800。评分采用“连续得分 + gold-band 阈值满分”口径：

- 普通提交按 RMSE、Spearman 和格式合法性连续计分
- 达到 gold-band 的优秀公开边界建模方案直接记为满分

| 指标          | 分值 |
| ------------- | ---- |
| RMSE 得分     | 500  |
| Spearman 得分 | 250  |
| 格式合法性    | 50   |

评分公式：

```text
raw_score = 500 * max(0, 1 - RMSE / rmse_ref) + 250 * max(0, Spearman) + 50
if RMSE <= 0.25 and Spearman >= 0.96:
    score = 800
else:
    score = raw_score
```

正式题包中 `rmse_ref = 0.80`。评分脚本保证最终分数范围为 0 到 800。

gold-band 的含义是：选手方法已经在隐藏测试集上同时达到足够低的回归误差和足够稳定的排序质量，可视为本题公开边界下的优秀满分解。该阈值不是要求恢复出隐藏 reference 的逐样本精确生成机制；

示例提交只用于说明格式，不代表合理成绩。建议选手使用 `scaffold_split.json` 或自行构造 scaffold-aware validation，不建议随机划分作为唯一验证依据。

### 无效提交情况

以下情况将得到 0 分：

- CSV 无法解析。
- 表头不是且仅不是 `id,prediction`。
- 测试 ID 缺失、重复或出现测试集之外 ID。
- `prediction` 不是有限数值。
- 压缩包内缺少 `results.csv` 或包含文件夹/额外文件。

## 条款

### AI 协助规则

本题 AI 协助等级为 **A1**。

### Writeup 补充要求

获奖候选队伍必须按通用 Writeup 公告提交方法概述、任务理解、关键改进、验证与复现、证据截图和代码包，并提供 AI 使用声明。A1 题若让 AI 接触测试输入并生成最终预测，或让 Agent 托管实验并生成提交，原则上视为违反等级规则。

Writeup 必须说明分子表示方式、本地验证方式，优先使用 scaffold split。必须提供特征生成脚本日志或预提取特征版本 hash，展示目标分布、异常 SMILES 处理和至少 3 个预测失败案例。

如果使用外部预训练模型或外部数据，必须说明来源、许可证、是否公开和是否含测试标签风险。

获奖候选队伍必须提交可复现证据，包括特征生成脚本、训练/推理脚本、环境依赖、随机种子、关键运行日志和 AI 使用声明。若使用 RDKit、分子预训练模型或外部分子库，必须列出版本、来源、许可证和是否可能包含本题测试 SMILES 的标签。

Writeup需包含最终提交压缩包文件 SHA256，指平台上传的 NS-2026-04-answer.zip 的 SHA256。