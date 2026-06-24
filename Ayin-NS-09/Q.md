## 题目描述

# NS-2026-09 观测之环：隐变量世界模型反事实预测

## 赛题任务

本题编号为 **NS-2026-09**。

### 任务介绍

环形观测站正在排查一组异常网格世界。巡检机器人只能看到身边很小的一块局部视野，传感器还会把许多事件压缩成粗粒度摘要；真正的动力学参数、场效应顺序和隐藏机制组合不会直接给出。你需要扮演世界模型工程师，根据有限 probe 轨迹还原这些隐变量，并预测同一隐藏世界中多个反事实 query 的后果。

“观测之环：隐变量世界模型反事实预测”是一道 **探索型 Hard 题**。每个测试 context 给出同一隐藏世界的完整初始地图、坐标锚点、若干 probe 轨迹、稀疏局部过程观测、probe 终局完整观测、聚合事件传感器，以及多个共享同一隐藏 profile 的反事实 query 动作序列。目标是从公开校准观测中推断一致的世界动力学解释，并预测每个 query 执行后的终局网格、事件标志、事件发生时段、事件发生顺序和终局类型。

你需要完成的核心工作包括：

- 从公开 train/valid 的 probe、query 和标签中学习世界机制与隐变量线索。
- 对 test 中每个 context 建立一致的隐藏机制解释，而不是孤立处理每个 query。
- 对每个 query 的未来动作序列做反事实 rollout，输出终局网格、事件集合、事件时段、事件顺序和终局类型。
- 在只获得平台总分反馈的条件下，通过本地验证、消融和可视化定位模型缺陷。

隐藏 profile 可能影响移动、场效应、实体交互、事件触发、状态转移和终局判定等多个方面。不同隐变量之间可能存在组合效应；同一 context 下多个 query 共享隐藏 profile 和初始隐状态，但未来动作不同，需要跨 query 保持机制解释一致。

### Hard 难度定位

官方不承诺公开边界 exact `1500 / 1500`。高分需要构建或训练能够从公开初始地图、probe 校准轨迹和 probe 终局观测中恢复隐变量的世界模型。

## 赛题数据

在本页下方“相关资源”处下载数据包，或在 https://modelscope.cn/datasets/SteamedFresh/NS-2026-AAA-data 处获取

```text
NS-2026-09/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── frames/
│   ├── train/
│   ├── valid/
│   └── test/
├── label_schema.json
├── DATA_NOTICE.md
├── example_submission/
│   └── results.json
└── tools/
    ├── baseline.py
    ├── check_format.py
    └── visualize_rollout.py
```

数据规模：

| split | context 数 | query 目标数 | 标签状态                                                     |
| ----- | ---------: | -----------: | ------------------------------------------------------------ |
| train |        720 |         1440 | 公开每个 query 的 `label.final_grid`、`label.events`、`label.event_timeline`、`label.event_order`、`label.terminal` |
| valid |        160 |          320 | 公开每个 query 的 `label.final_grid`、`label.events`、`label.event_timeline`、`label.event_order`、`label.terminal` |
| test  |        320 |          640 | 隐藏每个 query 终局目标                                      |

每个样本包含：

| 字段                                               | 类型         | 含义                                                         |
| -------------------------------------------------- | ------------ | ------------------------------------------------------------ |
| `id`                                               | string       | context 级样本 ID                                            |
| `observation_mode`                                 | string       | 固定为 `sparse_egocentric_5x5_plus_event_sensors`            |
| `coordinate_system`                                | object       | 坐标系说明：`x` 为列号、`y` 为行号，原点在左上角             |
| `initial_full_grid`                                | list[string] | 该 context 的完整初始地图和实体位置；test 也公开             |
| `initial_entities`                                 | object       | 从 `initial_full_grid` 解析出的初始实体坐标                  |
| `context_episodes`                                 | list[object] | 同一隐藏世界 profile 下的 probe 轨迹                         |
| `context_episodes[].actions`                       | list[string] | probe 动作序列                                               |
| `context_episodes[].observations`                  | list[object] | probe 观测序列；部分项含 `local_view`，部分项为 sensor-only 观测 |
| `context_episodes[].observed_final_full_grid`      | list[string] | 该 probe 动作序列执行后的公开终局完整地图                    |
| `context_episodes[].observed_final_events`         | object       | 该 probe 的公开终局事件集合                                  |
| `context_episodes[].observed_final_event_timeline` | object       | 该 probe 的公开事件首次发生时段                              |
| `context_episodes[].observed_final_event_order`    | list[string] | 该 probe 的公开前三个首次发生事件顺序                        |
| `context_episodes[].observed_final_terminal`       | string       | 该 probe 的公开终局类型                                      |
| `queries`                                          | list[object] | 同一 context 下多个反事实 query，共享同一个隐藏 dynamics profile |
| `queries[].query_id`                               | string       | 提交 `results.json` 中必须使用的预测 ID                      |
| `queries[].initial_observation`                    | object       | query 初始稀疏局部观测，始终包含 `local_view`                |
| `queries[].initial_full_grid`                      | list[string] | query 对应的完整初始地图，与 context 的 `initial_full_grid` 对齐 |
| `queries[].initial_entities`                       | object       | query 初始实体坐标                                           |
| `queries[].future_actions`                         | list[string] | 需要预测的反事实未来动作序列                                 |
| `queries[].query_horizon`                          | int          | query rollout 步数                                           |
| `queries[].label`                                  | object       | 仅 train/valid 提供的 query 终局标签                         |

公开包不含完整逐步结构化状态、完整 test query 终局答案、隐藏 profile、生成 seed、评分程序或内部答案材料。`initial_full_grid` 和 probe 终局完整观测是选手可见校准信息；`local_view` 是稀疏化后的局部过程视图。sensor-only 观测只提供 sensor、事件增量、`event_timeline_so_far` 和 `event_order_so_far` 等字段。`frames/` 中的 PNG 是 probe/query 初始稀疏局部观测帧；完整地图以 JSON 文本字段提供。

## 评测说明

### 提交说明

提交文件必须压缩为 `NS-2026-09-answer.zip`，压缩包内只包含 `results.json`。

### 提交格式

`results.json` 是 JSON list，每个测试 query 必须出现且只能出现一次。`id` 使用 `test.jsonl` 中 `queries[].query_id`，不是 context 级 `id`：

```json
[
  {
    "id": "wmli_t_0000_xxxxxxxx_q0",
    "final_grid": ["############", "#..A.......#", "..."],
    "events": {
      "goal_reached": false,
      "collision": true,
      "hazard": false,
      "box_on_goal": false,
      "key_collected": true,
      "portal_used": false
    },
    "event_timeline": {
      "goal_reached": "never",
      "collision": "early",
      "hazard": "never",
      "box_on_goal": "never",
      "key_collected": "mid",
      "portal_used": "never"
    },
    "event_order": ["collision", "key_collected", "none"],
    "terminal": "blocked"
  }
]
```

格式要求：

- `id` 必须与 `test.jsonl` 中的 `queries[].query_id` 完全一致。
- `final_grid` 必须是 `12` 行、每行 `12` 个字符，字符集合以 `label_schema.json` 为准。
- `events` 字段必须包含 `label_schema.json` 中列出的全部事件键，取值为 bool。
- `event_timeline` 字段必须包含全部事件键，取值为 `never`、`early`、`mid` 或 `late`，表示该事件第一次发生在 query rollout 的哪个时段。
- `event_order` 必须是长度为 `3` 的 list，每项为事件键或 `none`，表示 query rollout 中前三个首次发生事件的顺序；不足三项用 `none` 补齐。
- `terminal` 必须属于 `goal`、`hazard`、`blocked`、`active`、`timeout`。
- 缺失样本、重复样本、未知样本或非法字段会导致本次提交得 0 分。

### 得分计算

总分 1500，格式完全合法后计入 100 分基础格式分：

| 项目               | 分值 | 说明                                                         |
| ------------------ | ---: | ------------------------------------------------------------ |
| 动态网格准确率     |  225 | 只在提交或隐藏答案出现动态实体/门/钥匙的位置比较 `final_grid` |
| 关键实体位置准确率 |  225 | 从提交网格解析 `A/B/O/K` 的终局位置，与隐藏答案比较；双方都不存在的实体不计分 |
| 事件向量准确率     |  250 | 六个事件键必须整体一致才计为该样本事件正确                   |
| 事件时序准确率     |  250 | 对非 `never` 的事件比较首次发生时段 `early/mid/late`         |
| 事件顺序准确率     |  250 | 比较前三个首次发生事件的顺序，`none` 只作补位                |
| 终局类型准确率     |  200 | 比较 `terminal`                                              |
| 格式合法性         |  100 | 完整、合法、可评分                                           |

```text
score = 100
      + 225 * dynamic_grid_accuracy
      + 225 * entity_accuracy
      + 250 * event_vector_accuracy
      + 250 * event_timeline_accuracy
      + 250 * event_order_accuracy
      + 200 * terminal_accuracy
```

平台正式反馈只展示总分。

## 条款

### AI 协助规则

本题 AI 协助等级为 **A1**。最终测试集推理必须由选手本地可复现方法完成。

### Writeup 补充要求

获奖候选队伍需要提交可复现 Writeup，至少包含：

- 方法概述、任务理解、关键改进、验证与复现、AI 使用声明和必要证据截图。
- 数据文件、代码入口、依赖版本、随机种子、运行命令和环境说明。
- 稀疏局部观测编码、隐变量系统辨识方法、多 query 一致性约束、事件时序/事件顺序建模和 transition model 设计。
- train/valid 验证分数、主要消融、失败案例和至少 5 个 probe-to-query 反事实可视化。
- 若使用额外公开数据、公开预训练模型或开源模拟/视觉组件，说明来源、许可证、使用方式和与 test 标签无关的排查过程。
- AI 工具没有接触完整 test 输入生成最终答案的声明。

Writeup需要提供最终提交压缩包文件 SHA256，指平台上传的 NS-2026-09-answer.zip 的 SHA256。