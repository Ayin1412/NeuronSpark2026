## 题目描述

# NS-2026-07 触觉调度者：灵巧手接触丰富闭环仿真控制

## 赛题任务

本题编号为 **NS-2026-07**。

### 任务介绍

你需要提交一个以 `agent.py` 为入口的闭环控制包，让灵巧手在隐藏对象、隐藏环境和隐藏物理扰动下完成接触丰富的非抓取式重定位、工具使用和资源约束序贯操作。

隐藏评测器会导入提交包根目录下的 `agent.py`，实例化 `Agent`，每个任务开始时调用 `Agent.reset(task_info)`，随后在最多 128 个控制步内反复调用 `Agent.act(observation)` 获取动作。若提交同时实现 `reset_batch/act_batch`，评测器可以把多个跨任务 rollout 合并调用，隐藏评测的 batch 上限为 64，用于让 CNN/RNN/Transformer 等模型策略更有效地利用远程 GPU。

规则控制器、在线优化器、CEM/MPPI 类方法或小模型策略都可以提交；高分预期来自能够利用触觉时序、空间网格、多阶段目标、接触恢复和跨物理扰动泛化的离线模型或混合策略。本题为 Hard 难度闭环控制题，满分 1500；公开 valid/test 只用于本地 sanity 和调试，不进入正式榜单。

### 你需要解决的问题

- 从低维状态、姿态估计、触觉法向/切向/振动/接触信号、触觉历史、7x4 tactile heatmap、`tactile_image_7x8x8`、`vision_grid_16x16` 空间 token 和警告标志中做闭环决策。
- 在隐藏对象族、摩擦、质量、顺应性、动作延迟、触觉 dropout、姿态噪声、局部遮挡、障碍/窄通道和多阶段目标下保持稳健。
- 在力、手腕预算、保留手指、脆弱物体和接触丢失约束下选择动作，避免过大冲击、无效动作和资源浪费。
- 将离线训练、弱示范学习、规则后处理、世界模型或搜索策略收敛为一个可在禁网远程 4090D 环境中独立运行的 `agent.py + model/` 提交包。

### Hard 难度定位

本题为 Hard 难度闭环控制题，满分 1500。公开 valid/test 只用于本地 sanity 和调试，不进入正式榜单；正式成绩以平台隐藏私有程序生成的 rollout 和聚合评分为准。官方不承诺只凭公开 split 可稳定达到 exact `1500 / 1500`。

## 赛题数据

在本页下方“相关资源”处下载数据，或在 https://modelscope.cn/datasets/SteamedFresh/NS-2026-B-data 处获取。

```text
NS-2026-07/
├── action_schema.json
├── agent_api.md
├── DATA_NOTICE.md
├── DATA_PROVENANCE.md
├── GETTING_STARTED.md
├── HIGH_FIDELITY_BOUNDARY.md
├── RUNTIME_ENVIRONMENT.md
├── requirements-mujoco.txt
├── requirements-dev.txt
├── requirements-runtime.txt
├── task_schema.json
├── tasks/
│   ├── train_tasks.jsonl
│   ├── valid_tasks.jsonl
│   └── test_tasks.jsonl
├── demonstrations/
│   ├── manifest.json
│   └── weak_train_rollouts.jsonl
├── tactile_probes/
│   ├── manifest.json
│   ├── train/
│   ├── valid/
│   └── test/
├── simulator/
│   ├── dexsim_core.py
│   ├── gymnasium_env.py
│   └── mujoco_scene_template.xml
├── example_submission/
│   └── agent.py
└── tools/
    ├── check_format.py
    ├── run_public_eval.py
    ├── gym_smoke.py
    └── render_replay.py
```

### 数据规模

- `tasks/train_tasks.jsonl`：6000 条公开训练/生成任务描述。
- `tasks/valid_tasks.jsonl`：360 条本地 public validation sanity。
- `tasks/test_tasks.jsonl`：360 条本地 public test sanity，不是榜单正式 test。
- `demonstrations/weak_train_rollouts.jsonl`：1200 个弱规则 teacher episode，可用于离线训练或行为克隆起点。
- `tactile_probes/train`：48 个触觉/接触观测调试 probe。
- `tactile_probes/valid`：24 个触觉/接触验证调试 probe。
- `tactile_probes/test`：24 个触觉/接触测试调试 probe。

`tasks/test_tasks.jsonl` 中的 test 是公开本地 sanity split，不是线上隐藏评分任务。隐藏评分会使用私有 procedural rollouts、私有物理扰动和 withheld capability aggregation。`simulator/gymnasium_env.py` 提供 Gymnasium-style 本地训练/调试 wrapper；该 wrapper 不是正式提交 API，正式平台始终加载 `agent.py`。

## 评测说明

### 提交格式

提交文件名建议为 `NS-2026-07-answer.zip`。提交 zip 根目录必须直接包含 `agent.py`，不得把 `agent.py` 多包一层目录。使用模型时，可把模型权重、tokenizer、配置、查表文件和纯 Python 辅助代码放在 `model/`、`models/` 或等价本地目录中，并建议提供 `model_manifest.json` 记录文件名、大小、SHA256、训练来源、依赖和推理入口。

`agent.py` 必须定义：

```python
class Agent:
    def reset(self, task_info: dict) -> None:
        ...

    def act(self, observation: dict) -> dict:
        return {
            "primitive": "wait",
            "finger": "palm",
            "force": 0.0,
            "direction": [0.0, 0.0],
        }
```

可选批量接口：

```python
class Agent:
    def reset_batch(self, task_infos: list[dict]) -> None:
        ...

    def act_batch(self, observations: list[dict]) -> list[dict]:
        ...
```

只实现 `reset/act` 的提交仍然合法，会被顺序评测。若同时实现 `reset_batch/act_batch`，返回动作数量必须与 observation 数量一致。导入失败、缺少 `agent.py`、缺少 `Agent`、返回非法动作、运行异常、超时或资源超限，可能导致该次提交低分、0 分或评测失败，并计入提交次数。

### 动作集合

动作必须符合 `action_schema.json`。必填字段为 `primitive`、`finger`、`force`、`direction`。

- `primitive`：`brace`、`push`、`drag`、`pivot`、`roll`、`lift_edge`、`tap`、`stabilize`、`wait`、`finish` 之一。
- `finger`：`thumb`、`index`、`middle`、`ring`、`pinky`、`palm`、`wrist` 之一。
- `force`：0.0 到 1.0 的数值。
- `direction`：长度为 2 的数值数组。

### 评分

正式评测使用隐藏私有程序生成 rollout、触觉噪声、动作延迟、各向异性摩擦、高维紧凑空间网格、多阶段目标、视觉干扰、资源瓶颈、障碍/窄通道场景和鲁棒聚合评分。官方不承诺只凭公开 split 可稳定达到 exact `1500 / 1500`。

隐藏评测主要考虑：

- 任务目标完成度、最终状态质量和多阶段目标进展。
- 安全约束、力预算、手腕预算、保留手指、脆弱物体和接触丢失惩罚。
- 接触质量、滑移恢复、触觉反馈利用和无效动作控制。
- 资源/效率，包括控制步数、动作强度和重复动作。
- 跨任务、跨对象、跨物理扰动和隐藏 capability tier 的稳健性。

隐藏评分按三个隐含能力层聚合：基础状态反馈控制、触觉滑移/接触恢复、跨任务/跨对象/跨物理扰动泛化，并包含最差能力层短板项。每个任务的 capability label、hard-slice label、private physics 参数、隐藏 rollout id、隐藏分项和最优动作不会返回给选手。

本题保留 hidden evaluator 输出的 **原始评测分 rawScore**，但排行榜和比赛总分使用 **竞赛分 competitionScore**。平台会在每次本题提交完成后，以及定期排行榜同步时，按所有队伍当前本题最佳原始分重算竞赛分。原始评测分不因其他队伍提交而改变；竞赛分可能随榜单场内分布变化而重算。

满分 `F = 1500`。对每支队伍，只取该队本题所有 `COMPLETED` 提交中的最高 `rawScore` 作为当前最佳原始分 `R_i`；`rawScore <= 0` 的队伍本题竞赛分为 `0`。对所有 `R_i > 0` 的队伍按原始分从高到低形成原始分档，相同原始分属于同一档并获得相同竞赛分。第一档排名比例 `p_i = 1`，最后一档 `p_i = 0`；如果当前只有一个正分档，则 `p_i = 1`。设 `g_i = (R_i - R_min) / (R_max - R_min)`；若当前正分档没有差距，则 `g_i = 1`。

竞赛分计算为 `competitionScore = round(F * (0.25 + 0.55 * p_i + 0.20 * g_i), 3)`，并截断在 `0` 到 `F` 之间。排行榜、题目页进度条、总榜总分使用 `competitionScore`；提交详情、提交列表、资源报告和评测日志中会保留 `rawScore`，用于选手判断模型真实改进。总榜若出现相同竞赛分，仍按平台现有的达到当前总分时间等规则排序。

### 平台评测资源与安全边界

评测 sandbox 默认不提供公网、DNS 或内网访问。正式评测期间不得联网下载模型、安装依赖、调用外部 API、调用闭源模型服务、访问托管推理服务或探测平台网络。

本题保证可用的系统环境和第三方包如下。未列出的第三方包即使在当前镜像中偶然存在，也不作为题目承诺，选手不应依赖。

| 包或环境             | 版本或说明     |
| -------------------- | -------------- |
| Ubuntu               | 24.04.2 LTS    |
| glibc                | 2.39           |
| Linux 架构           | x86_64         |
| Python               | CPython 3.12.3 |
| NVIDIA driver        | 580.65.06      |
| PyTorch CUDA runtime | 12.8           |
| `torch`              | 2.11.0+cu128   |
| `mujoco`             | 3.3.7          |
| `numpy`              | 2.4.6          |
| `scipy`              | 1.17.1         |
| `tqdm`               | 4.67.3         |

额外纯 Python 代码应随提交包携带；额外 native binary/wheel 不保证可用，除非与 CPython 3.12、x86_64 Linux 和 glibc 2.39 兼容并能在禁网环境中直接导入。不要依赖 `pip install`、`apt install`、在线模型仓库、远程数据库或任何外部服务。

主要运行资源限制如下。若平台页面或组委会公告给出更新限制，以最新公告为准。

| 项目                         |                                     限制 |
| ---------------------------- | ---------------------------------------: |
| 远程 GPU                     | 1 x NVIDIA GeForce RTX 4090 D，24564 MiB |
| Docker sandbox CPU           |                                    8 CPU |
| Docker sandbox 内存          |                                    48 GB |
| 评测网络                     |            `none`，禁公网、DNS、内网访问 |
| 单次评测总超时               |                                  2700 秒 |
| hidden 控制步数              |                              最多 128 步 |
| hidden batch 上限            |                       最多 64 个 rollout |
| agent 启动超时               |                                    30 秒 |
| `Agent.reset` 单次超时       |                                    12 秒 |
| `Agent.reset_batch` 单次超时 |                                    20 秒 |
| `Agent.act` 单次超时         |                                     2 秒 |
| `Agent.act_batch` 单次超时   |                                    10 秒 |
| 评测日志上限                 |                                   512 KB |

题目页上传控件会按本题配置显示 8192 MB 上限，并通过大提交入口上传；组委会发布的官方提交脚本也使用同一大提交链路。旧的普通 50MB 上传接口不适用于本题。

提交进入远程评测队列前会先完成 zip 结构预检、远程 blob HMAC 鉴权、大小校验和 SHA256 校验。未通过预检的提交不会进入评测队列。zip 内路径不得为空、不得是绝对路径、不得包含 `..`，路径长度不超过 512 字符；不允许 symlink、设备文件、FIFO、socket 等异常条目。大权重文件请拆分为不超过 512MB 的 shard，并随包附带 manifest/hash，不要依赖评测时下载。

| 项目                 |            限制 |
| -------------------- | --------------: |
| 提交格式             |          `.zip` |
| 提交 zip 大小        |  不超过 8192 MB |
| 解压后总大小         | 不超过 16384 MB |
| 文件数量             |     不超过 8000 |
| 任一单文件大小       |   不超过 512 MB |
| 模型/策略资产总量    | 不超过 12288 MB |
| 模型/策略资产文件数  |       不超过 64 |
| 同队同题在途远程评测 |       最多 1 个 |
| 同队同题提交冷却     |           60 秒 |
| 每队每日提交         |      最多 30 次 |
| 每队总提交           |     最多 100 次 |
| 大提交原件保留       |      目标 21 天 |

每日限制从常规 20 次放宽到 30 次，是因为本题允许大模型/策略资产提交，选手可能因误判 GPU 显存、CUDA/ABI 兼容性、模型加载时间、CPU/RAM 占用或 per-call timeout 而浪费评测次数。资源耗尽、导入失败、超时和运行错误仍会计入次数；连续异常或明显资源滥用可能触发人工复核。

平台只返回总分、脱敏资源报告和有限诊断附件。资源报告可能包含运行时长、退出码、sandbox 后端、网络模式、镜像、峰值内存、CPU、GPU 显存、GPU 利用率、preflight 状态、初始 GPU 显存和磁盘余量等摘要，不包含竞争队伍身份、进程号、容器 ID、服务器路径、内部 IP、hidden task id、private rollout 参数或 hard-slice 标签。

诊断附件仅允许 `.json` / `.zip`，最多 3 个文件，总量不超过 64MB，保留 48 小时。官方 hidden replay 附件如启用，只来自不计分 hidden-diagnostic task；该 task 与 hidden scoring task 同属任务语法族，但不参与正式分数聚合。附件不会暴露 scored private rollout id、hard-slice label、hidden capability label、private physics 参数或最优动作，`.mp4` 视频不保证生成也不长期保存。

远程节点会在评测前执行有界 preflight，检查 Docker、运行镜像、禁网/超时配置、存储根目录、磁盘余量、`nvidia-smi`/GPU 可用性和初始 GPU 显存，并可能清理官方 evaluateapp 标记的退出或陈旧容器。它不会杀死任意宿主进程、删除无标签历史容器、重置 GPU driver 或重启节点。

### 本地命令

安装公开 MuJoCo simulator 依赖并运行 sanity：

```powershell
python -m pip install -r requirements-mujoco.txt
python tools/check_format.py example_submission
python tools/run_public_eval.py example_submission --tasks tasks/valid_tasks.jsonl
python tools/gym_smoke.py --tasks tasks/valid_tasks.jsonl --index 0 --steps 8
python tools/render_replay.py example_submission --tasks tasks/valid_tasks.jsonl --index 0 --output-dir replay_debug
```

## 条款

### AI 协助等级

本题按 **A1** 处理。

### Writeup 补充要求

获奖候选必须满足通用 Writeup 的 6 个部分：解题概述、关键改进、验证与复现、AI 使用声明、证据截图和代码包。除此之外，本题还必须披露：

- 状态表示、触觉时序、触觉热图、proxy render 或 spatial grid 特征使用方式。
- 控制、搜索、学习、BC/RL、world model、规则后处理或混合策略方法。
- 训练数据边界，包括是否使用 `train_tasks`、weak demonstrations、tactile probes、公开外部数据或开源模型。
- 模型容量、checkpoint/export 方式、推理入口、`model_manifest.json`、模型文件。
- public valid 分数、本地评估命令、随机 seed、硬件资源、运行时长、峰值内存/显存和主要消融结论。
- 线上提交反馈如何使用，是否触及闭源 API、远程 Agent、正式提交反馈或排行榜反馈。
- AI 使用声明、额外数据来源和许可证排查结论。

选手应自行保存最终提交包、模型 manifest、checkpoint hash、导出包 sha256 和 Writeup 证据。平台大提交原件和诊断附件有保留期限，不作为选手长期归档。

Writeup需要提交最终提交压缩包文件的 SHA256 和压缩包文件内 [agent.py](http://agent.py/) 的 SHA256；若包含 model/ 或 models/等，请在 WP 中列出模型文件清单，建议附各模型文件 SHA256。