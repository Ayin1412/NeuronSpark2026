# NeuronSpark 2026 竞赛解题代码库 — L.Corp

本仓库归档了 **L.Corp** 队伍（队长：`Ayin`）在 **NeuronSpark 2026** 竞赛中完成的全部 10 道题目的解题代码、复现 Writeup、证据截图及最终提交副本。

---

## 1. 竞赛题目与成绩汇总

团队在本次竞赛中完成了全部 10 道题目，涵盖了 OCR 与 NLP 文本分类、时间序列人流预测、基于 RAG 的问答系统、内容安全检测、小分子性质预测（化学计算）、声学事件检测、噪声标签清洗、灵巧手触觉闭环控制、黑箱遥测 AIOps 运维以及隐变量物理世界反事实预测等多元领域。

以下是各题目的成绩及核心方案汇总：

| 题号与子项目链接 | 题目名称 | 官网有效得分 | 满分 | 核心方案与所用技术 |
| :--- | :--- | :--- | :--- | :--- |
| [Ayin-NS-00](file:///d:/NeuronSpark2026/Ayin-NS-00/README.md) | 星图识别员：校园票据分类 | **500.00** | 500 | RapidOCR 文本提取 + `hfl/chinese-roberta-wwm-ext` 序列分类微调 |
| [Ayin-NS-01](file:///d:/NeuronSpark2026/Ayin-NS-01/README.md) | 智慧饭堂：人流与备餐预测 | **489.09** | 500 | 比例（ratio）预测空间转换 + 多维度历史 ratio 统计特征 + LightGBM 树回归 |
| [Ayin-NS-02](file:///d:/NeuronSpark2026/Ayin-NS-02/README.md) | 图书馆问答员：带引用的 RAG 知识库问答 | **745.66** | 800 | BM25 与 Dense Embedding (`Qwen3-Embedding`) 融合检索 + Qwen3.5 严格 JSON 约束生成 |
| [Ayin-NS-03](file:///d:/NeuronSpark2026/Ayin-NS-03/README.md) | 防线之内：Prompt Injection 与内容安全检测 | **742.72** | 800 | 200+条精准正则规则（针对 `user_input` 过滤） + TF-IDF 逻辑回归分类器兜底 |
| [Ayin-NS-04](file:///d:/NeuronSpark2026/Ayin-NS-04/README.md) | 分子炼金术：小分子性质预测 | **622.94** | 800 | 计数型与二值型分子指纹提取 + 骨架分组 5 折交叉验证（GroupKFold）+ LightGBM/XGBoost/CatBoost/MLP 融合 + BayesianRidge Stacking |
| [Ayin-NS-05](file:///d:/NeuronSpark2026/Ayin-NS-05/README.md) | 声场巡检员：校园声景事件检测 | **1200.00** | 1200 | PANNs (`Cnn14`) 全局音频 Embedding + Focal Loss 独立分类器 + Classifier Chain 联合共现建模 + F3-score/强推线自适应阈值优化 |
| [Ayin-NS-06](file:///d:/NeuronSpark2026/Ayin-NS-06/README.md) | 标注净化师：噪声标签清洗与鲁棒分类 | **1031.22** | 1200 | 两阶段置信学习（Cleanlab）数据清洗 + 噪声来源频次特征 + LightGBM/CatBoost 融合 + Nelder-Mead 最优决策阈值搜索 |
| [Ayin-NS-07](file:///d:/NeuronSpark2026/Ayin-NS-07/README.md) | 触觉调度者：灵巧手接触丰富闭环仿真控制 | **812.35** | - | 确定性重放重建高维数据集 + 多模态序列 Transformer 决策 + AWR 离线强化学习 + 卡尔曼/EMA滤波与势场避障后处理 |
| [Ayin-NS-08](file:///d:/NeuronSpark2026/Ayin-NS-08/README.md) | 星云值班台：黑箱遥测世界模型闭环处置 | **1210.85** | 1500 | 故障模式智能拓扑分析 + 拓扑邻居主动探测与过滤 + 动作全局去重机制 |
| [Ayin-NS-09](file:///d:/NeuronSpark2026/Ayin-NS-09/README.md) | 观测之环：隐变量世界模型反事实预测 | **1293.79** | 1500 | 探针结束状态多数投票反推隐变量起点 + 手写确定性网格世界物理模拟器（含冰面滑动、传送门及钥匙开门物理逻辑） |

---

## 2. 仓库目录结构

本仓库的 10 个解题包按题号划分，每个子包均包含独立的复现源码、Writeup 详解、环境配置以及验证证据：

```text
NeuronSpark2026/
├── README.md               # 本文件（仓库全局总览与复现导航）
├── Ayin-NS-00/             # 00题：校园票据分类
│   ├── README.md           # 该题解题思路与详细步骤 (Writeup)
│   ├── src/                # 源码目录 (OCR.py, train.py, inference.py)
│   ├── evidence/           # 平台提交截图及运行日志
│   └── submission/         # 提交结果副本
├── Ayin-NS-01/             # 01题：智慧饭堂人流预测
│   ├── README.md
│   ├── src/                # 源码目录 (main.py 等)
│   └── evidence/
├── Ayin-NS-02/             # 02题：带引用 RAG 知识库问答
│   ├── README.md
│   ├── src/
│   └── evidence/
├── Ayin-NS-03/             # 03题：Agent 内容安全检测
│   ├── README.md
│   ├── src/
│   └── evidence/
├── Ayin-NS-04/             # 04题：小分子性质预测
│   ├── README.md
│   ├── src/
│   ├── models/             # 训练好的核心模型权重
│   └── evidence/
├── Ayin-NS-05/             # 05题：校园声景事件检测
│   ├── README.md
│   ├── src/
│   └── evidence/
├── Ayin-NS-06/             # 06题：噪声标签清洗与分类
│   ├── README.md
│   ├── src/
│   └── evidence/
├── Ayin-NS-07/             # 07题：灵巧手接触丰富闭环仿真控制
│   ├── README.md
│   ├── src/
│   ├── models/             # 离线强化学习训练好的策略权重
│   └── evidence/
├── Ayin-NS-08/             # 08题：黑箱遥测世界模型闭环处置
│   ├── README.md
│   ├── src/
│   └── evidence/
└── Ayin-NS-09/             # 09题：隐变量世界模型反事实预测
    ├── README.md
    ├── src/                # 物理模拟器与预测主程序
    ├── logs/               # 本地评估和推理日志
    └── evidence/
```

---

## 3. 全局运行环境与依赖汇总

各子项目主要基于以下软硬件环境进行开发与运行：

### 硬件环境
- **CPU**: AMD Ryzen 7 9800X3D 8-Core Processor
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB) 或相同级别 GPU（如部分模型使用 RTX 4090 D 训练）
- **内存 (RAM)**: 48 GB 或以上
- **操作系统**: Windows 11

### 软件与核心库依赖
本仓库中的各个模块依赖以下核心 Python 包，具体依赖清单可在对应子目录的 `requirements.txt` 中查看：
- **基础与通用**: `Python 3.12.11`, `CUDA 13.2`
- **传统机器学习**: `lightgbm==4.6.0`, `xgboost>=2.0.0`, `catboost>=1.2.10`, `scikit-learn==1.7.0`, `pandas`, `numpy`
- **深度学习**: `torch==2.7.1+cu128`, `transformers==4.53.2`, `datasets==4.8.5`
- **文本与检索**: `jieba`, `rank_bm25` (BM25Okapi)
- **音频处理**: `librosa==0.11.0`, `panns-inference`
- **化学计算**: `rdkit==2026.3.2`
- **数据清洗**: `cleanlab==2.9.0`
- **物理仿真与控制**: `mujoco==3.3.7`, `scipy==1.15.3`
- **文本 OCR**: `rapidocr-onnxruntime==3.8.1`

---

## 4. 复现指南

本仓库中每一道题目的具体复现步骤均已在对应的子目录 `README.md` 中作了详细介绍。以下是通用的复现逻辑：

1. **准备 Python 环境**：建议在 Windows 系统上为各题建立干净的 Python 虚拟环境，并安装所需要的依赖。
2. **下载或整理数据集**：将官方平台提供的数据集（如 `train.csv` / `.jsonl`、测试集等）拷贝到对应的解题目录中。
3. **进入对应目录复现**：
   - 比如复现 **NS-2026-00 校园票据分类**：
     ```bash
     cd Ayin-NS-00
     pip install -r src/requirements.txt   # 若无 requirements.txt 则安装 README.md 中列出的依赖
     python src/OCR.py                     # OCR文本提取
     python src/train.py                   # 模型微调
     python src/inference.py               # 推理生成结果
     ```
   - 比如一键复现 **NS-2026-06 噪声标签清洗与鲁棒分类**：
     ```bash
     cd Ayin-NS-06
     pip install -r requirements.txt
     python src/main.py                    # 自动完成噪声清洗、训练融合与最优乘子搜索
     ```
   - 复现 **NS-2026-09 隐变量世界模型反事实预测**：
     ```bash
     cd Ayin-NS-09
     python src/solution.py --eval_train --train ../train.jsonl   # 验证训练集
     ```

更多细节请直接查阅对应子项目下的 [README.md](file:///d:/NeuronSpark2026/README.md)。

---

## 5. AI 使用声明

本队在 NeurSpark 2026 竞赛期间使用的 AI 工具包括 **Gemini** 和 **Claude**。
- **主要用途**：资料查询、辅助编写代码、排版优化、消融实验思路拓展、物理逻辑边界排查。
- **使用规范**：所有核心算法结构设计、阈值优化策略、规则去重方案及物理模拟器均由人工深入分析并最终决定，AI 主要扮演了研发助手角色。
- **逐题具体声明**：各道题目对 AI 的具体使用边界均已在子项目 `README.md` 中如实披露。
