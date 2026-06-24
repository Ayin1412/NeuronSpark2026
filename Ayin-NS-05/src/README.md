# NS-2026-05 声场巡检员 - 复现代码说明

本目录包含了本题的完整复现代码及环境依赖说明。

## 1. 目录内容

- **main.py**：核心复现脚本，包含音频特征提取、模型训练、融合与后处理预测的完整流程。
- **requirements.txt**：代码运行所需的 Python 依赖库列表。

## 2. 复现准备

### 1. 安装环境依赖
在终端中执行以下命令安装运行所需的依赖：
```bash
pip install -r requirements.txt
```
*(推荐使用干净的 Python 3.12.11 虚拟环境进行安装)*

### 2. 准备数据与权重
在运行脚本前，请准备好以下数据和权重：
1. **数据集文件**：将 `train.csv` 和 `test.csv` 放置在项目根目录（即 `writeup/` 目录下）。
2. **微观特征文件**：将官方提供的特征文件夹 `features/`（包含 `train_audio_features.csv` 和 `test_audio_features.csv`）放置在项目根目录下。
3. **音频数据**：准备好对应的训练集和测试集音频文件夹。
4. **PANNs 预训练权重**：
   - 官方预训练权重文件：`Cnn14_mAP=0.431.pth` (大小约 312MB)
   - 可在 [PANNs 官方发布页](https://github.com/qiuqiangkong/audioset_tagging_cnn/releases/download/32000Hz/Cnn14_mAP%3D0.431.pth) 下载。
   - 下载后请放置在本地任意目录。

### (3) 修改 main.py 中的路径配置
打开 `src/main.py`，将以下修改为实际的数据路径：
```python
TRAIN_AUDIO_DIR = "Path/to/train/audio"  # 替换训练集音频文件夹路径
TEST_AUDIO_DIR = "Path/to/test/audio"    # 替换测试集音频文件夹路径
```
同时，修改 `extract_panns_embeddings` ：
```python
checkpoint_path = r"Path/to/panns/checkpoint"  # 替换 Cnn14_mAP=0.431.pth 实际存放路径
```

## 3. 复现步骤

```bash
python src/main.py
```

## 4. 预期资源消耗
- **显存占用**：特征提取阶段使用 GPU 显存约 1~2 GB。
- **运行时间**：在配备 NVIDIA GeForce RTX 5090 GPU 的环境下，完整特征提取与模型训练总耗时约 10~15 分钟。
