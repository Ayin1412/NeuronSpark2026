# 模型文件清单与说明

本方案使用了预训练的开源音频理解大模型 **PANNs** 中的 **Cnn14** 模型用于提取音频的全局语义 Embedding 特征。

## 模型信息

- **模型名称**：PANNs Cnn14
- **参数规模**：约 80.8 M (80,780,104)
- **权重文件名称**：`Cnn14_mAP=0.431.pth`
- **文件大小**：327,428,481 字节 (~312 MB)
- **SHA256 哈希值**：`6d0eb7406aa6c85426bb9c1950f1450337c7ff4d468165cf45ff29c88283a886`
- **开源许可证**：MIT License
- **官方来源与项目地址**：[GitHub - qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)

## 下载与配置方式

1. **官方直接下载链接**：[Cnn14_mAP=0.431.pth](https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1)
2. **放置路径**：
   下载后，请将权重文件放置于运行机器的 PANNs 默认缓存目录（通常为 `~/panns_data/`）或修改 `src/main.py` 中的 `checkpoint_path` 变量指向该文件的绝对路径。
