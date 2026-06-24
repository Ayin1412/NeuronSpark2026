### 复现步骤
确保代码文件在数据解压包根目录下
```bash
# 1. 安装依赖
pip install rapidocr transformers datasets scikit-learn pandas torch tqdm

# 2. OCR 提取：对训练集和测试集图片提取文本，生成 train_text.csv 和 test_text.csv
python OCR.py

# 3. 训练分类模型：读取 train_text.csv，微调 chinese-roberta-wwm-ext，保存至 ./best_model
python train.py

# 4. 推理并生成提交文件：读取 test_text.csv，输出 results.csv
python inference.py
```