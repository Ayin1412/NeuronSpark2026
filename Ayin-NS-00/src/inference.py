import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备: {device.upper()}")

model_path = "./best_model"
test_csv = "test_text.csv"
output_csv = "results.csv"

# 载入权重与分词器
print(f"正在从 {model_path} 载入权重...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval() # 切换到评估模式，关闭 Dropout 等训练专用机制
print("准备进行批量推理...")

# 读取测试集
df = pd.read_csv(test_csv)
df['text'] = df['text'].astype(str).fillna("")

print(f"测试集加载成功，共 {len(df)} 条样本。开始批量预测...")

final_predictions = []
batch_size = 64


with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size), desc="Predicting Batches"):
        batch_texts = df['text'].iloc[i : i + batch_size].tolist()
        
        # 文本向量化与 Padding
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        # 搬运至对应硬件
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 模型前向传播
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        
        # 提取概率最大的索引
        preds = np.argmax(logits, axis=1)
        
        # 将数字索引还原为文本标签名
        for pred in preds:
            label_name = model.config.id2label[pred]
            final_predictions.append(label_name)

# 保存预测结果到 results.csv
df['label'] = final_predictions

# 保留了识别的内容，便于抽检
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"\n预测结果已导出至: {output_csv}")
print("前5条预测样本预览：")
print(df[['id', 'label', 'text']].head())
