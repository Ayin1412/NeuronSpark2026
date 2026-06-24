import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset


os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备: {device.upper()}")

# 读取文本数据集
csv_path = 'train_text.csv'

    
df = pd.read_csv(csv_path)

# 构建标签与数字索引的映射字典
unique_labels = sorted(df['label'].unique().tolist())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
print("标签与数字索引映射关系:")
for k, v in label2id.items():
    print(f"  {k} -> {v}")
    
df['label_id'] = df['label'].map(label2id)

# 规范化文本，并按 8:2 切分训练/验证集
df['text'] = df['text'].astype(str)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])
print(f"\n数据切分完毕： 训练集: {len(train_df)} 条 | 验证集: {len(val_df)} 条")

# 加载预训练模型与分词器
model_name = "hfl/chinese-roberta-wwm-ext"
print(f"\n加载预训练模型与分词器: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(unique_labels),
    label2id=label2id,
    id2label=id2label
)

#转换数据格式为 Hugging Face Datasets，方便后续 Trainer 使用
train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
val_dataset = Dataset.from_pandas(val_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))

# 定义分词闭包函数
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=256)
    
print("正在执行文本向量化分词...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 动态填充组件
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 定义计算指标的函数（这里以准确率为例）
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = np.sum(preds == labels) / len(labels)
    return {"accuracy": float(acc)}
    
# 超参数配置
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",       # 每个 epoch 结束后做一次验证
    save_strategy="epoch",       # 每个 epoch 结束后保存一次权重
    learning_rate=3e-5,          
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    num_train_epochs=5,          
    weight_decay=0.01,
    load_best_model_at_end=True, 
    metric_for_best_model="accuracy",
    fp16=True,
    logging_steps=20,            
    report_to="none"             
)

# 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 启动训练
print("\n开始训练模型...")
trainer.train()

print("\n训练完成")
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")
