import os
import pandas as pd
from tqdm import tqdm
from rapidocr import RapidOCR

ocr = RapidOCR()
print("OCR 管道初始化成功！准备开始提取。")

def extract_text_from_csv(csv_path, output_csv_path):
    if not os.path.exists(csv_path):
        print(f"找不到输入文件: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    extracted_texts = []
    
    print(f"\n[开始处理]: {csv_path} (共 {len(df)} 张图片)")
    
    # 使用 tqdm 打印进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        img_path = row['image']
        
        if not os.path.exists(img_path):
            extracted_texts.append("")
            continue
            
        try:
            # 3. 调用 RapidOCR 预测
            result = ocr(img_path)
            
            combined_text = ""
            # 4. 解析 RapidOCROutput 对象
            if result and hasattr(result, 'txts') and result.txts:
                # result.txts 是一个包含所有文本行字符串的元组/列表
                combined_text = " ".join(result.txts)
            
            extracted_texts.append(combined_text)
            
        except Exception as e:
            extracted_texts.append("")
            
    df['text'] = extracted_texts
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"成功生成文本数据集: {output_csv_path}")

# 执行批量跑数
if __name__ == "__main__":
    extract_text_from_csv('train.csv', 'train_text.csv')
    extract_text_from_csv('test.csv', 'test_text.csv')