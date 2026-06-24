import os
import zipfile
import hashlib
import numpy as np
import pandas as pd
import subprocess
import sys

DATA_DIR  = "."
MODEL_DIR = "models_full"

print("加载数据...")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

final_pred_opt = np.load(f"{MODEL_DIR}/final_test_pred_opt.npy")
print(f"测试集优化预测数量: {len(final_pred_opt)}")
print(f"预测值分布统计: mean={final_pred_opt.mean():.4f}, std={final_pred_opt.std():.4f}, "
      f"min={final_pred_opt.min():.4f}, max={final_pred_opt.max():.4f}")

output_df = pd.DataFrame({
    'id':         test_df['id'].values,
    'prediction': final_pred_opt,
})

assert output_df['prediction'].notna().all(), "存在 NaN 预测！"
assert np.isfinite(output_df['prediction'].values).all(), "存在无限值！"

output_df.to_csv("results.csv", index=False)
print("已保存")

