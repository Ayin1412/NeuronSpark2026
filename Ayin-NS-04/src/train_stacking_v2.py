import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import BayesianRidge

DATA_DIR  = "."
FEAT_DIR  = "features"
MODEL_DIR = "models_full"

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def spearman_r(y_true, y_pred):
    val = spearmanr(y_true, y_pred).statistic
    return val if not np.isnan(val) else 0.0

def evaluate(y_true, y_pred, prefix=""):
    r = rmse(y_true, y_pred)
    s = spearman_r(y_true, y_pred)
    score = 500 * max(0, 1 - r / 0.80) + 250 * max(0, s) + 50
    if r <= 0.25 and s >= 0.96:
        score = 800
    print(f"{prefix}RMSE={r:.4f}, Spearman={s:.4f}, Score={score:.1f}")
    return r, s, score

print("载入基模型数据与标签...")
y_train = pd.read_csv(f"{DATA_DIR}/train.csv")["target"].values

oof_lgb = np.load(f"{MODEL_DIR}/oof_lgb.npy")
oof_xgb = np.load(f"{MODEL_DIR}/oof_xgb.npy")
oof_cat = np.load(f"{MODEL_DIR}/oof_cat.npy")
oof_mlp = np.load(f"{MODEL_DIR}/oof_mlp.npy")

test_lgb = np.load(f"{MODEL_DIR}/test_lgb.npy")
test_xgb = np.load(f"{MODEL_DIR}/test_xgb.npy")
test_cat = np.load(f"{MODEL_DIR}/test_cat.npy")
test_mlp = np.load(f"{MODEL_DIR}/test_mlp.npy")

oof_all = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_mlp])
test_all = np.column_stack([test_lgb, test_xgb, test_cat, test_mlp])


def extract_meta_features(pred_matrix):
    mean = pred_matrix.mean(axis=1, keepdims=True)
    std  = pred_matrix.std(axis=1, keepdims=True)
    mx   = pred_matrix.max(axis=1, keepdims=True)
    mn   = pred_matrix.min(axis=1, keepdims=True)
    
    diffs = []
    n_cols = pred_matrix.shape[1]
    for i in range(n_cols):
        for j in range(i+1, n_cols):
            diffs.append(np.abs(pred_matrix[:, i] - pred_matrix[:, j]).reshape(-1, 1))
            
    return np.hstack([pred_matrix, mean, std, mx, mn] + diffs).astype(np.float32)

oof_meta = extract_meta_features(oof_all)
test_meta = extract_meta_features(test_all)
print(f"元特征矩阵维度: {oof_meta.shape}")


desc_train_norm = np.load(f"{FEAT_DIR}/rdkit_desc_train_norm.npy")
desc_test_norm  = np.load(f"{FEAT_DIR}/rdkit_desc_test_norm.npy")

corrs = []
for col_idx in range(desc_train_norm.shape[1]):
    c = np.corrcoef(desc_train_norm[:, col_idx], y_train)[0, 1]
    if np.isnan(c): c = 0.0
    corrs.append((col_idx, abs(c)))

# 按绝对相关系数由大到小排序，选取前 50 个
corrs.sort(key=lambda x: x[1], reverse=True)
top_50_features = [x[0] for x in corrs[:50]]
print(f"最强相关描述符索引: {top_50_features[:10]} ...")
print(f"最强关联强度: {[f'{x[1]:.4f}' for x in corrs[:10]]}")

desc_top_train = desc_train_norm[:, top_50_features]
desc_top_test  = desc_test_norm[:, top_50_features]

X_stack_train = np.hstack([oof_meta, desc_top_train])
X_stack_test  = np.hstack([test_meta, desc_top_test])
print(f"Stacking 输入特征维度: {X_stack_train.shape}")

# 拟合回归器
br_opt = BayesianRidge()
br_opt.fit(X_stack_train, y_train)

# 评估本地 OOF 得分
oof_pred = br_opt.predict(X_stack_train)
evaluate(y_train, oof_pred, prefix="优化 Stacking (OOF+Top50) OOF 评估: ")

# 预测测试集
test_pred = br_opt.predict(X_stack_test)
print(f"测试集预测范围: min={test_pred.min():.4f}, max={test_pred.max():.4f}")


with open(f"{MODEL_DIR}/stack_models_opt.pkl", 'wb') as f:
    pickle.dump({
        'br_opt': br_opt,
        'top_50_features': top_50_features
    }, f)

np.save(f"{MODEL_DIR}/final_test_pred_opt.npy", test_pred)
print("已保存")
