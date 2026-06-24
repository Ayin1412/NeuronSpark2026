import os
import json
import warnings
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

DATA_DIR = "./"  
RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

HIGH_PRIORITY_CLASSES = ['computer_hardware', 'medical_health', 'space_science']
ALL_CLASSES = ['computer_hardware', 'system_software', 'mobility_vehicle', 'sports_recreation', 'medical_health', 'space_science']

print("[1/6] 正在加载数据与配置...")
with open(os.path.join(DATA_DIR, 'label_map.json'), 'r', encoding='utf-8') as f:
    label_map = json.load(f)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
valid_df = pd.read_csv(os.path.join(DATA_DIR, 'trusted_valid.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

le = LabelEncoder()
le.fit(ALL_CLASSES)
NUM_CLASSES = len(le.classes_)

train_df['target'] = le.transform(train_df['weak_label'])
valid_df['target'] = le.transform(valid_df['label'])

feature_cols = [f'f{i:03d}' for i in range(96)]

print("[2/6] 正在提取多源群组先验特征...")

# 组合细粒度标注域特征
train_df['src_anno'] = train_df['source_group'].astype(str) + "_" + train_df['annotator_group'].astype(str)

for col in ['source_group', 'src_anno']:
    freq = train_df[col].value_counts().to_dict()
    train_df[f'{col}_freq'] = train_df[col].map(freq).fillna(0)

# 验证集缺失列平滑处理
valid_df['src_anno_freq'] = 0
if 'source_group' in valid_df.columns:
    freq_sg = train_df['source_group'].value_counts().to_dict()
    valid_df['source_group_freq'] = valid_df['source_group'].map(freq_sg).fillna(0)
else:
    valid_df['source_group_freq'] = 0

cleaning_features = feature_cols + ['weak_confidence', 'source_group_freq', 'src_anno_freq']

print("[3/6] 启动两阶段高级 Confident Learning 标签净化...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof_preds_lgb = np.zeros((len(train_df), NUM_CLASSES))
oof_preds_xgb = np.zeros((len(train_df), NUM_CLASSES))

lgb_params = {
    'objective': 'multiclass', 'num_class': NUM_CLASSES, 'metric': 'multi_logloss',
    'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6,
    'feature_fraction': 0.8, 'verbose': -1, 'random_state': RANDOM_SEED, 'n_jobs': -1
}

xgb_params = {
    'objective': 'multi:softprob', 'num_class': NUM_CLASSES, 'eval_metric': 'mlogloss',
    'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED, 'n_jobs': -1, 'tree_method': 'hist'
}

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
    X_tr, y_tr = train_df.iloc[train_idx][cleaning_features], train_df.iloc[train_idx]['target']
    X_va, y_va = train_df.iloc[val_idx][cleaning_features], train_df.iloc[val_idx]['target']
    
    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=300)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20, verbose=False)])
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_va)
    
    # Model 2: XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=300)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_va)

# 双模型融合交叉检验置信度
oof_preds = 0.5 * oof_preds_lgb + 0.5 * oof_preds_xgb

# 计算每类的动态自适应阈值
thresholds = np.zeros(NUM_CLASSES)
for c in range(NUM_CLASSES):
    c_mask = (train_df['target'] == c)
    if c_mask.sum() > 0:
        thresholds[c] = np.mean(oof_preds[c_mask, c])

clean_indices = []
suspect_samples = []

for idx, row in train_df.iterrows():
    weak_lbl = row['target']
    pred_probs = oof_preds[idx]
    max_pred_class = np.argmax(pred_probs)
    
    # 强化判定：结合原本类别的衰减阈值与最大候选类别的绝对置信度裕度 (Margin)
    if pred_probs[weak_lbl] < (thresholds[weak_lbl] * 0.60) and (pred_probs[max_pred_class] - pred_probs[weak_lbl] > 0.25):
        suspect_samples.append({
            'id': row['id'],
            'weak_label': row['weak_label'],
            'model_suggest': le.inverse_transform([max_pred_class])[0]
        })
    else:
        clean_indices.append(idx)

print(f"成功识别并过滤系统性噪声样本: {len(train_df) - len(clean_indices)} 条.")
clean_train_df = train_df.iloc[clean_indices].copy()

print("[4/6] 注入高权重可信集，并行训练 LightGBM & CatBoost 鲁棒集成体系...")

clean_train_df['sample_weight'] = 1.0
valid_as_train = valid_df.copy()
# 提高黄金验证集样本的权重以强制修正多维决策边界
valid_as_train['sample_weight'] = 6.0 

final_train_df = pd.concat([
    clean_train_df[feature_cols + ['target', 'sample_weight']], 
    valid_as_train[feature_cols + ['target', 'sample_weight']]
], axis=0).reset_index(drop=True)

test_preds_lgb = np.zeros((len(test_df), NUM_CLASSES))
valid_preds_lgb = np.zeros((len(valid_df), NUM_CLASSES))
test_preds_cat = np.zeros((len(test_df), NUM_CLASSES))
valid_preds_cat = np.zeros((len(valid_df), NUM_CLASSES))

final_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 88)

for fold, (train_idx, val_idx) in enumerate(final_skf.split(final_train_df, final_train_df['target'])):
    X_tr, y_tr, w_tr = final_train_df.iloc[train_idx][feature_cols], final_train_df.iloc[train_idx]['target'], final_train_df.iloc[train_idx]['sample_weight']
    X_va, y_va, w_va = final_train_df.iloc[val_idx][feature_cols], final_train_df.iloc[val_idx]['target'], final_train_df.iloc[val_idx]['sample_weight']
    
    # 最终模型 1: 深度优化版 LightGBM
    f_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=600)
    f_lgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], eval_sample_weight=[w_va], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    test_preds_lgb += f_lgb.predict_proba(test_df[feature_cols]) / 5.0
    valid_preds_lgb += f_lgb.predict_proba(valid_df[feature_cols]) / 5.0
    
    # 最终模型 2: CatBoost (天生具备对抗类条件噪声的对称树结构)
    f_cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=5, loss_function='MultiClass', random_seed=RANDOM_SEED, verbose=False)
    f_cat.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), early_stopping_rounds=30)
    
    test_preds_cat += f_cat.predict_proba(test_df[feature_cols]) / 5.0
    valid_preds_cat += f_cat.predict_proba(valid_df[feature_cols]) / 5.0

# 异构概率集成
valid_preds = 0.5 * valid_preds_lgb + 0.5 * valid_preds_cat
test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_cat

print("[5/6] 启动 Nelder-Mead 连续空间寻优进行后处理阈值校准...")

priority_indices = [le.transform([cls])[0] for cls in HIGH_PRIORITY_CLASSES]

# 损失函数定义：将宏观分直接映射为负分作为极小化目标
def objective_func(multipliers):
    # 惩罚负的乘子
    if np.any(multipliers < 0.2):
        return 0.0
    
    adjusted_preds = valid_preds * multipliers
    pred_labels = np.argmax(adjusted_preds, axis=1)
    
    macro_f1 = f1_score(valid_df['target'], pred_labels, average='macro')
    
    recalls = []
    for p_idx in priority_indices:
        true_pos = np.sum((valid_df['target'] == p_idx) & (pred_labels == p_idx))
        actual_pos = np.sum(valid_df['target'] == p_idx)
        recalls.append(true_pos / (actual_pos + 1e-8))
    priority_recall = np.mean(recalls)
    
    total_score = 900 * macro_f1 + 200 * priority_recall
    return -total_score

# 初始乘子：对高优先级类别赋予更高的初始探索权值
init_multipliers = np.array([1.2 if c in priority_indices else 1.0 for c in range(NUM_CLASSES)])

res = minimize(objective_func, init_multipliers, method='Nelder-Mead', options={'maxiter': 400})
best_multipliers = res.x

print(f"    [优化完成] 本地可信集最佳决策总分 (不含格式分): {-res.fun:.4f}")
print(f"    [优化结果] 各类别自适应乘子: {np.round(best_multipliers, 4)}")

print("[6/6] 导出最终预测结果并生成合法性检查...")

final_test_preds = test_preds * best_multipliers
test_pred_labels = np.argmax(final_test_preds, axis=1)
test_df['label'] = le.inverse_transform(test_pred_labels)

submission = test_df[['id', 'label']]
submission.to_csv('results.csv', index=False, encoding='utf-8')
print("🎉 结果文件 results.csv 已成功生成！")

# 评估本地混淆矩阵与每类细节
valid_final_labels = np.argmax(valid_preds * best_multipliers, axis=1)

print("\n" + "="*60 + "\n📝 组委会 Writeup 补充材料核心数据支撑汇总\n" + "="*60)
print(f"1. 最终 results.csv 的唯一 SHA-256 校验和:")
with open('results.csv', 'rb') as f:
    print(f"   {hashlib.sha256(f.read()).hexdigest()}")

print(f"\n2. 5个代表性清洗纠错样本明细:")
for i, sample in enumerate(suspect_samples[:5]):
    print(f"   - [ID]: {sample['id']} | 原弱标签: {sample['weak_label']} --> 清洗建议: {sample['model_suggest']} | [处理方式]: 剔除弱标签噪声，交由高权信任集联合纠正边缘")

print(f"\n3. 本地可信验证集分类完整表现报告 (Classification Report):")
print(classification_report(valid_df['target'], valid_final_labels, target_names=le.classes_, digits=4))

print(f"4. 本地可信验证集混淆矩阵 (Confusion Matrix):")
print(confusion_matrix(valid_df['target'], valid_final_labels))
print("="*60)