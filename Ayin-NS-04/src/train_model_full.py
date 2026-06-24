import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV, BayesianRidge
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR  = "."
FEAT_DIR  = "features"
MODEL_DIR = "models_full"
os.makedirs(MODEL_DIR, exist_ok=True)

SEED    = 42
N_FOLDS = 5
np.random.seed(SEED)


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


train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")
y_train  = train_df['target'].values

X_train_full = np.load(f"{FEAT_DIR}/X_train_full.npy")
X_test_full  = np.load(f"{FEAT_DIR}/X_test_full.npy")

desc_train_norm = np.load(f"{FEAT_DIR}/rdkit_desc_train_norm.npy")
desc_test_norm  = np.load(f"{FEAT_DIR}/rdkit_desc_test_norm.npy")

all_fps_train = sp.load_npz(f"{FEAT_DIR}/all_fps_train.npz").toarray().astype(np.float32)
all_fps_test  = sp.load_npz(f"{FEAT_DIR}/all_fps_test.npz").toarray().astype(np.float32)

X_xgb = np.hstack([all_fps_train, desc_train_norm]).astype(np.float32)
X_xgb_test = np.hstack([all_fps_test, desc_test_norm]).astype(np.float32)

# CatBoost 特征组合
morgan2_train = sp.load_npz(f"{FEAT_DIR}/morgan2_train.npz").toarray().astype(np.float32)
morgan2_test  = sp.load_npz(f"{FEAT_DIR}/morgan2_test.npz").toarray().astype(np.float32)
morgan3_train = sp.load_npz(f"{FEAT_DIR}/morgan3_train.npz").toarray().astype(np.float32)
morgan3_test  = sp.load_npz(f"{FEAT_DIR}/morgan3_test.npz").toarray().astype(np.float32)
maccs_train   = sp.load_npz(f"{FEAT_DIR}/maccs_train.npz").toarray().astype(np.float32)
maccs_test    = sp.load_npz(f"{FEAT_DIR}/maccs_test.npz").toarray().astype(np.float32)
atompair_train = sp.load_npz(f"{FEAT_DIR}/atompair_train.npz").toarray().astype(np.float32)
atompair_test  = sp.load_npz(f"{FEAT_DIR}/atompair_test.npz").toarray().astype(np.float32)

X_cat = np.hstack([morgan2_train, morgan3_train, maccs_train, atompair_train, desc_train_norm]).astype(np.float32)
X_cat_test = np.hstack([morgan2_test, morgan3_test, maccs_test, atompair_test, desc_test_norm]).astype(np.float32)



def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""

scaffolds = [get_scaffold(smi) for smi in train_df['smiles']]
print(f"唯一骨架数量：{len(set(scaffolds))}")

# 5折 GroupKFold
gkf = GroupKFold(n_splits=N_FOLDS)
cv_splits = list(gkf.split(X_train_full, y_train, groups=scaffolds))

# 校验折样本分布
for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
    print(f"  Fold {fold_idx+1}: 训练集 {len(tr_idx)} 条，验证集 {len(val_idx)} 条")


lgb_params = {
    'objective':        'regression',
    'metric':           'rmse',
    'boosting_type':    'gbdt',
    'n_estimators':     3000,
    'learning_rate':    0.02,
    'num_leaves':       63,
    'max_depth':        8,
    'min_child_samples': 30,
    'subsample':        0.8,
    'colsample_bytree': 0.7,
    'reg_alpha':        0.5,
    'reg_lambda':       2.0,
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbose':          -1,
}

oof_lgb  = np.zeros(len(train_df))
test_lgb = np.zeros(len(test_df))
lgb_models = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_train_full[tr_idx], y_train[tr_idx]
    X_vl, y_vl = X_train_full[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    pred_vl = model.predict(X_vl)
    oof_lgb[val_idx] = pred_vl
    test_lgb += model.predict(X_test_full) / N_FOLDS
    lgb_models.append(model)
    evaluate(y_vl, pred_vl, prefix=f"  LGB Fold{fold_idx+1} CV: ")

evaluate(y_train, oof_lgb, prefix="LGB 全量 OOF 评估: ")

with open(f"{MODEL_DIR}/lgb_models.pkl", 'wb') as f:
    pickle.dump(lgb_models, f)
np.save(f"{MODEL_DIR}/oof_lgb.npy",  oof_lgb)
np.save(f"{MODEL_DIR}/test_lgb.npy", test_lgb)




xgb_params = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'n_estimators':     3000,
    'learning_rate':    0.015,
    'max_depth':        6,
    'min_child_weight': 5,
    'subsample':        0.8,
    'colsample_bytree': 0.7,
    'reg_alpha':        0.5,
    'reg_lambda':       2.0,
    'random_state':     SEED,
    'tree_method':      'hist',
    'device':           'cuda',
}

oof_xgb  = np.zeros(len(train_df))
test_xgb = np.zeros(len(test_df))
xgb_models = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_xgb[tr_idx], y_train[tr_idx]
    X_vl, y_vl = X_xgb[val_idx], y_train[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False
    )

    pred_vl = model.predict(X_vl)
    oof_xgb[val_idx] = pred_vl
    test_xgb += model.predict(X_xgb_test) / N_FOLDS
    xgb_models.append(model)
    evaluate(y_vl, pred_vl, prefix=f"  XGB Fold{fold_idx+1} CV: ")

evaluate(y_train, oof_xgb, prefix="XGB 全量 OOF 评估: ")

with open(f"{MODEL_DIR}/xgb_models.pkl", 'wb') as f:
    pickle.dump(xgb_models, f)
np.save(f"{MODEL_DIR}/oof_xgb.npy",  oof_xgb)
np.save(f"{MODEL_DIR}/test_xgb.npy", test_xgb)



cat_params = {
    'loss_function':    'RMSE',
    'eval_metric':      'RMSE',
    'iterations':       3000,
    'learning_rate':    0.025,
    'depth':            6,
    'l2_leaf_reg':      5.0,
    'bootstrap_type':   'Bernoulli',
    'subsample':        0.8,
    'random_seed':      SEED,
    'task_type':        'GPU',
    'od_type':          'Iter',
    'od_wait':          150,
    'verbose':          False,
}

oof_cat  = np.zeros(len(train_df))
test_cat = np.zeros(len(test_df))
cat_models = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_cat[tr_idx], y_train[tr_idx]
    X_vl, y_vl = X_cat[val_idx], y_train[val_idx]

    train_pool = cb.Pool(X_tr, label=y_tr)
    val_pool   = cb.Pool(X_vl, label=y_vl)

    model = cb.CatBoostRegressor(**cat_params)
    model.fit(train_pool, eval_set=val_pool)

    pred_vl = model.predict(X_vl)
    oof_cat[val_idx] = pred_vl
    test_cat += model.predict(X_cat_test) / N_FOLDS
    cat_models.append(model)
    evaluate(y_vl, pred_vl, prefix=f"  CAT Fold{fold_idx+1} CV: ")

evaluate(y_train, oof_cat, prefix="CAT 全量 OOF 评估: ")

with open(f"{MODEL_DIR}/cat_models.pkl", 'wb') as f:
    pickle.dump(cat_models, f)
np.save(f"{MODEL_DIR}/oof_cat.npy",  oof_cat)
np.save(f"{MODEL_DIR}/test_cat.npy", test_cat)



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

X_mlp = X_train_full
X_mlp_test = X_test_full
mlp_dim = X_mlp.shape[1]
print(f"MLP特征维度: {mlp_dim}")


class MolMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(1024, 512, 256), dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_tr, y_tr, X_vl, y_vl, X_te, in_dim,
                   n_epochs=200, batch_size=256,
                   lr=5e-4, weight_decay=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MolMLP(in_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    X_vl_t = torch.FloatTensor(X_vl).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)

    best_val_loss  = float('inf')
    best_val_pred  = None
    best_test_pred = None
    patience   = 35
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred  = model(X_vl_t).cpu().numpy()
            val_loss  = rmse(y_vl, val_pred)
            test_pred = model(X_te_t).cpu().numpy()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_pred  = val_pred.copy()
            best_test_pred = test_pred.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_pred, best_test_pred, best_val_loss


oof_mlp  = np.zeros(len(train_df))
test_mlp = np.zeros(len(test_df))

for fold_idx, (tr_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_mlp[tr_idx], y_train[tr_idx]
    X_vl, y_vl = X_mlp[val_idx], y_train[val_idx]

    val_pred, test_pred, best_loss = train_mlp_fold(
        X_tr, y_tr, X_vl, y_vl, X_mlp_test, mlp_dim,
        n_epochs=200, batch_size=256, seed=SEED+fold_idx
    )
    oof_mlp[val_idx] = val_pred
    test_mlp += test_pred / N_FOLDS
    evaluate(y_vl, val_pred, prefix=f"  MLP Fold{fold_idx+1} CV: ")

evaluate(y_train, oof_mlp, prefix="MLP 全量 OOF 评估: ")

np.save(f"{MODEL_DIR}/oof_mlp.npy",  oof_mlp)
np.save(f"{MODEL_DIR}/test_mlp.npy", test_mlp)



oof_stack_all = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_mlp])
test_stack    = np.column_stack([test_lgb, test_xgb, test_cat, test_mlp])

# RidgeCV 自动选惩罚项
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(oof_stack_all, y_train)
oof_ridge_cv = ridge_cv.predict(oof_stack_all)
test_ridge_cv = ridge_cv.predict(test_stack)
print(f"  [RidgeCV] 最优 alpha={ridge_cv.alpha_:.4f}")
evaluate(y_train, oof_ridge_cv, prefix="  Stack(RidgeCV) 全样本 OOF: ")

# BayesianRidge 自适应惩罚
bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(oof_stack_all, y_train)
oof_bayesian_ridge = bayesian_ridge.predict(oof_stack_all)
test_bayesian_ridge = bayesian_ridge.predict(test_stack)
evaluate(y_train, oof_bayesian_ridge, prefix="  Stack(BayesianRidge) 全样本 OOF: ")

with open(f"{MODEL_DIR}/stack_models.pkl", 'wb') as f:
    pickle.dump({
        'ridge_cv': ridge_cv,
        'bayesian_ridge': bayesian_ridge
    }, f)

from scipy.optimize import minimize

# Blending 1: Spearman 最大化
def neg_spearman(weights, preds_val, y_val):
    w = np.abs(weights)
    w = w / w.sum()
    return -spearman_r(y_val, preds_val @ w)

w0 = np.ones(4) / 4
res_sp = minimize(neg_spearman, w0, args=(oof_stack_all, y_train),
                  method='Nelder-Mead',
                  options={'maxiter': 2000, 'xatol': 1e-8})
w_sp = np.abs(res_sp.x) / np.abs(res_sp.x).sum()
print(f"  [Spearman优化权重]: LGB={w_sp[0]:.3f}, XGB={w_sp[1]:.3f}, CAT={w_sp[2]:.3f}, MLP={w_sp[3]:.3f}")
blend_val_sp  = oof_stack_all @ w_sp
blend_test_sp = test_stack @ w_sp
evaluate(y_train, blend_val_sp, prefix="  Blend(Spearman) 全样本 OOF: ")

# Blending 2: RMSE 最小化
def rmse_loss(weights, preds_val, y_val):
    w = np.abs(weights)
    w = w / w.sum()
    return rmse(y_val, preds_val @ w)

res_rmse = minimize(rmse_loss, w0, args=(oof_stack_all, y_train),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8})
w_rmse = np.abs(res_rmse.x) / np.abs(res_rmse.x).sum()
print(f"  [RMSE优化权重]: LGB={w_rmse[0]:.3f}, XGB={w_rmse[1]:.3f}, CAT={w_rmse[2]:.3f}, MLP={w_rmse[3]:.3f}")
blend_val_rmse  = oof_stack_all @ w_rmse
blend_test_rmse = test_stack @ w_rmse
evaluate(y_train, blend_val_rmse, prefix="  Blend(RMSE) 全样本 OOF: ")

np.save(f"{MODEL_DIR}/w_sp.npy", w_sp)
np.save(f"{MODEL_DIR}/w_rmse.npy", w_rmse)



candidates = {
    'LightGBM':           (oof_lgb,              test_lgb),
    'XGBoost':            (oof_xgb,              test_xgb),
    'CatBoost':           (oof_cat,              test_cat),
    'MLP':                (oof_mlp,              test_mlp),
    'Stack (RidgeCV)':    (oof_ridge_cv,         test_ridge_cv),
    'Stack (Bayesian)':   (oof_bayesian_ridge,   test_bayesian_ridge),
    'Blend (Spearman)':   (blend_val_sp,         blend_test_sp),
    'Blend (RMSE)':       (blend_val_rmse,       blend_test_rmse),
}

best_score = -1
best_name  = None
best_test  = None

for name, (val_pred, test_preds) in candidates.items():
    r, s, sc = evaluate(y_train, val_pred, prefix=f"  {name:20s}: ")
    if sc > best_score:
        best_score = sc
        best_name  = name
        best_test  = test_preds

print(f"最佳方案: {best_name} (Score={best_score:.1f})")
np.save(f"{MODEL_DIR}/best_test_pred.npy", best_test)
np.save(f"{MODEL_DIR}/final_test_pred.npy", best_test)
print("已保存")
