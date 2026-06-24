import pandas as pd
import numpy as np
import warnings
import sys
import os
warnings.filterwarnings("ignore")


def load_data(data_dir="."):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    cal   = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    wea   = pd.read_csv(os.path.join(data_dir, "weather.csv"))
    menu  = pd.read_csv(os.path.join(data_dir, "menu.csv"))
    for df in [train, test, cal, wea, menu]:
        df["date"] = pd.to_datetime(df["date"])
    print(f"训练集: {len(train)}行  测试集: {len(test)}行")
    return train, test, cal, wea, menu



MENU_MAP    = {"noodle":0,"porridge":1,"light":2,"spicy":3,
               "special":4,"rice_set":5,"western":6}
WEATHER_MAP = {"clear":0,"cloudy":1,"rain":2,"storm":3}
MEAL_MAP    = {"breakfast":0,"lunch":1,"dinner":2}

def prep(df_in, cal, wea, menu):
    """合并辅助表，做基础编码和填充，计算 ratio（训练集专用）"""
    df = df_in.copy()
    df = df.merge(cal,  on="date",                         how="left")
    df = df.merge(wea,  on=["date","meal"],                how="left")
    df = df.merge(menu, on=["date","meal","canteen_area"], how="left")

    df["menu_type_enc"] = df["menu_type"].map(MENU_MAP).fillna(-1)
    df["weather_enc"]   = df["weather"].map(WEATHER_MAP).fillna(0)
    df["meal_enc"]      = df["meal"].map(MEAL_MAP)
    df["area_enc"]      = df["canteen_area"].map(
        {a: i for i, a in enumerate(sorted(df["canteen_area"].unique()))}
    )

    int_cols = ["is_holiday","is_exam_week","rain_level","campus_event_level",
                "is_promotion","is_weekend","is_makeup_day","weekday","semester_week"]
    for c in int_cols:
        df[c] = df[c].fillna(0).astype(float)

    med_cols = ["temperature","feels_like","humidity","wind_speed",
                "demand_index","menu_popularity"]
    for c in med_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    # 体感温差：捕捉"体感寒冷"对出行的影响
    df["feels_diff"] = df["feels_like"] - df["temperature"]

    # ratio 目标（仅训练集有 volume）
    if "volume" in df.columns:
        df["ratio"] = df["volume"] / df["demand_index"]

    return df


def add_ratio_stats(d, ref):
    """
    基于 ref（训练集）计算各粒度的历史 ratio 均值，作为统计特征。
    这是 ratio 目标法的核心：给模型提供"这个条件下历史 ratio 是多少"的先验。
    """
    gm = ref["ratio"].mean()

    stat_defs = [
        (["meal","canteen_area","weekday","is_holiday"],    "r_wd_hol"),
        (["meal","canteen_area","weekday","semester_week"], "r_wd_sw"),
        (["meal","canteen_area","weekday","is_exam_week"],  "r_wd_exam"),
        (["meal","canteen_area","rain_level"],              "r_rain"),
        (["meal","canteen_area","campus_event_level"],      "r_event"),
        (["meal","canteen_area","menu_type_enc"],           "r_menu"),
        (["meal","canteen_area","menu_popularity"],         "r_pop"),
        (["meal","canteen_area","weekday"],                 "r_wd"),
        (["meal","canteen_area"],                           "r_base"),
    ]

    for keys, name in stat_defs:
        t = ref.groupby(keys)["ratio"].mean().rename(name).reset_index()
        d = d.merge(t, on=keys, how="left")
        d[name] = d[name].fillna(gm)

    return d



FEATURE_COLS = [
    "meal_enc", "area_enc",
    "weekday", "semester_week",
    "is_holiday", "is_exam_week", "is_weekend", "is_makeup_day",
    "rain_level", "campus_event_level", "weather_enc",
    "temperature", "feels_like", "humidity", "wind_speed",
    "demand_index", "feels_diff",
    "menu_type_enc", "menu_popularity", "is_promotion",
    "r_wd_hol", "r_wd_sw", "r_wd_exam",
    "r_rain", "r_event", "r_menu", "r_pop",
    "r_wd", "r_base",
]



def train_model(tr_df, val_df=None, num_rounds=6000):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("请先安装 lightgbm: pip install lightgbm")

    feats  = [c for c in FEATURE_COLS if c in tr_df.columns]
    X_tr   = tr_df[feats]
    y_tr   = tr_df["ratio"]

    params = {
        "objective":         "regression_l1",  # 优化 MAE
        "metric":            ["mae", "rmse"],
        "learning_rate":     0.02,
        "num_leaves":        63,
        "min_child_samples": 5,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.05,
        "reg_lambda":        0.05,
        "verbose":           -1,
        "n_jobs":            -1,
    }

    callbacks = [lgb.log_evaluation(500)]

    if val_df is not None:
        feats_v = [c for c in FEATURE_COLS if c in val_df.columns]
        X_val   = val_df[feats_v]
        y_val   = val_df["ratio"]
        dtrain  = lgb.Dataset(X_tr, y_tr)
        dval    = lgb.Dataset(X_val, y_val, reference=dtrain)
        callbacks.append(lgb.early_stopping(200, verbose=False))
        model   = lgb.train(params, dtrain,
                            num_boost_round=num_rounds,
                            valid_sets=[dtrain, dval],
                            callbacks=callbacks)
        # 反算 volume 误差
        ratio_pred = model.predict(X_val)
        vol_pred   = np.clip(val_df["demand_index"] * ratio_pred, 0, None)
        y_vol      = val_df["volume"].values
        mae   = np.mean(np.abs(vol_pred - y_vol))
        rmse  = np.sqrt(np.mean((vol_pred - y_vol) ** 2))
        mape  = np.mean(np.abs((vol_pred - y_vol) / (y_vol + 1e-8)))
        s_mae  = 200 * max(0, 1 - mae  / 1000)
        s_rmse = 150 * max(0, 1 - rmse / 1500)
        s_mape = 150 * max(0, 1 - mape / 0.50)
        print(f"\n[验证集] MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.5f}")
        print(f"[得分估算] {s_mae:.1f} + {s_rmse:.1f} + {s_mape:.1f} = "
              f"{s_mae+s_rmse+s_mape:.1f} / 500")
    else:
        dtrain = lgb.Dataset(X_tr, y_tr)
        model  = lgb.train(params, dtrain,
                           num_boost_round=num_rounds,
                           callbacks=callbacks)

    return model, feats


def main(data_dir=".", use_val=True):
    print("\n加载数据")
    train, test, cal, wea, menu = load_data(data_dir)

    print("\n特征工程")
    train_f = prep(train, cal, wea, menu)
    test_f  = prep(test,  cal, wea, menu)

    # 验证（可选，use_val=True 时做本地验证）
    if use_val:
        all_dates = sorted(train_f["date"].unique())
        cutoff    = all_dates[-80]
        tr_part   = train_f[train_f["date"] <  cutoff].copy()
        val_part  = train_f[train_f["date"] >= cutoff].copy()
        print(f"\n验证集切分（后80天，{cutoff.date()} 起）")
        print(f"  训练子集: {len(tr_part)}行  验证子集: {len(val_part)}行")

        tr_part  = add_ratio_stats(tr_part,  tr_part)
        val_part = add_ratio_stats(val_part, tr_part)

        print("\n训练模型")
        model_val, _ = train_model(tr_part, val_part)

    print("\n全量训练")
    train_f = add_ratio_stats(train_f, train_f)
    test_f  = add_ratio_stats(test_f,  train_f)

    model_final, feats = train_model(train_f, num_rounds=6000)

    print("\n生成预测")
    X_test      = test_f[[c for c in feats if c in test_f.columns]]
    ratio_pred  = model_final.predict(X_test)
    vol_pred    = np.clip(test_f["demand_index"].values * ratio_pred, 0, None)

    sub = test[["date","meal","canteen_area"]].copy()
    sub["date"]   = sub["date"].dt.strftime("%Y-%m-%d")
    sub["volume"] = np.round(vol_pred, 2)
    sub.to_csv("results.csv", index=False)
    print(f"\nresults.csv 已生成")
    print(sub.head(12).to_string(index=False))


if __name__ == "__main__":
    main(use_val=True)