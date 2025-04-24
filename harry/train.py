import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

def train(train_path, model_type="lgbm"):
    # 讀取與前處理資料
    df = pd.read_csv(train_path)
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.select_dtypes(include=["number"])
    X = df.drop(columns=["飆股"])
    y = df["飆股"]

    # 補缺失
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 計算 scale_pos_weight
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    params = {
        "objective": "binary",
        "metric": "recall",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,  # 降低學習率，提高穩定性
        "num_leaves": 31,
        "max_depth": -1,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,  # 平衡類別
        "num_iterations": 1000,  # 增加訓練次數
    }

    # SHAP 分析 (僅對飆股進行)
    X_label1 = X_imputed[y == 1]
    dtrain = lgb.Dataset(X_imputed, label=y)
    model = lgb.train(params, dtrain)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_label1)

    # 選出 SHAP 平均重要性前 500 名特徵
    shap_df = pd.DataFrame({
        "feature": X_label1.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="mean_abs_shap", ascending=False)

    top500_features = shap_df.head(500)["feature"].tolist()

    # 篩選特徵重新訓練
    X_top500 = X_imputed[top500_features]

    # 交叉驗證與 threshold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    metrics = {"f1": [],
               "auc": [],
               "threshold": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_top500, y)):
        X_train, X_val = X_top500.iloc[train_idx], X_top500.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == "catboost":
            model = CatBoostClassifier(verbose=0, random_seed=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(params, dtrain, valid_sets=[dval])
            y_prob = model.predict(X_val)

        prec, rec, thres = precision_recall_curve(y_val, y_prob)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
        best_threshold = thres[np.argmax(f1)]
        models.append(model)

        metrics["f1"].append(np.max(f1))
        metrics["auc"].append(roc_auc_score(y_val, y_prob))
        metrics["threshold"].append(best_threshold)

        print(f"Fold {fold+1}: AUC={roc_auc_score(y_val, y_prob):.4f}, F1={np.max(f1):.4f}, Threshold={best_threshold:.4f}")

    # 統計最佳門檻
    f1_mean = np.mean(metrics["f1"])
    f1_std = np.std(metrics["f1"])
    auc_mean = np.mean(metrics["auc"])
    auc_std = np.std(metrics["auc"])
    threshold_mean = np.mean(metrics["threshold"])
    threshold_std = np.std(metrics["threshold"])
    stability_score = f1_mean - 0.5 * f1_std
    print(f"\n✅ F1 平均值: {f1_mean:.4f}, 標準差: {f1_std:.4f}")
    print(f"✅ AUC 平均值: {auc_mean:.4f}, 標準差: {auc_std:.4f}")
    print(f"✅ Threshold 平均值: {threshold_mean:.4f}, 標準差: {threshold_std:.4f}")
    print(f"✅ 綜合穩定性指標(F1 mean - 0.5*std): {stability_score:.4f}")

    # 用全資料重訓（僅 top 500 特徵）
    if model_type == "catboost":
        model_final = CatBoostClassifier(verbose=0, random_seed=42)
        model_final.fit(X_top500, y)
    else:
        dtrain_full = lgb.Dataset(X_top500, label=y)
        model_final = lgb.train(params, dtrain_full, num_boost_round=500)

    # 儲存模型與處理器
    joblib.dump(model_final, f"checkpoints/derived_{model_type}_model_top500.pkl")
    joblib.dump(imputer, f"checkpoints/derived_{model_type}_imputer_top500.pkl")
    joblib.dump(threshold_mean, f"checkpoints/derived_{model_type}_threshold_top500.pkl")

    # 額外輸出 top500 清單
    pd.Series(top500_features).to_csv(f"top_features/derived_{model_type}_top500_features.csv", index=False, header=False)
    print("✅ 模型、imputer、threshold、top500 特徵已儲存完畢")


if __name__ == "__main__":
    train(r"D:\data\38_Training_Data_Set_V2\derived_cleaned_01_training.csv", model_type="catboost")
    '''lightgbm
    Fold 1: AUC=0.9922, F1=0.7140, Threshold=0.8526
    Fold 2: AUC=0.9924, F1=0.7526, Threshold=0.8963
    Fold 3: AUC=0.9929, F1=0.7643, Threshold=0.8430
    Fold 4: AUC=0.9893, F1=0.7331, Threshold=0.8766
    Fold 5: AUC=0.9920, F1=0.7574, Threshold=0.8492

    ✅ F1 平均值: 0.7443, 標準差: 0.0183
    ✅ AUC 平均值: 0.9918, 標準差: 0.0013
    ✅ Threshold 平均值: 0.8635, 標準差: 0.0199
    ✅ 綜合穩定性指標(F1 mean - 0.5*std): 0.7351
    ✅ 模型、imputer、threshold、top500 特徵已儲存完畢
    '''
    '''catboost
    Fold 1: AUC=0.9933, F1=0.7992, Threshold=0.2102
    Fold 2: AUC=0.9920, F1=0.8247, Threshold=0.2445
    Fold 3: AUC=0.9936, F1=0.8145, Threshold=0.1386
    Fold 4: AUC=0.9911, F1=0.7977, Threshold=0.2240
    Fold 5: AUC=0.9932, F1=0.8063, Threshold=0.2291

    ✅ F1 平均值: 0.8085, 標準差: 0.0101
    ✅ AUC 平均值: 0.9926, 標準差: 0.0009
    ✅ Threshold 平均值: 0.2093, 標準差: 0.0370
    ✅ 綜合穩定性指標(F1 mean - 0.5*std): 0.8035
    ✅ 模型、imputer、threshold、top500 特徵已儲存完畢
    '''