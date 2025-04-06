import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

def train(train_path):
    # 讀取與前處理資料
    df = pd.read_csv(train_path)
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.select_dtypes(include=["number"])
    X = df.drop(columns=["飆股"])
    y = df["飆股"]

    # 補缺失
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # SHAP 分析 (僅對飆股進行)
    X_label1 = X_imputed[y == 1]
    dtrain = lgb.Dataset(X_imputed, label=y)
    model = lgb.train({
        "objective": "binary",
        "metric": "None",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "seed": 42
    }, dtrain, num_boost_round=300)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_label1)

    # 選出 SHAP 平均重要性前 50 名特徵
    shap_df = pd.DataFrame({
        "feature": X_label1.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="mean_abs_shap", ascending=False)

    top50_features = shap_df.head(50)["feature"].tolist()

    # 篩選特徵重新訓練
    X_top50 = X_imputed[top50_features]

    # 計算 scale_pos_weight
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42
    }

    # 交叉驗證與 threshold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds, models = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_top50, y)):
        X_train, X_val = X_top50.iloc[train_idx], X_top50.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(params, dtrain, num_boost_round=1000,
                        valid_sets=[dval],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                        )

        y_prob = model.predict(X_val)
        prec, rec, thres = precision_recall_curve(y_val, y_prob)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
        best_threshold = thres[np.argmax(f1)]
        thresholds.append(best_threshold)
        models.append(model)

        print(f"Fold {fold+1}: AUC={roc_auc_score(y_val, y_prob):.4f}, F1={np.max(f1):.4f}, Threshold={best_threshold:.4f}")

    # 統計最佳門檻
    mean_threshold = np.mean(thresholds)
    print(f"\n✅ 平均最佳 Threshold: {mean_threshold:.4f}")

    # 用全資料重訓（僅 top 50 特徵）
    dtrain_full = lgb.Dataset(X_top50, label=y)
    model_final = lgb.train(params, dtrain_full, num_boost_round=300)

    # 儲存模型與處理器
    joblib.dump(model_final, "lgbm_model_top50.pkl")
    joblib.dump(imputer, "imputer_top50.pkl")
    joblib.dump(mean_threshold, "threshold_top50.pkl")

    # 額外輸出 top50 清單
    pd.Series(top50_features).to_csv("top50_features.csv", index=False)
    print("✅ 模型、imputer、threshold、top50 特徵已儲存完畢")


if __name__ == "__main__":
    train(r"E:\side_project\stock_competition\data\38_Training_Data_Set_V2\cleaned_training_05.csv")