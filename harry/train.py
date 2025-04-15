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

    # 選出 SHAP 平均重要性前 200 名特徵
    shap_df = pd.DataFrame({
        "feature": X_label1.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="mean_abs_shap", ascending=False)

    top200_features = shap_df.head(200)["feature"].tolist()

    # 篩選特徵重新訓練
    X_top200 = X_imputed[top200_features]

    # 交叉驗證與 threshold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds, models = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_top200, y)):
        X_train, X_val = X_top200.iloc[train_idx], X_top200.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(params, dtrain, valid_sets=[dval])

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

    # 用全資料重訓（僅 top 200 特徵）
    dtrain_full = lgb.Dataset(X_top200, label=y)
    model_final = lgb.train(params, dtrain_full, num_boost_round=200)

    # 儲存模型與處理器
    joblib.dump(model_final, "lgbm_model_top200.pkl")
    joblib.dump(imputer, "imputer_top200.pkl")
    joblib.dump(mean_threshold, "threshold_top200.pkl")

    # 額外輸出 top200 清單
    pd.Series(top200_features).to_csv("top200_features.csv", index=False)
    print("✅ 模型、imputer、threshold、top200 特徵已儲存完畢")


if __name__ == "__main__":
    train(r"D:\data\38_Training_Data_Set_V2\cleaned_01_training.csv")
    '''
    Fold 1: AUC=0.9908, F1=0.7481, Threshold=0.3544
    Fold 2: AUC=0.9856, F1=0.7010, Threshold=0.6750
    Fold 3: AUC=0.9912, F1=0.7878, Threshold=0.4583
    Fold 4: AUC=0.9892, F1=0.7539, Threshold=0.4037
    Fold 5: AUC=0.9900, F1=0.7364, Threshold=0.3452

    ✅ F1 平均值: 0.7454, 標準差: 0.0280
    ✅ AUC 平均值: 0.9894, 標準差: 0.0020
    ✅ Threshold 平均值: 0.4473, 標準差: 0.1208
    ✅ 綜合穩定性指標(F1 mean - 0.5*std): 0.7314
    ✅ 模型、imputer、threshold、top500 特徵已儲存完畢
    '''
    '''
    Fold 1: AUC=0.9874, F1=0.5926, Threshold=0.7883
    Fold 2: AUC=0.9893, F1=0.6706, Threshold=0.8501
    Fold 3: AUC=0.9906, F1=0.6834, Threshold=0.7843
    Fold 4: AUC=0.9863, F1=0.6603, Threshold=0.8390
    Fold 5: AUC=0.9879, F1=0.6307, Threshold=0.8474

    ✅ F1 平均值: 0.6475, 標準差: 0.0325
    ✅ AUC 平均值: 0.9883, 標準差: 0.0015
    ✅ Threshold 平均值: 0.8218, 標準差: 0.0293
    ✅ 綜合穩定性指標(F1 mean - 0.5*std): 0.6313
    ✅ 模型、imputer、threshold、top2000 特徵已儲存完畢
    '''
    '''
    Fold 1: AUC=0.9883, F1=0.6124, Threshold=0.8495
    Fold 2: AUC=0.9905, F1=0.6763, Threshold=0.8438
    Fold 3: AUC=0.9919, F1=0.6861, Threshold=0.8330
    Fold 4: AUC=0.9875, F1=0.6731, Threshold=0.8725
    Fold 5: AUC=0.9896, F1=0.6419, Threshold=0.8536
    
    ✅ F1 平均值: 0.6580, 標準差: 0.0272
    ✅ AUC 平均值: 0.9896, 標準差: 0.0016
    ✅ Threshold 平均值: 0.8505, 標準差: 0.0130
    ✅ 綜合穩定性指標(F1 mean - 0.5*std): 0.6444
    ✅ 模型、imputer、threshold、top300 特徵已儲存完畢
    '''
    '''
    Fold 1: AUC=0.9885, F1=0.6381, Threshold=0.8487
    Fold 2: AUC=0.9909, F1=0.6804, Threshold=0.8569
    Fold 3: AUC=0.9913, F1=0.6766, Threshold=0.8426
    Fold 4: AUC=0.9876, F1=0.6735, Threshold=0.8964
    Fold 5: AUC=0.9905, F1=0.6529, Threshold=0.8649

    ✅ F1 平均值: 0.6643, 標準差: 0.0162
    ✅ AUC 平均值: 0.9898, 標準差: 0.0014
    ✅ Threshold 平均值: 0.8619, 標準差: 0.0188
    ✅ 綜合穩定性指標(F1 mean - 0.5*std): 0.6562
    ✅ 模型、imputer、threshold、top200 特徵已儲存完畢
    '''