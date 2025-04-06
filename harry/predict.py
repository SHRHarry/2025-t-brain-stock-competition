import pandas as pd
import numpy as np
import joblib

def predict(test_path, template_path):
    # 載入模型與處理器
    model = joblib.load("lgbm_model_top50.pkl")
    imputer = joblib.load("imputer_top50.pkl")
    threshold = joblib.load("threshold_top50.pkl")

    # 正確讀取 top50 特徵名稱為 list of strings
    top50_features = pd.read_csv("top50_features.csv", header=None).squeeze("columns").tolist()
    print("🔍 Top 50 features loaded:", top50_features[:5])

    # 載入測試資料
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df.drop(columns=["ID"], errors="ignore").select_dtypes(include=["number"])

    # 補上訓練時的完整欄位（保證與 imputer 對齊）
    X_test_full = X_test_raw.reindex(columns=imputer.feature_names_in_)

    # 補值
    X_test_imputed_full = pd.DataFrame(imputer.transform(X_test_full), columns=imputer.feature_names_in_)

    # 篩選 top50 特徵
    X_test_imputed_top50 = X_test_imputed_full[top50_features]

    # 推論
    y_prob = model.predict(X_test_imputed_top50)
    y_pred = (y_prob > threshold).astype(int)

    # 匯出 submission
    submission = pd.read_csv(template_path)
    submission["飆股"] = y_pred
    submission.to_csv("submission_supervised_top50.csv", index=False)
    print("✅ 已產生 submission_supervised_top50.csv（使用 SHAP 精選特徵）")

if __name__ == "__main__":
    predict(
        r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template_V2\public_x.csv",
        r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template_V2\submission_template_public.csv"
    )
