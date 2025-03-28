import pandas as pd
import joblib
import numpy as np

def predict(test_path, template_path):
    # === 1. 讀取資料 ===
    test_df = pd.read_csv(test_path)
    submission_template = pd.read_csv(template_path)

    # === 2. 載入模型與 Imputer ===
    model = joblib.load("lgbm_model.pkl")
    imputer = joblib.load("imputer.pkl")

    # print(f"model.feature_importances_ = {model.feature_importances_}")

    # === 3. 特徵預處理（與訓練邏輯一致） ===
    # 移除非數值欄位與 ID
    X_test = test_df.drop(columns=["ID"], errors="ignore")
    X_test = X_test.select_dtypes(include=["number"])

    # 只保留與訓練階段一致的欄位（Imputer 記錄的欄位）
    if hasattr(imputer, "feature_names_in_"):
        model_columns = imputer.feature_names_in_
        X_test = X_test.reindex(columns=model_columns)

    # 填補缺失值
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # === 4. 模型推論 ===
    y_pred = model.predict(X_test_imputed)

    # === 5. 套用到 submission 格式 ===
    submission = submission_template.copy()
    submission["飆股"] = y_pred

    # === 6. 輸出結果 ===
    submission.to_csv("submissions/submission.csv", index=False)
    print("✅ 已產生 submission.csv")

def feature_selection_predict(test_path, template_path, model_from_original=False):
    # === 1. 載入模型、Imputer、Top N 特徵名稱 ===
    if model_from_original:
        model = joblib.load("lgbm_model_top_from_original.pkl")
        imputer = joblib.load("imputer_top_from_original.pkl")
        top_features = joblib.load("top_features_from_original.pkl")
    else:
        model = joblib.load("lgbm_model_top.pkl")
        imputer = joblib.load("imputer_top.pkl")
        top_features = joblib.load("top_features.pkl")

    # === 2. 載入測試資料與提交範本 ===
    test_df = pd.read_csv(test_path)
    submission_template = pd.read_csv(template_path)

    # === 3. 資料前處理 ===
    X_test = test_df.drop(columns=["ID"], errors="ignore")
    X_test = X_test.select_dtypes(include=["number"])
    X_test = X_test.reindex(columns=top_features)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=top_features)

    # === 4. 模型推論 ===
    y_pred = model.predict(X_test_imputed)

    # === 5. 輸出 submission.csv ===
    submission = submission_template.copy()
    submission["飆股"] = y_pred
    if model_from_original:
        submission.to_csv("submissions/submission_from_original.csv", index=False)
        print("✅ 已產生 submission_from_original.csv")
    else:
        submission.to_csv("submissions/submission.csv", index=False)
        print("✅ 已產生 submission.csv")

if __name__ == "__main__":
    # predict(r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template\public_x.csv",
    #         r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template\submission_template_public.csv")
    feature_selection_predict(r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template\public_x.csv",
            r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template\submission_template_public.csv",
            model_from_original=True)