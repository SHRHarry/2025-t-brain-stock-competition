import pandas as pd
import numpy as np
import joblib

def ensemble_predict(test_path, template_path):
    # 載入模型與資源
    lgb_model = joblib.load("checkpoints/derived_lgbm_model_top500.pkl")
    cat_model = joblib.load("checkpoints/derived_catboost_model_top500.pkl")
    imputer = joblib.load("checkpoints/derived_imputer_top500.pkl")
    lgb_threshold = joblib.load("checkpoints/derived_threshold_top500.pkl")
    cat_threshold = joblib.load("checkpoints/derived_catboost_threshold_top500.pkl")

    # 使用平均 threshold（也可以改為驗證後找最佳）
    threshold = (lgb_threshold*0.4) + (cat_threshold*0.6)

    # 載入 top500 特徵名稱
    top500_features = pd.read_csv("top_features/derived_top500_features.csv", header=None).squeeze("columns").tolist()

    # 載入測試資料
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df.drop(columns=["ID"], errors="ignore").select_dtypes(include=["number"])
    X_test_full = X_test_raw.reindex(columns=imputer.feature_names_in_)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_full), columns=imputer.feature_names_in_)
    X_test_top500 = X_test_imputed[top500_features]

    # 預測機率（Soft Voting）
    y_prob_lgb = lgb_model.predict(X_test_top500)
    y_prob_cat = cat_model.predict_proba(X_test_top500)[:, 1]
    y_prob_avg = (y_prob_lgb*0.4) + (y_prob_cat*0.6)
    y_pred = (y_prob_avg > threshold).astype(int)

    # 匯出 submission
    submission = pd.read_csv(template_path)
    submission["飆股"] = y_pred
    submission.to_csv("submissions/submission_weights_private_ensemble_top500.csv", index=False)
    print("✅ 已產出 submission_weights_private_ensemble_top500.csv")

def predict(test_path, template_path):
    # 載入模型與處理器
    model = joblib.load("checkpoints/derived_catboost_model_top500.pkl")
    imputer = joblib.load("checkpoints/derived_catboost_imputer_top500.pkl")
    threshold = joblib.load("checkpoints/derived_catboost_threshold_top500.pkl")

    # 正確讀取 top500 特徵名稱為 list of strings
    top500_features = pd.read_csv("top_features/derived_catboost_top500_features.csv", header=None).squeeze("columns").tolist()
    print("🔍 Top 500 features loaded:", top500_features[:5])

    # 載入測試資料
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df.drop(columns=["ID"], errors="ignore").select_dtypes(include=["number"])

    # 補上訓練時的完整欄位（保證與 imputer 對齊）
    X_test_full = X_test_raw.reindex(columns=imputer.feature_names_in_)

    # 補值
    X_test_imputed_full = pd.DataFrame(imputer.transform(X_test_full), columns=imputer.feature_names_in_)

    # 篩選 top50 特徵
    X_test_imputed_top500 = X_test_imputed_full[top500_features]

    # 推論
    # y_prob = model.predict(X_test_imputed_top500)
    # print(f"y_prob = {y_prob}, threshold = {threshold}")
    # y_pred = (y_prob > threshold).astype(int)
    y_prob = model.predict_proba(X_test_imputed_top500)[:, 1]
    print(f"y_prob = {y_prob}, threshold = {threshold}")
    y_pred = (y_prob > threshold).astype(int)

    # 匯出 submission
    submission = pd.read_csv(template_path)
    submission["飆股"] = y_pred
    submission.to_csv("submissions/submission_private_x_derived_catboost_top500.csv", index=False)
    print("✅ 已產生 submission_private_x_derived_catboost_top500.csv（使用 SHAP 精選特徵）")

if __name__ == "__main__":
    # predict(
    #     r"D:\data\38_Public_Test_Set_and_Submmision_Template_V2\derived_public_x.csv",
    #     r"D:\data\38_Public_Test_Set_and_Submmision_Template_V2\submission_template_public.csv"
    # )

    # predict(
    #     r"D:\data\38_Private_Test_Set_and_Submission_Template_V2\derived_private_x.csv",
    #     r"D:\data\38_Private_Test_Set_and_Submission_Template_V2\submission_template_private.csv"
    # )

    # ensemble_predict(
    #     r"D:\data\38_Public_Test_Set_and_Submmision_Template_V2\derived_public_x.csv",
    #     r"D:\data\38_Public_Test_Set_and_Submmision_Template_V2\submission_template_public.csv"
    # )

    ensemble_predict(
        r"D:\data\38_Private_Test_Set_and_Submission_Template_V2\derived_private_x.csv",
        r"D:\data\38_Private_Test_Set_and_Submission_Template_V2\submission_template_private.csv"
    )