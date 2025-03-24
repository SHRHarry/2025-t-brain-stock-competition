import pandas as pd
import joblib
import numpy as np

# === 1. 讀取資料 ===
test_df = pd.read_csv(r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template\public_x.csv")
submission_template = pd.read_csv(r"E:\side_project\stock_competition\data\38_Public_Test_Set_and_Submmision_Template\submission_template_public.csv")

# === 2. 載入模型與 Imputer ===
model = joblib.load("lgbm_model.pkl")
imputer = joblib.load("imputer.pkl")

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
submission.to_csv("submission.csv", index=False)
print("✅ 已產生 submission.csv")
