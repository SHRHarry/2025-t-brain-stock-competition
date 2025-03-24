import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

# === 1. 載入資料 ===
df = pd.read_csv(r"E:\side_project\stock_competition\data\38_Training_Data_Set\training.csv")

# === 2. 移除缺失率 > 90% 的欄位 ===
df = df.loc[:, df.isnull().mean() < 0.9]

# === 3. 保留非空率前 300 名特徵 + 標籤欄位 ===
non_null_ratio = df.notnull().mean().sort_values(ascending=False)
top_columns = non_null_ratio.head(300).index.tolist()
if "飆股" not in top_columns:
    top_columns.append("飆股")
df = df[top_columns]

# === 4. 去除非數值欄位 + 切分資料 ===
df = df.select_dtypes(include=["number"])
X = df.drop(columns=["飆股"])
y = df["飆股"]

# 若 y 中只有一類，直接跳出提示
if y.nunique() < 2:
    raise ValueError("資料中僅包含單一類別，無法訓練模型")

# === 5. 填補缺失值（中位數） ===
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# === 6. 切分資料 ===
# 找一組包含兩類的切法（避免 SMOTE 錯誤）
for seed in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, stratify=y, test_size=0.2, random_state=seed
    )
    if y_train.nunique() == 2:
        print(f"[✓] 成功找到包含兩類別的訓練集 seed = {seed}")
        break
else:
    raise ValueError("無法找到同時包含 0/1 的訓練資料")

# === 7. SMOTE 上取樣 ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === 8. LightGBM 訓練（處理不平衡類別） ===
model = LGBMClassifier(random_state=42, is_unbalance=True)
model.fit(X_train_resampled, y_train_resampled)

# === 9. 預測與評估 ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n=== 📊 評估結果 ===")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(report)

# === 10. 儲存模型與前處理器 ===
joblib.dump(model, "lgbm_model.pkl")
joblib.dump(imputer, "imputer.pkl")
print("\n✅ 模型與 Imputer 已儲存！")
