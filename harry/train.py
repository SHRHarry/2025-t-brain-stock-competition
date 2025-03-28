import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train(train_path):
    # === 1. 載入資料 ===
    df = pd.read_csv(train_path)
    
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
    '''
        === 📊 評估結果 ===
    Accuracy: 0.9945
    AUC: 0.9818
    F1 Score: 0.5714
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     39879
           1       0.67      0.50      0.57       294

    accuracy                           0.99     40173
   macro avg       0.83      0.75      0.78     40173
weighted avg       0.99      0.99      0.99     40173
'''

def feature_selection_train(train_path, use_pretrained=False):
    # === 1. 讀取資料 ===
    df = pd.read_csv(train_path)

    # === 2. 清洗與前處理（與原始流程一致） ===
    df = df.loc[:, df.isnull().mean() < 0.9]
    non_null_ratio = df.notnull().mean().sort_values(ascending=False)
    top_columns = non_null_ratio.head(300).index.tolist()
    if "飆股" not in top_columns:
        top_columns.append("飆股")
    df = df[top_columns]
    df = df.select_dtypes(include=["number"])

    X = df.drop(columns=["飆股"])
    y = df["飆股"]

    if use_pretrained:
        # === 3. 填補缺失值（中位數） ===
        model_original = joblib.load("lgbm_model.pkl")
        imputer_original = joblib.load("imputer.pkl")
        X_imputed = pd.DataFrame(imputer_original.transform(X), columns=X.columns)

        # === 4. 使用原模型的重要性選出 Top N 特徵 ===
        importances = model_original.feature_importances_
    else:
        # === 3. 填補缺失值（中位數） ===
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # === 4. 使用原始模型找出 top N 特徵 ===
        base_model = LGBMClassifier(random_state=42, is_unbalance=True)
        base_model.fit(X_imputed, y)
        importances = base_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_n = 50
    top_features = importance_df.head(top_n)["feature"].tolist()

    # === 5. 畫圖顯示前 50 特徵 ===
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.figure(figsize=(10, 12))
    sns.barplot(data=importance_df.head(top_n), y="feature", x="importance", palette="viridis")
    
    if use_pretrained:
        plt.title(f"Top {top_n} Feature Importances (From Trained Model)")
        plt.tight_layout()
        plt.savefig("top_features_from_original_model.png")
    else:
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.savefig("top_features.png")
    print(f"✅ 已儲存前 {top_n} 特徵圖 top_features.png")

    # === 6. 使用 top N 特徵重建資料與 Imputer ===
    X_top = X[top_features]
    imputer_top = SimpleImputer(strategy="median")
    X_top_imputed = pd.DataFrame(imputer_top.fit_transform(X_top), columns=X_top.columns)

    # === 7. 切分訓練與測試集 ===
    for seed in range(100):
        X_train, X_test, y_train, y_test = train_test_split(
            X_top_imputed, y, stratify=y, test_size=0.2, random_state=seed)
        if y_train.nunique() == 2:
            break

    # === 8. SMOTE 處理不平衡資料 ===
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # === 9. 訓練新模型 ===
    model_top = LGBMClassifier(random_state=42, is_unbalance=True)
    model_top.fit(X_train_resampled, y_train_resampled)

    # === 10. 評估新模型 ===
    y_pred = model_top.predict(X_test)
    y_prob = model_top.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    if use_pretrained:
        print("\n=== 📊 使用原模型 importance 的 Top Features 評估結果 ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

        # === 10. 儲存模型與前處理器 ===
        joblib.dump(model_top, "lgbm_model_top_from_original.pkl")
        joblib.dump(imputer_top, "imputer_top_from_original.pkl")
        joblib.dump(top_features, "top_features_from_original.pkl")
        print("✅ 模型、Imputer、Top Features 已儲存 (from original model)")
        '''
        === 📊 使用原模型 importance 的 Top Features 評估結果 ===
        Accuracy: 0.9948
        AUC: 0.9800
        F1 Score: 0.5625
                      precision    recall  f1-score   support

                   0       1.00      1.00      1.00     39879
                   1       0.73      0.46      0.56       294

            accuracy                           0.99     40173
           macro avg       0.86      0.73      0.78     40173
        weighted avg       0.99      0.99      0.99     40173
        '''
    else:
        print("\n=== 📊 Top Features 模型評估結果 ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

        # === 11. 儲存模型與 Imputer 與 top feature 欄位名 ===
        joblib.dump(model_top, "lgbm_model_top.pkl")
        joblib.dump(imputer_top, "imputer_top.pkl")
        joblib.dump(top_features, "top_features.pkl")
        print("✅ 模型、Imputer 與欄位名稱已儲存")
        '''
            === 📊 Top Features 模型評估結果 ===
        Accuracy: 0.9777
        AUC: 0.9836
        F1 Score: 0.3421
                      precision    recall  f1-score   support

                   0       1.00      0.98      0.99     39879
                   1       0.22      0.79      0.34       294

            accuracy                           0.98     40173
           macro avg       0.61      0.89      0.67     40173
        weighted avg       0.99      0.98      0.98     40173
        '''

if __name__ == "__main__":
    # train(r"E:\side_project\stock_competition\data\38_Training_Data_Set\training.csv")
    feature_selection_train(r"E:\side_project\stock_competition\data\38_Training_Data_Set\training.csv",
                            use_pretrained=True)