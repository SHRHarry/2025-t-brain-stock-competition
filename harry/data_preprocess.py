import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import FunctionTransformer

def generate_derived_features(df: pd.DataFrame, top_features: list, corr_threshold: float = 0.85):
    """
    根據 top_features 產生以下衍生特徵：
    1. 高相關特徵對之差值、比值、乘積
    2. 高偏態特徵的對數與平方根轉換
    """
    df_result = df.copy()
    # df_result = pd.DataFrame()
    # df_result["飆股"] = df["飆股"]

    # 1. 高相關特徵組合
    corr_matrix = df[top_features].corr()
    pairs = []
    for i, j in combinations(top_features, 2):
        if abs(corr_matrix.loc[i, j]) >= corr_threshold:
            pairs.append((i, j))

    for f1, f2 in pairs:
        df_result[f"{f1}_minus_{f2}"] = df[f1] - df[f2]
        df_result[f"{f1}_div_{f2}"] = df[f1] / (df[f2] + 1e-6)
        df_result[f"{f1}_times_{f2}"] = df[f1] * df[f2]

    # 2. 高偏態特徵轉換
    skewness = df[top_features].skew()
    high_skew_features = skewness[skewness.abs() > 1].index.tolist()

    for f in high_skew_features:
        df_result[f"log1p_{f}"] = np.log1p(df[f].clip(lower=0))  # clip to avoid log(negative)
        df_result[f"sqrt_{f}"] = np.sqrt(df[f].clip(lower=0))

    return df_result


if __name__ == "__main__":
    # 範例使用方式
    df = pd.read_csv(r"D:\data\38_Private_Test_Set_and_Submission_Template_V2\private_x.csv")
    # df = pd.read_csv(r"D:\data\38_Public_Test_Set_and_Submmision_Template_V2\public_x.csv")
    # df = pd.read_csv(r"D:\data\38_Training_Data_Set_V2\cleaned_01_training.csv")
    df = df.select_dtypes(include=["number"])
    top_features = pd.read_csv("top200_features.csv", header=None).squeeze("columns").tolist()

    df_derived = generate_derived_features(df, top_features)
    df_derived.to_csv(r"D:\data\38_Private_Test_Set_and_Submission_Template_V2\derived_private_x.csv", index=False)
    # df_derived.to_csv(r"D:\data\38_Public_Test_Set_and_Submmision_Template_V2\derived_public_x.csv", index=False)
    # df_derived.to_csv(r"D:\data\38_Training_Data_Set_V2\derived_only_01_training.csv", index=False)
    print("✅ 已產出 derived_private_x.csv 含衍生特徵")
