import pandas as pd

def save_txt(dataframe, name):
    non_null_ratio = dataframe.notnull().mean().sort_values(ascending=False)
    top_columns = non_null_ratio.head(300).index.tolist()
    with open(f"{name}.txt", "w") as f:
        for i in top_columns:
            f.write(i+"\n")

if __name__ == "__main__":
    # 載入資料
    df = pd.read_csv(r"D:\data\38_Training_Data_Set_V2\training.csv")
    save_txt(df, "origin")
    print(f"origin size: {df.shape}")
    
    # for idx in range(9, 0, -1):
    #     tick = round(idx*0.1, 1)
    #     df_processed = df.loc[:, df.isnull().mean() < tick]
    #     print(f"{str(tick)} size: {df_processed.shape}")
    #     save_txt(df_processed, f"df_{str(tick)}")

    df_01 = df.loc[:, df.isnull().mean() < 0.1]
    print(f"Processed size: {df_01.shape}")
    df_01 = df_01.select_dtypes(include=["number"])
    df_01.to_csv(r"D:\data\38_Training_Data_Set_V2\cleaned_01_training.csv", index=False)
    
    # non_null_ratio = df.notnull().mean().sort_values(ascending=False)
    # top_columns = non_null_ratio.head(300).index.tolist()
    # if "飆股" not in top_columns:
    #     top_columns.append("飆股")
    # df = df[top_columns]
    # df = df.select_dtypes(include=["number"])
    # df.to_csv(r"E:\side_project\stock_competition\data\38_Training_Data_Set\cleaned_training.csv", index=False)
    # print("✅ 已儲存 cleaned_training.csv（移除缺失率 > 90% 的欄位、保留非空率前 300 名特徵 + 標籤欄位、去除非數值欄位）")