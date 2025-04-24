import pandas as pd

def save_txt(dataframe, name):
    non_null_ratio = dataframe.notnull().mean().sort_values(ascending=False)
    top_columns = non_null_ratio.head(300).index.tolist()
    with open(f"titles/{name}.txt", "w") as f:
        for i in top_columns:
            f.write(i+"\n")

def clean_data_range(df, start, end, tick):
    for idx in range(start, end, tick):
        tick = round(idx*0.1, 1)
        df_processed = df.loc[:, df.isnull().mean() < tick]
        print(f"{str(tick)} size: {df_processed.shape}")
        save_txt(df_processed, f"df_{str(tick)}")
    return df_processed

def clean_data_single_tick(df, tick):
    df_processed = df.loc[:, df.isnull().mean() < 0.1]
    print(f"Processed size: {df_processed.shape}")
    df_processed = df_processed.select_dtypes(include=["number"])
    return df_processed

if __name__ == "__main__":
    # 載入資料
    df = pd.read_csv(r"D:\data\38_Training_Data_Set_V2\training.csv")
    save_txt(df, "origin")
    print(f"origin size: {df.shape}")
    
    df_processed = clean_data_range(df, start=9, end=0, tick=-1)
    df_processed = clean_data_single_tick(df, tick=0.1)
    
    df_processed.to_csv(r"D:\data\38_Training_Data_Set_V2\cleaned_01_training.csv", index=False)
    print("✅ 已儲存 cleaned_01_training.csv（移除缺失率 > 90% 的欄位、保留非空率前 300 名特徵 + 標籤欄位、去除非數值欄位）")