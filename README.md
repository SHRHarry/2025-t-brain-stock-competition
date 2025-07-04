# 2025永豐AI GO競賽：股神對決

[![Competition](https://img.shields.io/badge/T--Brain-Competition-blue)](https://tbrain.trendmicro.com.tw/Competitions/Details/38)

## Introduction
> 訓練資料集欄位約10200欄，且大部分欄位有缺值，競賽目標是找出是否為飆股，計分方式採用`F1 score`，最終排名：`48/868`

## Experiment Logs
- 2025-04-22 Harry
    1. 做最後的掙扎，將lightGBM及CatBoost作soft voting，score on public testing data
        | LGBM:CatBoost | F1 Score |
        |---------------|----------|
        |      5:5      | `0.8013` |
        |      6:4      |  0.7933  |
        |      4:6      | `0.8013` |
        |      2:8      |  0.7987  |
    2. 最終比賽結果為`第48名`，score on private testing data
        | F1 Score |
        |----------|
        | `0.8076` |
- 2025-04-21 Harry
    1. 沿用04-17的作法，更換模型為CatBoost，多項指標達career high，performance on training data(8:2)
        | 資料量     | F1 平均值 | F1 標準差 | AUC 平均值 | AUC 標準差 | Threshold 平均值 | Threshold 標準差 | 綜合穩定性指標 |
        |------------|-----------|-----------|-------------|-------------|------------------|------------------|----------------|
        | 500筆資料  | `0.8085`  | `0.0101`   | `0.9926`    | `0.0009`    | 0.2093           | 0.0370           | `0.8035`       |
    2. Score on public testing data
        | F1 Score |
        |----------|
        | `0.7948` |
- 2025-04-17 Harry
    1. 沿用昨天作法，調整top筆數，performance on training data(8:2)
        | 資料量     | F1 平均值 | F1 標準差 | AUC 平均值 | AUC 標準差 | Threshold 平均值 | Threshold 標準差 | 綜合穩定性指標 |
        |------------|-----------|-----------|-------------|-------------|------------------|------------------|----------------|
        | 500筆資料  | 0.7443    | 0.0183     | 0.9918      | 0.0013      | 0.8635           | 0.0199           | `0.7351`       |
    2. Score on public testing data
        | F1 Score |
        |----------|
        | `0.7442` |
- 2025-04-16 Harry
    1. 衍生特徵實驗：
        - 產生一組「高相關特徵對的交互衍生特徵」
        - 產生「偏態修正特徵（如 log、sqrt）」
        - 利用衍生特徵+原特徵進行訓練
    2. performance on training data(8:2)
        | 資料量     | F1 平均值 | F1 標準差 | AUC 平均值 | AUC 標準差 | Threshold 平均值 | Threshold 標準差 | 綜合穩定性指標 |
        |------------|-----------|-----------|-------------|-------------|------------------|------------------|----------------|
        | 200筆資料  | 0.7292    | `0.0137`   | 0.9916      | 0.0013      | 0.8838           | 0.0189           | `0.7223`       |
    3. Score on public testing data
        | F1 Score |
        |----------|
        | `0.7365` |
- 2025-04-15 Harry
    1. 資料前處理實驗：刪除缺值率10%/20%/30%/.../90%的欄位會剩下多少欄位
        - 發現10%~80%刪除的欄位沒很多，但是到90%時斷崖式下降至2000多欄位。目前後續實驗主要基於此2000的欄位進行
    2. SHAP 分析 (僅對飆股類別進行)，選出 SHAP 平均重要性前200/300/500/2000比特徵篩選特徵重新訓練。觀察到200筆的F1 標準差下降幅度最大，泛用性最佳，故選用200筆的實驗結果
    4. performance on training data(8:2)
        | 資料量     | F1 平均值 | F1 標準差 | AUC 平均值 | AUC 標準差 | Threshold 平均值 | Threshold 標準差 | 綜合穩定性指標 |
        |------------|-----------|-----------|-------------|-------------|------------------|------------------|----------------|
        | 500筆資料  | 0.7454    | 0.0280     | 0.9894      | 0.0020      | 0.4473           | 0.1208           | 0.7314         |
        | 2000筆資料 | 0.6475    | 0.0325     | 0.9883      | 0.0015      | 0.8218           | 0.0293           | 0.6313         |
        | 300筆資料  | 0.6580    | 0.0272     | 0.9896      | 0.0016      | 0.8505           | 0.0130           | 0.6444         |
        | 200筆資料  | 0.6643    | `0.0162`   | 0.9898      | 0.0014      | 0.8619           | 0.0188           | 0.6562         |
    5. Score on public testing data(最終選用200筆的實驗結果)
        | F1 Score |
        |----------|
        | `0.6997` |
- 2025-04-06 Harry
    1. SHAP 分析 (僅對飆股類別進行)，選出 SHAP 平均重要性前 50 名特徵篩選特徵重新訓練
    2. 計算 scale_pos_weight，找出「是飆股」&「不是飆股」的比例
    3. 交叉驗證與並找出最佳threshold
    4. performance on training data(8:2) --> 不佳，不列出來
    5. Score on public testing data
        | F1 Score |
        |----------|
        | 0.5287   |
- 2025-03-28 Harry
    1. 原作法訓練好的模型作為pretrained，找出 top N importance的特徵，篩選特徵重新訓練
    2. 先訓練好一個base model用來找出top N importance的特徵，篩選特徵重新訓練
    3. performance on training data(8:2)
        | Accuracy | AUC    | F1 Score |
        |----------|--------|----------|
        | 0.9777   | 0.9836 | 0.3421   |
    4. Score on public testing data
        | F1 Score |
        |----------|
        | 0.5821   |
- 2025-03-24 Harry
    1. 移除缺失率 > 90% 的欄位
    2. 去除非數值欄位 + 切分資料
    3. SMOTE 上取樣飆股，處理不平衡資料
    4. lightGBM classifier with is_unbalance=True
    5. performance on training data(8:2)
        | Accuracy |
        |----------|
        | 0.9945   |
    5. Score on public testing data
        | F1 Score |
        |----------|
        | 0.5375   |
