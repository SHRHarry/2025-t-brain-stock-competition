# 2025-t-brain-stock-competition

## Experiment Logs
- 2025-3-24 Harry
    1. 移除缺失率 > 90% 的欄位
    2. 去除非數值欄位 + 切分資料
    3. SMOTE 上取樣飆股，處理不平衡資料
    4. lightgbm classifier with is_unbalance=True
    5. Accuracy: 0.9945 on training data(8:2)
