# 电商客户分群与 CRM 策略

基于 RFM + K-Means 的客户分群，为每个群体设计差异化运营策略。**业务决策选 K=4**（非数学最优 K=2），体现分析的业务导向。

## 文件结构

```
.
├── src/
│   └── step1_rfm_clustering.py       # RFM 计算 + K-Means 聚类 + 策略设计
├── data/
│   ├── ecommerce_transactions.csv    # 交易明细
│   ├── online_retail.xlsx            # 原始零售数据
│   └── customer_segments.csv         # 2,000 客户分群标签
├── charts/
│   └── 01_customer_segmentation.png  # 分群可视化（肘部法则 + 轮廓系数 + 散点图）
└── reports/
    ├── Customer_Segmentation_Report_June2026.docx  # 英文版报告
    └── 电商客户分群分析报告.docx                    # 中文版报告
```

## 运行

```bash
python src/step1_rfm_clustering.py
```

一次性完成：RFM 计算 → log 变换 → StandardScaler → Elbow Method + Silhouette Score 双重验证 → K=4 聚类 → 策略建议。

## 核心发现

- VIP 客户（24.4%）贡献 **86.1%** 收入（$7.3M）——帕累托效应
- 沉睡客户（30.3%）持有 $73.5 万历史收入——最大召回机会（预期 12–18× ROI）
- 数学最优 K=2，但**业务决策选 K=4**——K=2 只分「好/坏」，无法驱动差异化运营
