# 电信客户流失预测模型

构建逻辑回归模型预测客户流失，AUC = 0.848。识别并矫正辛普森悖论。

## 文件结构

```
.
├── src/                              # 5 步分析脚本
│   ├── step1_eda.py                  # 探索性分析 + 流失画像
│   ├── step2_new_customer_churn.py   # 新客户专项分析（首年流失率 47.4%）
│   ├── step3_strategy_quantify.py    # 挽留策略量化（预计降 37% 流失）
│   ├── step4_modeling.py             # 逻辑回归建模（L1 正则化，AUC=0.848）
│   ├── step4b_model_optimization.py  # 模型优化 + 阈值调优
│   └── step5_critical_list.py        # 输出 1,697 名高风险客户名单
├── data/
│   ├── telco_churn.csv               # Kaggle 原始数据（7,043 条）
│   ├── churn_risk_scores.csv         # 全员风险评分
│   ├── critical_risk_customers.csv   # 需挽留客户名单
│   └── churn_predictions_optimized.csv # 优化后预测结果
├── charts/
│   ├── 01_churn_overview.png         # 流失概况
│   ├── 02_new_customer_churn.png     # 新客户流失分析
│   ├── 03_retention_strategies.png   # 策略量化效果
│   ├── 04_churn_prediction_model.png # ROC 曲线 + 混淆矩阵
│   └── 05_model_optimization.png     # 阈值优化
└── reports/
    ├── CEO_Churn_Report_June2026.docx # 英文版 CEO 报告
    └── 电信客户流失预测分析报告.docx   # 中文版分析报告
```

## 运行顺序

```bash
python src/step1_eda.py                # 必须先跑，生成 EDA 图表
python src/step2_new_customer_churn.py
python src/step3_strategy_quantify.py
python src/step4_modeling.py           # 训练模型
python src/step4b_model_optimization.py
python src/step5_critical_list.py      # 输出挽留名单
```

## 核心发现

- 识别**辛普森悖论**：光纤用户总流失率高，但真正根因是「光纤 + 月付合约」组合
- 新客户首年流失率 **47.4%**，第一年是「生死窗口」
- 锁定 **1,697 名**临界风险客户，涉及 **$1.74M** 可挽留营收
