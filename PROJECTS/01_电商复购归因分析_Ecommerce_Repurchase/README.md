# 电商复购率下降归因分析

定位某电商平台月度复购率下降 8% 的根因。这是我的第一个数据分析项目。

## 文件结构

```
.
├── src/                            # 分析脚本（按难度递进）
│   ├── mini_project_step1.py       # 数据清洗 + 基础指标
│   ├── mini_project_step2.py       # 分国家对比分析
│   ├── mini_project_step3.py       # 品类级根因追踪
│   ├── mini_project_full.py        # 完整分析流程
│   ├── analyze_gmv.py              # GMV 趋势分析
│   ├── analyze_gmv_improved.py     # GMV 趋势（改进版）
│   ├── conversion_analysis.py      # 转化漏斗分析
│   ├── user_profile_analysis.py    # 用户画像
│   └── warm_users_analysis.py      # 活跃用户分析
├── data/                           # 原始数据 + 中间结果
│   ├── mini_project_data.csv       # 原始订单数据
│   ├── repurchase_by_month.csv     # 月度复购率
│   ├── repurchase_by_country.csv   # 分国家复购率
│   ├── gmv_by_month.csv            # 月度 GMV
│   ├── gmv_by_month_improved.csv   # GMV 趋势（改进版）
│   └── user_profiles.csv           # 用户画像数据
├── charts/                         # 12 张可视化图表
│   ├── 01_monthly_repurchase_rate.png  # 月度复购率趋势
│   ├── 02_country_repurchase_rate.png  # 国家对比
│   ├── 03_electronics_price_trend.png  # 电子品类价格暴涨 188%
│   ├── 04_orders_vs_repurchase.png     # 订单量与复购率关系
│   ├── 05_spain_vs_uk.png              # 西班牙 vs 英国对比
│   ├── 06_gmv_trend.png                # GMV 趋势
│   ├── 07_gmv_trend_improved.png       # GMV 趋势（改进版）
│   ├── 08_user_segmentation.png        # 用户分层
│   ├── 09_spend_freq_distribution.png  # 消费-频次分布
│   ├── 10_conversion_analysis.png      # 转化分析
│   ├── 11_conversion_funnel.png        # 转化漏斗
│   └── 12_warm_users_analysis.png      # 活跃用户分析
└── reports/
    └── REPORT_repurchase_analysis.md   # 完整分析报告
```

## 运行顺序

```bash
python src/mini_project_step1.py   # 数据清洗 + 基础指标
python src/mini_project_step2.py   # 分国家分析
python src/mini_project_step3.py   # 品类根因追踪
python src/mini_project_full.py    # 完整流程
```

## 核心发现

- 西班牙复购率 **0%**（所有用户仅购买一次）
- 电子品类 3 月起单价暴涨 **188%**，直接抑制复购
- 4 月核心问题是订单量下降，而非价格因素
