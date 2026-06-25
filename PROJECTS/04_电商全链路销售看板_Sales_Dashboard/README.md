# 电商全链路经营 BI 看板

面向管理层的 Tableau 看板，覆盖经营总览、促销效率、客户价值三个决策场景。

## 文件结构

```
.
├── data/                          # 19 个分析就绪 CSV 数据集
│   ├── kpi_summary.csv            # KPI 汇总
│   ├── monthly_trend.csv          # 月度趋势
│   ├── category_abc.csv           # ABC 品类分析
│   ├── region_analysis.csv        # 区域分布
│   ├── channel_analysis.csv       # 渠道分析
│   ├── promo_vs_normal.csv        # 促销 vs 正常对比
│   ├── discount_profit_impact.csv # 折扣盈利影响
│   ├── profitability_matrix.csv   # 盈利矩阵
│   ├── customer_segment_summary.csv # 客户分群汇总
│   ├── province_analysis.csv      # 省份分析
│   ├── yoy_growth.csv             # 同比增长
│   └── ...                        # 更多维度数据集
├── notebooks/
│   ├── step0_generate_data.py     # 生成 66,925 条订单模拟数据
│   └── step1_full_analysis.py     # 全维度 EDA 分析
├── sql/
│   └── analytics_queries.sql      # 10 条面试级 SQL（窗口函数、CTE、多表 JOIN）
├── tableau/
│   └── Ecommerce_Sales_Dashboard.twb  # Tableau 工作簿（3 个仪表板）
└── reports/
    └── charts/                    # 7 张分析图表
        ├── 01_monthly_dashboard.png
        ├── 02_category_abc.png
        ├── 03_region_ranking.png
        ├── 04_channel_analysis.png
        ├── 05_customer_segments.png
        ├── 06_promotion_analysis.png
        └── 07_profitability_matrix.png
```

## 运行

```bash
# 1. 生成数据（需要先跑）
python notebooks/step0_generate_data.py

# 2. 全维度探索性分析
python notebooks/step1_full_analysis.py

# 3. 用 Tableau Desktop 打开
# 文件：tableau/Ecommerce_Sales_Dashboard.twb
```

## 三个仪表板

| 仪表板 | 关注维度 | 核心洞察 |
|--------|---------|---------|
| 经营总览 | KPI 卡片 + 月度趋势 + 品类构成 | 手机数码占 GMV 46.3%、APP 渠道占 35% |
| 促销效率 | 盈利矩阵 + 折扣影响 | 促销订单占 23.8%，折扣 > 10% 即亏损 |
| 客户价值 | RFM 分群 + 区域分布 | 一线城市贡献 52% GMV |

## SQL 面试考点

`sql/analytics_queries.sql` 包含 10 条可独立运行的 SQL 查询，覆盖窗口函数（ROW_NUMBER、RANK、LAG）、CTE、多表 JOIN、RFM 计算——均为数据分析面试高频考点。
