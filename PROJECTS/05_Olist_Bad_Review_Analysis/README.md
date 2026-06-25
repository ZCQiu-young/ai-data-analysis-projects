# Olist 电商差评根因分析

9 表联查、六步分析链路（EDA→SQL→统计检验→NLP→LDA→策略），定位差评根因。

## 文件结构

```
.
├── data/                               # 9 张原始 CSV + SQLite 数据库 + Tableau 数据
│   ├── olist_orders_dataset.csv        # 订单主表（99,441 条）
│   ├── olist_order_items_dataset.csv   # 订单明细
│   ├── olist_order_payments_dataset.csv # 支付信息
│   ├── olist_order_reviews_dataset.csv # 用户评分 + 评论文本
│   ├── olist_customers_dataset.csv     # 客户地理位置
│   ├── olist_products_dataset.csv      # 商品品类/属性
│   ├── olist_sellers_dataset.csv       # 卖家信息
│   ├── olist_geolocation_dataset.csv   # 巴西邮编经纬度
│   ├── product_category_name_translation.csv # 品类名翻译（葡→英）
│   ├── olist.db                       # SQLite 数据库（导入后多表联查）
│   ├── td_kpi_summary.csv             # Tableau KPI 预聚合数据
│   ├── td_monthly_trend.csv           # 月度差评趋势
│   ├── td_delay_dose.csv              # 延迟剂量-反应数据
│   ├── td_group_compare.csv           # 差评/非差评分组对比
│   └── td_category_risk.csv           # 品类风险分层
├── notebooks/                          # 六步分析脚本（按顺序运行）
│   ├── 01_eda.py                       # 全貌探索
│   ├── 02_sql_analysis.py             # SQL 多表联查
│   ├── 03_statistical_analysis.py     # 统计检验（卡方/分层/逻辑回归）
│   ├── 04_nlp_analysis.py             # NLP 评论文本分析
│   ├── 05_lda_topic_modeling.py       # LDA 主题建模
│   └── 06_category_risk_analysis.py   # 品类关联规则 + 三级干预策略
├── visuals/                            # EDA 可视化
│   ├── eda_review_score.png           # 评分分布
│   ├── eda_time_status.png           # 交付状态
│   ├── eda_category_gmv.png          # 品类 GMV
│   ├── eda_payment.png               # 支付方式
│   ├── eda_seller_pareto.png         # 卖家帕累托
│   └── eda_delay_distribution.png    # 延迟分布
├── tableau/
│   └── Olist_Dashboard.twb           # Tableau 工作簿
└── reports/
    ├── eda_summary_2026-06-17.md     # EDA 阶段小结
    └── olist_analysis_report.md      # 完整六步分析报告
```

## 运行顺序

```bash
# 按顺序运行，每步输出是下步的输入
python notebooks/01_eda.py
python notebooks/02_sql_analysis.py
python notebooks/03_statistical_analysis.py
python notebooks/04_nlp_analysis.py
python notebooks/05_lda_topic_modeling.py
python notebooks/06_category_risk_analysis.py
```

## 六步分析链路

每一步回答一个递进的问题：

```
全貌探索 → 差评有多严重？
SQL 联查 → 配送延迟是元凶吗？
统计检验 → 错过了多少非延迟差评？
NLP 分析 → 他们到底在骂什么？
LDA 建模 → 能不能自动归类？
品类策略 → 不同品类该怎么管？
```

## 核心发现

- **差评三大类型（LDA 自动聚类）：** 配送（49%）/ 产品（36%）/ 服务（15%）
- 延迟订单差评率 62.4%，但 **70.3% 差评来自准时到货订单**——产品描述不符才是基本盘
- 统计链路：χ² 检验 → 分层分析 → 逻辑回归 OR=1.89 → **RR=6.9×**
- 品类三级干预策略：🔴 fashion_shoes / 🟡 bed_bath_table / 🟢 health_beauty
