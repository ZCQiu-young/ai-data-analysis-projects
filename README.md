# 数据分析项目 Portfolio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python" alt="Python 3.9">
  <img src="https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-2.x-green?logo=pandas" alt="pandas">
  <img src="https://img.shields.io/badge/SQL-SQLite-blue?logo=sqlite" alt="SQL">
  <img src="https://img.shields.io/badge/Tableau-10.5-blue?logo=tableau" alt="Tableau">
  <img src="https://img.shields.io/badge/NLP-LDA-green?logo=nltk" alt="NLP">
  <img src="https://img.shields.io/badge/Stats-χ²_OR_RR-red" alt="Statistics">
</p>

**5 个端到端数据分析项目**，覆盖假设驱动分析、有监督学习、无监督学习、BI 看板、NLP 文本分析。从真实数据到业务策略，全链路闭环。

> 📄 简历精选了其中 3 个最硬核的项目（差评归因 / 流失预测 / BI 看板）。完整代码和报告见各子目录 [PROJECTS/](PROJECTS/)。

---

## 项目概览

| # | 项目 | 核心方法 | 关键成果 |
|---|------|---------|---------|
| 1 | [电商复购率下降归因分析](PROJECTS/01_电商复购归因分析_Ecommerce_Repurchase) | EDA · 假设驱动分析 · 可视化 | 定位西班牙 0% 复购率、电子品类涨价 188% 根因 |
| 2 | [电信客户流失预测模型](PROJECTS/02_电信客户流失预测_Telecom_Churn) | 逻辑回归 · L1 正则化 · 特征工程 | AUC 0.848 · 辛普森悖论 · 锁定 1,697 名临界风险客户 |
| 3 | [电商客户分群与 CRM 策略](PROJECTS/03_电商客户分群_Customer_Segmentation) | RFM · K-Means · 无监督学习 | 4 类可运营群体 · VIP 用 24.4% 人数贡献 86.1% 收入 |
| 4 | [电商全链路经营 BI 看板](PROJECTS/04_电商全链路销售看板_Sales_Dashboard) | SQL · Tableau · 9 维分析框架 | 3 个交互仪表板 · 10 条面试级 SQL 查询 |
| 5 | [Olist 巴西电商差评分析](PROJECTS/05_Olist_Bad_Review_Analysis) | EDA→SQL→统计检验→NLP→LDA→策略 | 六步链路 · 差评三类型定位 · RR=6.9× · 品类三级干预 |

---

## 技术栈

| 层级 | 工具与方法 |
|------|-----------|
| **编程** | Python（pandas、NumPy、scikit-learn、matplotlib、seaborn） |
| **机器学习** | 逻辑回归（L1 正则化）、K-Means 聚类、LDA 主题建模、特征工程、交叉验证 |
| **统计学** | 卡方检验、分层分析、逻辑回归（OR）、相对风险（RR）、剂量反应关系 |
| **自然语言处理** | CountVectorizer、停用词过滤、n-gram、关键词提取、LDA 主题建模 |
| **SQL** | 窗口函数（ROW_NUMBER/RANK/LAG）、CTE、子查询、多表 JOIN（6+ 表联查） |
| **BI 与可视化** | Tableau 10.5、matplotlib、seaborn |
| **分析方法论** | EDA、假设驱动分析、RFM 模型、MECE 拆解、辛普森悖论识别、六步分析链路 |

---

## 仓库结构

```
PROJECTS/
├── 01_电商复购归因分析_Ecommerce_Repurchase/    # 项目1：复购率下降归因
│   ├── src/       9 个分析脚本
│   ├── data/      原始数据 + 中间结果
│   ├── charts/    12 张可视化图表
│   └── reports/   完整分析报告
│
├── 02_电信客户流失预测_Telecom_Churn/           # 项目2：流失预测模型
│   ├── src/       6 步分析脚本
│   ├── data/      Kaggle 数据 + 模型输出
│   ├── charts/    5 张评估图表
│   └── reports/   中文 + 英文双版报告
│
├── 03_电商客户分群_Customer_Segmentation/       # 项目3：RFM + K-Means 分群
│   ├── src/       分群脚本
│   ├── data/      交易数据 + 分群标签
│   ├── charts/    分群可视化
│   └── reports/   中文 + 英文双版报告
│
├── 04_电商全链路销售看板_Sales_Dashboard/        # 项目4：BI 经营看板
│   ├── data/      19 个分析就绪 CSV
│   ├── notebooks/ 数据生成 + EDA 脚本
│   ├── sql/       10 条面试级 SQL 查询
│   ├── tableau/   Tableau 工作簿（3 个仪表板）
│   └── reports/   7 张分析图表
│
└── 05_Olist_Bad_Review_Analysis/                # 项目5：差评驱动因素全链路分析
    ├── data/      9 张原始表 + SQLite 数据库 + Tableau 数据
    ├── notebooks/ 6 步分析脚本（EDA→SQL→统计→NLP→LDA→策略）
    ├── tableau/   Tableau 工作簿
    ├── visuals/   EDA 可视化
    └── reports/   完整六步分析报告
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/ZCQiu-young/ai-data-analysis-projects.git
cd ai-data-analysis-projects

# 安装依赖
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# 运行项目
cd "PROJECTS/02_电信客户流失预测_Telecom_Churn/src"
python step1_eda.py
python step4_modeling.py
```

Tableau 看板用 Tableau Desktop 打开各项目 `tableau/` 目录下的 `.twb` 文件。

> ⚠️ 大文件（SQLite 数据库、部分 CSV）已加入 `.gitignore`。如需完整数据集，请参考各项目脚本从原始数据重新生成。
