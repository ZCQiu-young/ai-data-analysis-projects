# %% [markdown]
# 项目5：Olist 巴西电商 — 探索性数据分析（EDA）
# 
# **分析目标**：
# 1. 理解9张表的数据结构、缺失情况、时间跨度
# 2. 回答3个核心业务问题：
#    - 订单延迟有多严重？什么因素跟延迟相关？
#    - 用户评分分布如何？差评跟什么有关？
#    - 销售/卖家/品类有什么特征？
# 3. 确定后续深挖方向（SQL查询 + 建模）

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'D:/AI_Dateannaly/PROJECTS/project_olist_analysis/data/'

# %% [markdown]
# ## 1. 数据加载与初探

# %%
# 加载所有表
orders = pd.read_csv(DATA_DIR + 'olist_orders_dataset.csv')
customers = pd.read_csv(DATA_DIR + 'olist_customers_dataset.csv')
items = pd.read_csv(DATA_DIR + 'olist_order_items_dataset.csv')
payments = pd.read_csv(DATA_DIR + 'olist_order_payments_dataset.csv')
reviews = pd.read_csv(DATA_DIR + 'olist_order_reviews_dataset.csv')
products = pd.read_csv(DATA_DIR + 'olist_products_dataset.csv')
sellers = pd.read_csv(DATA_DIR + 'olist_sellers_dataset.csv')
geo = pd.read_csv(DATA_DIR + 'olist_geolocation_dataset.csv')
cat_trans = pd.read_csv(DATA_DIR + 'product_category_name_translation.csv')

print('='*60)
print('[STATS] 数据集总览')
print('='*60)
datasets = {
    'orders': orders, 'customers': customers, 'order_items': items,
    'payments': payments, 'reviews': reviews, 'products': products,
    'sellers': sellers, 'geolocation': geo, 'category_translation': cat_trans
}
for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    print(f'{name:20s}  {df.shape[0]:>10,} rows × {df.shape[1]} cols  |  缺失值: {missing:>8,}')

# %%
# 时间转换
time_cols = ['order_purchase_timestamp', 'order_approved_at', 
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in time_cols:
    orders[col] = pd.to_datetime(orders[col])

# %% [markdown]
# ## 2. 时间范围 & 订单状态

# %%
print('[TIME] 数据时间跨度：')
print(f'  最早订单: {orders["order_purchase_timestamp"].min()}')
print(f'  最晚订单: {orders["order_purchase_timestamp"].max()}')

# 按月统计订单量
orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
monthly = orders.groupby('order_month').size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 月度趋势
monthly.plot(kind='bar', ax=axes[0], color='steelblue', width=0.8)
axes[0].set_title('月度订单量趋势', fontsize=13, fontweight='bold')
axes[0].set_ylabel('订单数')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)

# 订单状态分布
status_counts = orders['order_status'].value_counts()
status_counts.plot(kind='barh', ax=axes[1], color=['#2E86AB','#A23B72','#F18F01','#C73E1D','#6A994E','#386641'])
axes[1].set_title('订单状态分布', fontsize=13, fontweight='bold')
for i, v in enumerate(status_counts.values):
    axes[1].text(v + 500, i, f'{v:,} ({v/len(orders)*100:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('D:/AI_Dateannaly/PROJECTS/project_olist_analysis/visuals/eda_time_status.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **关键发现**：
# - 数据覆盖 2016.09 ~ 2018.08，共 24 个月
# - 97%+ 订单已交付（delivered），数据质量适合分析
# - 2017 年底有明显增长高峰（黑色星期五？圣诞节？）

# %% [markdown]
# ## 3. 核心问题①：订单延迟分析

# %%
# 只分析已交付订单
delivered = orders[orders['order_status'] == 'delivered'].copy()

# 计算实际交付天数 vs 预估交付天数
delivered['actual_delivery_days'] = (
    delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']
).dt.days

delivered['estimated_delivery_days'] = (
    delivered['order_estimated_delivery_date'] - delivered['order_purchase_timestamp']
).dt.days

# 延迟 = 实际 - 预估（正数=延迟，负数=提前）
delivered['delay_days'] = delivered['actual_delivery_days'] - delivered['estimated_delivery_days']

# 基本统计
print('[DELAY] 交付时效统计：')
print(f'  平均实际交付: {delivered["actual_delivery_days"].mean():.1f} 天')
print(f'  平均预估交付: {delivered["estimated_delivery_days"].mean():.1f} 天')
print(f'  中位延迟天数: {delivered["delay_days"].median():.1f} 天')

# 延迟率
delayed_pct = (delivered['delay_days'] > 0).mean() * 100
on_time_pct = (delivered['delay_days'] <= 0).mean() * 100
severe_delay_pct = (delivered['delay_days'] > 10).mean() * 100

print(f'\n[STATS] 延迟分布：')
print(f'  按时/提前交付: {on_time_pct:.1f}%')
print(f'  延迟交付:       {delayed_pct:.1f}%')
print(f'  严重延迟(>10天): {severe_delay_pct:.1f}%')

# 可视化延迟分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 直方图
axes[0].hist(delivered['delay_days'].clip(-30, 50), bins=80, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='准时线')
axes[0].set_title('交付延迟天数分布', fontsize=13, fontweight='bold')
axes[0].set_xlabel('延迟天数（负=提前，正=延迟）')
axes[0].set_ylabel('订单数')
axes[0].legend()

# 箱线图
delay_by_status = [
    delivered.loc[delivered['delay_days'] <= 0, 'delay_days'],
    delivered.loc[(delivered['delay_days'] > 0) & (delivered['delay_days'] <= 10), 'delay_days'],
    delivered.loc[delivered['delay_days'] > 10, 'delay_days']
]
bp = axes[1].boxplot(delay_by_status, labels=['按时/提前', '轻度延迟(≤10天)', '严重延迟(>10天)'],
                     patch_artist=True)
for patch, color in zip(bp['boxes'], ['#6A994E', '#F18F01', '#C73E1D']):
    patch.set_facecolor(color)
axes[1].set_title('不同延迟等级的交付天数分布', fontsize=13, fontweight='bold')
axes[1].set_ylabel('实际交付天数')

plt.tight_layout()
plt.savefig('D:/AI_Dateannaly/PROJECTS/project_olist_analysis/visuals/eda_delay_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **[*] 分析思维**：
# 
# 延迟是电商的致命问题——巴西物流基础设施不如中国，延迟会导致：
# 1. 差评 → 复购率下降
# 2. 退货/退款 → 成本增加
# 3. NPS 下降 → 口碑变差
# 
# **下一步方向**：延迟跟什么因素相关？州？卖家？品类？支付方式？

# %% [markdown]
# ## 4. 核心问题②：评分分析

# %%
print('[RATING] 评分分布：')
score_dist = reviews['review_score'].value_counts().sort_index()
for s, c in score_dist.items():
    bar = '█' * int(c / 1000)
    print(f'  {s}分: {c:>7,} ({c/len(reviews)*100:5.1f}%) {bar}')

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#C73E1D','#F18F01','#F2C94C','#A5C882','#386641']
axes[0].bar(score_dist.index, score_dist.values, color=colors, edgecolor='white')
axes[0].set_title('评分分布（1-5分）', fontsize=13, fontweight='bold')
axes[0].set_xlabel('评分')
axes[0].set_ylabel('评论数')
for i, (s, c) in enumerate(zip(score_dist.index, score_dist.values)):
    axes[0].text(s, c + 500, f'{c/len(reviews)*100:.1f}%', ha='center', fontsize=10)

# 差评 vs 好评 = 1-2分 vs 4-5分
reviews['sentiment'] = pd.cut(reviews['review_score'], 
                               bins=[0, 2, 3, 5], 
                               labels=['差评(1-2)', '中评(3)', '好评(4-5)'])
sentiment_counts = reviews['sentiment'].value_counts()
axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=['#C73E1D','#F2C94C','#386641'], startangle=90, explode=(0.05, 0, 0))
axes[1].set_title('好评 vs 差评占比', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/AI_Dateannaly/PROJECTS/project_olist_analysis/visuals/eda_review_score.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\n差评率: {(reviews["review_score"] <= 2).mean()*100:.1f}%')
print(f'好评率: {(reviews["review_score"] >= 4).mean()*100:.1f}%')

# %% [markdown]
# **关键发现**：
# - 巴西消费者评分偏高（4-5分占 75%+），整体满意度不错
# - 但差评（1-2分）也有 ~10%+ ，每10个订单就有1个不满意
# - **核心问题**：差评的原因是什么？延迟？产品质量？客服？
# 
# → 后面会 JOIN reviews + orders 来看延迟跟评分的关系

# %% [markdown]
# ## 5. 商品 & 品类分析

# %%
# 合并品类英文名
products_en = products.merge(cat_trans, on='product_category_name', how='left')
products_en['category_en'] = products_en['product_category_name_english'].fillna(products_en['product_category_name'])

# 合并订单明细 → 品类
items_with_cat = items.merge(products_en[['product_id', 'category_en']], on='product_id', how='left')

# 品类 GMV 排名
cat_gmv = items_with_cat.groupby('category_en').agg(
    订单数=('order_id', 'count'),
    总GMV=('price', 'sum'),
    均价=('price', 'mean'),
    总运费=('freight_value', 'sum')
).sort_values('总GMV', ascending=False)

# 品类单价 vs 运费（找高运费品类）
cat_gmv['运费占比'] = (cat_gmv['总运费'] / cat_gmv['总GMV'] * 100).round(1)
cat_gmv['均运费'] = (cat_gmv['总运费'] / cat_gmv['订单数']).round(1)

print('[CATEGORY] 品类 GMV Top 10：')
print(cat_gmv.head(10).to_string())

# 可视化 Top 15 品类
top15 = cat_gmv.head(15)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# GMV 柱状图
axes[0].barh(range(len(top15)), top15['总GMV'].values/1e6, color='steelblue')
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels(top15.index)
axes[0].set_xlabel('GMV (百万雷亚尔)')
axes[0].set_title('Top 15 品类 GMV', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()

# 运费占比
top15_by_freight = cat_gmv.nlargest(15, '运费占比')
axes[1].barh(range(len(top15_by_freight)), top15_by_freight['运费占比'].values, color='#A23B72')
axes[1].set_yticks(range(len(top15_by_freight)))
axes[1].set_yticklabels(top15_by_freight.index)
axes[1].set_xlabel('运费占GMV比例 (%)')
axes[1].set_title('Top 15 运费占比最高的品类', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('D:/AI_Dateannaly/PROJECTS/project_olist_analysis/visuals/eda_category_gmv.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. 支付方式分析

# %%
# 支付方式分布（每个订单可能有多笔支付）
payment_type = payments['payment_type'].value_counts()
installments = payments.groupby('payment_type')['payment_installments'].mean().round(1)

print('[PAYMENT] 支付方式分布：')
for pt, cnt in payment_type.items():
    print(f'  {pt:20s}: {cnt:>7,} 笔 ({cnt/len(payments)*100:5.1f}%)  |  平均分期: {installments[pt]}期')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].pie(payment_type.values, labels=payment_type.index, autopct='%1.1f%%',
            colors=['#2E86AB','#A23B72','#F18F01','#6A994E','#C73E1D'], startangle=90)
axes[0].set_title('支付方式占比', fontsize=13, fontweight='bold')

# 分期分布
axes[1].hist(payments[payments['payment_installments'] <= 12]['payment_installments'], 
             bins=12, color='steelblue', edgecolor='white', rwidth=0.85)
axes[1].set_title('分期数分布（≤12期）', fontsize=13, fontweight='bold')
axes[1].set_xlabel('分期数')
axes[1].set_ylabel('交易笔数')
axes[1].set_xticks(range(1,13))

plt.tight_layout()
plt.savefig('D:/AI_Dateannaly/PROJECTS/project_olist_analysis/visuals/eda_payment.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. 卖家集中度分析

# %%
# 卖家订单量/收入集中度
seller_stats = items.groupby('seller_id').agg(
    订单数=('order_id', 'count'),
    总GMV=('price', 'sum'),
    均价=('price', 'mean')
).sort_values('总GMV', ascending=False)

seller_stats['GMV累计占比'] = seller_stats['总GMV'].cumsum() / seller_stats['总GMV'].sum() * 100
seller_stats['排名百分位'] = (np.arange(1, len(seller_stats)+1) / len(seller_stats) * 100)

# 帕累托分析
top1_pct = 1
top5_pct = 5
top10_pct = 10
top1_gmv = seller_stats.head(int(len(seller_stats)*top1_pct/100))['总GMV'].sum() / seller_stats['总GMV'].sum() * 100
top5_gmv = seller_stats.head(int(len(seller_stats)*top5_pct/100))['总GMV'].sum() / seller_stats['总GMV'].sum() * 100
top10_gmv = seller_stats.head(int(len(seller_stats)*top10_pct/100))['总GMV'].sum() / seller_stats['总GMV'].sum() * 100

print('[SELLER] 卖家集中度（帕累托分析）：')
print(f'  Top 1% 卖家贡献 GMV: {top1_gmv:.1f}%')
print(f'  Top 5% 卖家贡献 GMV: {top5_gmv:.1f}%')
print(f'  Top 10% 卖家贡献 GMV: {top10_gmv:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GMV 洛伦兹曲线
axes[0].plot(seller_stats['排名百分位'], seller_stats['GMV累计占比'], 'steelblue', linewidth=2)
axes[0].plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, label='完全平均')
axes[0].axvline(10, color='red', linestyle=':', alpha=0.7, label='Top 10% 卖家')
axes[0].legend()
axes[0].set_xlabel('卖家累计占比 (%)')
axes[0].set_ylabel('GMV 累计占比 (%)')
axes[0].set_title('卖家 GMV 洛伦兹曲线', fontsize=13, fontweight='bold')

# 订单量分布
axes[1].hist(seller_stats['订单数'].clip(upper=500), bins=50, color='steelblue', edgecolor='white')
axes[1].set_title('卖家订单量分布', fontsize=13, fontweight='bold')
axes[1].set_xlabel('订单数')
axes[1].set_ylabel('卖家数')

plt.tight_layout()
plt.savefig('D:/AI_Dateannaly/PROJECTS/project_olist_analysis/visuals/eda_seller_pareto.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. EDA 总结

# %%
print('='*60)
print('[SUMMARY] EDA 核心发现总结')
print('='*60)
print(f'''
1. [CATEGORY] 数据规模：~10万订单，覆盖2016.09-2018.08共24个月
2. [DELAY] 延迟问题：{delayed_pct:.1f}% 订单延迟，{severe_delay_pct:.1f}% 严重延迟(>10天)
3. [RATING] 评分分布：好评(4-5分)占{(reviews["review_score"]>=4).mean()*100:.1f}%，差评(1-2分)占{(reviews["review_score"]<=2).mean()*100:.1f}%
4. [CATEGORY] 品类特征：头部品类集中（需确认具体品类），部分品类运费占比偏高 → 需深挖
5. [PAYMENT] 支付偏好：信用卡占主导，分期普遍
6. [SELLER] 卖家集中：Top 10% 卖家贡献 {top10_gmv:.1f}% GMV

→ 下一步：SQL 多表 JOIN，回答"延迟跟评分到底有没有关系？"
''')
