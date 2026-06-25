"""
Step1: 电商全链路销售分析 - EDA + 全维度洞察
输出：分析报告 + Tableau 就绪数据集
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA = r'D:\AI_Dateannaly\PROJECTS\project_sales_dashboard\data'
OUT_CHARTS = r'D:\AI_Dateannaly\PROJECTS\project_sales_dashboard\reports\charts'
OUT_DATA = r'D:\AI_Dateannaly\PROJECTS\project_sales_dashboard\data'
import os
os.makedirs(OUT_CHARTS, exist_ok=True)

# ====== 加载数据 ======
df = pd.read_csv(f'{DATA}/orders.csv')
df['order_date'] = pd.to_datetime(df['order_date'])
valid = df[df['is_returned']==0].copy()
valid['year_month'] = valid['order_date'].dt.to_period('M')
valid['year_quarter'] = valid['order_date'].dt.to_period('Q')

print(f"数据加载完成：{len(df)} 条订单，{len(valid)} 条有效")

# ====== 1. 整体KPI看板 ======
print("\n" + "="*50)
print("[1] KPI Dashboard")
print("="*50)

total_gmv = valid['total_amount'].sum()
total_profit = valid['profit'].sum()
total_orders = len(valid)
total_customers = valid['customer_id'].nunique()
profit_margin = total_profit / total_gmv * 100
aov = total_gmv / total_orders
return_rate = df['is_returned'].sum() / len(df) * 100
promo_order_pct = valid['is_promotion'].sum() / len(valid) * 100
avg_discount = valid.loc[valid['discount_pct']>0, 'discount_pct'].mean() * 100

kpi_df = pd.DataFrame({
    '指标': ['总GMV(万)', '总利润(万)', '总订单', '客户数', '利润率', '客单价', '退货率', '促销订单占比', '平均折扣'],
    '数值': [
        f'{total_gmv/10000:,.0f}',
        f'{total_profit/10000:,.0f}',
        f'{total_orders:,}',
        f'{total_customers:,}',
        f'{profit_margin:.1f}%',
        f'¥{aov:,.0f}',
        f'{return_rate:.1f}%',
        f'{promo_order_pct:.1f}%',
        f'{avg_discount:.1f}%'
    ]
})
print(kpi_df.to_string(index=False))
kpi_df.to_csv(f'{OUT_DATA}/kpi_summary.csv', index=False, encoding='utf-8-sig')

# ====== 2. 时间序列分析 ======
print("\n" + "="*50)
print("[2] Time Series Trend")
print("="*50)

monthly = valid.groupby('year_month').agg(
    GMV=('total_amount', 'sum'),
    Orders=('order_id', 'count'),
    Profit=('profit', 'sum'),
    AOV=('total_amount', 'mean'),
    Customers=('customer_id', 'nunique')
).reset_index()
monthly['year_month'] = monthly['year_month'].astype(str)
monthly['margin'] = (monthly['Profit'] / monthly['GMV'] * 100).round(1)
monthly.to_csv(f'{OUT_DATA}/monthly_trend.csv', index=False, encoding='utf-8-sig')

# Year-over-year
quarterly = valid.groupby('year_quarter').agg(
    GMV=('total_amount', 'sum'),
    Orders=('order_id', 'count'),
    Profit=('profit', 'sum')
).reset_index()
quarterly['year_quarter'] = quarterly['year_quarter'].astype(str)
quarterly.to_csv(f'{OUT_DATA}/quarterly_trend.csv', index=False, encoding='utf-8-sig')

print("月度趋势前5行：")
print(monthly.head().to_string())

# 找到GMV最高月、最低月
print(f"GMV最高月：{monthly.loc[monthly['GMV'].idxmax(), 'year_month']}，{monthly['GMV'].max()/10000:.0f}万")
print(f"GMV最低月：{monthly.loc[monthly['GMV'].idxmin(), 'year_month']}，{monthly['GMV'].min()/10000:.0f}万")

# ====== 3. 品类分析 (ABC + 利润率) ======
print("\n" + "="*50)
print("[3] Category ABC Analysis")
print("="*50)

cat = valid.groupby('category').agg(
    GMV=('total_amount', 'sum'),
    订单数=('order_id', 'count'),
    利润=('profit', 'sum'),
    客单价=('total_amount', 'mean'),
    利润率=('profit', lambda x: x.sum() / valid.loc[x.index, 'total_amount'].sum() * 100)
).sort_values('GMV', ascending=False)
cat['GMV占比'] = (cat['GMV'] / total_gmv * 100).round(1)
cat['利润占比'] = (cat['利润'] / total_profit * 100).round(1)
cat['GMV累计占比'] = cat['GMV占比'].cumsum()
cat['ABC'] = cat['GMV累计占比'].apply(lambda x: 'A' if x <= 70 else ('B' if x <= 90 else 'C'))
cat.to_csv(f'{OUT_DATA}/category_abc.csv', encoding='utf-8-sig')
print(cat.to_string())

# 子品类
subcat = valid.groupby(['category', 'sub_category']).agg(
    GMV=('total_amount', 'sum'),
    利润=('profit', 'sum'),
    订单数=('order_id', 'count')
).sort_values('GMV', ascending=False)
subcat.to_csv(f'{OUT_DATA}/subcategory_detail.csv', encoding='utf-8-sig')

# ====== 4. 渠道分析 ======
print("\n" + "="*50)
print("[4] Channel Analysis")
print("="*50)

chan = valid.groupby('channel').agg(
    GMV=('total_amount', 'sum'),
    订单数=('order_id', 'count'),
    利润=('profit', 'sum'),
    客单价=('total_amount', 'mean'),
    利润率=('profit', lambda x: x.sum() / valid.loc[x.index, 'total_amount'].sum() * 100)
).sort_values('GMV', ascending=False)
chan['GMV占比'] = (chan['GMV'] / total_gmv * 100).round(1)
chan.to_csv(f'{OUT_DATA}/channel_analysis.csv', encoding='utf-8-sig')
print(chan.to_string())

# 渠道×品类交叉
chan_cat = valid.groupby(['channel', 'category'])['total_amount'].sum().unstack(fill_value=0)
chan_cat_pct = chan_cat.div(chan_cat.sum(axis=1), axis=0) * 100
chan_cat_pct.to_csv(f'{OUT_DATA}/channel_category_cross.csv', encoding='utf-8-sig')

# ====== 5. 地区分析 ======
print("\n" + "="*50)
print("[5] Region Analysis")
print("="*50)

region = valid.groupby('province').agg(
    GMV=('total_amount', 'sum'),
    订单数=('order_id', 'count'),
    利润=('profit', 'sum'),
    客户数=('customer_id', 'nunique'),
    客单价=('total_amount', 'mean'),
    利润率=('profit', lambda x: x.sum() / valid.loc[x.index, 'total_amount'].sum() * 100)
).sort_values('GMV', ascending=False)
region['GMV占比'] = (region['GMV'] / total_gmv * 100).round(1)
region.to_csv(f'{OUT_DATA}/region_analysis.csv', encoding='utf-8-sig')
print(region.to_string())

# 城市Top20
city = valid.groupby('city').agg(
    GMV=('total_amount', 'sum'),
    利润=('profit', 'sum'),
    订单数=('order_id', 'count')
).sort_values('GMV', ascending=False).head(20)
city.to_csv(f'{OUT_DATA}/city_top20.csv', encoding='utf-8-sig')
print(f"\nTop 5 城市：\n{city.head().to_string()}")

# 地区×品类交叉
region_cat = valid.groupby(['province', 'category'])['total_amount'].sum().unstack(fill_value=0)
region_cat.to_csv(f'{OUT_DATA}/region_category_cross.csv', encoding='utf-8-sig')

# ====== 6. 促销分析 ======
print("\n" + "="*50)
print("[6] Promotion Impact")
print("="*50)

promo = valid.groupby('is_promotion').agg(
    GMV=('total_amount', 'sum'),
    订单数=('order_id', 'count'),
    利润=('profit', 'sum'),
    客单价=('total_amount', 'mean'),
    利润率=('profit', lambda x: x.sum() / valid.loc[x.index, 'total_amount'].sum() * 100),
    平均折扣=('discount_pct', 'mean')
)
print(promo.to_string())

# 促销月份 vs 非促销月份对比
valid['is_promo_month'] = valid['is_promotion'].apply(lambda x: '促销期' if x==1 else '非促销期')
promo_monthly = valid.groupby(['year_month', 'is_promo_month']).agg(
    GMV=('total_amount', 'sum'),
    利润率=('profit', lambda x: x.sum() / valid.loc[x.index, 'total_amount'].sum() * 100)
).reset_index()
promo_monthly.to_csv(f'{OUT_DATA}/promo_vs_normal.csv', index=False, encoding='utf-8-sig')

# ====== 7. 客户分析 ======
print("\n" + "="*50)
print("[7] Customer Segmentation")
print("="*50)

cust = valid.groupby('customer_id').agg(
    GMV=('total_amount', 'sum'),
    订单数=('order_id', 'count'),
    利润=('profit', 'sum'),
    首单日期=('order_date', 'min'),
    末单日期=('order_date', 'max'),
    购买品类数=('category', 'nunique'),
    主要品类=('category', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
    主要渠道=('channel', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
).reset_index()

cust['生命周期(天)'] = (cust['末单日期'] - cust['首单日期']).dt.days
cust['年均GMV'] = cust['GMV'] / ((cust['生命周期(天)'] / 365).clip(lower=0.25))

# 客户分层
def segment(gmv, orders):
    if orders >= 10 and gmv >= 50000:
        return 'VIP'
    elif orders >= 5 and gmv >= 15000:
        return '高频客户'
    elif orders >= 2 and gmv >= 3000:
        return '活跃客户'
    elif orders == 1:
        return '新客(仅一次)'
    else:
        return '普通客户'

cust['分层'] = cust.apply(lambda r: segment(r['GMV'], r['订单数']), axis=1)
cust.to_csv(f'{OUT_DATA}/customer_segments.csv', index=False, encoding='utf-8-sig')

seg = cust.groupby('分层').agg(
    人数=('customer_id', 'count'),
    总GMV=('GMV', 'sum'),
    人均GMV=('GMV', 'mean'),
    人均订单=('订单数', 'mean')
).sort_values('总GMV', ascending=False)
seg['GMV占比'] = (seg['总GMV'] / cust['GMV'].sum() * 100).round(1)
seg['人数占比'] = (seg['人数'] / len(cust) * 100).round(1)
print(seg.to_string())
seg.to_csv(f'{OUT_DATA}/customer_segment_summary.csv', encoding='utf-8-sig')

# ====== 8. 利润率深度分析 ======
print("\n" + "="*50)
print("[8] Profitability Deep Dive")
print("="*50)

# 各品类-渠道利润率
profit_matrix = valid.groupby(['category', 'channel']).agg(
    GMV=('total_amount', 'sum'),
    利润=('profit', 'sum')
).reset_index()
profit_matrix['利润率'] = (profit_matrix['利润'] / profit_matrix['GMV'] * 100).round(1)
profit_pivot = profit_matrix.pivot_table(values='利润率', index='category', columns='channel', 
                                          aggfunc='first').round(1)
profit_pivot.to_csv(f'{OUT_DATA}/profitability_matrix.csv', encoding='utf-8-sig')
print(profit_pivot.to_string())

# 折扣与利润的关系
disc_impact = valid.groupby(pd.cut(valid['discount_pct'], bins=[0, 0.05, 0.10, 0.15, 0.20, 0.31],
                                     labels=['0%', '5%', '10%', '15%', '20-30%'], include_lowest=True)).agg(
    订单数=('order_id', 'count'),
    GMV=('total_amount', 'sum'),
    利润=('profit', 'sum')
).reset_index()
disc_impact['利润率'] = (disc_impact['利润'] / disc_impact['GMV'] * 100).round(1)
disc_impact['订单占比'] = (disc_impact['订单数'] / disc_impact['订单数'].sum() * 100).round(1)
disc_impact.to_csv(f'{OUT_DATA}/discount_profit_impact.csv', index=False, encoding='utf-8-sig')
print(disc_impact.to_string())

# ====== 9. 增长率分析 ======
print("\n" + "="*50)
print("[9] YoY Growth Analysis")
print("="*50)

# YoY 月度对比
monthly_2024 = monthly[monthly['year_month'].str.startswith('2024')].copy()
monthly_2025 = monthly[monthly['year_month'].str.startswith('2025')].copy()
monthly_2024['month_num'] = monthly_2024['year_month'].str[-2:].astype(int)
monthly_2025['month_num'] = monthly_2025['year_month'].str[-2:].astype(int)

yoy = monthly_2024[['month_num', 'GMV']].merge(
    monthly_2025[['month_num', 'GMV']], on='month_num', suffixes=('_2024', '_2025'))
yoy['YoY增长'] = ((yoy['GMV_2025'] - yoy['GMV_2024']) / yoy['GMV_2024'] * 100).round(1)
yoy.to_csv(f'{OUT_DATA}/yoy_growth.csv', index=False, encoding='utf-8-sig')
print(yoy[['month_num', 'GMV_2024', 'GMV_2025', 'YoY增长']].to_string())

# ====== 10. 生成可视化图表 ======
print("\n" + "="*50)
print("[10] Generating Charts")
print("="*50)

# Chart 1: Monthly Trend Dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('月度销售趋势看板', fontsize=18, fontweight='bold')

# GMV + Profit
ax = axes[0, 0]
months = monthly['year_month'].tolist()
x = range(len(months))
ax.bar(x, monthly['GMV']/10000, color='#4A90D9', alpha=0.8, label='GMV(万)')
ax.plot(x, monthly['Profit']/10000, color='#E74C3C', marker='o', linewidth=2, label='利润(万)')
ax.set_title('GMV与利润月度趋势')
ax.set_xticks(x[::3])
ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Orders + AOV
ax = axes[0, 1]
ax2 = ax.twinx()
ax.bar(x, monthly['Orders'], color='#27AE60', alpha=0.6, label='订单数')
ax2.plot(x, monthly['AOV'], color='#F39C12', marker='s', linewidth=2, label='客单价')
ax.set_title('订单数与客单价趋势')
ax.set_xticks(x[::3])
ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, fontsize=8)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Margin trend
ax = axes[1, 0]
ax.plot(x, monthly['margin'], color='#8E44AD', marker='D', linewidth=2)
ax.set_title('利润率月度趋势 (%)')
ax.set_xticks(x[::3])
ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, fontsize=8)
ax.axhline(y=profit_margin, color='red', linestyle='--', alpha=0.5, label=f'均值 {profit_margin:.1f}%')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Customers
ax = axes[1, 1]
ax.fill_between(x, monthly['Customers'], alpha=0.6, color='#3498DB')
ax.plot(x, monthly['Customers'], color='#2980B9', linewidth=2)
ax.set_title('月活跃客户数')
ax.set_xticks(x[::3])
ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/01_monthly_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Monthly Dashboard")

# Chart 2: Category ABC
fig, ax = plt.subplots(figsize=(14, 7))
colors = {'A': '#27AE60', 'B': '#F39C12', 'C': '#E74C3C'}
bars = ax.barh(cat.index, cat['GMV']/10000, color=[colors[c] for c in cat['ABC']], alpha=0.85)
ax.set_xlabel('GMV (万元)')
ax.set_title('品类 ABC 分析（帕累托）', fontsize=14, fontweight='bold')
for i, (v, pct, abc) in enumerate(zip(cat['GMV']/10000, cat['GMV占比'], cat['ABC'])):
    ax.text(v + 50, i, f'{pct}% ({abc})', va='center', fontsize=10)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/02_category_abc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Category ABC")

# Chart 3: Region Heatmap Data
fig, ax = plt.subplots(figsize=(14, 6))
regions_sorted = region.sort_values('GMV', ascending=True)
bars = ax.barh(regions_sorted.index, regions_sorted['GMV']/10000, color='#4A90D9', alpha=0.8)
ax.set_xlabel('GMV (万元)')
ax.set_title('地区 GMV 排名', fontsize=14, fontweight='bold')
for bar, gmv in zip(bars, regions_sorted['GMV']/10000):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, f'{gmv:.0f}万', va='center', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/03_region_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Region Ranking")

# Chart 4: Channel Mix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Pie
wedges, texts, autotexts = axes[0].pie(chan['GMV'], labels=chan.index, autopct='%1.1f%%',
                                         colors=['#4A90D9','#27AE60','#F39C12','#8E44AD'],
                                         startangle=90, explode=(0.05, 0, 0, 0))
axes[0].set_title('渠道 GMV 占比', fontsize=13, fontweight='bold')
# Bar: Margin by channel
axes[1].bar(chan.index, chan['利润率'], color=['#4A90D9','#27AE60','#F39C12','#8E44AD'], alpha=0.8)
axes[1].set_title('各渠道利润率 (%)', fontsize=13, fontweight='bold')
axes[1].axhline(y=profit_margin, color='red', linestyle='--', alpha=0.5, label=f'整体 {profit_margin:.1f}%')
for i, v in enumerate(chan['利润率']):
    axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=10)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/04_channel_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Channel Analysis")

# Chart 5: Customer Segment
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Pie
seg_data = seg.sort_index()
axes[0].pie(seg_data['人数'], labels=seg_data.index, autopct='%1.1f%%',
            colors=['#E74C3C','#F39C12','#3498DB','#27AE60','#95A5A6'],
            startangle=90)
axes[0].set_title('客户分层（人数）', fontsize=13, fontweight='bold')
# Bar: GMV contribution
axes[1].bar(seg_data.index, seg_data['GMV占比'], 
            color=['#E74C3C','#F39C12','#3498DB','#27AE60','#95A5A6'], alpha=0.85)
axes[1].set_title('各层 GMV 贡献占比 (%)', fontsize=13, fontweight='bold')
for i, v in enumerate(seg_data['GMV占比']):
    axes[1].text(i, v+1, f'{v}%', ha='center', fontsize=11)
axes[1].set_ylim(0, max(seg_data['GMV占比'])+15)
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/05_customer_segments.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Customer Segments")

# Chart 6: Promo Impact
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Monthly GMV with promo highlight
months_list = monthly['year_month'].tolist()
x = range(len(months_list))
promo_monthly_gmv = promo_monthly.pivot(index='year_month', columns='is_promo_month', values='GMV').fillna(0)
promo_monthly_gmv = promo_monthly_gmv.reindex(months_list).fillna(0)

axes[0].fill_between(x, 0, promo_monthly_gmv.get('促销期', [0]*len(x))/10000, 
                      color='#E74C3C', alpha=0.3, label='促销期GMV')
axes[0].fill_between(x, 0, promo_monthly_gmv.get('非促销期', [0]*len(x))/10000, 
                      color='#4A90D9', alpha=0.3, label='非促销期GMV')
axes[0].set_title('促销期 vs 非促销期 GMV', fontsize=13, fontweight='bold')
axes[0].set_xticks(x[::3])
axes[0].set_xticklabels([months_list[i] for i in range(0, len(months_list), 3)], rotation=45, fontsize=8)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Discount vs margin
axes[1].bar(disc_impact['discount_pct'].astype(str), disc_impact['订单占比'], color='#3498DB', alpha=0.7, label='订单占比%')
ax2 = axes[1].twinx()
ax2.plot(range(len(disc_impact)), disc_impact['利润率'], color='#E74C3C', marker='o', linewidth=2.5, label='利润率%')
axes[1].set_title('折扣力度 vs 利润率', fontsize=13, fontweight='bold')
axes[1].set_xlabel('折扣率')
axes[1].set_ylabel('订单占比 (%)')
ax2.set_ylabel('利润率 (%)')
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1+lines2, labels1+labels2, loc='upper right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/06_promotion_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Promotion Analysis")

# Chart 7: Profitability Heatmap - category × channel
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(profit_pivot.values, cmap='RdYlGn', aspect='auto', vmin=5, vmax=25)
ax.set_xticks(range(len(profit_pivot.columns)))
ax.set_xticklabels(profit_pivot.columns, fontsize=10)
ax.set_yticks(range(len(profit_pivot.index)))
ax.set_yticklabels(profit_pivot.index, fontsize=10)
for i in range(len(profit_pivot.index)):
    for j in range(len(profit_pivot.columns)):
        val = profit_pivot.iloc[i, j]
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white' if val < 12 else 'black')
ax.set_title('品类 × 渠道 利润率矩阵 (%)', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8, label='利润率 %')
plt.tight_layout()
plt.savefig(f'{OUT_CHARTS}/07_profitability_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Profitability Heatmap")

print(f"\n[DONE] All analysis complete! Charts: {OUT_CHARTS}")
print(f"[DONE] Tableau data sources: {OUT_DATA}")
print("\n=== Tableau 就绪数据文件 ===")
for f in os.listdir(OUT_DATA):
    if f.endswith('.csv'):
        size = os.path.getsize(f'{OUT_DATA}/{f}')
        print(f"  {f} ({size/1024:.1f} KB)")
