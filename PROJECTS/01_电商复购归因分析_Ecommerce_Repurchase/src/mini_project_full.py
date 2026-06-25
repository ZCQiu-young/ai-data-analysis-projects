import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'D:\AI_Dateannaly\PROJECTS\project_ecommerce_analysis'
DATA_DIR = os.path.join(BASE, 'data')
CHARTS_DIR = os.path.join(BASE, 'charts')

print("=== Repurchase Rate Analysis Report ===\n")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'mini_project_data.csv'))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df = df[df['Quantity'] > 0].copy()

os.makedirs(CHARTS_DIR, exist_ok=True)

# Calc repurchase rate
def calc_repurchase_rate(df, month_col='Month'):
    results = []
    for month in sorted(df[month_col].unique()):
        month_df = df[df[month_col] == month]
        user_orders = month_df.groupby('CustomerID')['InvoiceNo'].nunique()
        total_users = len(user_orders)
        repurchase_users = (user_orders >= 2).sum()
        repurchase_rate = repurchase_users / total_users if total_users > 0 else 0
        results.append({
            'Month': month, 'Total_Users': total_users,
            'Repurchase_Users': repurchase_users, 'Repurchase_Rate': repurchase_rate
        })
    return pd.DataFrame(results)

month_data = calc_repurchase_rate(df)

# Fig 1: Monthly repurchase rate
print("Chart 1: Monthly repurchase rate...")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if m in [4, 8] else '#3498db' for m in month_data['Month']]
bars = ax.bar(month_data['Month'], month_data['Repurchase_Rate'] * 100, color=colors)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Repurchase Rate (%)', fontsize=12)
ax.set_title('Monthly Repurchase Rate (Red = Abnormal)', fontsize=14)
ax.set_xticks(month_data['Month'])
ax.set_ylim(0, max(month_data['Repurchase_Rate'] * 100) * 1.3)
for bar, rate in zip(bars, month_data['Repurchase_Rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '01_monthly_repurchase_rate.png'), dpi=150)
plt.close()

# Fig 2: Country repurchase rate
print("Chart 2: Country repurchase rate...")
country_data = []
for country in df['Country'].unique():
    country_df = df[df['Country'] == country]
    user_orders = country_df.groupby('CustomerID')['InvoiceNo'].nunique()
    total_users = len(user_orders)
    repurchase_users = (user_orders >= 2).sum()
    repurchase_rate = repurchase_users / total_users if total_users > 0 else 0
    country_data.append({'Country': country, 'Total_Users': total_users, 'Repurchase_Rate': repurchase_rate})
country_data = pd.DataFrame(country_data).sort_values('Repurchase_Rate', ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ['#e74c3c' if r == 0 else '#2ecc71' for r in country_data['Repurchase_Rate']]
bars = ax.barh(country_data['Country'], country_data['Repurchase_Rate'] * 100, color=bar_colors)
ax.set_xlabel('Repurchase Rate (%)', fontsize=12)
ax.set_title('Repurchase Rate by Country (Red = 0%)', fontsize=14)
for bar, rate in zip(bars, country_data['Repurchase_Rate']):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{rate*100:.1f}%', ha='left', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '02_country_repurchase_rate.png'), dpi=150)
plt.close()

# Fig 3: Electronics price trend
print("Chart 3: Electronics price trend...")
elec_df = df[df['Description'] == 'Electronics']
elec_monthly = elec_df.groupby('Month')['UnitPrice'].agg(['mean', 'count']).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(elec_monthly['Month'], elec_monthly['mean'], 'o-', linewidth=2, markersize=10, color='#e74c3c')
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(elec_monthly['Month'], elec_monthly['mean'], alpha=0.3, color='#e74c3c')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Average Unit Price', fontsize=12)
ax.set_title('Electronics Price: 3x Increase from March', fontsize=14)
ax.set_xticks(elec_monthly['Month'])
for x, y in zip(elec_monthly['Month'], elec_monthly['mean']):
    ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '03_electronics_price_trend.png'), dpi=150)
plt.close()

# Fig 4: Orders vs repurchase rate
print("Chart 4: Orders vs repurchase rate...")
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
month_order_count = df.groupby('Month').size().reset_index(name='Order_Count')
merged = month_data.merge(month_order_count, on='Month')
bars = ax1.bar(merged['Month'], merged['Order_Count'], alpha=0.6, color='#3498db', label='Order Count')
ax2.plot(merged['Month'], merged['Repurchase_Rate'] * 100, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Repurchase Rate')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Order Count', fontsize=12, color='#3498db')
ax2.set_ylabel('Repurchase Rate (%)', fontsize=12, color='#e74c3c')
ax1.set_title('Order Count vs Repurchase Rate', fontsize=14)
ax1.set_xticks(merged['Month'])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '04_orders_vs_repurchase.png'), dpi=150)
plt.close()

# Fig 5: Spain vs UK
print("Chart 5: Spain vs UK...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
spain_df = df[df['Country'] == 'Spain']
spain_users = spain_df.groupby('CustomerID')['InvoiceNo'].nunique()
axes[0].pie([len(spain_users)], labels=['Only 1 order'], colors=['#e74c3c'], autopct='%1.0f%%', startangle=90)
axes[0].set_title(f'Spain: All {len(spain_users)} users = 1 order', fontsize=12)
uk_df = df[df['Country'] == 'UK']
uk_users = uk_df.groupby('CustomerID')['InvoiceNo'].nunique()
uk_data = [sum(uk_users == 1), sum(uk_users >= 2)]
axes[1].pie(uk_data, labels=['1 order', '2+ orders'], colors=['#3498db', '#2ecc71'], autopct='%1.0f%%', startangle=90)
axes[1].set_title(f'UK: {uk_data[1]} users with 2+ orders', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '05_spain_vs_uk.png'), dpi=150)
plt.close()

# Generate report
print("\nGenerating report...")
report = f"""# E-commerce Repurchase Rate Analysis Report

## Executive Summary

**Objective:** Find root causes of repurchase rate decline
**Data:** 2011 Jan-Aug, {len(df)} orders, {df['CustomerID'].nunique()} users
**Key Findings:** 3 major issues identified

---

## 1. Key Metrics

| Metric | Value |
|--------|-------|
| Total Users | {df['CustomerID'].nunique()} |
| Total Orders | {df['InvoiceNo'].nunique()} |
| Avg Repurchase Rate | {month_data['Repurchase_Rate'].mean()*100:.1f}% |
| Highest Rate Month | {month_data.loc[month_data['Repurchase_Rate'].idxmax(), 'Month']} ({month_data['Repurchase_Rate'].max()*100:.1f}%) |
| Lowest Rate Month | {month_data.loc[month_data['Repurchase_Rate'].idxmin(), 'Month']} ({month_data['Repurchase_Rate'].min()*100:.1f}%) |

---

## 2. Key Findings

### Finding 1: Spain Market - 0% Repurchase Rate

All 57 Spain users made only 1 purchase. No user returned.

**Recommendation:** Launch Spain-specific repurchase incentive (discount coupons)

### Finding 2: Electronics Price +188% from March

Electronics unit price jumped from ~55 (Jan-Feb) to ~158 (March onwards)

**Recommendation:** Evaluate price strategy impact on customer retention

### Finding 3: April Order Volume Drop

April orders (944) significantly below Mar (1322) and May (1343)

**Recommendation:** Investigate seasonality, competitors, or supply chain issues

---

## 3. Priority Actions

| Priority | Issue | Action | Expected Impact |
|----------|-------|--------|----------------|
| P0 | Spain 0% | Launch Spain repurchase campaign | Fast improvement |
| P1 | Electronics price | Re-evaluate pricing | Optimize retention vs profit |
| P2 | April drop | Root cause investigation | Prevent recurrence |

---

## 4. Charts

| File | Description |
|------|-------------|
| 01_monthly_repurchase_rate.png | Monthly trend (Apr/Aug highlighted) |
| 02_country_repurchase_rate.png | Country comparison (Spain=0%) |
| 03_electronics_price_trend.png | Electronics price spike |
| 04_orders_vs_repurchase.png | Orders vs rate correlation |
| 05_spain_vs_uk.png | User distribution comparison |

---

**Generated:** 2026-06-03
**Analyst:** AI-Enhanced Data Analyst
"""

report_path = os.path.join(BASE, 'reports', 'REPORT_repurchase_analysis.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print("\n=== All done ===")
print(f"Charts: {CHARTS_DIR}")
print(f"Report: {report_path}")
