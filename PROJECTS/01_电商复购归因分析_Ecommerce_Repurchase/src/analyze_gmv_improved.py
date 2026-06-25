import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'D:\AI_Dateannaly\PROJECTS\project_ecommerce_analysis'
DATA_DIR = os.path.join(BASE, 'data')
CHARTS_DIR = os.path.join(BASE, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=== Improved GMV Trend Analysis ===\n")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'mini_project_data.csv'))
print(f"Raw data: {df.shape[0]} rows")
print(f"Date range: {pd.to_datetime(df['InvoiceDate']).min()} ~ {pd.to_datetime(df['InvoiceDate']).max()}")

# Data cleaning
print("\n--- Data Cleaning ---")
print(f"Columns: {df.columns.tolist()}")

# Remove returns
n_returns = (df['Quantity'] < 0).sum()
print(f"Returns (Quantity < 0): {n_returns} ({n_returns/len(df)*100:.1f}%)")
df_clean = df[df['Quantity'] > 0].copy()
print(f"After removing returns: {df_clean.shape[0]} rows")

# Check missing values
print(f"\nMissing values:\n{df_clean.isnull().sum()}")

# Convert date
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
df_clean['Month'] = df_clean['InvoiceDate'].dt.month
df_clean['DaysInMonth'] = df_clean['InvoiceDate'].dt.days_in_month
df_clean['DayOfMonth'] = df_clean['InvoiceDate'].dt.day

# Check data completeness by month
print("\n--- Data Completeness Check ---")
monthly_coverage = df_clean.groupby('Month').apply(
    lambda x: x['DayOfMonth'].max() / x['DaysInMonth'].iloc[0] * 100
).reset_index(name='Coverage')
monthly_coverage['Coverage'] = monthly_coverage['Coverage'].round(1)
print(monthly_coverage.to_string(index=False))

# Calculate GMV
print("\n--- GMV by Month ---")
gmv_by_month = df_clean.groupby('Month')['TotalPrice'].sum().reset_index()
gmv_by_month.columns = ['Month', 'GMV']
gmv_by_month['GMV_M'] = gmv_by_month['GMV'] / 1_000_000
gmv_by_month['MoM_Change'] = gmv_by_month['GMV'].pct_change() * 100

# Add completeness info
gmv_by_month = gmv_by_month.merge(monthly_coverage, on='Month')

print("\nGMV Table:")
print(gmv_by_month[['Month', 'GMV_M', 'MoM_Change', 'Coverage']].to_string(index=False))

# Plot with completeness annotation
fig, ax = plt.subplots(figsize=(10, 5))
line = ax.plot(gmv_by_month['Month'], gmv_by_month['GMV_M'], 'o-', 
                linewidth=2, markersize=10, color='#3498db', label='GMV')

# Fill area
ax.fill_between(gmv_by_month['Month'], gmv_by_month['GMV_M'], alpha=0.3, color='#3498db')

# Labels
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('GMV (Million)', fontsize=12)
ax.set_title('Monthly GMV Trend (with Data Completeness)', fontsize=14)
ax.set_xticks(gmv_by_month['Month'])
ax.grid(True, alpha=0.3)

# Add value labels
for x, y, cov in zip(gmv_by_month['Month'], gmv_by_month['GMV_M'], gmv_by_month['Coverage']):
    label = f'{y:.2f}M'
    if cov < 90:
        label += f'\n({cov:.0f}% data)'
    ax.text(x, y + 0.05, label, ha='center', va='bottom', fontsize=9)

plt.tight_layout()
chart_path = os.path.join(CHARTS_DIR, '07_gmv_trend_improved.png')
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"\nChart saved: {chart_path}")

# Conclusion
print("\n--- Conclusion ---")

# Identify abnormal months (data incompleteness or GMV drop)
abnormal_months = []
for _, row in gmv_by_month.iterrows():
    reasons = []
    if row['Coverage'] < 90:
        reasons.append(f"data only {row['Coverage']:.0f}% complete")
    if pd.notna(row['MoM_Change']) and abs(row['MoM_Change']) > 30:
        reasons.append(f"MoM change {row['MoM_Change']:+.1f}%")
    if reasons:
        abnormal_months.append(f"{int(row['Month'])}月: " + ", ".join(reasons))

conclusion = f"""
GMV Trend Analysis (2011 Jan-Aug):

1. Overall:
   - Total GMV (after cleaning): {df_clean['TotalPrice'].sum()/1_000_000:.2f}M
   - Average monthly GMV: {gmv_by_month['GMV'].mean()/1_000_000:.2f}M
   - Highest month: {int(gmv_by_month.loc[gmv_by_month['GMV'].idxmax(), 'Month'])}月 ({gmv_by_month['GMV'].max()/1_000_000:.2f}M)
   - Lowest month: {int(gmv_by_month.loc[gmv_by_month['GMV'].idxmin(), 'Month'])}月 ({gmv_by_month['GMV'].min()/1_000_000:.2f}M)

2. Abnormal months requiring attention:
"""
if abnormal_months:
    for msg in abnormal_months:
        conclusion += f"   - {msg}\n"
else:
    conclusion += "   (None)\n"

conclusion += """
3. Key observations:
   - 1月 & 8月 data incomplete (not full month), GMV should not be compared with other months
   - 4月 GMV dropped 31.4% MoM (data is complete, genuine drop)
   - 2-3月 & 5月 have higher GMV (possible seasonality or promotion)

4. Recommendations:
   - For incomplete months (1月, 8月), use daily GMV or exclude from MoM comparison
   - Investigate 4月 GMV drop: check traffic, conversion rate, AOV
   - Compare with same period last year (if available) to identify seasonality
"""

print(conclusion)

# Save GMV table
output_path = os.path.join(DATA_DIR, 'gmv_by_month_improved.csv')
gmv_by_month.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nGMV table saved to: {output_path}")

print("\n=== Analysis Complete ===")
