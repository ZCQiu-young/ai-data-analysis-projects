import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'D:\AI_Dateannaly\PROJECTS\project_ecommerce_analysis'
DATA_DIR = os.path.join(BASE, 'data')
CHARTS_DIR = os.path.join(BASE, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=== GMV Trend Analysis ===\n")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'mini_project_data.csv'))
print(f"Raw data: {df.shape[0]} rows")

# Data cleaning
print("\n--- Data Cleaning ---")

# 1. Check column names
print(f"Columns: {df.columns.tolist()}")

# 2. Handle returns (Quantity < 0)
n_returns = (df['Quantity'] < 0).sum()
print(f"Returns (Quantity < 0): {n_returns} ({n_returns/len(df)*100:.1f}%)")
df_clean = df[df['Quantity'] > 0].copy()
print(f"After removing returns: {df_clean.shape[0]} rows")

# 3. Check for missing values
print(f"\nMissing values:\n{df_clean.isnull().sum()}")

# 4. Convert date
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
df_clean['Month'] = df_clean['InvoiceDate'].dt.month

# Calculate GMV (TotalPrice = Quantity * UnitPrice for non-return records)
# TotalPrice column already exists in the data
print(f"\nGMV definition: sum of TotalPrice (excluding returns)")
print(f"Total GMV: {df_clean['TotalPrice'].sum():,.2f}")

# GMV by month
print("\n--- GMV by Month ---")
gmv_by_month = df_clean.groupby('Month')['TotalPrice'].sum().reset_index()
gmv_by_month.columns = ['Month', 'GMV']
gmv_by_month['GMV_M'] = gmv_by_month['GMV'] / 1_000_000  # Convert to millions

print(gmv_by_month.to_string(index=False))

# Calculate MoM change
gmv_by_month['MoM_Change'] = gmv_by_month['GMV'].pct_change() * 100
print("\nMoM Change (%):")
for _, row in gmv_by_month.iterrows():
    if pd.notna(row['MoM_Change']):
        print(f"  {int(row['Month'])}月: {row['MoM_Change']:+.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
line = ax.plot(gmv_by_month['Month'], gmv_by_month['GMV_M'], 'o-', 
                linewidth=2, markersize=10, color='#3498db', label='GMV')

# Fill area
ax.fill_between(gmv_by_month['Month'], gmv_by_month['GMV_M'], alpha=0.3, color='#3498db')

# Labels
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('GMV (Million)', fontsize=12)
ax.set_title('Monthly GMV Trend', fontsize=14)
ax.set_xticks(gmv_by_month['Month'])
ax.grid(True, alpha=0.3)

# Add value labels
for x, y in zip(gmv_by_month['Month'], gmv_by_month['GMV_M']):
    ax.text(x, y + 0.05, f'{y:.2f}M', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
chart_path = os.path.join(CHARTS_DIR, '06_gmv_trend.png')
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"\nChart saved: {chart_path}")

# Analysis conclusion
print("\n--- Conclusion ---")
max_gmv_month = gmv_by_month.loc[gmv_by_month['GMV'].idxmax()]
min_gmv_month = gmv_by_month.loc[gmv_by_month['GMV'].idxmin()]
avg_gmv = gmv_by_month['GMV'].mean()

conclusion = """GMV Trend Analysis (2011 Jan-Aug):

1. Overall trend: GMV fluctuated between months
   - Highest: {}月 ({:.2f}M)
   - Lowest:  {}月 ({:.2f}M)
   - Average:  {:.2f}M/month

2. Notable changes:
""".format(
    int(max_gmv_month['Month']), max_gmv_month['GMV']/1_000_000,
    int(min_gmv_month['Month']), min_gmv_month['GMV']/1_000_000,
    avg_gmv/1_000_000
)

for _, row in gmv_by_month.iterrows():
    if pd.notna(row['MoM_Change']):
        conclusion += f"   - {int(row['Month'])}月: {row['GMV']/1_000_000:.2f}M ({row['MoM_Change']:+.1f}% MoM)\n"

conclusion += """
3. Possible reasons for GMV fluctuation:
   - Seasonality (e.g., holiday seasons)
   - Marketing campaigns
   - Product launches
   - External factors (competitors, economy)

Recommendation: Investigate months with significant drops (e.g., MoM decline >10%) 
to identify root causes (traffic, conversion rate, or AOV).
"""

print(conclusion)

print(f"\nGMV data saved to: {os.path.join(DATA_DIR, 'gmv_by_month.csv')}")
gmv_by_month.to_csv(os.path.join(DATA_DIR, 'gmv_by_month.csv'), index=False, encoding='utf-8-sig')

print("\n=== Analysis Complete ===")
