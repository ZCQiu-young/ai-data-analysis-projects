import pandas as pd
import numpy as np
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

print("=== Low-to-High Frequency Conversion Analysis ===\n")

# Load and clean
df = pd.read_csv(os.path.join(DATA_DIR, 'mini_project_data.csv'))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[df['Quantity'] > 0].copy()

# Build user-level timeline
print("--- Building per-user purchase timeline ---")

user_data = df.groupby('CustomerID').agg(
    First_Purchase=('InvoiceDate', 'min'),
    Last_Purchase=('InvoiceDate', 'max'),
    Total_Orders=('InvoiceNo', 'nunique'),
    Total_Spend=('TotalPrice', 'sum'),
    Country=('Country', 'first'),
    Avg_Order_Value=('TotalPrice', 'mean')
).reset_index()

# Key metric: active days (time between first and last)
user_data['Active_Days'] = (user_data['Last_Purchase'] - user_data['First_Purchase']).dt.days
user_data['Active_Days'] = user_data['Active_Days'].replace(0, 1)  # same-day to 1

# Frequency classification
def classify_freq(n):
    if n >= 5: return 'High (5+)'
    elif n >= 3: return 'Medium (3-4)'
    else: return 'Low (1-2)'

user_data['Freq_Segment'] = user_data['Total_Orders'].apply(classify_freq)

# ========== Analysis 1: Time gap analysis ==========
print("\n=== 1. Time Between Purchases ===")

# For users with 2+ orders, calculate average time between orders
user_data['Avg_Days_Between'] = user_data['Active_Days'] / (user_data['Total_Orders'] - 1)
user_data['Avg_Days_Between'] = user_data['Avg_Days_Between'].replace([np.inf, -np.inf], np.nan)

print("\nAverage days between purchases by segment:")
for seg in ['Low (1-2)', 'Medium (3-4)', 'High (5+)']:
    seg_data = user_data[user_data['Freq_Segment'] == seg]['Avg_Days_Between'].dropna()
    if len(seg_data) > 0:
        print(f"  {seg}: {seg_data.median():.0f} days (median)")

# ========== Analysis 2: "Still Active?" ==========
print("\n=== 2. Activity Status ===")

# Calculate recency: days since last purchase
latest_date = df['InvoiceDate'].max()
user_data['Days_Since_Last'] = (latest_date - user_data['Last_Purchase']).dt.days

# Activity classification
def activity_status(days):
    if days <= 30: return 'Active (<30d)'
    elif days <= 90: return 'Warm (30-90d)'
    else: return 'Cold (>90d)'

user_data['Activity'] = user_data['Days_Since_Last'].apply(activity_status)

print("\nActivity by frequency segment:")
cross_activity = pd.crosstab(user_data['Freq_Segment'], user_data['Activity'], margins=True)
print(cross_activity.to_string())

# ========== Analysis 3: Conversion Potential ==========
print("\n=== 3. Conversion Potential Analysis ===")

low_users = user_data[user_data['Freq_Segment'] == 'Low (1-2)']

# Breakdown: 1 order vs 2 orders
one_order = low_users[low_users['Total_Orders'] == 1]
two_orders = low_users[low_users['Total_Orders'] == 2]

print(f"\nLow frequency users breakdown:")
print(f"  1 order only: {len(one_order)} ({len(one_order)/len(low_users)*100:.1f}%)")
print(f"  2 orders: {len(two_orders)} ({len(two_orders)/len(low_users)*100:.1f}%)")

# Among 2-order users: how many are still active?
print(f"\n2-order users activity:")
two_active = two_orders[two_orders['Activity'] == 'Active (<30d)']
two_cold = two_orders[two_orders['Activity'] == 'Cold (>90d)']
print(f"  Still active: {len(two_active)} ({len(two_active)/len(two_orders)*100:.1f}%) -> potential to convert")
print(f"  Gone cold: {len(two_cold)} ({len(two_cold)/len(two_orders)*100:.1f}%) -> likely lost")

# Among 2-order active users: what's their spend level?
print(f"\n2-order active user profile:")
print(f"  Avg total spend: {two_active['Total_Spend'].mean():,.2f}")
print(f"  Avg order value: {two_active['Avg_Order_Value'].mean():,.2f}")
print(f"  Avg days since last: {two_active['Days_Since_Last'].mean():.0f}")
print(f"  Countries: {two_active['Country'].value_counts().head(3).to_dict()}")

# ========== Analysis 4: What separates 2-order from 3+ order users ==========
print("\n=== 4. What Separates 2-Order from 3+ Order Users? ===")

# Compare users who stopped at 2 vs users who went to 3+
stopped_at_2 = user_data[user_data['Total_Orders'] == 2]
went_to_3plus = user_data[user_data['Total_Orders'] >= 3]

print(f"\nUsers who stopped at 2 orders: {len(stopped_at_2)}")
print(f"Users who went to 3+ orders: {len(went_to_3plus)}")

print(f"\nComparison:")
print(f"  {'Metric':<25} {'Stopped at 2':>15} {'Went to 3+':>15}")
print(f"  {'-'*55}")
print(f"  {'Days between 1st-2nd':>25} {stopped_at_2['Avg_Days_Between'].median():>15.0f} {went_to_3plus['Avg_Days_Between'].median():>15.0f}")
print(f"  {'Avg Order Value':>25} {stopped_at_2['Avg_Order_Value'].mean():>15.2f} {went_to_3plus['Avg_Order_Value'].mean():>15.2f}")
print(f"  {'Avg Unit Price':>25} {stopped_at_2['Avg_Order_Value'].median():>15.2f} {went_to_3plus['Avg_Order_Value'].median():>15.2f}")

# ========== Analysis 5: Product category of low-frequency users ==========
print("\n=== 5. What Do Low-Frequency Users Buy? ===")
low_user_ids = low_users['CustomerID'].tolist()
low_df = df[df['CustomerID'].isin(low_user_ids)]

# Category preference of low-frequency users
low_category = low_df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)
all_category = df.groupby('Description')['TotalPrice'].sum()

print("\nCategory spend share comparison:")
for cat in all_category.index:
    low_share = low_category.get(cat, 0) / low_category.sum() * 100
    all_share = all_category[cat] / all_category.sum() * 100
    diff = low_share - all_share
    print(f"  {cat:<15} Low users: {low_share:>5.1f}%  All: {all_share:>5.1f}%  Diff: {diff:>+5.1f}%")

# ========== Charts ==========
print("\n--- Generating Charts ---")

# Chart 1: User distribution by orders
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

order_dist = user_data['Total_Orders'].value_counts().sort_index()
axes[0].bar(order_dist.index, order_dist.values, color='#3498db', edgecolor='white')
axes[0].axvline(x=2.5, color='#e74c3c', linestyle='--', linewidth=2, label='Conversion line')
axes[0].set_xlabel('Number of Orders')
axes[0].set_ylabel('Number of Users')
axes[0].set_title('Order Count Distribution\n(Dashed line = 2-to-3 conversion point)')
axes[0].legend()
for i, (x, y) in enumerate(zip(order_dist.index, order_dist.values)):
    if i < 10:
        axes[0].text(x, y + 5, str(y), ha='center', fontsize=9)

# Activity by segment
activity_counts = pd.crosstab(user_data['Freq_Segment'], user_data['Activity'])
activity_pct = activity_counts.div(activity_counts.sum(axis=1), axis=0) * 100
activity_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                  color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1].set_xlabel('Frequency Segment')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_title('User Activity Status by Frequency')
axes[1].legend(title='Activity')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '10_conversion_analysis.png'), dpi=150)
plt.close()

# Chart 2: Conversion funnel
fig, ax = plt.subplots(figsize=(10, 5))

stages = ['1 order', '2 orders', '3 orders', '4 orders', '5+ orders']
counts = [
    len(user_data[user_data['Total_Orders'] == 1]),
    len(user_data[user_data['Total_Orders'] == 2]),
    len(user_data[user_data['Total_Orders'] == 3]),
    len(user_data[user_data['Total_Orders'] == 4]),
    len(user_data[user_data['Total_Orders'] >= 5]),
]

colors = ['#e74c3c', '#e74c3c', '#3498db', '#3498db', '#2ecc71']
bars = ax.bar(stages, counts, color=colors, edgecolor='white')

# Add drop-off rates
for i in range(len(stages) - 1):
    if counts[i] > 0:
        drop_rate = (counts[i] - counts[i+1]) / counts[i] * 100
        ax.annotate(f'Drop: {drop_rate:.0f}%', 
                    xy=(i+0.5, (counts[i] + counts[i+1])/2),
                    ha='center', fontsize=11, color='#e74c3c', fontweight='bold')

for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontsize=11)

ax.set_xlabel('Purchase Frequency')
ax.set_ylabel('Number of Users')
ax.set_title('User Conversion Funnel (1 order -> 5+ orders)', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '11_conversion_funnel.png'), dpi=150)
plt.close()

print("\n=== Analysis Complete ===")
print(f"Charts saved to: {CHARTS_DIR}")
