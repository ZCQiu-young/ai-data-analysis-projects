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

print("=== Warm User Activation Analysis ===\n")

# Load and clean
df = pd.read_csv(os.path.join(DATA_DIR, 'mini_project_data.csv'))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[df['Quantity'] > 0].copy()

# Build user profiles
latest_date = df['InvoiceDate'].max()

user_data = df.groupby('CustomerID').agg(
    First_Purchase=('InvoiceDate', 'min'),
    Last_Purchase=('InvoiceDate', 'max'),
    Total_Orders=('InvoiceNo', 'nunique'),
    Total_Spend=('TotalPrice', 'sum'),
    Avg_Order_Value=('TotalPrice', 'mean'),
    Max_Order_Value=('TotalPrice', 'max'),
    Country=('Country', 'first'),
    Total_Items=('Quantity', 'sum'),
    Avg_UnitPrice=('UnitPrice', 'mean')
).reset_index()

user_data['Active_Days'] = (user_data['Last_Purchase'] - user_data['First_Purchase']).dt.days.replace(0, 1)
user_data['Days_Since_Last'] = (latest_date - user_data['Last_Purchase']).dt.days
user_data['Avg_Days_Between'] = user_data['Active_Days'] / (user_data['Total_Orders'] - 1)

# Identify warm users: low freq (1-2 orders) + last purchase 30-90 days ago
warm_users = user_data[(user_data['Total_Orders'] <= 2) & 
                        (user_data['Days_Since_Last'] >= 30) & 
                        (user_data['Days_Since_Last'] <= 90)]

print(f"Warm users identified: {len(warm_users)}")

# Breakdown: 1 order vs 2 orders
warm_1order = warm_users[warm_users['Total_Orders'] == 1]
warm_2orders = warm_users[warm_users['Total_Orders'] == 2]

print(f"\nBreakdown:")
print(f"  1 order (new, never returned): {len(warm_1order)}")
print(f"  2 orders (returned once, then stopped): {len(warm_2orders)}")

# ========== Profile Analysis ==========
print("\n=== 1. Warm User Profile ===")

print(f"\n1-order warm users:")
print(f"  Avg spend: {warm_1order['Total_Spend'].mean():,.2f}")
print(f"  Avg order value: {warm_1order['Avg_Order_Value'].mean():,.2f}")
print(f"  Median days since purchase: {warm_1order['Days_Since_Last'].median():.0f}")
print(f"  Countries:")
print(warm_1order['Country'].value_counts().to_string())

print(f"\n2-order warm users:")
print(f"  Avg total spend: {warm_2orders['Total_Spend'].mean():,.2f}")
print(f"  Avg order value: {warm_2orders['Avg_Order_Value'].mean():,.2f}")
print(f"  Median days since last: {warm_2orders['Days_Since_Last'].median():.0f}")
print(f"  Median days between 1st-2nd: {warm_2orders['Avg_Days_Between'].median():.0f}")
print(f"  Countries:")
print(warm_2orders['Country'].value_counts().to_string())

# ========== Spend brackets ==========
print("\n=== 2. Spend Distribution of Warm Users ===")

bins = [0, 500, 1000, 2000, 3000, 10000]
labels = ['0-500', '500-1000', '1000-2000', '2000-3000', '3000+']
warm_users['Spend_Bracket'] = pd.cut(warm_users['Total_Spend'], bins=bins, labels=labels)

print("\nSpend bracket distribution:")
print(warm_users['Spend_Bracket'].value_counts().sort_index().to_string())

# ========== Time-based urgency ==========
print("\n=== 3. Urgency Tiers ===")

# Within 40-day window (the golden window from earlier analysis)
urgent = warm_users[warm_users['Days_Since_Last'] <= 40]
moderate = warm_users[(warm_users['Days_Since_Last'] > 40) & (warm_users['Days_Since_Last'] <= 60)]
slow = warm_users[warm_users['Days_Since_Last'] > 60]

print(f"\nUrgency tiers:")
print(f"  Urgent (<40d since last): {len(urgent)} users - act NOW")
print(f"  Moderate (40-60d): {len(moderate)} users - act soon")
print(f"  Slow (60-90d): {len(slow)} users - last chance")

# ========== Category Analysis ==========
print("\n=== 4. What Warm Users Bought ===")

warm_ids = warm_users['CustomerID'].tolist()
warm_df = df[df['CustomerID'].isin(warm_ids)]

# Category preference
warm_cats = warm_df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False)
all_cats = df.groupby('Description')['TotalPrice'].sum()

print("\nCategory preference (warm users vs all):")
for cat in all_cats.index:
    warm_share = warm_cats.get(cat, 0) / warm_cats.sum() * 100
    all_share = all_cats[cat] / all_cats.sum() * 100
    diff = warm_share - all_share
    marker = " <<< OPPORTUNITY" if diff > 2 else ""
    print(f"  {cat:<15} Warm: {warm_share:>5.1f}%  All: {all_share:>5.1f}%  Diff: {diff:>+5.1f}%{marker}")

# ========== Country deep-dive ==========
print("\n=== 5. Country Breakdown ===")

# Compare warm users by country
country_warm = warm_users['Country'].value_counts()
country_all = user_data['Country'].value_counts()

print("\nWarm user share by country:")
for country in country_warm.index[:5]:
    warm_count = country_warm[country]
    all_count = country_all[country]
    warm_pct = warm_count / all_count * 100
    print(f"  {country:<15} {warm_count}/{all_count} warm ({warm_pct:.1f}% of all {country} users)")

# ========== Charts ==========
print("\n--- Generating Charts ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Chart 1: Urgency pie
urgency_data = [len(urgent), len(moderate), len(slow)]
axes[0, 0].pie(urgency_data, labels=['Urgent (<40d)', 'Moderate (40-60d)', 'Slow (60-90d)'],
               autopct='%1.0f%%', colors=['#e74c3c', '#f39c12', '#3498db'], startangle=90)
axes[0, 0].set_title('Activation Urgency Tiers\n(140 Warm Users)', fontsize=13)

# Chart 2: Spend bracket
spend_order = warm_users['Spend_Bracket'].value_counts().sort_index()
axes[0, 1].bar(spend_order.index.astype(str), spend_order.values, color='#2ecc71', edgecolor='white')
axes[0, 1].set_xlabel('Total Spend Bracket')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_title('Spend Distribution of Warm Users', fontsize=13)
for i, (x, y) in enumerate(zip(spend_order.index, spend_order.values)):
    axes[0, 1].text(i, y + 1, str(y), ha='center', fontsize=10)

# Chart 3: 1-order vs 2-order
order_types = ['1 order', '2 orders']
type_counts = [len(warm_1order), len(warm_2orders)]
axes[1, 0].bar(order_types, type_counts, color=['#e74c3c', '#3498db'], edgecolor='white')
axes[1, 0].set_ylabel('Number of Users')
axes[1, 0].set_title('Warm Users: New vs Returning', fontsize=13)
for i, (x, y) in enumerate(zip(order_types, type_counts)):
    axes[1, 0].text(i, y + 2, f'{y} ({y/len(warm_users)*100:.0f}%)', ha='center', fontsize=11)

# Chart 4: Spending potential
# Compare warm users 1-order vs 2-order spending
spend_1order = warm_1order['Total_Spend'].mean()
spend_2order = warm_2orders['Total_Spend'].mean()
axes[1, 1].bar(['1-order users', '2-order users'], [spend_1order, spend_2order], 
               color=['#e74c3c', '#2ecc71'], edgecolor='white')
axes[1, 1].set_ylabel('Avg Total Spend')
axes[1, 1].set_title('Average Spend Comparison', fontsize=13)
for i, (x, y) in enumerate(zip(['1-order', '2-order'], [spend_1order, spend_2order])):
    axes[1, 1].text(i, y + 20, f'{y:,.0f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '12_warm_users_analysis.png'), dpi=150)
plt.close()

# ========== Final summary ==========
print("\n=== 6. Activation Strategy Summary ===")

# Calculate ROI potential
if len(warm_2orders) > 0:
    avg_spend_2order = warm_2orders['Total_Spend'].mean()
    print(f"\nActivation ROI Estimate:")
    print(f"  If 2-order users come back for a 3rd order,")
    print(f"  they spend avg {avg_spend_2order:,.0f} in first 2 orders.")
    print(f"  A 3rd order at similar value could add ~{avg_spend_2order/2:,.0f} per user.")

print(f"\nRecommended Action Plan:")
print(f"  1. [Immediate] {len(urgent)} urgent users: push notification + discount")
print(f"  2. [This week] {len(moderate)} moderate users: email campaign")
print(f"  3. [This month] {len(slow)} slow users: retargeting ads")

# Priority users: 2-order urgent (highest conversion chance)
priority = warm_users[(warm_users['Total_Orders'] == 2) & (warm_users['Days_Since_Last'] <= 40)]
print(f"\n  PRIORITY TARGET: {len(priority)} users (2 orders + returned within 40 days + still warm)")
print(f"  These are the MOST likely to convert to 3+ orders.")
if len(priority) > 0:
    print(f"  Avg spend: {priority['Total_Spend'].mean():,.2f}")
    print(f"  Avg days since last: {priority['Days_Since_Last'].mean():.0f}")

print("\n=== Analysis Complete ===")
