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

print("=== User Profiling Analysis ===\n")

# 1. Load & Clean
df = pd.read_csv(os.path.join(DATA_DIR, 'mini_project_data.csv'))
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"Raw data: {df.shape[0]} rows")

# Remove returns
n_returns = (df['Quantity'] < 0).sum()
df_clean = df[df['Quantity'] > 0].copy()
print(f"Returns removed: {n_returns} -> Clean: {df_clean.shape[0]} rows")

# 2. Build user profile (one row per customer)
print("\n--- Building User Profiles ---")

user_profile = df_clean.groupby('CustomerID').agg(
    Purchase_Count=('InvoiceNo', 'nunique'),          # order count
    Total_Spend=('TotalPrice', 'sum'),                 # total spend
    Avg_Order_Value=('TotalPrice', 'mean'),            # average order value
    Avg_UnitPrice=('UnitPrice', 'mean'),               # average unit price
    First_Purchase=('InvoiceDate', 'min'),             # first purchase date
    Last_Purchase=('InvoiceDate', 'max'),              # last purchase date
    Country=('Country', 'first')                       # country
).reset_index()

n_users = len(user_profile)
print(f"Total unique customers: {n_users}")

# 3. Spend Segmentation (Percentile-based)
print("\n--- Spend Segmentation ---")
p33 = user_profile['Total_Spend'].quantile(0.33)
p67 = user_profile['Total_Spend'].quantile(0.67)
print(f"Spend thresholds: Low < {p33:.2f}, Medium {p33:.2f}-{p67:.2f}, High > {p67:.2f}")

def spend_segment(spend):
    if spend > p67:
        return 'High'
    elif spend >= p33:
        return 'Medium'
    else:
        return 'Low'

user_profile['Spend_Segment'] = user_profile['Total_Spend'].apply(spend_segment)

spend_dist = user_profile['Spend_Segment'].value_counts()
print("\nSpend segment distribution:")
for seg in ['High', 'Medium', 'Low']:
    count = spend_dist.get(seg, 0)
    pct = count / n_users * 100
    avg_spend = user_profile[user_profile['Spend_Segment'] == seg]['Total_Spend'].mean()
    print(f"  {seg}: {count} users ({pct:.1f}%), Avg Spend: {avg_spend:,.2f}")

# 4. Purchase Frequency Segmentation
print("\n--- Purchase Frequency Segmentation ---")

def freq_segment(count):
    if count >= 5:
        return 'High (5+)'
    elif count >= 3:
        return 'Medium (3-4)'
    else:
        return 'Low (1-2)'

user_profile['Freq_Segment'] = user_profile['Purchase_Count'].apply(freq_segment)

freq_dist = user_profile['Freq_Segment'].value_counts()
print("\nPurchase frequency distribution:")
for seg in ['High (5+)', 'Medium (3-4)', 'Low (1-2)']:
    count = freq_dist.get(seg, 0)
    pct = count / n_users * 100
    avg_count = user_profile[user_profile['Freq_Segment'] == seg]['Purchase_Count'].mean()
    print(f"  {seg}: {count} users ({pct:.1f}%), Avg Orders: {avg_count:.1f}")

# 5. Cross-tab: Spend x Frequency
print("\n--- Cross Analysis: Spend x Frequency ---")
cross = pd.crosstab(user_profile['Freq_Segment'], user_profile['Spend_Segment'])
print(cross.to_string())

# 6. Charts
print("\n--- Generating Charts ---")

# Chart 1: Spend segment pie chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie
spend_colors = {'High': '#2ecc71', 'Medium': '#3498db', 'Low': '#e74c3c'}
sizes = [spend_dist.get(s, 0) for s in ['High', 'Medium', 'Low']]
labels = ['High Value', 'Medium Value', 'Low Value']
explode = (0.05, 0, 0)
axes[0].pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, 
            colors=['#2ecc71', '#3498db', '#e74c3c'], explode=explode)
axes[0].set_title('User Value Segmentation\n(by Total Spend)', fontsize=13)

# Bar
freq_colors = {'High (5+)': '#2ecc71', 'Medium (3-4)': '#3498db', 'Low (1-2)': '#e74c3c'}
freq_order = ['Low (1-2)', 'Medium (3-4)', 'High (5+)']
freq_sizes = [freq_dist.get(s, 0) for s in freq_order]
bars = axes[1].bar(freq_order, freq_sizes, color=['#e74c3c', '#3498db', '#2ecc71'])
axes[1].set_title('User Purchase Frequency\n(by Order Count)', fontsize=13)
axes[1].set_ylabel('Number of Users')
for bar, val in zip(bars, freq_sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val}\n({val/n_users*100:.0f}%)', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '08_user_segmentation.png'), dpi=150)
plt.close()

# Chart 2: Spend distribution histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(user_profile['Total_Spend'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(p33, color='#e74c3c', linestyle='--', label=f'33% = {p33:.0f}')
axes[0].axvline(p67, color='#2ecc71', linestyle='--', label=f'67% = {p67:.0f}')
axes[0].set_xlabel('Total Spend')
axes[0].set_ylabel('Number of Users')
axes[0].set_title('Total Spend Distribution')
axes[0].legend()

axes[1].hist(user_profile['Purchase_Count'], bins=range(1, 20), color='#2ecc71', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Number of Orders')
axes[1].set_ylabel('Number of Users')
axes[1].set_title('Purchase Frequency Distribution')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '09_spend_freq_distribution.png'), dpi=150)
plt.close()

# 7. Detailed stats
print("\n--- Detailed User Profile Stats ---")
print(f"\nTotal Spend Stats:")
stats = user_profile['Total_Spend'].describe(percentiles=[.25, .5, .75, .9, .95])
print(stats.to_string())

print(f"\nPurchase Count Stats:")
stats = user_profile['Purchase_Count'].describe(percentiles=[.25, .5, .75, .9, .95])
print(stats.to_string())

# Top 10% users contribution
top10 = user_profile.nlargest(int(n_users * 0.1), 'Total_Spend')
top10_spend_share = top10['Total_Spend'].sum() / user_profile['Total_Spend'].sum() * 100
print(f"\nTop 10% users contribute: {top10_spend_share:.1f}% of total spend")
print(f"Top 10% avg purchase count: {top10['Purchase_Count'].mean():.1f}")

# Bottom 50% users contribution  
bottom50 = user_profile.nsmallest(int(n_users * 0.5), 'Total_Spend')
bottom50_spend_share = bottom50['Total_Spend'].sum() / user_profile['Total_Spend'].sum() * 100
print(f"Bottom 50% users contribute: {bottom50_spend_share:.1f}% of total spend")

# High value user features
print("\n--- High Value User Profile ---")
high_users = user_profile[user_profile['Spend_Segment'] == 'High']
print(f"Total users: {len(high_users)}")
print(f"Avg spend: {high_users['Total_Spend'].mean():,.2f}")
print(f"Avg purchase count: {high_users['Purchase_Count'].mean():.1f}")
print(f"Avg order value: {high_users['Avg_Order_Value'].mean():,.2f}")
print(f"Top countries:")
print(high_users['Country'].value_counts().head().to_string())

# Low value user features  
print("\n--- Low Value User Profile ---")
low_users = user_profile[user_profile['Spend_Segment'] == 'Low']
print(f"Total users: {len(low_users)}")
print(f"Avg spend: {low_users['Total_Spend'].mean():,.2f}")
print(f"Avg purchase count: {low_users['Purchase_Count'].mean():.1f}")
print(f"Avg order value: {low_users['Avg_Order_Value'].mean():,.2f}")
print(f"Top countries:")
print(low_users['Country'].value_counts().head().to_string())

# Save results
user_profile.to_csv(os.path.join(DATA_DIR, 'user_profiles.csv'), index=False, encoding='utf-8-sig')
print(f"\nUser profiles saved to: {os.path.join(DATA_DIR, 'user_profiles.csv')}")

print("\n=== Analysis Complete ===")
