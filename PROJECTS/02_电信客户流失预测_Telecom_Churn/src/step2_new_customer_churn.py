import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'D:\AI_Dateannaly\PROJECTS\project_churn_prediction'
DATA_DIR = os.path.join(BASE, 'data')
CHARTS_DIR = os.path.join(BASE, 'charts')

print("=== Step 2: New Customer (0-12m) Churn Profile ===\n")

# Load and clean
df = pd.read_csv(os.path.join(DATA_DIR, 'telco_churn.csv'))

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Filter new customers
new_cust = df[df['tenure'] <= 12].copy()
print(f"New customers (0-12m): {len(new_cust)} of {len(df)} total ({len(new_cust)/len(df)*100:.1f}%)")

churn_new = new_cust['Churn'].value_counts()
print(f"  Stayed: {churn_new.get('No', 0)}")
print(f"  Churned: {churn_new.get('Yes', 0)} ({churn_new.get('Yes', 0)/len(new_cust)*100:.1f}%)")

# Split
new_churn = new_cust[new_cust['Churn'] == 'Yes']
new_stay = new_cust[new_cust['Churn'] == 'No']

print(f"\n=== 1. Contract Type (Most Important) ===")
for col in ['Contract', 'PaymentMethod', 'InternetService']:
    print(f"\n  [{col}] comparison:")
    for val in new_cust[col].unique():
        total = len(new_cust[new_cust[col] == val])
        churned = (new_cust[(new_cust[col] == val) & (new_cust['Churn'] == 'Yes')])
        cr = len(churned) / total * 100
        marker = " <<< HIGH" if cr > 40 else ""
        print(f"    {val:<30s}: {len(churned):>3d}/{total:>3d} = {cr:.1f}%{marker}")

print(f"\n=== 2. Monthly Charges ===")
print(f"  Stayed customers - avg monthly: {new_stay['MonthlyCharges'].mean():.2f}")
print(f"  Churned customers - avg monthly: {new_churn['MonthlyCharges'].mean():.2f}")
print(f"  Difference: +{new_churn['MonthlyCharges'].mean() - new_stay['MonthlyCharges'].mean():.2f}")

# Monthly charges brackets
bins = [0, 30, 60, 90, 120]
labels = ['Low (<$30)', 'Medium ($30-60)', 'High ($60-90)', 'Very High (>$90)']
new_cust['Charge_Bracket'] = pd.cut(new_cust['MonthlyCharges'], bins=bins, labels=labels)
print(f"\n  Churn rate by charge bracket:")
for bracket in labels:
    subset = new_cust[new_cust['Charge_Bracket'] == bracket]
    if len(subset) > 0:
        cr = (subset['Churn'] == 'Yes').mean() * 100
        print(f"    {bracket:<25s}: {cr:.1f}% ({len(subset)} users)")

print(f"\n=== 3. Services ===")
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in service_cols:
    if col in new_cust.columns:
        for val in new_cust[col].unique()[:3]:
            subset = new_cust[new_cust[col] == val]
            cr = (subset['Churn'] == 'Yes').mean() * 100
            diff = cr - 47.7  # baseline
            marker = f" ({diff:+.1f}% vs avg)" if abs(diff) > 5 else ""
            print(f"  {col}={val:<15s}: {cr:.1f}%{marker}")

print(f"\n=== 4. Demographics ===")
for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
    if col in new_cust.columns:
        for val in new_cust[col].unique():
            subset = new_cust[new_cust[col] == val]
            cr = (subset['Churn'] == 'Yes').mean() * 100
            print(f"  {col}={val}: {cr:.1f}% ({len(subset)} users)")

print(f"\n=== 5. PaperlessBilling ===")
for val in new_cust['PaperlessBilling'].unique():
    subset = new_cust[new_cust['PaperlessBilling'] == val]
    cr = (subset['Churn'] == 'Yes').mean() * 100
    print(f"  PaperlessBilling={val}: {cr:.1f}% ({len(subset)} users)")

# Top risk combination
print(f"\n=== 6. Top Risk Combination ===")
risk = new_cust[
    (new_cust['Contract'] == 'Month-to-month') &
    (new_cust['MonthlyCharges'] > 60)
]
print(f"  Month-to-month + Monthly > $60: {len(risk)} users")
cr = (risk['Churn'] == 'Yes').mean() * 100
print(f"  Churn rate: {cr:.1f}%")
print(f"  This is {risk['Churn'].value_counts().get('Yes', 0)} churned of {len(risk)} in this segment")

# Add Fiber optic risk
risk2 = new_cust[
    (new_cust['Contract'] == 'Month-to-month') &
    (new_cust['InternetService'] == 'Fiber optic')
]
cr2 = (risk2['Churn'] == 'Yes').mean() * 100
print(f"\n  Month-to-month + Fiber optic: {len(risk2)} users, churn={cr2:.1f}%")

# === Charts ===
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Chart 1: Contract type for new customers
contract_order = ['Month-to-month', 'One year', 'Two year']
contract_data = {}
for ct in contract_order:
    if ct in new_cust['Contract'].values:
        subset = new_cust[new_cust['Contract'] == ct]
        contract_data[ct] = {
            'total': len(subset),
            'churn_pct': (subset['Churn'] == 'Yes').mean() * 100,
            'churn_n': (subset['Churn'] == 'Yes').sum()
        }

ct_labels = list(contract_data.keys())
ct_totals = [contract_data[k]['total'] for k in ct_labels]
ct_churn_pct = [contract_data[k]['churn_pct'] for k in ct_labels]

x = np.arange(len(ct_labels))
width = 0.35
bars1 = axes[0, 0].bar(x, ct_totals, width, color='#3498db', edgecolor='white', label='Total Users')
ax2 = axes[0, 0].twinx()
bars2 = ax2.bar(x + width, ct_churn_pct, width, color='#e74c3c', edgecolor='white', label='Churn Rate (%)')
axes[0, 0].set_xticks(x + width/2)
axes[0, 0].set_xticklabels(ct_labels, fontsize=9)
axes[0, 0].set_ylabel('Number of Users')
ax2.set_ylabel('Churn Rate (%)')
axes[0, 0].set_title('New Customers: Contract Type Distribution', fontsize=13)
for bar, val in zip(bars1, ct_totals):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontsize=10)
for bar, val in zip(bars2, ct_churn_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# Chart 2: Monthly charges distribution
axes[0, 1].hist([new_stay['MonthlyCharges'], new_churn['MonthlyCharges']],
                bins=20, alpha=0.7, label=['Stayed', 'Churned'],
                color=['#2ecc71', '#e74c3c'])
axes[0, 1].axvline(new_stay['MonthlyCharges'].mean(), color='#2ecc71', linestyle='--', linewidth=2)
axes[0, 1].axvline(new_churn['MonthlyCharges'].mean(), color='#e74c3c', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Monthly Charges ($)')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_title('Monthly Charges: Stayed vs Churned', fontsize=13)
axes[0, 1].legend()

# Chart 3: Service adoption (churn vs stay)
service_impact = []
for col in service_cols:
    if col in new_cust.columns:
        for val in ['Yes', 'No']:
            subset = new_cust[new_cust[col] == val]
            cr = (subset['Churn'] == 'Yes').mean() * 100
            diff = cr - 47.7
            service_impact.append((f'{col}={val}', diff, cr))

service_impact.sort(key=lambda x: abs(x[1]), reverse=True)
top_services = service_impact[:10]
axes[1, 0].barh([s[0] for s in reversed(top_services)],
                [s[1] for s in reversed(top_services)],
                color=['#e74c3c' if s[1] > 0 else '#2ecc71' for s in reversed(top_services)])
axes[1, 0].axvline(x=0, color='black', linewidth=1)
axes[1, 0].set_xlabel('Deviation from Avg Churn Rate (%)')
axes[1, 0].set_title('Service Impact on Churn\n(vs avg 47.7%)', fontsize=13)

# Chart 4: Risk Matrix
risk_segments = {
    'Month-to-month\n+ High Monthly (>$80)': None,
    'Month-to-month\n+ Medium Monthly': None,
    'Month-to-month\n+ Low Monthly (<$50)': None,
    '1yr contract\n+ Any Monthly': None,
    '2yr contract\n+ Any Monthly': None,
}

risk_data = {}
for label in risk_segments:
    if 'Month-to-month' in label and 'High' in label:
        subset = new_cust[(new_cust['Contract'] == 'Month-to-month') & (new_cust['MonthlyCharges'] > 80)]
    elif 'Month-to-month' in label and 'Low' in label:
        subset = new_cust[(new_cust['Contract'] == 'Month-to-month') & (new_cust['MonthlyCharges'] < 50)]
    elif 'Month-to-month' in label and 'Medium' in label:
        subset = new_cust[(new_cust['Contract'] == 'Month-to-month') & 
                          (new_cust['MonthlyCharges'] >= 50) & (new_cust['MonthlyCharges'] <= 80)]
    elif '1yr' in label:
        subset = new_cust[new_cust['Contract'] == 'One year']
    elif '2yr' in label:
        subset = new_cust[new_cust['Contract'] == 'Two year']
    else:
        subset = new_cust
    
    if len(subset) > 0:
        risk_data[label] = (len(subset), (subset['Churn'] == 'Yes').mean() * 100)

risk_labels = list(risk_data.keys())
risk_counts = [risk_data[k][0] for k in risk_labels]
risk_rates = [risk_data[k][1] for k in risk_labels]
risk_colors = ['#e74c3c' if r > 50 else '#f39c12' if r > 30 else '#2ecc71' for r in risk_rates]

scatter = axes[1, 1].scatter(risk_counts, risk_rates, s=[c*0.5 for c in risk_counts],
                              c=risk_colors, alpha=0.7, edgecolors='black')
axes[1, 1].axhline(y=47.7, color='gray', linestyle='--', alpha=0.5, label='Avg churn')

for i, label in enumerate(risk_labels):
    axes[1, 1].annotate(label, (risk_counts[i], risk_rates[i]),
                        xytext=(10, 5), textcoords='offset points', fontsize=8)

axes[1, 1].set_xlabel('Number of Users in Segment')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].set_title('Risk Matrix: Contract × Monthly Charges', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '02_new_customer_churn.png'), dpi=150)
plt.close()

print(f"\n=== Charts saved ===")
print("\n=== Step 2 Complete ===")
