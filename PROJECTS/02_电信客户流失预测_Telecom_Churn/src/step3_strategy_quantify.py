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

print("=== Step 3: Quantify Retention Strategies ===\n")

# Load and clean
df = pd.read_csv(os.path.join(DATA_DIR, 'telco_churn.csv'))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Focus on month-to-month users
mtm = df[df['Contract'] == 'Month-to-month'].copy()
print(f"Month-to-month users: {len(mtm)} ({len(mtm)/len(df)*100:.1f}% of total)")
print(f"  Churned: {(mtm['Churn']=='Yes').sum()} ({(mtm['Churn']=='Yes').mean()*100:.1f}%)")
print(f"  Stayed: {(mtm['Churn']=='No').sum()} ({(mtm['Churn']=='No').mean()*100:.1f}%)")

# Revenue lost from churned month-to-month users
mtm_churned = mtm[mtm['Churn'] == 'Yes']
total_lost_revenue = mtm_churned['TotalCharges'].sum()
print(f"\n  Revenue lost from churned MTM users: ${total_lost_revenue:,.0f}")
print(f"  Avg lifetime revenue per churned MTM user: ${mtm_churned['TotalCharges'].mean():,.0f}")

# ========== Analysis 1: Who stays on Month-to-Month? ==========
print("\n=== 1. What Keeps Month-to-Month Users? ===")

mtm_stay = mtm[mtm['Churn'] == 'No']
mtm_churn = mtm[mtm['Churn'] == 'Yes']

# Services that protect
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'StreamingTV', 'StreamingMovies', 'MultipleLines']

print("\n  Service adoption: Stay vs Churn (% with Yes):")
for col in service_cols:
    stay_yes = (mtm_stay[col] == 'Yes').mean() * 100
    churn_yes = (mtm_churn[col] == 'Yes').mean() * 100
    diff = stay_yes - churn_yes
    bar = "█" * max(0, int(abs(diff)/2))
    print(f"  {col:<20s}: Stay={stay_yes:5.1f}%  Churn={churn_yes:5.1f}%  Δ={diff:+5.1f}% {bar}")

# ========== Analysis 2: Contract conversion impact ==========
print("\n=== 2. If MTM → 1-Year: Estimated Impact ===")

# Current 1-year contract churn rate (all users, not just new)
one_year = df[df['Contract'] == 'One year']
one_year_churn = (one_year['Churn'] == 'Yes').mean() * 100
print(f"  Current 1-year contract churn rate: {one_year_churn:.1f}%")
print(f"  Current MTM churn rate: {(mtm['Churn']=='Yes').mean()*100:.1f}%")

# Simulate: convert X% of MTM users to 1-year
mtm_total = len(mtm)
mtm_churn_n = (mtm['Churn'] == 'Yes').sum()

for conversion_rate in [10, 20, 30]:
    converted = int(mtm_total * conversion_rate / 100)
    saved_from_churn = converted * ((mtm['Churn']=='Yes').mean() - one_year_churn/100)
    new_churn_n = mtm_churn_n - saved_from_churn
    new_churn_pct = new_churn_n / mtm_total * 100
    print(f"\n  Convert {conversion_rate}% ({converted} users):")
    print(f"    Previously churned: {int(converted * (mtm['Churn']=='Yes').mean())}")
    print(f"    Would save: {saved_from_churn:.0f} customers")
    print(f"    New MTM churn rate: {new_churn_pct:.1f}% (was {(mtm['Churn']=='Yes').mean()*100:.1f}%)")

# ========== Analysis 3: Security service bundle impact ==========
print("\n=== 3. Security/Tech Support Bundle: Retention Impact ===")

# Users with BOTH security and tech support
has_both = mtm[(mtm['OnlineSecurity'] == 'Yes') & (mtm['TechSupport'] == 'Yes')]
has_one = mtm[((mtm['OnlineSecurity'] == 'Yes') & (mtm['TechSupport'] != 'Yes')) |
              ((mtm['OnlineSecurity'] != 'Yes') & (mtm['TechSupport'] == 'Yes'))]
has_none = mtm[(mtm['OnlineSecurity'] != 'Yes') & (mtm['TechSupport'] != 'Yes')]

print(f"\n  MTM users by security/support status:")
print(f"    Both Security+TechSupport: {len(has_both)} users, churn={(has_both['Churn']=='Yes').mean()*100:.1f}%")
print(f"    One of them: {len(has_one)} users, churn={(has_one['Churn']=='Yes').mean()*100:.1f}%")
print(f"    Neither: {len(has_none)} users, churn={(has_none['Churn']=='Yes').mean()*100:.1f}%")

# Potential savings: give security+tech to "neither" group
if len(has_none) > 0:
    current_churn = (has_none['Churn'] == 'Yes').sum()
    if len(has_both) > 0:
        target_churn = (has_both['Churn'] == 'Yes').mean()
        saved = current_churn - int(len(has_none) * target_churn)
    else:
        saved = 0
    print(f"\n  If 'Neither' group got Security+TechSupport:")
    print(f"    Current churn: {current_churn}")
    print(f"    Estimated churn after: {int(len(has_none) * target_churn)}")
    print(f"    Would save: {saved} customers")
    # Revenue impact
    avg_monthly = has_none['MonthlyCharges'].mean()
    print(f"    Avg monthly charge: ${avg_monthly:.2f}")
    print(f"    Potential annual revenue saved: ${saved * avg_monthly * 12:,.0f}")

# ========== Analysis 4: Payment method conversion ==========
print("\n=== 4. Electronic Check → Automatic Payment: Impact ===")

echeck = mtm[mtm['PaymentMethod'] == 'Electronic check']
echeck_churn = (echeck['Churn'] == 'Yes').mean() * 100

auto_pay = mtm[mtm['PaymentMethod'].isin(['Credit card (automatic)', 'Bank transfer (automatic)'])]
auto_churn = (auto_pay['Churn'] == 'Yes').mean() * 100

print(f"  Electronic check MTM users: {len(echeck)}, churn={echeck_churn:.1f}%")
print(f"  Auto-pay MTM users: {len(auto_pay)}, churn={auto_churn:.1f}%")

if len(echeck) > 0:
    saved_by_switch = int(len(echeck) * (echeck_churn/100 - auto_churn/100))
    print(f"  If all echeck → auto-pay: would save ~{saved_by_switch} customers")

# ========== Analysis 5: Fiber optic deep dive ==========
print("\n=== 5. Fiber Optic: Why So High Churn? ===")

fiber_mtm = mtm[mtm['InternetService'] == 'Fiber optic']
fiber_churn_rate = (fiber_mtm['Churn'] == 'Yes').mean() * 100

print(f"  MTM + Fiber optic: {len(fiber_mtm)} users, churn={fiber_churn_rate:.1f}%")
print(f"  Avg monthly: ${fiber_mtm['MonthlyCharges'].mean():.2f}")
print(f"  Avg tenure: {fiber_mtm['tenure'].mean():.1f} months")

# Does security/support help fiber users?
fiber_with_sec = fiber_mtm[(fiber_mtm['OnlineSecurity'] == 'Yes') | (fiber_mtm['TechSupport'] == 'Yes')]
fiber_without_sec = fiber_mtm[(fiber_mtm['OnlineSecurity'] != 'Yes') & (fiber_mtm['TechSupport'] != 'Yes')]

print(f"\n  Fiber + Security/TechSupport: {len(fiber_with_sec)} users, churn={(fiber_with_sec['Churn']=='Yes').mean()*100:.1f}%")
print(f"  Fiber w/o Security/TechSupport: {len(fiber_without_sec)} users, churn={(fiber_without_sec['Churn']=='Yes').mean()*100:.1f}%")

# ========== Charts ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Chart 1: Stay vs Churn - service adoption gap
service_data = []
for col in service_cols:
    stay_pct = (mtm_stay[col] == 'Yes').mean() * 100
    churn_pct = (mtm_churn[col] == 'Yes').mean() * 100
    service_data.append((col, stay_pct, churn_pct, stay_pct - churn_pct))

service_data.sort(key=lambda x: x[3], reverse=True)
for i, (name, stay, churn, diff) in enumerate(service_data):
    axes[0, 0].barh(i*2, stay, 0.8, color='#2ecc71', label='Stay' if i == 0 else '')
    axes[0, 0].barh(i*2+0.8, churn, 0.8, color='#e74c3c', label='Churn' if i == 0 else '')
    axes[0, 0].text(max(stay, churn) + 2, i*2+0.8, f'Δ={diff:+.0f}%', va='center', fontweight='bold',
                    color='#e74c3c' if diff < 0 else '#2ecc71')

axes[0, 0].set_yticks([i*2+0.4 for i in range(len(service_data))])
axes[0, 0].set_yticklabels([s[0] for s in service_data], fontsize=9)
axes[0, 0].set_xlabel('Adoption Rate (%)')
axes[0, 0].set_title('MTM Users: Service Adoption\n(Stay vs Churn)', fontsize=13)
axes[0, 0].legend()

# Chart 2: Churn rate by security bundle
bundle_data = {
    'Both': (len(has_both), (has_both['Churn']=='Yes').mean()*100),
    'One': (len(has_one), (has_one['Churn']=='Yes').mean()*100),
    'Neither': (len(has_none), (has_none['Churn']=='Yes').mean()*100)
}

bundle_labels = list(bundle_data.keys())
bundle_counts = [bundle_data[k][0] for k in bundle_labels]
bundle_rates = [bundle_data[k][1] for k in bundle_labels]
bundle_colors = ['#2ecc71', '#f39c12', '#e74c3c']

for i, (label, count, rate, color) in enumerate(zip(bundle_labels, bundle_counts, bundle_rates, bundle_colors)):
    axes[0, 1].bar(i, count, color=color, alpha=0.6, edgecolor='white')
    axes[0, 1].text(i, count + 10, f'{count}\n({rate:.0f}% churn)', ha='center', fontsize=10)

axes[0, 1].set_xticks(range(len(bundle_labels)))
axes[0, 1].set_xticklabels(['Security\n+TechSupport', 'One\nof them', 'Neither'])
axes[0, 1].set_ylabel('Number of MTM Users')
axes[0, 1].set_title('Churn Rate by Security Bundle\n(MTM Users)', fontsize=13)

# Chart 3: Contract conversion scenario
conversion_rates = [0, 10, 20, 30, 50]
churn_after = []
for cr in conversion_rates:
    converted = int(mtm_total * cr / 100)
    saved = converted * ((mtm['Churn']=='Yes').mean() - one_year_churn/100)
    new_churn = (mtm_churn_n - saved) / mtm_total * 100
    churn_after.append(new_churn)

axes[1, 0].plot(conversion_rates, churn_after, 'o-', linewidth=3, markersize=12, color='#3498db')
axes[1, 0].fill_between(conversion_rates, churn_after, alpha=0.2, color='#3498db')
axes[1, 0].axhline(y=one_year_churn, color='#2ecc71', linestyle='--', 
                   label=f'1-year contract churn ({one_year_churn:.0f}%)')
axes[1, 0].set_xlabel('MTM Users Converted to 1-Year (%)')
axes[1, 0].set_ylabel('MTM Churn Rate (%)')
axes[1, 0].set_title('Impact of Converting MTM → 1-Year', fontsize=13)
axes[1, 0].legend()
for i, (x, y) in enumerate(zip(conversion_rates, churn_after)):
    axes[1, 0].annotate(f'{y:.1f}%', (x, y), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=10)

# Chart 4: ROI comparison of strategies
# Strategy 1: Convert 20% MTM → 1yr
s1_users = int(mtm_total * 0.2)
s1_saved = s1_users * ((mtm['Churn']=='Yes').mean() - one_year_churn/100)
# Strategy 2: Give security+tech to "neither" users
s2_users = len(has_none)
if len(has_both) > 0:
    target_rate = (has_both['Churn'] == 'Yes').mean()
    s2_saved = int((has_none['Churn']=='Yes').mean() - target_rate) * s2_users
else:
    s2_saved = 0
# Strategy 3: Switch echeck to auto-pay
s3_users = len(echeck)
s3_saved = int(s3_users * (echeck_churn - auto_churn) / 100)

strategies = [
    ('Convert 20% MTM\nto 1-Year', s1_saved, s1_users),
    ('Give Security+Tech\nw/o services', s2_saved, s2_users),
    ('Switch eCheck\nto Auto-Pay', s3_saved, s3_users)
]

strat_labels = [s[0] for s in strategies]
strat_saved = [max(0, s[1]) for s in strategies]
strat_users = [s[2] for s in strategies]

colors_roi = ['#3498db', '#2ecc71', '#f39c12']
bars = axes[1, 1].barh(strat_labels, strat_saved, color=colors_roi, edgecolor='white')
axes[1, 1].set_xlabel('Estimated Customers Saved')
axes[1, 1].set_title('Strategy ROI Comparison\n(Estimated Customers Retained)', fontsize=13)

for bar, saved, total in zip(bars, strat_saved, strat_users):
    axes[1, 1].text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{int(saved)} saved\n(from {total} target)', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '03_retention_strategies.png'), dpi=150)
plt.close()

# ========== Final strategy summary ==========
print(f"\n=== 6. Strategy Impact Summary ===")
print(f"""
Strategy Comparison:

1. CONTRACT CONVERSION (convert 20% MTM → 1-year):
   - Target: {s1_users} users
   - Save: ~{s1_saved:.0f} customers from churning
   - Cost: Revenue loss from discounts (TBD)
   - Risk: Low (proven success of 1-year contracts)

2. SECURITY + TECH SUPPORT BUNDLE (for MTM users lacking both):
   - Target: {s2_users} users
   - Save: ~{s2_saved:.0f} customers
   - Cost: Service delivery cost
   - Risk: Low (these services already exist)

3. PAYMENT METHOD MIGRATION (eCheck → Auto-pay):
   - Target: {s3_users} users
   - Save: ~{int(s3_saved)} customers  
   - Cost: Incentive to switch
   - Risk: Medium (users may resist auto-pay)

COMBINED POTENTIAL (conservative estimate):
   ~{int(s1_saved + s2_saved + s3_saved)} customers saved
   = ~{(s1_saved + s2_saved + s3_saved) / mtm_churn_n * 100:.0f}% reduction in MTM churn
""")

print("=== Step 3 Complete ===")
