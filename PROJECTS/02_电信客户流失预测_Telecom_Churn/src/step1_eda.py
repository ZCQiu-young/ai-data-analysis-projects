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

print("=== Project 1: Telco Customer Churn - Step 1 ===\n")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'telco_churn.csv'))
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# ========== 1. Basic Info ==========
print("\n=== 1. Data Overview ===")
print(f"\nColumn list ({df.shape[1]} columns):")
for i, col in enumerate(df.columns):
    col_type = str(df[col].dtype)
    n_unique = df[col].nunique()
    n_null = df[col].isnull().sum()
    null_info = f", NULL:{n_null}" if n_null > 0 else ""
    print(f"  {i+1:2d}. {col:<25s} type={col_type:<10s} unique={n_unique:4d}{null_info}")

# ========== 2. Target: Churn distribution ==========
print("\n=== 2. Target Variable: Churn ===")
churn_dist = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(f"  No:  {churn_dist['No']:>5d} ({churn_pct['No']:.1f}%)")
print(f"  Yes: {churn_dist['Yes']:>5d} ({churn_pct['Yes']:.1f}%)")

# ========== 3. Missing values ==========
print("\n=== 3. Missing Values ===")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  No missing values")
else:
    print(missing.to_string())

# Check for blank strings / special missing markers
print("\n--- Checking for blank/space values ---")
for col in df.columns:
    if df[col].dtype == object:
        blank_count = (df[col].astype(str).str.strip() == '').sum()
        space_count = (df[col].astype(str).str.strip() == ' ').sum()
        if blank_count > 0:
            print(f"  {col}: {blank_count} blank values")

# ========== 4. Numeric summary ==========
print("\n=== 4. Numeric Columns Summary ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n  Numeric columns: {numeric_cols}")
print(df[numeric_cols].describe().to_string())

# ========== 5. Categorical summary ==========
print("\n=== 5. Categorical Columns Summary ===")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('customerID')  # remove ID column
if 'Churn' in cat_cols:
    cat_cols.remove('Churn')

for col in cat_cols:
    print(f"\n  [{col}]")
    vc = df[col].value_counts()
    for val, count in vc.items():
        pct = count / len(df) * 100
        print(f"    {val:<30s}: {count:>4d} ({pct:.1f}%)")

# ========== 6. Churn by key demographics ==========
print("\n=== 6. Churn Rate by Key Features ===")

# By Contract
print("\n  Churn Rate by Contract Type:")
for contract_type in df['Contract'].unique():
    subset = df[df['Contract'] == contract_type]
    churn_rate = (subset['Churn'] == 'Yes').mean() * 100
    print(f"    {contract_type:<20s}: {churn_rate:.1f}% ({len(subset)} users)")

# By Internet Service
if 'InternetService' in df.columns:
    print("\n  Churn Rate by Internet Service:")
    for is_type in df['InternetService'].unique():
        subset = df[df['InternetService'] == is_type]
        churn_rate = (subset['Churn'] == 'Yes').mean() * 100
        print(f"    {is_type:<20s}: {churn_rate:.1f}% ({len(subset)} users)")

# By Payment Method
if 'PaymentMethod' in df.columns:
    print("\n  Churn Rate by Payment Method:")
    for pm_type in df['PaymentMethod'].unique():
        subset = df[df['PaymentMethod'] == pm_type]
        churn_rate = (subset['Churn'] == 'Yes').mean() * 100
        print(f"    {pm_type:<35s}: {churn_rate:.1f}% ({len(subset)} users)")

# By Senior Citizen
print("\n  Churn Rate by Senior Citizen:")
for sc in df['SeniorCitizen'].unique():
    subset = df[df['SeniorCitizen'] == sc]
    churn_rate = (subset['Churn'] == 'Yes').mean() * 100
    label = 'Senior' if sc == 1 else 'Non-senior'
    print(f"    {label:<20s}: {churn_rate:.1f}% ({len(subset)} users)")

# By Partner/Dependents
for col in ['Partner', 'Dependents']:
    if col in df.columns:
        print(f"\n  Churn Rate by {col}:")
        for val in df[col].unique():
            subset = df[df[col] == val]
            churn_rate = (subset['Churn'] == 'Yes').mean() * 100
            print(f"    {val:<20s}: {churn_rate:.1f}% ({len(subset)} users)")

# ========== 7. Data quality: TotalCharges check ==========
print("\n=== 7. Data Quality Checks ===")

# TotalCharges might have non-numeric values
if 'TotalCharges' in df.columns:
    # Find non-numeric
    non_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce').isnull() & df['TotalCharges'].notna()
    if non_numeric.sum() > 0:
        print(f"  TotalCharges has {non_numeric.sum()} non-numeric values")
        print(f"  Examples: {df.loc[non_numeric, 'TotalCharges'].head().tolist()}")

# Tenure distribution
if 'tenure' in df.columns:
    print(f"\n  Tenure range: {df['tenure'].min()} - {df['tenure'].max()} months")
    print(f"  Tenure distribution (by chunks):")
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12m', '12-24m', '24-36m', '36-48m', '48-60m', '60-72m']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    for group in labels:
        subset = df[df['Tenure_Group'] == group]
        count = len(subset)
        churn_rate = (subset['Churn'] == 'Yes').mean() * 100
        print(f"    {group}: {count:>4d} users, churn rate={churn_rate:.1f}%")

# ========== 8. First Chart: Churn overview ==========
print("\n=== 8. Generating Charts ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Churn pie
churn_counts = [churn_dist.get('No', 0), churn_dist.get('Yes', 0)]
axes[0, 0].pie(churn_counts, labels=['Stay', 'Churn'], autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.05))
axes[0, 0].set_title('Overall Churn Rate', fontsize=14)

# Chart 2: Contract type vs Churn
contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=True)
bars = axes[0, 1].bar(contract_churn.index, contract_churn.values,
                      color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='white')
axes[0, 1].set_ylabel('Churn Rate (%)')
axes[0, 1].set_title('Churn Rate by Contract Type', fontsize=14)
for bar, val in zip(bars, contract_churn.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Chart 3: Tenure vs Churn
tenure_churn = df.groupby('Tenure_Group')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
)
axes[1, 0].plot(range(len(tenure_churn)), tenure_churn.values, 'o-',
                color='#e74c3c', linewidth=2, markersize=10)
axes[1, 0].set_xticks(range(len(tenure_churn)))
axes[1, 0].set_xticklabels(tenure_churn.index, rotation=45)
axes[1, 0].set_ylabel('Churn Rate (%)')
axes[1, 0].set_title('Churn Rate by Tenure (months)', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# Chart 4: Top factors (horizontal bar)
factor_churn = {}
for col in df.columns:
    if df[col].dtype == object and col not in ['customerID', 'Churn']:
        for val in df[col].unique():
            subset = df[df[col] == val]
            churn_rate = (subset['Churn'] == 'Yes').mean() * 100
            factor_churn[f'{col}: {val}'] = (churn_rate, len(subset))

# Get top 10 highest churn segments (min 30 users)
high_churn = [(k, v[0], v[1]) for k, v in factor_churn.items() if v[1] >= 30]
high_churn = sorted(high_churn, key=lambda x: x[1], reverse=True)[:10]

labels_top = [x[0] for x in high_churn]
values_top = [x[1] for x in high_churn]
bars = axes[1, 1].barh(range(len(labels_top)), values_top, color='#e74c3c')
axes[1, 1].set_yticks(range(len(labels_top)))
axes[1, 1].set_yticklabels(labels_top, fontsize=9)
axes[1, 1].set_xlabel('Churn Rate (%)')
axes[1, 1].set_title('Top 10 Highest Churn Segments\n(min 30 users)', fontsize=14)
axes[1, 1].axvline(x=churn_pct['Yes'], color='#3498db', linestyle='--', label=f'Avg: {churn_pct["Yes"]:.1f}%')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '01_churn_overview.png'), dpi=150)
plt.close()
print(f"  Chart saved: 01_churn_overview.png")

print("\n=== Step 1 Complete ===")
