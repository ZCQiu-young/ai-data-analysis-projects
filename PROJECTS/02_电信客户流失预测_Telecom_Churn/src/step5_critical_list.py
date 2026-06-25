import pandas as pd
import numpy as np
import os

BASE = r'D:\AI_Dateannaly\PROJECTS\project_churn_prediction'
DATA_DIR = os.path.join(BASE, 'data')

print("=== Step 5: Critical Risk Customer List ===\n")

# Load prediction results
df_pred = pd.read_csv(os.path.join(DATA_DIR, 'churn_predictions_optimized.csv'))
df = pd.read_csv(os.path.join(DATA_DIR, 'telco_churn.csv'))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Merge predictions with original data
df_merged = df.merge(df_pred[['customerID', 'Churn_Probability_Opt', 'Risk_Tier_Opt']], 
                      on='customerID', how='left')

# Filter critical risk
critical = df_merged[df_merged['Risk_Tier_Opt'] == 'Critical'].copy()

# Sort by churn probability (most at risk first)
critical = critical.sort_values('Churn_Probability_Opt', ascending=False)

print(f"Critical risk customers: {len(critical)}")
print(f"  Churn probability range: {critical['Churn_Probability_Opt'].min():.2%} - {critical['Churn_Probability_Opt'].max():.2%}")
print(f"  Revenue at risk (TotalCharges): ${critical['TotalCharges'].sum():,.0f}")
print(f"  Average tenure: {critical['tenure'].mean():.1f} months")

# Priority levels within critical
critical['Priority'] = pd.cut(
    critical['Churn_Probability_Opt'],
    bins=[0.70, 0.80, 0.90, 1.01],
    labels=['P3-High', 'P2-Very High', 'P1-Extreme']
)

print(f"\n  Priority breakdown:")
for p in ['P1-Extreme', 'P2-Very High', 'P3-High']:
    subset = critical[critical['Priority'] == p]
    if len(subset) > 0:
        print(f"    {p}: {len(subset)} customers, avg prob={subset['Churn_Probability_Opt'].mean():.2%}")

# Create actionable columns
critical_out = critical[[
    'customerID', 'Priority', 'Churn_Probability_Opt',
    'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
    'InternetService', 'PaymentMethod', 
    'OnlineSecurity', 'TechSupport', 'OnlineBackup',
    'StreamingTV', 'StreamingMovies',
    'SeniorCitizen', 'Partner', 'Dependents',
    'gender', 'PaperlessBilling'
]].copy()

critical_out.columns = [
    '客户ID', '优先级', '流失概率',
    '在网月数', '合约类型', '月费', '历史总消费',
    '上网服务', '支付方式',
    '在线安全', '技术支持', '在线备份',
    '流媒体电视', '流媒体电影',
    '老年人', '有伴侣', '有子女',
    '性别', '电子账单'
]

# Convert probabilities to percentages
critical_out['流失概率'] = (critical_out['流失概率'] * 100).round(1).astype(str) + '%'

# Save
output_path = os.path.join(DATA_DIR, 'critical_risk_customers.csv')
critical_out.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n  Saved: critical_risk_customers.csv")

# Summary stats for operations team
print(f"\n=== Operations Summary ===")
print(f"\n  Contact Priority:")
print(f"    P1 (Extreme, >90% probability):  {len(critical[critical['Priority']=='P1-Extreme'])} customers - CALL TODAY")
print(f"    P2 (Very High, 80-90%):          {len(critical[critical['Priority']=='P2-Very High'])} customers - CALL THIS WEEK")
print(f"    P3 (High, 70-80%):               {len(critical[critical['Priority']=='P3-High'])} customers - EMAIL CAMPAIGN")

print(f"\n  Key Segments in Critical List:")
print(f"    Month-to-month:     {(critical['Contract']=='Month-to-month').sum()} ({(critical['Contract']=='Month-to-month').mean()*100:.0f}%)")
print(f"    Fiber optic:        {(critical['InternetService']=='Fiber optic').sum()} ({(critical['InternetService']=='Fiber optic').mean()*100:.0f}%)")
print(f"    Electronic check:   {(critical['PaymentMethod']=='Electronic check').sum()} ({(critical['PaymentMethod']=='Electronic check').mean()*100:.0f}%)")
print(f"    No security:        {(critical['OnlineSecurity']!='Yes').sum()} ({(critical['OnlineSecurity']!='Yes').mean()*100:.0f}%)")
print(f"    No tech support:    {(critical['TechSupport']!='Yes').sum()} ({(critical['TechSupport']!='Yes').mean()*100:.0f}%)")

print(f"\n  Suggested Actions per Segment:")
mtm_fiber = critical[(critical['Contract']=='Month-to-month') & (critical['InternetService']=='Fiber optic')]
print(f"    > {len(mtm_fiber)} Fiber+MTM -> Offer 1-year contract + free security bundle")
no_sec = critical[(critical['OnlineSecurity']!='Yes') & (critical['TechSupport']!='Yes')]
print(f"    > {len(no_sec)} No Security -> Free 3-month OnlineSecurity + TechSupport trial")
ec_no = critical[critical['PaymentMethod']=='Electronic check']
print(f"    > {len(ec_no)} E-check -> Incentive to switch to auto-pay")

print(f"\n  Revenue at stake: ${critical['TotalCharges'].sum():,.0f}")
print(f"  Avg per customer: ${critical['TotalCharges'].mean():,.0f}")

print("\n=== Step 5 Complete ===")
