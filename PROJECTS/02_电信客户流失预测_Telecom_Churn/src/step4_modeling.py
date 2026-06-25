import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'D:\AI_Dateannaly\PROJECTS\project_churn_prediction'
DATA_DIR = os.path.join(BASE, 'data')
CHARTS_DIR = os.path.join(BASE, 'charts')

print("=== Step 4: Churn Prediction Modeling ===\n")

# Load
df = pd.read_csv(os.path.join(DATA_DIR, 'telco_churn.csv'))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# ========== 1. Feature Engineering ==========
print("=== 1. Feature Engineering ===")

df_model = df.copy()

# Binary encoding for Yes/No columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_model[col] = (df_model[col] == 'Yes').astype(int)

# Service columns
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'MultipleLines']
for col in service_cols:
    df_model[col] = df_model[col].apply(lambda x: 1 if x == 'Yes' else 
                                        (0 if x == 'No' else -1))

# Create composite features
df_model['Has_Security'] = (
    (df_model['OnlineSecurity'] == 1) | (df_model['TechSupport'] == 1)
).astype(int)

df_model['Service_Count'] = (
    (df_model['PhoneService'] == 1).astype(int) +
    (df_model['OnlineSecurity'] == 1).astype(int) +
    (df_model['OnlineBackup'] == 1).astype(int) +
    (df_model['DeviceProtection'] == 1).astype(int) +
    (df_model['TechSupport'] == 1).astype(int) +
    (df_model['StreamingTV'] == 1).astype(int) +
    (df_model['StreamingMovies'] == 1).astype(int) +
    (df_model['MultipleLines'] == 1).astype(int)
)

# Charges ratio
df_model['Charges_Ratio'] = df_model['MonthlyCharges'] / (df_model['TotalCharges'] + 1)

# One-hot encode multi-category columns
cat_encode = ['Contract', 'InternetService', 'PaymentMethod', 'gender']
df_encoded = pd.get_dummies(df_model, columns=cat_encode, drop_first=True)

# Target
y = (df_encoded['Churn'] == 'Yes').astype(int)

# Feature matrix
drop_cols = ['customerID', 'Churn', 'Tenure_Group', 'Charge_Bracket', 
             'TotalCharges']  # drop TotalCharges (collinear with tenure+monthly)
X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])

# Handle any remaining object columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

print(f"  Features: {X.shape[1]}")
print(f"  Samples: {X.shape[0]}")
print(f"  Churn rate: {y.mean()*100:.1f}%")

# ========== 2. Train/Test Split ==========
print("\n=== 2. Train/Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]}, Churn={y_train.mean()*100:.1f}%")
print(f"  Test: {X_test.shape[0]}, Churn={y_test.mean()*100:.1f}%")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 3. Logistic Regression ==========
print("\n=== 3. Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Stay', 'Churn']))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"  Confusion Matrix:")
print(f"    TN={cm_lr[0,0]}, FP={cm_lr[0,1]}")
print(f"    FN={cm_lr[1,0]}, TP={cm_lr[1,1]}")

auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"  ROC-AUC: {auc_lr:.4f}")

# Cross-validation
cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"  5-fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_[0]
}).sort_values('Coefficient', ascending=False)

print(f"\n  Top 10 Positive Factors (increase churn risk):")
for i, row in feature_importance.head(10).iterrows():
    print(f"    {row['Feature']:<35s}: {row['Coefficient']:+.4f}")

print(f"\n  Top 10 Protective Factors (decrease churn risk):")
for i, row in feature_importance.tail(10).iterrows():
    print(f"    {row['Feature']:<35s}: {row['Coefficient']:+.4f}")

# ========== 4. Random Forest ==========
print("\n=== 4. Random Forest (for comparison) ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced',
                            max_depth=10, min_samples_leaf=10)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"  ROC-AUC: {auc_rf:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Stay', 'Churn']))

# RF Feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  Top 10 RF Features:")
for i, row in rf_importance.head(10).iterrows():
    print(f"    {row['Feature']:<35s}: {row['Importance']:.4f}")

# ========== 5. Risk Scoring ==========
print("\n=== 5. Risk Scoring for All Customers ===")

# Score all customers
X_all_scaled = scaler.transform(X)
df['Churn_Probability'] = lr.predict_proba(X_all_scaled)[:, 1]

# Risk tiers
def risk_tier(p):
    if p >= 0.7:
        return 'Critical'
    elif p >= 0.5:
        return 'High'
    elif p >= 0.3:
        return 'Medium'
    else:
        return 'Low'

df['Risk_Tier'] = df['Churn_Probability'].apply(risk_tier)

print(f"\n  Risk tier distribution:")
for tier in ['Critical', 'High', 'Medium', 'Low']:
    count = (df['Risk_Tier'] == tier).sum()
    churn_in_tier = (df[df['Risk_Tier'] == tier]['Churn'] == 'Yes').sum()
    pct = count / len(df) * 100
    print(f"    {tier:<10s}: {count:>4d} ({pct:.1f}%), actual churn: {churn_in_tier}")

# Top risk profile
critical = df[df['Risk_Tier'] == 'Critical']
print(f"\n  Critical risk profile:")
print(f"    Count: {len(critical)}")
print(f"    Avg tenure: {critical['tenure'].mean():.1f} months")
print(f"    Avg monthly: ${critical['MonthlyCharges'].mean():.2f}")
print(f"    Month-to-month: {(critical['Contract']=='Month-to-month').mean()*100:.1f}%")
print(f"    Fiber optic: {(critical['InternetService']=='Fiber optic').mean()*100:.1f}%")
print(f"    E-check: {(critical['PaymentMethod']=='Electronic check').mean()*100:.1f}%")

# ========== 6. Charts ==========
print("\n=== 6. Charts ===")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'LR (AUC={auc_lr:.3f})')
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
axes[0, 0].plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'RF (AUC={auc_rf:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Confusion Matrix (LR)
from matplotlib.colors import LinearSegmentedColormap
cmap = plt.cm.Blues
axes[0, 1].imshow(cm_lr, interpolation='nearest', cmap=cmap)
for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, str(cm_lr[i, j]), ha='center', va='center', 
                       fontsize=16, fontweight='bold',
                       color='white' if cm_lr[i, j] > cm_lr.max()/2 else 'black')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(['Pred Stay', 'Pred Churn'])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_yticklabels(['Actual Stay', 'Actual Churn'])
axes[0, 1].set_title(f'Confusion Matrix (Logistic Regression)', fontsize=13)

# Top features (LR)
top_n = 15
top_features = feature_importance.head(top_n)
axes[0, 2].barh(range(top_n), top_features['Coefficient'].values[::-1],
                color=['#e74c3c' if c > 0 else '#2ecc71' for c in top_features['Coefficient'].values[::-1]])
axes[0, 2].set_yticks(range(top_n))
axes[0, 2].set_yticklabels(top_features['Feature'].values[::-1], fontsize=9)
axes[0, 2].set_xlabel('Coefficient (+ = increases churn risk)')
axes[0, 2].set_title('Top Churn Drivers (Logistic Regression)', fontsize=13)
axes[0, 2].axvline(x=0, color='black', linewidth=1)

# Risk distribution
risk_counts = [len(df[df['Risk_Tier'] == t]) for t in ['Low', 'Medium', 'High', 'Critical']]
risk_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
axes[1, 0].pie(risk_counts, labels=['Low', 'Medium', 'High', 'Critical'],
               autopct='%1.0f%%', colors=risk_colors, startangle=90)
axes[1, 0].set_title('Customer Risk Distribution', fontsize=13)

# Churn probability histogram
axes[1, 1].hist(df[df['Churn']=='Yes']['Churn_Probability'], bins=30, alpha=0.7, 
                color='#e74c3c', label='Churned')
axes[1, 1].hist(df[df['Churn']=='No']['Churn_Probability'], bins=30, alpha=0.7,
                color='#2ecc71', label='Stayed')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Decision boundary')
axes[1, 1].set_xlabel('Predicted Churn Probability')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].set_title('Churn Probability Distribution', fontsize=13)
axes[1, 1].legend()

# RF Feature importance
top_rf = rf_importance.head(15)
axes[1, 2].barh(range(top_n), top_rf['Importance'].values[::-1], color='#3498db')
axes[1, 2].set_yticks(range(top_n))
axes[1, 2].set_yticklabels(top_rf['Feature'].values[::-1], fontsize=9)
axes[1, 2].set_xlabel('Feature Importance')
axes[1, 2].set_title('Top Features (Random Forest)', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '04_churn_prediction_model.png'), dpi=150)
plt.close()

# Save risk scores
df[['customerID', 'Churn', 'Churn_Probability', 'Risk_Tier']].to_csv(
    os.path.join(DATA_DIR, 'churn_risk_scores.csv'), index=False, encoding='utf-8-sig')
print(f"  Risk scores saved: churn_risk_scores.csv")

# Model comparison
print(f"\n=== Model Comparison ===")
print(f"  Logistic Regression AUC: {auc_lr:.4f}")
print(f"  Random Forest AUC: {auc_rf:.4f}")
print(f"\n  Logistic Regression is preferred for:")
print(f"    - Interpretability (you can explain WHY)")
print(f"    - Business deployment (simple, fast)")
print(f"    - Similar performance to RF")

print("\n=== Step 4 Complete ===")
