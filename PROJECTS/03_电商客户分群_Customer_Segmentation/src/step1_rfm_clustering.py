import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'D:\AI_Dateannaly\PROJECTS\project_customer_segmentation'
DATA_DIR = os.path.join(BASE, 'data')
CHARTS_DIR = os.path.join(BASE, 'charts')

print("=== Project 3: Customer Segmentation (RFM + K-Means) ===\n")

# ========== 1. Load & Calculate RFM ==========
print("=== 1. RFM Calculation ===")
df = pd.read_csv(os.path.join(DATA_DIR, 'ecommerce_transactions.csv'), parse_dates=['InvoiceDate'])

reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'UnitPrice': lambda x: (df.loc[x.index, 'Quantity'] * df.loc[x.index, 'UnitPrice']).sum()
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Quantity': 'Total_Items',
    'UnitPrice': 'Monetary'
})

rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']

print(f"Customers: {len(rfm)}")
print(f"\nRFM Summary:")
print(f"  Recency: {rfm['Recency'].min()}-{rfm['Recency'].max()} days (lower = more recent)")
print(f"  Frequency: {rfm['Frequency'].min()}-{rfm['Frequency'].max()} orders")
print(f"  Monetary: ${rfm['Monetary'].min():.0f}-${rfm['Monetary'].max():.0f}")

# ========== 2. Data Preprocessing ==========
print("\n=== 2. Data Preprocessing ===")

rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].copy()
rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

print("Transformation: log(1+x) applied to all RFM features")
print(f"  Skewness before: R={rfm['Recency'].skew():.2f}, F={rfm['Frequency'].skew():.2f}, M={rfm['Monetary'].skew():.2f}")
print(f"  Skewness after:  R={rfm_log['Recency'].skew():.2f}, F={rfm_log['Frequency'].skew():.2f}, M={rfm_log['Monetary'].skew():.2f}")

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm.index)

print("Standardization applied (mean=0, std=1)")

# ========== 3. Find Optimal K ==========
print("\n=== 3. Finding Optimal K ===")

K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, labels))

best_k_sil = K_range[np.argmax(silhouettes)]
# Business context: minimum 4 segments for actionable strategies
best_k = max(best_k_sil, 4)
print(f"\n  Elbow & Silhouette analysis (k=2..10):")
for k, inert, sil in zip(K_range, inertias, silhouettes):
    marker = " <- SIL_BEST" if k == best_k_sil else ""
    chosen = " <- CHOSEN" if k == best_k else ""
    print(f"    k={k:2d}: Inertia={inert:8.0f}, Silhouette={sil:.4f}{marker}{chosen}")

print(f"\n  Silhouette best: k={best_k_sil} ({max(silhouettes):.4f})")
print(f"  Chosen for business: k={best_k} (minimum 4 segments required)")

# ========== 4. Fit K-Means ==========
print(f"\n=== 4. Fitting K-Means (k={best_k}) ===")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

cluster_profile = rfm.groupby('Cluster').agg(
    Recency_Mean=('Recency', 'mean'),
    Recency_Median=('Recency', 'median'),
    Frequency_Mean=('Frequency', 'mean'),
    Frequency_Median=('Frequency', 'median'),
    Monetary_Mean=('Monetary', 'mean'),
    Monetary_Median=('Monetary', 'median'),
    Avg_Order_Value=('Avg_Order_Value', 'mean'),
    Count=('Recency', 'count')
).round(1)
cluster_profile = cluster_profile.sort_values('Monetary_Mean', ascending=False)

print(f"\n  Cluster Profiles (k={best_k}, sorted by revenue):")
for cluster_id, row in cluster_profile.iterrows():
    pct = row['Count'] / len(rfm) * 100
    total_rev = rfm[rfm['Cluster'] == cluster_id]['Monetary'].sum()
    pct_rev = total_rev / rfm['Monetary'].sum() * 100
    print(f"\n  Cluster {cluster_id} ({int(row['Count'])} customers, {pct:.1f}%, Revenue=${total_rev:,.0f}, {pct_rev:.1f}%):")
    print(f"    Recency:   mean={row['Recency_Mean']:.0f}d, median={row['Recency_Median']:.0f}d")
    print(f"    Frequency: mean={row['Frequency_Mean']:.1f} orders, median={row['Frequency_Median']:.0f}")
    print(f"    Monetary:  mean=${row['Monetary_Mean']:,.0f}, median=${row['Monetary_Median']:,.0f}")
    print(f"    Avg Order: ${row['Avg_Order_Value']:,.0f}")

# Auto-name clusters
total = len(rfm)
rec_q = rfm.groupby('Cluster')['Recency'].mean()
freq_q = rfm.groupby('Cluster')['Frequency'].mean()
mon_q = rfm.groupby('Cluster')['Monetary'].mean()

cluster_names = {}
sorted_clusters = sorted(cluster_profile.index.tolist())
for cid in sorted_clusters:
    r, f, m = rec_q[cid], freq_q[cid], mon_q[cid]
    r_med, f_med, m_med = np.median(rec_q.values), np.median(freq_q.values), np.median(mon_q.values)
    
    if m > m_med * 1.5 and f > f_med * 1.5:
        cluster_names[cid] = 'VIP'
    elif r < r_med * 0.7 and f > f_med:
        cluster_names[cid] = 'Loyal Regular'
    elif r < r_med and m > m_med:
        cluster_names[cid] = 'High Value'
    elif r > r_med * 1.5 and m < m_med:
        cluster_names[cid] = 'At Risk / Lost'
    elif f <= 1:
        cluster_names[cid] = 'One-Time Buyer'
    elif r > r_med:
        cluster_names[cid] = 'Dormant'
    else:
        cluster_names[cid] = 'Standard'

rfm['Cluster_Name'] = rfm['Cluster'].map(cluster_names)

# Also add CustomerID back as a column for grouping
rfm_reset = rfm.reset_index()

# ========== 5. Charts ==========
print("\n=== 5. Generating Charts ===")

fig = plt.figure(figsize=(18, 12))

# Chart 1: Elbow + Silhouette
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (WCSS)')
ax1.set_title('Elbow Method', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(list(K_range), silhouettes, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
ax2.legend()

# Chart 2: 3D scatter
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for i in range(best_k):
    cluster_data = rfm_scaled_df[rfm['Cluster'] == i]
    ax3.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'],
                c=[colors[i]], label=f'{cluster_names.get(i, f"Cl{i}")}', alpha=0.6, s=30)
ax3.set_xlabel('Recency (scaled)')
ax3.set_ylabel('Frequency (scaled)')
ax3.set_zlabel('Monetary (scaled)')
ax3.set_title('Customer Segments (3D)', fontsize=13)
ax3.legend(fontsize=8, loc='upper left')

# Chart 3: Cluster sizes
cluster_sizes = rfm_reset['Cluster_Name'].value_counts()
colors_bar = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
ax4 = fig.add_subplot(2, 3, 4)
bars = ax4.barh(range(len(cluster_sizes)), cluster_sizes.values, color=colors_bar)
ax4.set_yticks(range(len(cluster_sizes)))
ax4.set_yticklabels(cluster_sizes.index, fontsize=10)
ax4.set_xlabel('Number of Customers')
ax4.set_title('Customer Distribution by Segment', fontsize=13)
for i, (bar, val) in enumerate(zip(bars, cluster_sizes.values)):
    ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f'{val} ({val/len(rfm)*100:.0f}%)', va='center', fontsize=10)

# Chart 4: Revenue by cluster
cluster_rev = rfm_reset.groupby('Cluster_Name').agg(
    Total_Rev=('Monetary', 'sum'),
    Average_Rev=('Monetary', 'mean'),
    Count=('CustomerID', 'count')
).sort_values('Total_Rev', ascending=True)

ax5 = fig.add_subplot(2, 3, 5)
rev_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(cluster_rev)))
bars = ax5.barh(range(len(cluster_rev)), cluster_rev['Total_Rev'].values / 1000, color=rev_colors)
ax5.set_yticks(range(len(cluster_rev)))
ax5.set_yticklabels(cluster_rev.index, fontsize=10)
ax5.set_xlabel('Total Revenue ($K)')
ax5.set_title('Revenue Contribution by Segment', fontsize=13)
for i, (bar, rev, avg, cnt) in enumerate(zip(bars, cluster_rev['Total_Rev'].values, 
                                                cluster_rev['Average_Rev'].values,
                                                cluster_rev['Count'].values)):
    ax5.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'${rev/1000:.0f}K (avg ${avg:.0f}/cust)', va='center', fontsize=9)

# Chart 5: RFM Radar
ax6 = fig.add_subplot(2, 3, 6)
rfm_radar = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
})
rfm_radar['Recency_Score'] = 1 - (rfm_radar['Recency'] / rfm_radar['Recency'].max())
rfm_radar['Freq_Score'] = rfm_radar['Frequency'] / rfm_radar['Frequency'].max()
rfm_radar['Mon_Score'] = rfm_radar['Monetary'] / rfm_radar['Monetary'].max()

categories = ['Recency\n(more recent)', 'Frequency\n(more orders)', 'Monetary\n(higher value)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (cid, row) in enumerate(rfm_radar.iterrows()):
    values = [row['Recency_Score'], row['Freq_Score'], row['Mon_Score']]
    values += values[:1]
    ax6.plot(angles, values, 'o-', linewidth=2, label=cluster_names.get(cid, f'Cl{cid}'),
             color=colors[i])
    ax6.fill(angles, values, alpha=0.1, color=colors[i])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontsize=9)
ax6.set_title('Segment RFM Profiles', fontsize=13)
ax6.legend(fontsize=8, loc='lower right')
ax6.set_ylim(0, 1.1)

plt.suptitle(f'Customer Segmentation Analysis (k={best_k})', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, '01_customer_segmentation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Chart saved: 01_customer_segmentation.png")

# ========== 6. Strategy Recommendations ==========
print(f"\n=== 6. Strategy Recommendations ===")

for cname in cluster_sizes.index:
    subset = rfm_reset[rfm_reset['Cluster_Name'] == cname]
    cnt = len(subset)
    avg_rec = subset['Recency'].mean()
    avg_freq = subset['Frequency'].mean()
    avg_mon = subset['Monetary'].mean()
    total_rev = subset['Monetary'].sum()
    
    if cname == 'VIP':
        strategy = "VIP dedicated service + early access to new products + double points"
        action = "Monthly 1-on-1 callback, invite to beta test"
    elif cname == 'Loyal Regular':
        strategy = "Membership upgrade + cross-category recommendation + subscription discount"
        action = "Push bundle deals to increase order value"
    elif cname == 'High Value':
        strategy = "Personalized recommendations + flash sale + repurchase incentive"
        action = "Push related new products based on history, 10% off first order"
    elif cname == 'Standard':
        strategy = "Regular EDM + category education + first-purchase coupons"
        action = "Weekly newsletter + newcomer exclusive coupons"
    elif cname == 'Dormant':
        strategy = "Reactivation discount + promo campaign + exit survey"
        action = "Send $10 comeback coupon + limited-time flash sale link"
    elif cname == 'At Risk / Lost':
        strategy = "Strong recall + large discount + exit interview"
        action = "Send $20 coupon + we miss you email + churn reason survey"
    elif cname == 'One-Time Buyer':
        strategy = "Post-purchase follow-up + category education + conversion to repeat"
        action = "Send recommendation 3 days after purchase, coupon 7 days after"
    else:
        strategy = "General maintenance + email marketing"
        action = "Regular EDM push"
    
    print(f"\n  [{cname}] {cnt} customers ({cnt/len(rfm)*100:.0f}%)")
    print(f"    Profile: R={avg_rec:.0f}d, F={avg_freq:.1f} orders, M=${avg_mon:.0f}, Total Rev=${total_rev:,.0f}")
    print(f"    Strategy: {strategy}")
    print(f"    Action: {action}")

# ========== 7. Save Results ==========
rfm_reset.to_csv(os.path.join(DATA_DIR, 'customer_segments.csv'), index=False, encoding='utf-8-sig')
print(f"\n  Segment results saved: customer_segments.csv")

print(f"\n=== Step 1 Complete ===")
