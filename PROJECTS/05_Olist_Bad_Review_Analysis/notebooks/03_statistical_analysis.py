# -*- coding: utf-8 -*-
"""
Project 5: Olist 巴西电商 — Step 3 统计检验 + 逻辑回归
从"看起来相关"升级为"统计显著 + 效应量化"
"""
import sqlite3
import pandas as pd
import numpy as np
import os, sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DB = r"D:\AI_Dateannaly\PROJECTS\project_olist_analysis\data\olist.db"

def build_dataset(conn):
    """构建完整的分析用宽表"""
    sql = """
    SELECT
        o.order_id,
        CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) AS delay_days,
        CASE WHEN CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) > 0
            THEN 1 ELSE 0 END AS is_delayed,
        CASE WHEN CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) > 4
            THEN 1 ELSE 0 END AS is_severely_delayed,
        ROUND(AVG(r.review_score), 2) AS avg_score,
        CASE WHEN AVG(r.review_score) <= 2 THEN 1 ELSE 0 END AS is_bad,
        SUM(oi.price) AS total_price,
        SUM(oi.freight_value) AS total_freight,
        SUM(oi.freight_value) * 100.0 / NULLIF(SUM(oi.price), 0) AS freight_ratio,
        COUNT(DISTINCT oi.product_id) AS item_count,
        MAX(pay.payment_type) AS pay_type,
        MAX(pay.payment_installments) AS installments,
        ct.product_category_name_english AS main_category
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN reviews r ON o.order_id = r.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_trans ct ON p.product_category_name = ct.product_category_name
    LEFT JOIN (
        SELECT order_id, payment_type, payment_installments,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY payment_value DESC) AS rn
        FROM payments
    ) pay ON o.order_id = pay.order_id AND pay.rn = 1
    WHERE o.order_status = 'delivered'
        AND o.order_delivered_customer_date IS NOT NULL
        AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY o.order_id
    """
    return pd.read_sql(sql, conn)


def chi2_test(df, col, target='is_bad'):
    """卡方独立性检验"""
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    cramer = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape) - 1)))
    return chi2, p, dof, cramer


def main():
    conn = sqlite3.connect(DB)
    df = build_dataset(conn)
    conn.close()
    print(f"[数据] 有效订单: {len(df):,} 行 x {df.shape[1]} 列")
    print(f"       差评率: {df['is_bad'].mean() * 100:.1f}%")
    print(f"       延迟率: {df['is_delayed'].mean() * 100:.1f}%")
    print(f"       严重延迟率(>4天): {df['is_severely_delayed'].mean() * 100:.1f}%")

    # ===================================================================
    print("\n" + "=" * 70)
    print("T1: 卡方检验 —— 延迟 vs 差评（独立性检验）")
    print("=" * 70)

    chi2, p, dof, cramer = chi2_test(df, 'is_delayed')
    print(f"  chi2 = {chi2:.1f}, p = {p:.2e}, dof = {dof}, Cramer's V = {cramer:.3f}")
    print(f"  H0: 延迟与差评独立  →  p={p:.2e} {'<< 0.001, 拒绝H0 ✅' if p < 0.001 else '不能拒绝H0'}")

    chi2s, ps, dofs, cramers = chi2_test(df, 'is_severely_delayed')
    print(f"\n  严重延迟(>4天) vs 差评:")
    print(f"  chi2 = {chi2s:.1f}, p = {ps:.2e}, Cramer's V = {cramers:.3f}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("T2: 分层分析 —— 控制运费后，延迟效应是否仍显著？")
    print("=" * 70)

    for freight_bucket, label in [(15, 'Low Freight(<=15%)'), (15, 'High Freight(>15%)')]:
        subset = df[df['freight_ratio'] <= freight_bucket] if 'Low' in label else df[df['freight_ratio'] > freight_bucket]
        bad_delayed = subset[subset['is_delayed'] == 1]['is_bad'].mean()
        bad_ontime = subset[subset['is_delayed'] == 0]['is_bad'].mean()
        print(f"  {label:25s} 延迟差评率={bad_delayed*100:.1f}%  准时差评率={bad_ontime*100:.1f}%  差值={abs(bad_delayed-bad_ontime)*100:.1f}pp")

    # ===================================================================
    print("\n" + "=" * 70)
    print("T3: 逻辑回归 —— 量化各因素对差评的贡献（OR值）")
    print("=" * 70)

    # 准备特征
    feats = df[['delay_days', 'is_delayed', 'total_price', 'freight_ratio',
                 'item_count', 'installments']].copy()
    feats['pay_type_credit'] = (df['pay_type'] == 'credit_card').astype(int)

    # 处理缺失值
    feats = feats.fillna(0)
    feats['installments'] = feats['installments'].fillna(1)
    y = df['is_bad']

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feats)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feats.columns)

    # 拟合
    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_scaled_df, y)

    # Odds Ratio
    coef_df = pd.DataFrame({
        'Feature': feats.columns,
        'Coef': lr.coef_[0],
        'OR': np.exp(lr.coef_[0]),
        'OR_interpretation': [
            f'延迟每+1天, 差评风险×{np.exp(lr.coef_[0][0]):.2f}',
            f'延迟(vs准时), 差评风险×{np.exp(lr.coef_[0][1]):.2f}',
            f'客单价每+1std, 差评风险×{np.exp(lr.coef_[0][2]):.2f}',
            f'运费占比每+1std, 差评风险×{np.exp(lr.coef_[0][3]):.2f}',
            f'商品数每+1std, 差评风险×{np.exp(lr.coef_[0][4]):.2f}',
            f'分期每+1std, 差评风险×{np.exp(lr.coef_[0][5]):.2f}',
            f'信用卡(vs其他), 差评风险×{np.exp(lr.coef_[0][6]):.2f}',
        ]
    })
    coef_df = coef_df.sort_values('OR', ascending=False)
    print(coef_df[['Feature', 'OR', 'OR_interpretation']].to_string(index=False))

    # ===================================================================
    print("\n" + "=" * 70)
    print("T4: 延迟天数分组的相对风险 (Relative Risk)")
    print("=" * 70)

    df['delay_group'] = pd.cut(df['delay_days'],
        bins=[-200, -10, -3, 0, 3, 10, 200],
        labels=['提前10天+', '提前3-10天', '准时/提前1-3天', '延迟1-3天', '延迟4-10天', '延迟>10天'])

    baseline = df[df['delay_group'] == '准时/提前1-3天']['is_bad'].mean()
    print(f"  Baseline (准时): 差评率 = {baseline*100:.1f}%")
    print()
    for grp in df['delay_group'].cat.categories:
        rate = df[df['delay_group'] == grp]['is_bad'].mean()
        rr = rate / baseline if baseline > 0 else float('inf')
        n = len(df[df['delay_group'] == grp])
        print(f"  {grp:20s} n={n:>6,}  差评率={rate*100:5.1f}%  RR={rr:5.1f}x  {'⚠️' if rr > 3 else '✅' if rr <= 1.2 else '  '}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("T5: 逻辑回归 —— 仅用延迟天数预测差评（AUC）")
    print("=" * 70)

    from sklearn.metrics import roc_auc_score, classification_report
    X_simple = df[['delay_days']].values
    lr_simple = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    lr_simple.fit(X_simple, y)
    y_prob = lr_simple.predict_proba(X_simple)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"  AUC (仅延迟天数): {auc:.3f}")
    print(f"  解释：仅凭延迟天数一个变量，就能区分差评/非差评的AUC={auc:.3f}")
    print(f"  对比：项目2电信流失预测AUC=0.848用了更多特征")

    # ===================================================================
    print("\n" + "=" * 70)
    print("T6: 延迟 vs 差评的阈值分析 —— 几天是临界点？")
    print("=" * 70)

    for threshold in [1, 2, 3, 4, 5, 7, 10]:
        delayed = df[df['delay_days'] > threshold]
        ontime = df[df['delay_days'] <= threshold]
        if len(delayed) > 100:
            bad_d = delayed['is_bad'].mean()
            bad_o = ontime['is_bad'].mean()
            lift = bad_d / bad_o if bad_o > 0 else float('inf')
            print(f"  延迟>{threshold:2d}天: 差评率={bad_d*100:5.1f}% vs 准时={bad_o*100:5.1f}%  Lift={lift:.1f}x  n={len(delayed):,}")

    print("\n[完成] 统计分析全部完成")


if __name__ == '__main__':
    main()
