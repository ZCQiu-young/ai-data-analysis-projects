# -*- coding: utf-8 -*-
"""
Project 5: Olist 巴西电商 — Step 6 关联规则 + 品类延迟引爆分析
高敏问题: 哪些品类延迟后差评率最高？有没有"延迟引爆器"品类？
"""
import sqlite3
import pandas as pd
import numpy as np
import os, sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\AI_Dateannaly\PROJECTS\project_olist_analysis\data\olist.db"

def main():
    conn = sqlite3.connect(DB)

    # === 宽表 ===
    sql = """
    SELECT
        o.order_id,
        CASE WHEN CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) > 0
            THEN 1 ELSE 0 END AS is_delayed,
        CASE WHEN AVG(r.review_score) <= 2 THEN 1 ELSE 0 END AS is_bad,
        COALESCE(ct.product_category_name_english, 'Unknown') AS category,
        SUM(oi.price) AS gmv,
        SUM(oi.freight_value) AS freight,
        COUNT(*) AS items
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN reviews r ON o.order_id = r.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_trans ct ON p.product_category_name = ct.product_category_name
    WHERE o.order_status = 'delivered'
        AND o.order_delivered_customer_date IS NOT NULL
        AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY o.order_id
    """
    df = pd.read_sql(sql, conn)
    conn.close()

    print(f"[数据] 有效订单: {len(df):,}")
    print(f"       品类数: {df['category'].nunique()}")
    print(f"       整体延迟率: {df['is_delayed'].mean()*100:.1f}%")
    print(f"       整体差评率: {df['is_bad'].mean()*100:.1f}%")

    # ===================================================================
    print("\n" + "=" * 70)
    print("R1: 各品类延迟率 & 差评率排名")
    print("=" * 70)

    cat_stats = df.groupby('category').agg(
        orders=('order_id', 'count'),
        delay_rate=('is_delayed', 'mean'),
        bad_rate=('is_bad', 'mean'),
        avg_gmv=('gmv', 'mean'),
    ).reset_index()
    cat_stats['delay_rate'] = (cat_stats['delay_rate'] * 100).round(1)
    cat_stats['bad_rate'] = (cat_stats['bad_rate'] * 100).round(1)
    cat_stats = cat_stats[cat_stats['orders'] >= 100]

    overall_delay = df['is_delayed'].mean()
    overall_bad = df['is_bad'].mean()

    print(f"\n{'Category':30s} {'Orders':>7s} {'延迟率':>7s} {'差评率':>7s} {'vs整体延迟':>10s} {'vs整体差评':>10s}")
    print("-" * 80)
    for _, r in cat_stats.sort_values('delay_rate', ascending=False).head(10).iterrows():
        d_flag = '↑' if r['delay_rate'] > overall_delay*100 else '↓'
        b_flag = '↑' if r['bad_rate'] > overall_bad*100 else '↓'
        print(f"  {r['category']:30s} {r['orders']:>7,} {r['delay_rate']:>6.1f}% {r['bad_rate']:>6.1f}% {d_flag}{abs(r['delay_rate']-overall_delay*100):.1f}pp {' ' + b_flag}{abs(r['bad_rate']-overall_bad*100):.1f}pp")

    # ===================================================================
    print("\n" + "=" * 70)
    print("R2: 🔥 '延迟引爆器' — 各品类延迟后的差评增幅 (Lift)")
    print("=" * 70)

    cat_lift = []
    for cat in cat_stats['category'].unique():
        sub = df[df['category'] == cat]
        if len(sub) < 200:
            continue
        bad_ontime = sub[sub['is_delayed'] == 0]['is_bad'].mean()
        bad_delayed = sub[sub['is_delayed'] == 1]['is_bad'].mean()
        delay_rate = sub['is_delayed'].mean()
        
        if delay_rate > 0.03 and len(sub[sub['is_delayed']==1]) >= 10:
            lift = bad_delayed / bad_ontime if bad_ontime > 0 else float('inf')
            delta = bad_delayed - bad_ontime
            cat_lift.append({
                'category': cat,
                'orders': len(sub),
                'delay_rate': delay_rate * 100,
                'bad_ontime': bad_ontime * 100,
                'bad_delayed': bad_delayed * 100,
                'delta_pp': delta * 100,
                'lift': lift,
            })

    lift_df = pd.DataFrame(cat_lift).sort_values('delta_pp', ascending=False)

    print(f"\n{'Category':30s} {'Orders':>7s} {'延迟率':>7s} {'准时差评':>8s} {'延迟差评':>8s} {'增幅':>7s} {'Lift':>7s}")
    print("-" * 85)
    for _, r in lift_df.head(15).iterrows():
        bar = '█' * int(r['delta_pp'] / 3)
        print(f"  {r['category']:30s} {r['orders']:>7,.0f} {r['delay_rate']:>6.1f}% {r['bad_ontime']:>7.1f}% {r['bad_delayed']:>7.1f}% +{r['delta_pp']:>5.1f}pp {r['lift']:>5.1f}x {bar}")
    
    print(f"\n  平均基线: 准时差评率 = {df[df['is_delayed']==0]['is_bad'].mean()*100:.1f}% | 延迟差评率 = {df[df['is_delayed']==1]['is_bad'].mean()*100:.1f}%")

    # ===================================================================
    print("\n" + "=" * 70)
    print("R3: 高延迟×高差评增幅 的 '高危品类' 交叉排名")
    print("=" * 70)

    cat_risk = []
    for cat in cat_stats['category'].unique():
        sub = df[df['category'] == cat]
        if len(sub) < 200:
            continue
        d_r = sub['is_delayed'].mean()
        b_r = sub['is_bad'].mean()
        bad_d = sub[sub['is_delayed']==1]['is_bad'].mean()
        bad_o = sub[sub['is_delayed']==0]['is_bad'].mean()
        delta = bad_d - bad_o if len(sub[sub['is_delayed']==1]) >= 10 else 0
        
        # 综合风险分 = 延迟率 × 延迟差评增幅 (归一化)
        risk_score = d_r * (bad_d - bad_o) * 100
        cat_risk.append({
            'category': cat, 'orders': len(sub),
            'delay_rate': d_r*100, 'bad_delayed': bad_d*100,
            'bad_ontime': bad_o*100, 'delta_pp': delta*100,
            'risk_score': risk_score
        })

    risk_df = pd.DataFrame(cat_risk).sort_values('risk_score', ascending=False).head(12)
    print(f"\n{'Category':30s} {'延迟率':>7s} {'准时差评':>8s} {'延迟差评':>8s} {'增幅':>7s} {'风险分':>7s}")
    print("-" * 75)
    for _, r in risk_df.iterrows():
        bar = '⚠️' * min(int(r['risk_score']*10), 5)
        print(f"  {r['category']:30s} {r['delay_rate']:>6.1f}% {r['bad_ontime']:>7.1f}% {r['bad_delayed']:>7.1f}% +{r['delta_pp']:>5.1f}pp {r['risk_score']:>6.3f} {bar}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("R4: 单品延迟 + 差评数 ≫ 最大损失品类（差评总量贡献）")
    print("=" * 70)

    cat_volume = df.groupby('category').agg(
        orders=('order_id', 'count'),
        delayed_orders=('is_delayed', 'sum'),
        bad_orders=('is_bad', 'sum'),
        delayed_bad=('is_bad', lambda x: (x[df['is_delayed']==1]).sum()),
        ontime_bad=('is_bad', lambda x: (x[df['is_delayed']==0]).sum()),
        total_gmv=('gmv', 'sum'),
    ).reset_index()

    cat_volume['bad_from_delayed_pct'] = (cat_volume['delayed_bad'] / cat_volume['bad_orders'] * 100).fillna(0)
    cat_volume = cat_volume[cat_volume['orders'] >= 200]
    cat_volume = cat_volume.sort_values('delayed_bad', ascending=False)

    print(f"\n{'Category':30s} {'订单':>7s} {'差评总数':>8s} {'延迟差评':>8s} {'准时差评':>8s} {'延迟差评占比':>10s}")
    print("-" * 85)
    for _, r in cat_volume.head(12).iterrows():
        pct = r['bad_from_delayed_pct']
        bar = '=' * int(pct / 5) if pct > 0 else ''
        print(f"  {r['category']:30s} {r['orders']:>7,.0f} {r['bad_orders']:>8,.0f} {r['delayed_bad']:>8,.0f} {r['ontime_bad']:>8,.0f} {pct:>9.1f}% {bar}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("R5: 🎯 最终建议 — 三类品类的不同策略")
    print("=" * 70)

    # 综合风险分 + 差评总量
    risk_df['share_of_bad'] = risk_df['category'].map(
        cat_volume.set_index('category')['bad_orders']
    ).fillna(0)

    high_risk_high_volume = risk_df[
        (risk_df['risk_score'] > risk_df['risk_score'].median()) & 
        (risk_df['share_of_bad'] > cat_volume['bad_orders'].median())
    ].sort_values('risk_score', ascending=False)

    high_risk_low_volume = risk_df[
        (risk_df['risk_score'] > risk_df['risk_score'].median()) & 
        (risk_df['share_of_bad'] <= cat_volume['bad_orders'].median())
    ]

    high_volume_low_risk = risk_df[
        (risk_df['risk_score'] <= risk_df['risk_score'].median()) & 
        (risk_df['share_of_bad'] > cat_volume['bad_orders'].median())
    ]

    print("\n  [A] 高风险 × 高差评量 —— 优先整治品类（延迟引爆+量大）")
    for _, r in high_risk_high_volume.iterrows():
        print(f"      {r['category']:30s} 延迟率{r['delay_rate']:.1f}% 差评增幅+{r['delta_pp']:.1f}pp 差评量{r['share_of_bad']:.0f}")

    print("\n  [B] 高风险 × 低差评量 —— 重点监控品类（延迟引爆但量小）")
    for _, r in high_risk_low_volume.head(5).iterrows():
        print(f"      {r['category']:30s} 延迟率{r['delay_rate']:.1f}% 差评增幅+{r['delta_pp']:.1f}pp 差评量{r['share_of_bad']:.0f}")

    print("\n  [C] 低风险 × 高差评量 —— 产品优化品类（准时到货也差评）")
    for _, r in high_volume_low_risk.head(5).iterrows():
        print(f"      {r['category']:30s} 延迟率{r['delay_rate']:.1f}% 差评增幅+{r['delta_pp']:.1f}pp 差评量{r['share_of_bad']:.0f}")

    print("\n[完成] 关联规则/品类引爆分析完成")


if __name__ == '__main__':
    main()
