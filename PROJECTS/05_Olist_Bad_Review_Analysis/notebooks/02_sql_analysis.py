# -*- coding: utf-8 -*-
"""
Project 5: Olist 巴西电商 — Step 2 SQL 多表分析
核心问题: 差评到底跟什么有关？
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

DATA = r"D:\AI_Dateannaly\PROJECTS\project_olist_analysis\data"
DB = r"D:\AI_Dateannaly\PROJECTS\project_olist_analysis\data\olist.db"

CSV_MAP = {
    "orders":        "olist_orders_dataset.csv",
    "order_items":   "olist_order_items_dataset.csv",
    "products":      "olist_products_dataset.csv",
    "payments":      "olist_order_payments_dataset.csv",
    "reviews":       "olist_order_reviews_dataset.csv",
    "customers":     "olist_customers_dataset.csv",
    "sellers":       "olist_sellers_dataset.csv",
    "geolocation":   "olist_geolocation_dataset.csv",
    "category_trans":"product_category_name_translation.csv",
}

def load_all(conn):
    """加载所有CSV到SQLite"""
    for table, fname in CSV_MAP.items():
        fp = os.path.join(DATA, fname)
        df = pd.read_csv(fp)
        df.to_sql(table, conn, if_exists='replace', index=False)
        print(f"[OK] {table}: {df.shape[0]:,} rows x {df.shape[1]} cols")

def run():
    conn = sqlite3.connect(DB)
    load_all(conn)

    print("\n" + "=" * 70)
    print("Q1: 延迟与差评 —— 晚到的真的评价更低吗？")
    print("=" * 70)

    q1 = """
    WITH order_delays AS (
        SELECT
            o.order_id,
            CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) AS delay_days,
            o.order_status
        FROM orders o
        WHERE o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL
            AND o.order_estimated_delivery_date IS NOT NULL
    ),
    order_reviews AS (
        SELECT
            order_id,
            AVG(review_score) AS avg_score
        FROM reviews
        GROUP BY order_id
    )
    SELECT
        CASE
            WHEN d.delay_days <= -10 THEN '>>提前10天+'
            WHEN d.delay_days < -3  THEN '>提前3-10天'
            WHEN d.delay_days <= -1 THEN '>提前1-3天'
            WHEN d.delay_days = 0   THEN '=准时'
            WHEN d.delay_days <= 3   THEN '<延迟1-3天'
            WHEN d.delay_days <= 10  THEN '<延迟4-10天'
            ELSE '<延迟>10天'
        END AS delay_bucket,
        COUNT(*) AS order_cnt,
        ROUND(AVG(r.avg_score), 2) AS avg_review_score,
        ROUND(SUM(CASE WHEN r.avg_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_rate_pct
    FROM order_delays d
    JOIN order_reviews r ON d.order_id = r.order_id
    GROUP BY delay_bucket
    ORDER BY MIN(d.delay_days)
    """
    df = pd.read_sql(q1, conn)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Q2: 运费占比与差评 —— 运费贵 ≠ 更容易差评？")
    print("=" * 70)

    q2 = """
    WITH order_freight AS (
        SELECT
            oi.order_id,
            SUM(oi.freight_value) AS total_freight,
            SUM(oi.price) AS total_price,
            SUM(oi.freight_value) * 100.0 / NULLIF(SUM(oi.price), 0) AS freight_ratio
        FROM order_items oi
        GROUP BY oi.order_id
    ),
    order_reviews AS (
        SELECT order_id, AVG(review_score) AS avg_score FROM reviews GROUP BY order_id
    )
    SELECT
        CASE
            WHEN f.freight_ratio = 0 THEN '0%'
            WHEN f.freight_ratio <= 5 THEN '0-5%'
            WHEN f.freight_ratio <= 10 THEN '5-10%'
            WHEN f.freight_ratio <= 20 THEN '10-20%'
            WHEN f.freight_ratio <= 30 THEN '20-30%'
            ELSE '30%+'
        END AS freight_ratio_bucket,
        COUNT(*) AS orders,
        ROUND(AVG(r.avg_score), 2) AS avg_score,
        ROUND(SUM(CASE WHEN r.avg_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_rate_pct,
        ROUND(AVG(f.total_price), 0) AS avg_order_value
    FROM order_freight f
    JOIN order_reviews r ON f.order_id = r.order_id
    GROUP BY freight_ratio_bucket
    ORDER BY MIN(f.freight_ratio)
    """
    df = pd.read_sql(q2, conn)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Q3: 品类 × 差评率 —— 哪些品类是差评重灾区？")
    print("=" * 70)

    q3 = """
    WITH order_category AS (
        SELECT
            oi.order_id,
            ct.product_category_name_english AS category
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN category_trans ct ON p.product_category_name = ct.product_category_name
    ),
    order_reviews AS (
        SELECT order_id, AVG(review_score) AS avg_score FROM reviews GROUP BY order_id
    )
    SELECT
        COALESCE(oc.category, 'Unknown') AS category,
        COUNT(DISTINCT oc.order_id) AS orders,
        ROUND(AVG(r.avg_score), 2) AS avg_score,
        ROUND(SUM(CASE WHEN r.avg_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_rate_pct
    FROM order_category oc
    JOIN order_reviews r ON oc.order_id = r.order_id
    GROUP BY oc.category
    HAVING orders >= 100
    ORDER BY bad_rate_pct DESC
    LIMIT 15
    """
    df = pd.read_sql(q3, conn)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Q4: 支付方式 × 差评 —— 分期数会影响评分吗？")
    print("=" * 70)

    q4 = """
    WITH order_pay AS (
        SELECT
            order_id,
            MAX(payment_type) AS pay_type,
            MAX(payment_installments) AS installments
        FROM payments
        GROUP BY order_id
    ),
    order_reviews AS (
        SELECT order_id, AVG(review_score) AS avg_score FROM reviews GROUP BY order_id
    )
    SELECT
        op.pay_type,
        CASE
            WHEN op.installments = 1 THEN '1期'
            WHEN op.installments <= 3 THEN '2-3期'
            WHEN op.installments <= 6 THEN '4-6期'
            WHEN op.installments <= 10 THEN '7-10期'
            ELSE '10期+'
        END AS installment_bucket,
        COUNT(*) AS orders,
        ROUND(AVG(r.avg_score), 2) AS avg_score,
        ROUND(SUM(CASE WHEN r.avg_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_rate_pct
    FROM order_pay op
    JOIN order_reviews r ON op.order_id = r.order_id
    WHERE op.pay_type IN ('credit_card', 'boleto')
    GROUP BY op.pay_type, installment_bucket
    ORDER BY op.pay_type, MIN(op.installments)
    """
    df = pd.read_sql(q4, conn)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Q5: 延迟 × 运费 × 品类 —— 多变量交叉：谁才是真凶？")
    print("=" * 70)

    q5 = """
    WITH o AS (
        SELECT
            oi.order_id,
            CASE WHEN CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) > 0
                 THEN 1 ELSE 0 END AS is_delayed,
            ROUND(AVG(r.review_score), 2) AS avg_score,
            CASE WHEN AVG(r.review_score) <= 2 THEN 1 ELSE 0 END AS is_bad,
            SUM(oi.freight_value) * 100.0 / NULLIF(SUM(oi.price), 0) AS freight_ratio,
            SUM(oi.price) + SUM(oi.freight_value) AS order_gmv
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN reviews r ON oi.order_id = r.order_id
        WHERE o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL
        GROUP BY oi.order_id
    )
    SELECT
        'All Orders' AS segment,
        COUNT(*) AS n,
        ROUND(AVG(avg_score), 2) AS avg_score,
        ROUND(SUM(is_bad) * 100.0 / COUNT(*), 1) AS bad_rate,
        ROUND(AVG(is_delayed) * 100, 1) AS delay_rate,
        ROUND(AVG(freight_ratio), 1) AS avg_freight_ratio
    FROM o
    UNION ALL
    SELECT
        'Delayed' AS segment,
        COUNT(*) AS n,
        ROUND(AVG(avg_score), 2) AS avg_score,
        ROUND(SUM(is_bad) * 100.0 / COUNT(*), 1) AS bad_rate,
        ROUND(AVG(is_delayed) * 100, 1) AS delay_rate,
        ROUND(AVG(freight_ratio), 1) AS avg_freight_ratio
    FROM o WHERE is_delayed = 1
    UNION ALL
    SELECT
        'Not Delayed' AS segment,
        COUNT(*) AS n,
        ROUND(AVG(avg_score), 2) AS avg_score,
        ROUND(SUM(is_bad) * 100.0 / COUNT(*), 1) AS bad_rate,
        ROUND(AVG(is_delayed) * 100, 1) AS delay_rate,
        ROUND(AVG(freight_ratio), 1) AS avg_freight_ratio
    FROM o WHERE is_delayed = 0
    UNION ALL
    SELECT
        'High Freight(>20%)' AS segment,
        COUNT(*) AS n,
        ROUND(AVG(avg_score), 2) AS avg_score,
        ROUND(SUM(is_bad) * 100.0 / COUNT(*), 1) AS bad_rate,
        ROUND(AVG(is_delayed) * 100, 1) AS delay_rate,
        ROUND(AVG(freight_ratio), 1) AS avg_freight_ratio
    FROM o WHERE freight_ratio > 20
    """
    df = pd.read_sql(q5, conn)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Q6: 差评订单的延迟天数分布 —— 差评 vs 好评，延迟差多少？")
    print("=" * 70)

    q6 = """
    WITH od AS (
        SELECT
            o.order_id,
            CAST(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS INTEGER) AS delay_days
        FROM orders o
        WHERE o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL
    ),
    review_agg AS (
        SELECT order_id, AVG(review_score) AS avg_r FROM reviews GROUP BY order_id
    )
    SELECT
        CASE WHEN ra.avg_r <= 2 THEN 'Bad(1-2分)' WHEN ra.avg_r >= 4 THEN 'Good(4-5分)' ELSE 'Neutral(3分)' END AS review_group,
        COUNT(*) AS orders,
        ROUND(AVG(od.delay_days), 1) AS avg_delay_days,
        ROUND(AVG(CASE WHEN od.delay_days > 0 THEN od.delay_days ELSE NULL END), 1) AS avg_delay_when_late,
        ROUND(SUM(CASE WHEN od.delay_days > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS delay_rate_pct
    FROM od
    JOIN review_agg ra ON od.order_id = ra.order_id
    GROUP BY review_group
    ORDER BY MIN(ra.avg_r)
    """
    df = pd.read_sql(q6, conn)
    print(df.to_string(index=False))

    conn.close()
    print(f"\n[完成] SQL分析结果已输出")

if __name__ == '__main__':
    run()
