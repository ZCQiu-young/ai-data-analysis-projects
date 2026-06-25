-- ============================================================
-- 项目4：电商全链路销售分析 — SQL 查询集
-- 数据表：orders（63,645 条订单，2024-01 至 2026-05）
-- 用途：面试展示 SQL 能力 + 业务分析思维
-- ============================================================

-- --------------------------------------------------
-- Q1: 月度核心指标仪表盘（KPI Dashboard）
-- 场景：CEO 每月看的经营数据总览
-- --------------------------------------------------
SELECT 
    order_year,
    order_month,
    COUNT(DISTINCT order_id)   AS total_orders,
    COUNT(DISTINCT customer_id) AS active_customers,
    ROUND(SUM(total_amount), 2) AS total_gmv,
    ROUND(SUM(profit), 2)       AS total_profit,
    ROUND(AVG(total_amount), 2) AS avg_order_value,
    ROUND(SUM(profit) / NULLIF(SUM(total_amount), 0) * 100, 1) AS profit_margin_pct,
    ROUND(SUM(CASE WHEN is_returned = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS return_rate_pct
FROM orders
WHERE is_returned = 0  -- 只看有效订单
GROUP BY order_year, order_month
ORDER BY order_year, order_month;


-- --------------------------------------------------
-- Q2: 品类 ABC 分析（帕累托分析）
-- 场景：识别核心品类，优化资源配置
-- --------------------------------------------------
WITH category_gmv AS (
    SELECT 
        category,
        SUM(total_amount) AS gmv,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE is_returned = 0
    GROUP BY category
),
ranked AS (
    SELECT 
        *,
        ROUND(gmv / SUM(gmv) OVER() * 100, 1) AS gmv_pct,
        ROUND(SUM(gmv) OVER(ORDER BY gmv DESC) / SUM(gmv) OVER() * 100, 1) AS cumulative_pct
    FROM category_gmv
)
SELECT 
    category,
    gmv,
    orders,
    gmv_pct,
    cumulative_pct,
    CASE 
        WHEN cumulative_pct <= 80 THEN 'A类-核心'
        WHEN cumulative_pct <= 95 THEN 'B类-成长'
        ELSE 'C类-长尾'
    END AS abc_class
FROM ranked
ORDER BY gmv DESC;


-- --------------------------------------------------
-- Q3: 区域销售排名 + 同比增长
-- 场景：区域经理月度复盘，看自己区域表现
-- --------------------------------------------------
WITH region_monthly AS (
    SELECT 
        province,
        order_year,
        order_month,
        SUM(total_amount) AS monthly_gmv,
        COUNT(DISTINCT customer_id) AS customers
    FROM orders
    WHERE is_returned = 0
    GROUP BY province, order_year, order_month
),
yoy_compare AS (
    SELECT 
        cur.province,
        cur.order_year,
        cur.order_month,
        cur.monthly_gmv,
        cur.customers,
        prev.monthly_gmv AS prev_year_gmv,
        ROUND((cur.monthly_gmv - prev.monthly_gmv) / NULLIF(prev.monthly_gmv, 0) * 100, 1) AS yoy_gmv_growth_pct
    FROM region_monthly cur
    LEFT JOIN region_monthly prev 
        ON cur.province = prev.province 
        AND cur.order_year = prev.order_year + 1
        AND cur.order_month = prev.order_month
)
SELECT * FROM yoy_compare
ORDER BY province, order_year, order_month;


-- --------------------------------------------------
-- Q4: 渠道转化效率分析
-- 场景：市场部评估各渠道的投入产出比
-- --------------------------------------------------
SELECT 
    channel,
    COUNT(DISTINCT order_id)   AS orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    ROUND(SUM(total_amount), 2) AS total_gmv,
    ROUND(SUM(profit), 2)       AS total_profit,
    ROUND(AVG(total_amount), 2) AS avg_order_value,
    ROUND(SUM(profit) / NULLIF(SUM(total_amount), 0) * 100, 1) AS margin_pct,
    -- 每个渠道各品类GMV占比（子查询形式展示 Multi-Table Join）
    ROUND(SUM(CASE WHEN category = '手机数码' THEN total_amount ELSE 0 END) / NULLIF(SUM(total_amount), 0) * 100, 1) AS digital_pct
FROM orders
WHERE is_returned = 0
GROUP BY channel
ORDER BY total_gmv DESC;


-- --------------------------------------------------
-- Q5: 用户生命周期分析 — 月度复购率
-- 场景：衡量用户粘性，复购率是高价值业务的北极星指标
-- --------------------------------------------------
WITH customer_monthly AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', order_date) AS order_month
    FROM orders
    WHERE is_returned = 0
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
),
customer_retention AS (
    SELECT 
        a.customer_id,
        a.order_month AS first_month,
        b.order_month AS return_month
    FROM customer_monthly a
    LEFT JOIN customer_monthly b 
        ON a.customer_id = b.customer_id 
        AND b.order_month > a.order_month
),
retention_stats AS (
    SELECT 
        first_month,
        COUNT(DISTINCT customer_id) AS new_customers,
        COUNT(DISTINCT CASE WHEN return_month IS NOT NULL THEN customer_id END) AS returned_customers
    FROM customer_retention
    GROUP BY first_month
)
SELECT 
    first_month,
    new_customers,
    returned_customers,
    ROUND(returned_customers * 100.0 / NULLIF(new_customers, 0), 1) AS repurchase_rate_pct
FROM retention_stats
ORDER BY first_month;


-- --------------------------------------------------
-- Q6: 用户分层 — RFM 模型（SQL版本）
-- 场景：CRM 团队需要分层名单做精准营销
-- 使用 2026-05-31 作为分析基准日
-- --------------------------------------------------
WITH rfm_raw AS (
    SELECT 
        customer_id,
        DATEDIFF('day', MAX(order_date), '2026-05-31') AS recency,
        COUNT(DISTINCT order_id)    AS frequency,
        ROUND(SUM(total_amount), 2) AS monetary
    FROM orders
    WHERE is_returned = 0
    GROUP BY customer_id
),
rfm_scored AS (
    SELECT 
        customer_id,
        recency,
        frequency,
        monetary,
        NTILE(4) OVER (ORDER BY recency DESC)   AS r_score,  -- 越小越好
        NTILE(4) OVER (ORDER BY frequency ASC)  AS f_score,  -- 越大越好
        NTILE(4) OVER (ORDER BY monetary ASC)   AS m_score   -- 越大越好
    FROM rfm_raw
)
SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    r_score, f_score, m_score,
    CONCAT(r_score, f_score, m_score) AS rfm_cell,
    CASE 
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'VIP'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'At Risk/Lost'
        WHEN r_score <= 2 THEN 'Dormant'
        ELSE 'Standard'
    END AS segment
FROM rfm_scored
ORDER BY monetary DESC;


-- --------------------------------------------------
-- Q7: 促销效率分析 — 打折打出了什么？
-- 场景：运营想知道打折真的值吗？
-- --------------------------------------------------
SELECT 
    CASE 
        WHEN discount_pct = 0 THEN '无折扣'
        WHEN discount_pct <= 0.1 THEN '微折(0-10%)'
        WHEN discount_pct <= 0.2 THEN '小折(10-20%)'
        WHEN discount_pct <= 0.3 THEN '中折(20-30%)'
        ELSE '深折(>30%)'
    END AS discount_bucket,
    COUNT(DISTINCT order_id)   AS orders,
    ROUND(SUM(total_amount), 2) AS total_gmv,
    ROUND(AVG(total_amount), 2) AS avg_gmv_per_order,
    ROUND(SUM(profit), 2)       AS total_profit,
    ROUND(SUM(profit) / NULLIF(SUM(total_amount), 0) * 100, 1) AS margin_pct,
    ROUND(SUM(profit) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS profit_per_order
FROM orders
WHERE is_returned = 0
GROUP BY discount_bucket
ORDER BY MIN(discount_pct);


-- --------------------------------------------------
-- Q8: 产品盈利矩阵 — 四象限分析
-- 场景：品类经理决定哪些产品该推、哪些该砍
-- --------------------------------------------------
WITH product_stats AS (
    SELECT 
        product_id,
        product_name,
        category,
        COUNT(DISTINCT order_id)   AS orders,
        SUM(quantity)               AS units_sold,
        ROUND(SUM(total_amount), 2) AS total_gmv,
        ROUND(AVG(profit / NULLIF(total_amount, 0)) * 100, 2) AS avg_margin_pct
    FROM orders
    WHERE is_returned = 0
    GROUP BY product_id, product_name, category
),
avg_gmv AS (
    SELECT AVG(total_gmv) AS avg_gmv_val FROM product_stats
),
avg_margin AS (
    SELECT AVG(avg_margin_pct) AS avg_margin_val FROM product_stats
)
SELECT 
    ps.product_name,
    ps.category,
    ps.total_gmv,
    ps.avg_margin_pct,
    CASE 
        WHEN ps.total_gmv >= ag.avg_gmv_val AND ps.avg_margin_pct >= am.avg_margin_val THEN '现金牛'
        WHEN ps.total_gmv <  ag.avg_gmv_val AND ps.avg_margin_pct >= am.avg_margin_val THEN '明星产品'
        WHEN ps.total_gmv <  ag.avg_gmv_val AND ps.avg_margin_pct <  am.avg_margin_val THEN '问题产品'
        ELSE '瘦狗产品'
    END AS quadrant
FROM product_stats ps, avg_gmv ag, avg_margin am
ORDER BY ps.total_gmv DESC;


-- --------------------------------------------------
-- Q9: 用户留存漏斗 — 首购后第N月留存
-- 场景：评估新用户的长期价值
-- --------------------------------------------------
WITH first_purchase AS (
    SELECT 
        customer_id,
        MIN(order_date) AS first_order_date,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    WHERE is_returned = 0
    GROUP BY customer_id
),
cohort_orders AS (
    SELECT 
        fp.customer_id,
        fp.cohort_month,
        o.order_date,
        DATEDIFF('month', fp.first_order_date, o.order_date) AS month_number
    FROM first_purchase fp
    JOIN orders o ON fp.customer_id = o.customer_id AND o.is_returned = 0
)
SELECT 
    cohort_month,
    COUNT(DISTINCT customer_id) AS cohort_size,
    ROUND(COUNT(DISTINCT CASE WHEN month_number = 1 THEN customer_id END) * 100.0 / COUNT(DISTINCT customer_id), 1) AS m1_retention,
    ROUND(COUNT(DISTINCT CASE WHEN month_number = 2 THEN customer_id END) * 100.0 / COUNT(DISTINCT customer_id), 1) AS m2_retention,
    ROUND(COUNT(DISTINCT CASE WHEN month_number = 3 THEN customer_id END) * 100.0 / COUNT(DISTINCT customer_id), 1) AS m3_retention,
    ROUND(COUNT(DISTINCT CASE WHEN month_number = 6 THEN customer_id END) * 100.0 / COUNT(DISTINCT customer_id), 1) AS m6_retention
FROM cohort_orders
GROUP BY cohort_month
ORDER BY cohort_month;


-- --------------------------------------------------
-- Q10: 异常检测 — 同比波动超过2个标准差
-- 场景：自动预警，找出需要关注的异常月份
-- --------------------------------------------------
WITH monthly_stats AS (
    SELECT 
        order_year,
        order_month,
        SUM(total_amount) AS monthly_gmv
    FROM orders
    WHERE is_returned = 0
    GROUP BY order_year, order_month
),
stats AS (
    SELECT 
        AVG(monthly_gmv) AS mean_gmv,
        STDDEV(monthly_gmv) AS std_gmv
    FROM monthly_stats
)
SELECT 
    ms.order_year,
    ms.order_month,
    ms.monthly_gmv,
    ROUND((ms.monthly_gmv - st.mean_gmv) / NULLIF(st.std_gmv, 0), 2) AS z_score,
    CASE 
        WHEN ABS((ms.monthly_gmv - st.mean_gmv) / NULLIF(st.std_gmv, 0)) > 2 
        THEN '异常-需关注'
        ELSE '正常'
    END AS alert
FROM monthly_stats ms, stats st
ORDER BY ms.order_year, ms.order_month;