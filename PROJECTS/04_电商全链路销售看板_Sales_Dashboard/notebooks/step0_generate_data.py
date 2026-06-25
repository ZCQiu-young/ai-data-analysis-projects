"""
电商全链路销售数据生成
生成 55,000 条订单数据，用于销售分析 + Tableau 可视化
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ========== 产品主数据 ==========
categories = {
    "手机数码": {"sub": ["手机", "平板", "耳机", "智能手表", "充电器"], "cost_range": (300, 6000), "margin": (0.08, 0.20)},
    "服装鞋包": {"sub": ["男装", "女装", "童装", "鞋靴", "箱包"], "cost_range": (40, 800), "margin": (0.35, 0.55)},
    "食品生鲜": {"sub": ["零食", "饮料", "生鲜水果", "粮油调味", "乳制品"], "cost_range": (5, 200), "margin": (0.15, 0.30)},
    "家用电器": {"sub": ["冰箱", "洗衣机", "空调", "小家电", "厨房电器"], "cost_range": (100, 4000), "margin": (0.10, 0.22)},
    "美妆个护": {"sub": ["护肤品", "彩妆", "洗发护发", "口腔护理", "香水"], "cost_range": (15, 600), "margin": (0.40, 0.65)},
    "运动户外": {"sub": ["运动鞋", "运动服", "健身器材", "户外装备", "骑行"], "cost_range": (30, 1500), "margin": (0.25, 0.45)},
}

products = []
pid = 1
for cat, info in categories.items():
    for sub in info["sub"]:
        for _ in range(6):  # 每个子类6个SKU
            cost = round(random.uniform(*info["cost_range"]), 2)
            price = round(cost * (1 + random.uniform(*info["margin"])), 2)
            products.append({
                "product_id": f"P{pid:04d}",
                "product_name": f"{sub}{random.choice(['Plus','Pro','Lite','Max','Air','S'])}-{random.choice(['A','B','C','X','Y'])}{random.randint(1,9)}",
                "category": cat,
                "sub_category": sub,
                "cost_price": cost,
                "list_price": price,
            })
            pid += 1
products_df = pd.DataFrame(products)
# 共 6*5*6 = 180 个产品

# ========== 城市数据 ==========
city_data = [
    ("北京", "华北", 1.3), ("天津", "华北", 0.5), ("石家庄", "华北", 0.3), ("太原", "华北", 0.25),
    ("上海", "华东", 1.4), ("杭州", "华东", 0.8), ("南京", "华东", 0.55), ("苏州", "华东", 0.45), ("宁波", "华东", 0.3),
    ("广州", "华南", 1.1), ("深圳", "华南", 1.0), ("东莞", "华南", 0.4), ("厦门", "华南", 0.3),
    ("成都", "西南", 0.7), ("重庆", "西南", 0.55), ("昆明", "西南", 0.3), ("贵阳", "西南", 0.2),
    ("武汉", "华中", 0.6), ("长沙", "华中", 0.4), ("郑州", "华中", 0.35), ("合肥", "华中", 0.25),
    ("西安", "西北", 0.35), ("兰州", "西北", 0.15), ("乌鲁木齐", "西北", 0.12),
    ("沈阳", "东北", 0.3), ("哈尔滨", "东北", 0.2), ("大连", "东北", 0.2),
]
city_weights = [w for _, _, w in city_data]

channels = ["APP", "微信小程序", "PC网页", "线下门店"]
channel_weights = [0.35, 0.30, 0.20, 0.15]
payment_methods = ["微信支付", "支付宝", "银行卡", "货到付款"]
pay_weights = [0.45, 0.35, 0.12, 0.08]

# ========== 客户池 ==========
n_customers = 3000
customers = []
for i in range(n_customers):
    city, region, _ = random.choices(city_data, weights=city_weights, k=1)[0]
    reg_date = datetime(2023, random.randint(1, 12), random.randint(1, 28))
    customers.append({
        "customer_id": f"C{i+1:05d}",
        "city": city,
        "province": region,
        "registration_date": reg_date,
        "acquisition_channel": random.choices(["搜索", "社交推荐", "广告投放", "地推"], weights=[0.3, 0.35, 0.25, 0.1], k=1)[0],
    })
customers_df = pd.DataFrame(customers)

# ========== 生成订单 ==========
def get_seasonal_factor(month):
    """模拟季节性因子"""
    base = {
        1: 0.75, 2: 0.70, 3: 0.95, 4: 1.00,
        5: 1.10, 6: 1.25, 7: 1.08, 8: 1.05,
        9: 1.02, 10: 1.10, 11: 1.50, 12: 1.20
    }
    return base[month]

def get_promo_factor(date):
    """促销活动因子"""
    m, d = date.month, date.day
    # 618
    if m == 6 and 1 <= d <= 20: return random.uniform(1.1, 1.4)
    # 双11
    if m == 11 and 1 <= d <= 15: return random.uniform(1.3, 1.7)
    # 双12
    if m == 12 and 10 <= d <= 15: return random.uniform(1.1, 1.3)
    # 年货节
    if m == 1 and 15 <= d <= 30: return random.uniform(1.1, 1.3)
    # 38节
    if m == 3 and 3 <= d <= 10: return random.uniform(1.05, 1.2)
    # 99
    if m == 9 and 5 <= d <= 12: return random.uniform(1.1, 1.25)
    return 1.0

orders = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 5, 31)
current = start_date
day_idx = 0

daily_base = 55  # 基础日均订单

while current <= end_date:
    sf = get_seasonal_factor(current.month)
    pf = get_promo_factor(current)
    day_of_week = current.weekday()
    dow_factor = 1.0 if day_of_week < 5 else random.uniform(1.1, 1.25)  # 周末略高
    
    daily_count = int(daily_base * sf * pf * dow_factor * random.uniform(0.85, 1.15))
    # 加上3%的年增长趋势
    year_progress = (current - start_date).days / (end_date - start_date).days
    daily_count = int(daily_count * (1 + year_progress * 0.45))
    
    for _ in range(max(1, daily_count)):
        cust = customers_df.sample(1).iloc[0]
        prod = products_df.sample(1).iloc[0]
        qty = random.choices([1, 2, 3, 4, 5], weights=[0.55, 0.25, 0.12, 0.05, 0.03], k=1)[0]
        
        # 价格波动（季节/促销）
        base_price = prod["list_price"]
        discount_pct = 0
        if pf > 1.1:
            discount_pct = random.choices([0, 0.05, 0.10, 0.15, 0.20, 0.30], 
                                          weights=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05], k=1)[0]
        unit_price = round(base_price * (1 - discount_pct), 2)
        
        total_amount = round(unit_price * qty, 2)
        cost = round(prod["cost_price"] * qty, 2)
        profit = round(total_amount - cost, 2)
        
        # 退货率 ~5%
        is_returned = random.random() < 0.05
        
        orders.append({
            "order_id": f"ORD{202400000 + len(orders) + 1}",
            "order_date": current.strftime("%Y-%m-%d"),
            "order_year": current.year,
            "order_month": current.month,
            "order_quarter": (current.month - 1) // 3 + 1,
            "order_dayofweek": day_of_week,
            "order_hour": random.randint(8, 23),
            "customer_id": cust["customer_id"],
            "city": cust["city"],
            "province": cust["province"],
            "channel": random.choices(channels, weights=channel_weights, k=1)[0],
            "payment_method": random.choices(payment_methods, weights=pay_weights, k=1)[0],
            "product_id": prod["product_id"],
            "product_name": prod["product_name"],
            "category": prod["category"],
            "sub_category": prod["sub_category"],
            "quantity": qty,
            "list_price": base_price,
            "discount_pct": discount_pct,
            "unit_price": unit_price,
            "total_amount": total_amount if not is_returned else -total_amount,
            "cost_price": cost,
            "profit": profit if not is_returned else -cost,
            "is_promotion": 1 if pf > 1.1 else 0,
            "is_returned": 1 if is_returned else 0,
        })
    
    current += timedelta(days=1)

df = pd.DataFrame(orders)
print(f"共生成 {len(df)} 条订单")
print(f"有效订单（非退货）：{(df['is_returned']==0).sum()}")
print(f"日期范围：{df['order_date'].min()} ~ {df['order_date'].max()}")
print(f"总GMV：{df.loc[df['is_returned']==0, 'total_amount'].sum():,.0f}")
print(f"总利润：{df.loc[df['is_returned']==0, 'profit'].sum():,.0f}")

# 保存
out = "D:/AI_Dateannaly/PROJECTS/project_sales_dashboard/data/orders.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n数据已保存：{out}")
print(f"文件列：{list(df.columns)}")

# 生成客户汇总表（用于Tableau关联）
cust_summary = df[df['is_returned']==0].groupby('customer_id').agg(
    total_orders=('order_id', 'nunique'),
    total_spent=('total_amount', 'sum'),
    total_profit=('profit', 'sum'),
    first_order=('order_date', 'min'),
    last_order=('order_date', 'max'),
    fav_category=('category', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
    fav_channel=('channel', lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
).reset_index()
cust_summary = cust_summary.merge(customers_df[['customer_id', 'city', 'province', 'registration_date', 'acquisition_channel']], on='customer_id', how='left')
cust_out = "D:/AI_Dateannaly/PROJECTS/project_sales_dashboard/data/customers.csv"
cust_summary.to_csv(cust_out, index=False, encoding="utf-8-sig")
print(f"客户汇总已保存：{cust_out} ({len(cust_summary)} 条)")

# 保存产品表
prod_out = "D:/AI_Dateannaly/PROJECTS/project_sales_dashboard/data/products.csv"
products_df.to_csv(prod_out, index=False, encoding="utf-8-sig")
print(f"产品表已保存：{prod_out} ({len(products_df)} 条)")

# 快速概览
print("\n===== 数据概览 =====")
print(f"品类分布：\n{df[df['is_returned']==0].groupby('category')['total_amount'].agg(['sum','count']).sort_values('sum',ascending=False)}")
print(f"\n渠道分布：\n{df[df['is_returned']==0].groupby('channel')['total_amount'].sum()}")
print(f"\n地区分布：\n{df[df['is_returned']==0].groupby('province')['total_amount'].sum().sort_values(ascending=False)}")
