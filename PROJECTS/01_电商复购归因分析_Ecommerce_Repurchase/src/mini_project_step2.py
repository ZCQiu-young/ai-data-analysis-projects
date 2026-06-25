import pandas as pd
import numpy as np
from datetime import datetime

print("=== Step 2: 鏁版嵁娓呮礂 & EDA ===")

# 鍔犺浇鏁版嵁
df = pd.read_csv('D:/AI_Dateannaly/PROJECTS/project_ecommerce_analysis/data/mini_project_data.csv')
print(f'\n鍘熷鏁版嵁: {df.shape[0]} 琛?)

# 鏌ョ湅鍒楀悕
print('\n鍒楀悕:', df.columns.tolist())

# 1. 澶勭悊閫€璐э紙Quantity 涓鸿礋鏁扮殑琛岋級
print('\n=== 1. 澶勭悊閫€璐?===')
n_returns = (df['Quantity'] < 0).sum()
print(f'閫€璐ц褰曟暟: {n_returns} ({n_returns/len(df)*100:.1f}%)')

# 鏂规锛氬垹闄ら€€璐ц褰曪紙鎴栬€呬繚鐣欎絾鏍囪锛岃繖閲岄€夋嫨鍒犻櫎锛?df = df[df['Quantity'] > 0].copy()
print(f'鍒犻櫎閫€璐у悗: {df.shape[0]} 琛?)

# 2. 妫€鏌ョ己澶卞€?print('\n=== 2. 妫€鏌ョ己澶卞€?===')
print(df.isnull().sum())

# 3. 杞崲鏃ユ湡鏍煎紡
print('\n=== 3. 杞崲鏃ユ湡鏍煎紡 ===')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
df['Month'] = df['InvoiceDate'].dt.month

print('鏃ユ湡鑼冨洿:', df['InvoiceDate'].min(), '~', df['InvoiceDate'].max())
print('鏈堜唤鍒楄〃:', sorted(df['Month'].unique()))

# 4. 璁＄畻澶嶈喘鐜囷紙鎸夋湀浠斤級
print('\n=== 4. 璁＄畻鍚勬湀澶嶈喘鐜?===')

def calc_repurchase_rate(df, month_col='Month'):
    """
    璁＄畻澶嶈喘鐜囷細
    澶嶈喘鐢ㄦ埛 = 璐拱娆℃暟 >= 2 鐨勭敤鎴?    澶嶈喘鐜?= 澶嶈喘鐢ㄦ埛鏁?/ 鎬荤敤鎴锋暟
    """
    results = []
    
    for month in sorted(df[month_col].unique()):
        month_df = df[df[month_col] == month]
        
        # 璁＄畻璇ユ湀姣忎釜鐢ㄦ埛鐨勮喘涔版鏁?        user_orders = month_df.groupby('CustomerID')['InvoiceNo'].nunique()
        
        total_users = len(user_orders)
        repurchase_users = (user_orders >= 2).sum()
        repurchase_rate = repurchase_users / total_users if total_users > 0 else 0
        
        results.append({
            'Month': month,
            'Total_Users': total_users,
            'Repurchase_Users': repurchase_users,
            'Repurchase_Rate': repurchase_rate
        })
    
    return pd.DataFrame(results)

# 璁＄畻澶嶈喘鐜?repurchase_by_month = calc_repurchase_rate(df)
print('\n鍚勬湀澶嶈喘鐜?')
print(repurchase_by_month.to_string(index=False))

# 5. 璁＄畻鍚勫浗瀹跺璐巼
print('\n=== 5. 璁＄畻鍚勫浗瀹跺璐巼 ===')
repurchase_by_country = []

for country in df['Country'].unique():
    country_df = df[df['Country'] == country]
    user_orders = country_df.groupby('CustomerID')['InvoiceNo'].nunique()
    
    total_users = len(user_orders)
    repurchase_users = (user_orders >= 2).sum()
    repurchase_rate = repurchase_users / total_users if total_users > 0 else 0
    
    repurchase_by_country.append({
        'Country': country,
        'Total_Users': total_users,
        'Repurchase_Users': repurchase_users,
        'Repurchase_Rate': repurchase_rate
    })

repurchase_by_country = pd.DataFrame(repurchase_by_country)
print('\n鍚勫浗瀹跺璐巼:')
print(repurchase_by_country.sort_values('Repurchase_Rate', ascending=False).to_string(index=False))

# 6. 璁＄畻鍚勭被鍒璐巼
print('\n=== 6. 璁＄畻鍚勭被鍒璐巼锛圱op 5锛?===')
repurchase_by_category = []

for category in df['Description'].unique()[:5]:  # 鍙湅鍓?涓被鍒?    cat_df = df[df['Description'] == category]
    user_orders = cat_df.groupby('CustomerID')['InvoiceNo'].nunique()
    
    total_users = len(user_orders)
    repurchase_users = (user_orders >= 2).sum()
    repurchase_rate = repurchase_users / total_users if total_users > 0 else 0
    
    repurchase_by_category.append({
        'Category': category,
        'Total_Users': total_users,
        'Repurchase_Users': repurchase_users,
        'Repurchase_Rate': repurchase_rate
    })

repurchase_by_category = pd.DataFrame(repurchase_by_category)
print('\n鍚勭被鍒璐巼:')
print(repurchase_by_category.sort_values('Repurchase_Rate', ascending=False).to_string(index=False))

# 7. 淇濆瓨缁撴灉
print('\n=== 7. 淇濆瓨缁撴灉 ===')
repurchase_by_month.to_csv('D:/AI_Dateannaly/repurchase_by_month.csv', index=False, encoding='utf-8-sig')
repurchase_by_country.to_csv('D:/AI_Dateannaly/repurchase_by_country.csv', index=False, encoding='utf-8-sig')
print('缁撴灉宸蹭繚瀛?)

print('\n=== Step 2 瀹屾垚 ===')
print('涓嬩竴姝? 瑙ｈ缁撴灉锛屾壘鍑哄璐巼寮傚父鐨勫師鍥?)
