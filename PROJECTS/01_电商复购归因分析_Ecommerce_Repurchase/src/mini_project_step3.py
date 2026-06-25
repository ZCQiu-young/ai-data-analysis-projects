import pandas as pd
import numpy as np

print("=== Step 3: 楠岃瘉鍋囪 ===\n")

# 鍔犺浇娓呮礂鍚庣殑鏁版嵁
df = pd.read_csv('D:/AI_Dateannaly/PROJECTS/project_ecommerce_analysis/data/mini_project_data.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month

# 鍒犻櫎閫€璐?df = df[df['Quantity'] > 0].copy()

# 鏁翠綋骞冲潎鍊硷紙鐢ㄤ簬瀵规瘮锛?overall_avg_amount = df['TotalPrice'].mean()
print(f"鏁翠綋骞冲潎璁㈠崟閲戦: {overall_avg_amount:.2f}\n")

# ========== 鍋囪1: 4鏈堝拰8鏈堝璐巼浣庯紝鏄惁鍥犱负璁㈠崟閲戦鍋忛珮锛?==========
print("=" * 60)
print("鍋囪1楠岃瘉: 4鏈堝拰8鏈堝璐巼浣庯紝鏄惁鍥犱负璁㈠崟閲戦鍋忛珮锛?)
print("=" * 60)

month_stats = df.groupby('Month')['TotalPrice'].agg(['mean', 'count']).reset_index()
month_stats.columns = ['Month', 'Avg_Amount', 'Order_Count']
month_stats['vs_Overall'] = (month_stats['Avg_Amount'] / overall_avg_amount - 1) * 100

print("\n鍚勬湀璁㈠崟閲戦:")
print(f"{'Month':<8} {'Avg_Amount':>12} {'Order_Count':>12} {'vs_Overall':>12}")
print("-" * 44)
for _, row in month_stats.iterrows():
    print(f"{int(row['Month']):<8} {row['Avg_Amount']:>12.2f} {int(row['Order_Count']):>12} {row['vs_Overall']:>+11.1f}%")

# 鎵惧嚭澶嶈喘鐜囦綆鐨勬湀浠?low_repurchase_months = [4, 8]
print(f"\n澶嶈喘鐜囦綆鐨勬湀浠斤紙4鏈堝拰8鏈堬級:")
for m in low_repurchase_months:
    row = month_stats[month_stats['Month'] == m].iloc[0]
    status = "[楂樹簬骞冲潎]" if row['Avg_Amount'] > overall_avg_amount else "[浣庝簬骞冲潎]"
    print(f"  {m}鏈? 骞冲潎璁㈠崟閲戦 {row['Avg_Amount']:.2f}, {status}")

# ========== 鍋囪2: Spain澶嶈喘鐜囦负0锛屾槸鍚︽墍鏈塖pain鐢ㄦ埛鍙拱浜?娆★紵 ==========
print("\n" + "=" * 60)
print("鍋囪2楠岃瘉: Spain澶嶈喘鐜囦负0锛屾槸鍚︽墍鏈塖pain鐢ㄦ埛鍙拱浜?娆★紵")
print("=" * 60)

spain_df = df[df['Country'] == 'Spain']

if len(spain_df) == 0:
    print("Spain 娌℃湁鏁版嵁")
else:
    spain_user_orders = spain_df.groupby('CustomerID')['InvoiceNo'].nunique()
    
    print(f"\nSpain 鐢ㄦ埛鎬绘暟: {len(spain_user_orders)}")
    print(f"Spain 璁㈠崟鎬绘暟: {len(spain_df)}")
    print(f"\n鍚勭敤鎴疯喘涔版鏁板垎甯?")
    print(spain_user_orders.value_counts().sort_index())
    
    users_with_1_order = (spain_user_orders == 1).sum()
    users_with_2plus = (spain_user_orders >= 2).sum()
    
    print(f"\n鍙喘涔?娆＄殑鐢ㄦ埛: {users_with_1_order} ({users_with_1_order/len(spain_user_orders)*100:.1f}%)")
    print(f"璐拱2娆″強浠ヤ笂鐨勭敤鎴? {users_with_2plus} ({users_with_2plus/len(spain_user_orders)*100:.1f}%)")
    
    if users_with_2plus == 0:
        print("\n[楠岃瘉鎴愬姛] Spain 鎵€鏈夌敤鎴烽兘鍙拱浜?娆★紝澶嶈喘鐜囪嚜鐒朵负0")

# ========== 鍋囪3: Electronics澶嶈喘鐜囦綆锛屾槸鍚﹀洜涓轰环鏍间笂娑紵 ==========
print("\n" + "=" * 60)
print("鍋囪3楠岃瘉: Electronics澶嶈喘鐜囦綆锛屾槸鍚﹀洜涓轰环鏍间笂娑紵")
print("=" * 60)

# 鍏堢湅鏁翠綋 Electronics 鐨勪环鏍艰秼鍔匡紙鎸夋湀锛?elec_df = df[df['Description'] == 'Electronics'].copy()

if len(elec_df) > 0:
    elec_monthly = elec_df.groupby('Month')['UnitPrice'].agg(['mean', 'count']).reset_index()
    elec_monthly.columns = ['Month', 'Avg_UnitPrice', 'Order_Count']
    
    # 鏁翠綋 Electronics 骞冲潎浠锋牸
    overall_elec_price = elec_df['UnitPrice'].mean()
    print(f"\nElectronics 鏁翠綋骞冲潎鍗曚环: {overall_elec_price:.2f}")
    
    print(f"\nElectronics 鍚勬湀鍗曚环鍙樺寲:")
    print(f"{'Month':<8} {'Avg_UnitPrice':>14} {'Order_Count':>12} {'vs_Overall':>12}")
    print("-" * 46)
    for _, row in elec_monthly.iterrows():
        diff = (row['Avg_UnitPrice'] / overall_elec_price - 1) * 100
        print(f"{int(row['Month']):<8} {row['Avg_UnitPrice']:>14.2f} {int(row['Order_Count']):>12} {diff:>+11.1f}%")
    
    # 鎵惧嚭3鏈堣捣鐨勪环鏍煎彉鍖?    march_price = elec_monthly[elec_monthly['Month'] == 3]['Avg_UnitPrice'].values[0]
    print(f"\n3鏈?Electronics 骞冲潎鍗曚环: {march_price:.2f}")
    print("锛?鏈堝悗浠锋牸涓婃定锛屾寜鐞嗚搴旇褰卞搷澶嶈喘鐜囷級")
    
    # 瀵规瘮 Electronics vs 鍏朵粬绫诲埆
    print("\n鍚勬湀浠?Electronics vs 鏁翠綋鍗曚环瀵规瘮:")
    print(f"{'Month':<8} {'Electronics':>14} {'鏁翠綋骞冲潎':>12} {'婧环鐜?:>10}")
    print("-" * 44)
    for _, row in elec_monthly.iterrows():
        elec_price = row['Avg_UnitPrice']
        overall_price = df[df['Month'] == row['Month']]['TotalPrice'].mean()
        premium = (elec_price / overall_price - 1) * 100
        print(f"{int(row['Month']):<8} {elec_price:>14.2f} {overall_price:>12.2f} {premium:>+9.1f}%")

print("\n=== 鍋囪楠岃瘉瀹屾垚 ===")
