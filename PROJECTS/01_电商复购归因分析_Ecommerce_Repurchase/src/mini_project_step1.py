import pandas as pd
import numpy as np

print("=== Step 1: 加载数据 ===")
# UCI Online Retail 数据集（本地缓存或下载）
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'

try:
    df = pd.read_excel(url)
    print("✓ 数据下载成功")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("尝试使用备选数据源...")
    # 备选：直接读取常见本地路径
    import os
    local_paths = [
        r'D:\AI_Dateannaly\Online_Retail.xlsx',
        r'C:\Users\Amar\Online_Retail.xlsx'
    ]
    df = None
    for path in local_paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            print(f"✓ 从本地加载: {path}")
            break
    if df is None:
        print("✗ 无法加载数据，请手动下载数据集")
        exit(1)

print(f'\n数据形状: {df.shape[0]} 行 × {df.shape[1]} 列')
print('\n=== 前5行 ===')
print(df.head())

print('\n=== 列信息 ===')
print(df.info())

print('\n=== 缺失值统计 ===')
print(df.isnull().sum())

print('\n=== 描述统计 ===')
print(df.describe())

print('\n=== 唯一值统计 ===')
for col in df.columns:
    print(f'{col}: {df[col].nunique()} 个唯一值')

print('\n=== 数据预览（检查异常值）===')
print('StockCode 示例:', df['StockCode'].unique()[:10])
print('Quantity 最小值:', df['Quantity'].min())
print('UnitPrice 最小值:', df['UnitPrice'].min())

print('\n=== Step 1 完成 ===')
print('下一步: 数据清洗（处理退货、缺失值、异常值）')
