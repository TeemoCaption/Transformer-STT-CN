import os
import pandas as pd
import numpy as np

df = pd.read_csv('./dataset/validated.tsv', sep='\t')

print("\n各欄位缺失值數量：")
print(df.isnull().sum())

cols_to_keep = ['client_id', 'path', 'sentence']
df = df[cols_to_keep]

print(f"資料筆數：{len(df)} 筆")
print(df.head())