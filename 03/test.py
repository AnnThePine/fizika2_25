import pandas as pd

df = pd.read_csv("ld3_1.txt", sep='\t', decimal=',', encoding='utf-8-sig')

print(df.head())