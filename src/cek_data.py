import pandas as pd

df = pd.read_csv("data/raw/heart_disease_uci.csv")
print(df.columns.tolist())
print(df.head())
print(df.isnull().sum())
print(df["num"].value_counts())