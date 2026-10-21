import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/heart_disease_uci.csv")
print("Bentuk awal:", df.shape)

# --- Buang kolom yang tidak berguna & duplikat ---
df = df.drop(columns=["id", "dataset"], errors="ignore")
df = df.drop_duplicates().reset_index(drop=True)

# --- Target jadi biner: 0 = sehat, 1 = ada penyakit jantung ---
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

# --- Nilai 0 yang tidak masuk akal dianggap data kosong ---
for c in ["trestbps", "chol"]:
    df.loc[df[c] == 0, c] = np.nan

# --- Pisahkan kolom numerik & kategorikal ---
fitur = [c for c in df.columns if c != "target"]
num_cols = df[fitur].select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in fitur if c not in num_cols]
print("Numerik    :", num_cols)
print("Kategorikal:", cat_cols)

# --- Tangani data kosong ---
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0]).astype(str)

# --- Mengubah kategori menjadi angka (Label Encoding) untuk kolom biner ---
biner = [c for c in cat_cols if df[c].nunique() == 2]
for c in biner:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    print(f"{c}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- Encoding kategorikal (One-Hot) untuk kolom dengan >2 kategori ---
multi = [c for c in cat_cols if c not in biner]
df = pd.get_dummies(df, columns=multi, drop_first=True, dtype=int)

# --- Simpan ---
X = df.drop(columns=["target"])
y = df["target"]
X.to_csv("data/processed/X_clean.csv", index=False)
y.to_csv("data/processed/y_clean.csv", index=False)
joblib.dump(num_cols, "models/num_cols.pkl")

print("Sisa data kosong:", df.isnull().sum().sum())
print("Bentuk akhir X:", X.shape)