import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

os.makedirs("output", exist_ok=True)

# --- Load data uji ---
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# --- Load model yang sudah disimpan ---
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
}

# --- Hitung metrik tiap model ---
hasil = []
for nama, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    hasil.append({
        "Model": nama,
        "Akurasi": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })

tabel = pd.DataFrame(hasil).set_index("Model").round(4)
print("\n=== PERBANDINGAN MODEL ===")
print(tabel)

# --- Simpan tabel ke CSV ---
tabel.to_csv("output/perbandingan_model.csv")

# --- Grafik batang perbandingan ---
tabel.plot(kind="bar", figsize=(9, 5), ylim=(0.6, 1.0))
plt.title("Perbandingan Model Klasifikasi Penyakit Jantung")
plt.ylabel("Skor")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("output/perbandingan_model.png", dpi=150)
plt.show()

print("\nTabel dan grafik tersimpan di folder output/")