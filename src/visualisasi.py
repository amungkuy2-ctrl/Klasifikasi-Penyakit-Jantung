import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Folder untuk menyimpan gambar (dibuat otomatis kalau belum ada)
os.makedirs("output", exist_ok=True)

# --- Load data uji & model ---
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
model = joblib.load("models/random_forest.pkl")

y_pred = model.predict(X_test)

# --- 1. Confusion Matrix (heatmap) ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Sehat", "Sakit"],
    yticklabels=["Sehat", "Sakit"],
)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150)
plt.show()

# --- 2. Feature Importance (10 teratas) ---
importance = pd.Series(model.feature_importances_, index=X_test.columns)
top10 = importance.sort_values(ascending=True).tail(10)

plt.figure(figsize=(8, 5))
top10.plot(kind="barh", color="steelblue")
plt.xlabel("Tingkat kepentingan")
plt.title("10 Fitur Terpenting - Random Forest")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150)
plt.show()

print("Gambar tersimpan di folder output/")