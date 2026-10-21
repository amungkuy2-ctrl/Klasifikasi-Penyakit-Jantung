import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("data/processed/X_clean.csv")
y = pd.read_csv("data/processed/y_clean.csv").squeeze()
num_cols = joblib.load("models/num_cols.pkl")

# --- Membagi data latih & uji (80:20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Skalasi fitur (fit hanya di data latih) ---
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# --- Simpan ---
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
joblib.dump(scaler, "models/scaler.pkl")

print("Data latih:", X_train.shape, "| Data uji:", X_test.shape)
print("Proporsi kelas latih:\n", y_train.value_counts(normalize=True))