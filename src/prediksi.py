import pandas as pd
import joblib

# ============================================
# DATA PASIEN BARU (ubah sesuai kebutuhan)
# ============================================
pasien = {
    "age": 58,
    "sex": "Male",                    # Male / Female
    "cp": "asymptomatic",             # typical angina / atypical angina / non-anginal / asymptomatic
    "trestbps": 140,                  # tekanan darah istirahat (mm Hg)
    "chol": 240,                      # kolesterol (mg/dl)
    "fbs": False,                     # gula darah puasa > 120 mg/dl (True/False)
    "restecg": "lv hypertrophy",      # normal / st-t abnormality / lv hypertrophy
    "thalch": 120,                    # detak jantung maksimum
    "exang": True,                    # angina akibat olahraga (True/False)
    "oldpeak": 2.0,                   # ST depression
    "slope": "flat",                  # upsloping / flat / downsloping
    "ca": 1,                          # jumlah pembuluh utama (0-3)
    "thal": "reversable defect",      # normal / fixed defect / reversable defect
}

# ============================================
# LOAD MODEL, SCALER, DAN DAFTAR KOLOM
# ============================================
X_train = pd.read_csv("data/processed/X_train.csv")
kolom_model = X_train.columns.tolist()
num_cols = joblib.load("models/num_cols.pkl")
scaler = joblib.load("models/scaler.pkl")
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
}

# ============================================
# PREPROCESSING (sama persis dengan saat pelatihan)
# ============================================
baris = {c: 0 for c in kolom_model}

# Kolom numerik
for c in num_cols:
    baris[c] = pasien[c]

# Label encoding kolom biner
baris["sex"] = 1 if pasien["sex"] == "Male" else 0
baris["fbs"] = 1 if pasien["fbs"] else 0
baris["exang"] = 1 if pasien["exang"] else 0

# One-hot encoding kolom kategorikal
for c in ["cp", "restecg", "slope", "thal"]:
    nama_kolom = f"{c}_{pasien[c]}"
    if nama_kolom in baris:
        baris[nama_kolom] = 1
    # kalau tidak ada, berarti itu kategori acuan (drop_first), semua kolom tetap 0

X_new = pd.DataFrame([baris])[kolom_model]

# Skalasi fitur numerik
X_new[num_cols] = scaler.transform(X_new[num_cols])

# ============================================
# PREDIKSI
# ============================================
print("\n=== HASIL PREDIKSI ===")
for nama, model in models.items():
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]
    label = "ADA indikasi penyakit jantung" if pred == 1 else "TIDAK ada indikasi penyakit jantung"
    print(f"{nama:20s}: {label} (probabilitas sakit: {proba:.1%})")

print("\nCatatan: ini hasil model latihan, bukan diagnosis medis.")