import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

print("Loading CICIDS dataset...")
df = pd.read_csv("artifacts/cicids_preprocessed.csv", nrows=50000, low_memory=False)

print("Encoding labels...")
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

joblib.dump(le, "artifacts/cicids_label_encoder.pkl")
print("✔ Label encoder saved at artifacts/cicids_label_encoder.pkl")
print("Classes:", le.classes_)
