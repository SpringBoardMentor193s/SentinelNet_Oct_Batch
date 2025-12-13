import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n===================================")
print("        🚀 NIDS MODEL EVALUATION")
print("===================================\n")

# ---------------------------------------------------
# LOAD MODELS + LABEL ENCODER
# ---------------------------------------------------
NSL_MODEL = joblib.load("artifacts/nslkdd_model.pkl")
CICIDS_MODEL = joblib.load("artifacts/cicids_model.pkl")
CICIDS_ENCODER = joblib.load("artifacts/cicids_label_encoder.pkl")

# All 15 CICIDS attack family labels (final list)
ALL_LABELS = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", "Heartbleed",
    "Infiltration", "PortScan", "SSH-Patator",
    "Web Attack – Brute Force", "Web Attack – Sql Injection", "Web Attack – XSS"
]

# ---------------------------------------------------
# NSL-KDD EVALUATION
# ---------------------------------------------------
print("\n===============================")
print("🔷 Evaluating NSL-KDD Model")
print("===============================\n")

nsl = pd.read_csv("artifacts/nslkdd_test.csv")

# REMOVE unwanted label column if exists
if "label" in nsl.columns:
    print("⚠ Removing extra 'label' column from NSL test data...")
    nsl = nsl.drop(columns=["label"])

X_nsl = nsl.drop("attack_type", axis=1)
y_nsl = nsl["attack_type"]

pred_nsl = NSL_MODEL.predict(X_nsl)

print("Accuracy:", accuracy_score(y_nsl, pred_nsl))
print("\nClassification Report:")
print(classification_report(y_nsl, pred_nsl))
print("\nConfusion Matrix:")
print(confusion_matrix(y_nsl, pred_nsl))

# ---------------------------------------------------
# CICIDS EVALUATION
# ---------------------------------------------------
print("\n===============================")
print("🔶 Evaluating CICIDS Model")
print("===============================\n")

# Load only 30k rows for evaluation
df = pd.read_csv("artifacts/cicids_preprocessed.csv", nrows=30000)
df.columns = df.columns.str.strip()

# Keep only BENIGN & DDoS since model trained on 2 classes
df = df[df["Label"].isin(["BENIGN", "DDoS"])]

y_true = df["Label"]
X = df.drop(columns=["Label"])

# Ensure numeric conversion
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Predict directly (model outputs BENIGN / DDoS)
y_pred = CICIDS_MODEL.predict(X)

print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
