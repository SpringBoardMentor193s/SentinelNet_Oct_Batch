import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

DATA = "artifacts/cicids_small.csv"
df = pd.read_csv(DATA)

print("Dataset loaded:", df.shape)

# Keep only BENIGN & DDoS
df = df[df["Label"].isin(["BENIGN", "DDoS"])]
print("Filtered (BENIGN + DDoS):", df.shape)

# Split
benign = df[df["Label"] == "BENIGN"]
ddos   = df[df["Label"] == "DDoS"]

# Balance both classes
m = min(len(benign), len(ddos))
print("Balancing to:", m)

benign_s = resample(benign, replace=False, n_samples=m, random_state=42)
ddos_s   = resample(ddos, replace=False, n_samples=m, random_state=42)

train_df = pd.concat([benign_s, ddos_s])
print("Final training dataset:", train_df.shape)

X = train_df.drop(columns=["Label"])
y = train_df["Label"]

# Force numeric (important)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# ⭐⭐ Save final feature list for prediction
CICIDS_FEATURES = list(X.columns)
joblib.dump(CICIDS_FEATURES, "artifacts/cic_features.pkl")
print("✔ Saved CICIDS feature list:", len(CICIDS_FEATURES))

print("Training RandomForest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "artifacts/cicids_model.pkl")
print("✔ Model saved!")
