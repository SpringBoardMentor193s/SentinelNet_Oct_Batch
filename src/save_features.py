import pandas as pd
import joblib

print("Saving NSL-KDD and CICIDS feature lists...")

# === NSL-KDD FEATURES ===
df_nsl = pd.read_csv("artifacts/nslkdd_preprocessed.csv", nrows=5)

nsl_features = [col for col in df_nsl.columns if col not in ["label", "attack_type"]]
joblib.dump(nsl_features, "artifacts/nsl_features.pkl")

print("NSL Features Saved:", len(nsl_features))


# === CICIDS FEATURES ===
df_cic = pd.read_csv("artifacts/cicids_preprocessed.csv", nrows=5)

cic_features = [col for col in df_cic.columns if col not in ["Label", " Label"]]
joblib.dump(cic_features, "artifacts/cic_features.pkl")

print("CICIDS Features Saved:", len(cic_features))

print("\n✔ Feature files generated successfully!")
