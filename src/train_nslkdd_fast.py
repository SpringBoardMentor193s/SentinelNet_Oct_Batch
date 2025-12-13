import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "artifacts")

TRAIN_PATH = os.path.join(ART, "nsl_train.csv")
FEATURES_PATH = os.path.join(ART, "nsl_features.pkl")
MODEL_PATH = os.path.join(ART, "nslkdd_model.pkl")

print("Loading train file:", TRAIN_PATH)
df = pd.read_csv(TRAIN_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns)

# Load feature list
NSL_FEATURES = joblib.load(FEATURES_PATH)

# Keep selected features only
df = df[NSL_FEATURES + ["label"]]

# Encode categorical values
cat_cols = ["protocol_type", "service", "flag"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("label", axis=1)
y = df["label"]

# Encode target
y = LabelEncoder().fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training RandomForest (optimized for low RAM)...")

model = RandomForestClassifier(
    n_estimators=160,
    max_depth=18,
    min_samples_split=4,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("✔ Training Done!")
print("Accuracy:", acc)

# Save model
joblib.dump(model, MODEL_PATH)
print("✔ Model saved at:", MODEL_PATH)
