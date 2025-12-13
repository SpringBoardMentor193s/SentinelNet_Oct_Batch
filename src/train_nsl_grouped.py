# src/train_nsl_grouped.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT = "artifacts/nsl_train_grouped.csv"
MODEL_OUT = "artifacts/nslkdd_grouped_model.pkl"
ENCODER_OUT = "artifacts/nsl_label_encoder.pkl"

print("Loading:", INPUT)
df = pd.read_csv(INPUT)
print("Shape:", df.shape)

# Features = everything except label columns
X = df.drop(columns=["label", "class"], errors="ignore")
X = X.select_dtypes(include="number").fillna(0)

y = df["class"]

# Encode 5 target classes
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Train-test split (stratified so each class preserved)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print("Training RandomForest...")
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n✔ Accuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model + encoder
joblib.dump(model, MODEL_OUT)
joblib.dump(encoder, ENCODER_OUT)

print("\n✔ Model saved at:", MODEL_OUT)
print("✔ Encoder saved at:", ENCODER_OUT)

# Show confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
