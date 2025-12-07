import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("\nTraining NSL-KDD model...\n")

df = pd.read_csv("artifacts/nslkdd_preprocessed.csv")

# Drop raw attack name (not needed for training)
df = df.drop(["label"], axis=1)

# MULTI-CLASS TARGET = attack_type
y = df["attack_type"]
X = df.drop(["attack_type"], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Save model
joblib.dump(model, "artifacts/nslkdd_model.pkl")

print("\nNSL-KDD Multi-Class Model Saved Successfully → artifacts/nslkdd_model.pkl\n")
