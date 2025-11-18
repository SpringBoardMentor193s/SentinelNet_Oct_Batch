import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================
# 1. LOAD TRAIN & TEST DATA
# ===============================================
train_data = pd.read_csv("kdd_train.csv")
test_data  = pd.read_csv("kdd_test.csv")

print("Train Dataset Size :", train_data.shape)
print("Test Dataset Size  :", test_data.shape)

# ===============================================
# 2. CREATE BINARY TARGET LABEL
# ===============================================
train_data["is_attack"] = train_data["labels"].apply(lambda x: 0 if x == "normal" else 1)
test_data["is_attack"]  = test_data["labels"].apply(lambda x: 0 if x == "normal" else 1)

cat_columns = ["protocol_type", "service", "flag"]
label_enc = LabelEncoder()

for col in cat_columns:
    train_data[col] = label_enc.fit_transform(train_data[col])
    test_data[col] = label_enc.transform(test_data[col])

TARGET = "is_attack"
FEATURE_COLS = [c for c in train_data.columns if c not in ["is_attack", "labels"]]

# ===============================================
# 3. SPLIT FEATURES AND TARGET
# ===============================================
X_train = train_data[FEATURE_COLS]
y_train = train_data[TARGET]

X_test  = test_data[FEATURE_COLS]
y_test  = test_data[TARGET]

# ===============================================
# 4. FEATURE SCALING
# ===============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===============================================
# 5. CORRELATION HEATMAP
# ===============================================
num_data = train_data.select_dtypes(include=['number'])

plt.figure(figsize=(14, 11))
sns.heatmap(num_data.corr(), cmap="viridis", linewidths=0.3)
plt.title("Correlation Heatmap - Training Dataset")
plt.show()

# ===============================================
# 6. MODEL DEFINITIONS
# ===============================================
models = {
    "Linear OLS": LinearRegression(),
    "SGD Classifier": SGDClassifier(max_iter=2000, tol=1e-4, random_state=42),
    "Logistic Model": LogisticRegression(max_iter=1500),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42)
}

# ===============================================
# 7. TRAIN + EVALUATE ALL MODELS
# ===============================================
results = {}

for model_name, model in models.items():
    print(f"\n---- Training Model: {model_name} ----")

    model.fit(X_train_scaled, y_train)

    if model_name == "Linear OLS":
        y_cont = model.predict(X_test_scaled)
        y_pred = (y_cont >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy Score : {acc:.4f}")
    print(f"Recall (Attack): {rep['1']['recall']:.4f}")
    print("Confusion Matrix:\n", cm)

    results[model_name] = {
        "cm": cm,
        "acc": acc,
        "report": rep
    }

# ===============================================
# 8. PLOT CONFUSION MATRICES
# ===============================================
rows, cols = 3, 2
fig, axes = plt.subplots(rows, cols, figsize=(15, 16))
axes = axes.flatten()

for idx, (model_name, metrics) in enumerate(results.items()):
    disp = ConfusionMatrixDisplay(metrics["cm"], display_labels=["Normal", "Attack"])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"{model_name}\nAccuracy = {metrics['acc']:.3f}")

for i in range(idx + 1, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# ===============================================
# 9. SAVE RESULTS TO TEXT FILE
# ===============================================
summary_file = "updated_model_results.txt"

with open(summary_file, "w") as f:
    f.write("UPDATED MODEL PERFORMANCE SUMMARY\n")
    f.write("=" * 65 + "\n\n")

    for model_name, info in results.items():
        rep = info["report"]
        cm = info["cm"]

        f.write(f"MODEL: {model_name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy       : {info['acc']:.4f}\n")
        f.write(f"Precision (Atk): {rep['1']['precision']:.4f}\n")
        f.write(f"Recall (Atk)   : {rep['1']['recall']:.4f}\n")
        f.write(f"F1 Score (Atk) : {rep['1']['f1-score']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"TN = {cm[0,0]}, FP = {cm[0,1]}\n")
        f.write(f"FN = {cm[1,0]}, TP = {cm[1,1]}\n")
        f.write("\n" + "-" * 50 + "\n")

print(f"\nSummary file saved as → {summary_file}")
