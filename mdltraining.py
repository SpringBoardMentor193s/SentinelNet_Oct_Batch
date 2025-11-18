# 1 IMPORT REQUIRED LIBRARIES
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


# 2 LOAD DATA
train_df = pd.read_csv("kdd_train.csv")
test_df  = pd.read_csv("kdd_test.csv")

print("Training Set Shape:", train_df.shape)
print("Testing Set Shape :", test_df.shape)


# 3 CREATE BINARY TARGET COLUMN
train_df["attack_binary"] = train_df["labels"].apply(lambda x: 0 if x == "normal" else 1)
test_df["attack_binary"]  = test_df["labels"].apply(lambda x: 0 if x == "normal" else 1)

categorical_cols = ["protocol_type", "service", "flag"]
encoder = LabelEncoder()

for col in categorical_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

TARGET = "attack_binary"
FEATURES = [col for col in train_df.columns if col not in ["attack_binary", "labels"]]


# 4 SPLIT INTO X AND y
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]


# 5 SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# 6 FEATURE CORRELATION HEATMAP (ONLY NUMERIC COLUMNS)
numeric_df = train_df.select_dtypes(include=['number'])

plt.figure(figsize=(15, 12))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", linewidths=0.4)
plt.title("Correlation Heatmap (Training Data)")
plt.show()


# 7 DEFINE MODELS
models = {
    "OLS Regression": LinearRegression(),
    "Stochastic GD": SGDClassifier(max_iter=2000, tol=1e-4, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1500),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42)
}


# 8 TRAIN AND EVALUATE ALL MODELS
results = {}

for name, model in models.items():
    print(f"\n{'='*20} Training: {name} {'='*20}")

    model.fit(X_train_scaled, y_train)

    if name == "OLS Regression":
        y_pred_cont = model.predict(X_test_scaled)
        y_pred = (y_pred_cont >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(f"Attack Recall: {rep['1']['recall']:.4f}")
    print("Confusion Matrix:")
    print(cm)

    results[name] = {
        "cm": cm,
        "acc": acc,
        "report": rep
    }


# 9 PLOT CONFUSION MATRICES
rows, cols = 3, 2
fig, axes = plt.subplots(rows, cols, figsize=(14, 15))
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    disp = ConfusionMatrixDisplay(res["cm"], display_labels=["Normal", "Attack"])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"{name}\nAccuracy: {res['acc']:.3f}")

for i in range(idx + 1, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()


# 10 SAVE TEXT SUMMARY
output_file = "model_results_summary.txt"

with open(output_file, "w") as f:
    f.write("MODEL PERFORMANCE SUMMARY\n")
    f.write("="*60 + "\n\n")

    for name, m in results.items():
        rep = m["report"]
        cm = m["cm"]

        f.write(f"MODEL: {name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy        : {m['acc']:.4f}\n")
        f.write(f"Precision (Atk) : {rep['1']['precision']:.4f}\n")
        f.write(f"Recall (Atk)    : {rep['1']['recall']:.4f}\n")
        f.write(f"F1 Score (Atk)  : {rep['1']['f1-score']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"TN={cm[0,0]}, FP={cm[0,1]}\n")
        f.write(f"FN={cm[1,0]}, TP={cm[1,1]}\n")
        f.write("\n" + "-"*40 + "\n")

print(f"\nSaved Summary → {output_file}")
