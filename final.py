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
from sklearn.svm import SVC
from xgboost import XGBClassifier

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


# 7 DEFINE MODELS (ALL 8)

models = {
    "Linear Regression (OLS)": LinearRegression(),
    "Gradient Descent (SGD)": SGDClassifier(max_iter=2000, tol=1e-4, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
    "Naïve Bayes (Gaussian)": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42)
}


# 8 TRAIN AND EVALUATE ALL MODELS

results = {}
table_rows = []

print("\nMODEL PERFORMANCE SUMMARY\n")
print(f"{'Model':25s} {'Train_Accuracy':>15s} {'Test_Accuracy':>15s} {'Gap':>10s} "
      f"{'Precision':>12s} {'Recall':>12s} {'F1_Score':>12s} "
      f"{'TN':>6s} {'FP':>6s} {'FN':>6s} {'TP':>6s}")
print("-" * 140)

for name, model in models.items():
    print(f"\n{'='*20} Training: {name} {'='*20}")

    # Fit model
    model.fit(X_train_scaled, y_train)

    # Train predictions
    if name == "Linear Regression (OLS)":
        y_train_cont = model.predict(X_train_scaled)
        y_train_pred = (y_train_cont >= 0.5).astype(int)
    else:
        y_train_pred = model.predict(X_train_scaled)

    # Test predictions
    if name == "Linear Regression (OLS)":
        y_pred_cont = model.predict(X_test_scaled)
        y_pred = (y_pred_cont >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_scaled)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test, y_pred)
    gap = train_acc - test_acc

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    rep = classification_report(y_test, y_pred, output_dict=True)
    precision = rep['1']['precision']
    recall    = rep['1']['recall']
    f1        = rep['1']['f1-score']

    print(f"Accuracy: {test_acc:.4f}")
    print(f"Attack Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save per-model info in results (for plots and file summary)
    results[name] = {
        "cm": cm,
        "acc": test_acc,
        "report": rep,
        "train_acc": train_acc,
        "gap": gap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    # Row for the big table
    table_rows.append([
        name, train_acc, test_acc, gap,
        precision, recall, f1, tn, fp, fn, tp
    ])

    # Print row in table format
    print(f"{name:25s} {train_acc:15.6f} {test_acc:15.6f} {gap:10.6f} "
          f"{precision:12.6f} {recall:12.6f} {f1:12.6f} "
          f"{tn:6d} {fp:6d} {fn:6d} {tp:6d}")


# 8.1 PRINT FINAL TABLE NEATLY

print("\n\nFINAL CONSOLIDATED TABLE\n")
print(f"{'Model':25s} {'Train_Accuracy':>15s} {'Test_Accuracy':>15s} {'Gap':>10s} "
      f"{'Precision':>12s} {'Recall':>12s} {'F1_Score':>12s} "
      f"{'TN':>6s} {'FP':>6s} {'FN':>6s} {'TP':>6s}")
print("-" * 140)

for row in table_rows:
    name, train_acc, test_acc, gap, precision, recall, f1, tn, fp, fn, tp = row
    print(f"{name:25s} {train_acc:15.6f} {test_acc:15.6f} {gap:10.6f} "
          f"{precision:12.6f} {recall:12.6f} {f1:12.6f} "
          f"{tn:6d} {fp:6d} {fn:6d} {tp:6d}")

# Also keep as DataFrame if you want to export
summary_df = pd.DataFrame(table_rows, columns=[
    "Model", "Train_Accuracy", "Test_Accuracy", "Gap",
    "Precision", "Recall", "F1_Score", "TN", "FP", "FN", "TP"
])
summary_df.to_csv("model_performance_table.csv", index=False)
print("\nSaved consolidated table → model_performance_table.csv")


# 9 PLOT CONFUSION MATRICES

rows, cols = 3, 3  # 8 models → 3x3 grid
fig, axes = plt.subplots(rows, cols, figsize=(16, 15))
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
        f.write(f"Train Accuracy   : {m['train_acc']:.6f}\n")
        f.write(f"Test Accuracy    : {m['acc']:.6f}\n")
        f.write(f"Gap              : {m['gap']:.6f}\n")
        f.write(f"Precision (Atk)  : {m['precision']:.6f}\n")
        f.write(f"Recall (Atk)     : {m['recall']:.6f}\n")
        f.write(f"F1 Score (Atk)   : {m['f1']:.6f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"TN={cm[0,0]}, FP={cm[0,1]}\n")
        f.write(f"FN={cm[1,0]}, TP={cm[1,1]}\n")
        f.write("\n" + "-"*40 + "\n")

print(f"\nSaved detailed summary → {output_file}")