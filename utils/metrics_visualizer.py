import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# ---------------------------
# Prediction Distribution
# ---------------------------
def plot_prediction_distribution(preds):
    unique, counts = np.unique(preds, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=unique, y=counts, ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")

    return fig

# ---------------------------
# Metrics Bar Chart
# ---------------------------
def plot_metrics_bar(metrics_dict):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_title("Model Performance Metrics")
    ax.set_ylim(0, 1)

    return fig

# ---------------------------
# Confusion Matrix Plot
# ---------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")

    return fig

# ---------------------------
# AUC Curve
# ---------------------------
def plot_auc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return fig
