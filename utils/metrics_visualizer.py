import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np


# Prediction Distribution Bar Chart
def plot_prediction_distribution(preds):
    unique, counts = np.unique(preds, return_counts=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique, y=counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    st.pyplot(plt)

# Metrics Bar Chart
def plot_metrics_bar(metrics_dict):
    labels = metrics_dict.keys()
    values = metrics_dict.values()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(labels), y=list(values))
    plt.title("Model Performance Metrics")
    plt.ylim(0, 1)
    st.pyplot(plt)

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    st.pyplot(plt)

# AUC Curve
def plot_auc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)
