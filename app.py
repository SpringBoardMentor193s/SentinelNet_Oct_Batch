import streamlit as st
import pandas as pd
import numpy as np

from utils.preprocessing import preprocess_live_data
from utils.model_loader import load_model
from utils.metrics_visualizer import (
    plot_prediction_distribution,
    plot_metrics_bar,
    plot_confusion_matrix,
    plot_auc_curve
)

from utils.predictions import predict_with_model


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="wide",
    page_icon="🔐"
)

st.title("🔐 Intrusion Detection System (IDS)")
st.write("Choose a mode to begin")

# ---------------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------------
mode = st.sidebar.radio(
    "Select Mode",
    ["Live Data Testing (Random Forest)", "Model Metrics Analysis (CSV Input)"]
)

# ---------------------------------------------------------
# MODE 1 — LIVE DATA TESTING
# ---------------------------------------------------------
if mode == "Live Data Testing (Random Forest)":

    st.header("🟢 Live Data Testing (Random Forest Only)")

    uploaded_file = st.file_uploader("Upload live traffic CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        # Preprocessing
        X = preprocess_live_data(df)

        # Load RF Model
        model = load_model("models/random_forest.pkl")
        scaler = load_model("models/scaler.pkl")

        X_scaled = scaler.transform(X)

        # Predictions
        preds = predict_with_model(model, X_scaled)

        st.subheader("🔍 Prediction Distribution")
        plot_prediction_distribution(preds)

        st.subheader("📊 Random Forest Performance")
        st.write("*(Based on pre-trained metrics)*")

        # Optional: Show Confusion Matrix, ROC Curve
        # if you have them saved as numpy arrays

