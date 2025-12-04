# app.py — FINAL WORKING VERSION (December 2025)
# Live IDS with Scapy + Streamlit + Random Forest

import streamlit as st
import threading
import time
import pandas as pd
import numpy as np
import os
from utils.realtime_sniffer import start_sniffing, stop_sniffing
from utils.model_loader import load_model
from utils.predictions import predict_with_model
from utils.metrics_visualizer import plot_prediction_distribution
from utils.realtime_sniffer import start_sniffing, stop_sniffing
# ===============================
# Global live packet buffer (thread-safe)
# ===============================
live_buffer = []
buffer_lock = threading.Lock()

def live_packet_callback(feature_dict):
    """Called for every captured packet"""
    with buffer_lock:
        live_buffer.append({
            "timestamp": time.time(),
            "features": feature_dict
        })
    print(f"[LIVE] Packet captured → Total in buffer: {len(live_buffer)}")

# ===============================
# Preprocessing
# ===============================
def preprocess_live_data(df):
    return df.select_dtypes(include=[np.number]).fillna(0)

# ===============================
# Safe model loader
# ===============================
@st.cache_resource
def safe_load(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

# ===============================
# Constants
# ===============================
AVAILABLE_MODELS_FILES = {
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "SVM": "models/svm.pkl",
    "Decision Tree": "models/decision_tree.pkl"
}
SCALER_PATH = "models/scaler.pkl"

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="SentinelNet IDS", layout="wide")
st.title("SentinelNet – Real-Time Intrusion Detection System")
st.markdown("### Live Network Monitoring + Offline CSV Analysis")

mode = st.selectbox("Choose Mode", ["Live Predicting (Real-Time)", "CSV Predicting (Offline)"])

# ==================================================================
# LIVE MODE
# ==================================================================
# In app.py — Replace the entire LIVE MODE block with this:

if mode == "Live Predicting (Real-Time)":
    st.header("SentinelNet – Live Intrusion Detection")
    st.info("Real-time packet capture → Random Forest prediction")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Controls")
        start_btn = st.button("Start Live Detection", type="primary")
        stop_btn = st.button("Stop Detection", type="secondary")
        refresh_btn = st.button("Refresh Results Now")
        interval = st.slider("Auto-refresh (sec)", 10, 60, 20)

    with col2:
        status = st.empty()
        result = st.empty()
        chart = st.empty()

    # Initialize
    if "live_buffer" not in st.session_state:
        st.session_state.live_buffer = []
    if "sniffer_running" not in st.session_state:
        st.session_state.sniffer_running = False

    lock = threading.Lock()

    def live_callback(features):
        with lock:
            st.session_state.live_buffer.append(features)

    # Start
    if start_btn:
        stop_sniffing()
        st.session_state.live_buffer = []
        start_sniffing(live_callback)
        st.session_state.sniffer_running = True
        st.success("LIVE DETECTION STARTED")

    # Stop
    if stop_btn:
        stop_sniffing()
        st.session_state.sniffer_running = False
        st.warning("Detection Stopped")

    # Load model
    rf_model = safe_load("models/random_forest.pkl")
    scaler = safe_load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None

    # Process buffer
    if st.session_state.sniffer_running and rf_model and (refresh_btn or (time.time() - st.session_state.get("last_refresh", 0) > interval)):
        st.session_state.last_refresh = time.time()
        with lock:
            packets = st.session_state.live_buffer.copy()
            st.session_state.live_buffer.clear()

        if packets:
            df = pd.DataFrame(packets)
            X = df[FEATURE_COLUMNS]
            X_scaled = scaler.transform(X) if scaler else X.values
            preds = predict_with_model(rf_model, X_scaled)
            attacks = int(preds.sum())
            normal = len(preds) - attacks

            status.success(f"Processed {len(preds)} packets")
            result.markdown(f"""
            ### LIVE RESULT
            **Attacks Detected:** `{attacks}`  
            **Normal Traffic:** `{normal}`  
            **Total Packets:** `{len(preds)}`
            """)
            fig = plot_prediction_distribution(preds)
            chart.pyplot(fig)
        else:
            status.info("Capturing packets... Open YouTube or browse")

# ==================================================================
# OFFLINE CSV MODE
# ==================================================================
else:
    st.header("Offline CSV Analysis")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    model_name = st.selectbox("Model", list(AVAILABLE_MODELS_FILES.keys()))
    run_btn = st.button("Run Prediction")

    if uploaded and run_btn:
        df = pd.read_csv(uploaded)
        X = preprocess_live_data(df)
        model = safe_load(AVAILABLE_MODELS_FILES[model_name])
        scaler = safe_load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

        if model:
            X_in = scaler.transform(X) if scaler else X.values
            preds = predict_with_model(model, X_in)
            attacks = int(preds.sum())

            st.success(f"**{attacks} Attacks** detected out of {len(preds)} records")
            result_df = pd.DataFrame({
                "No": range(1, len(preds)+1),
                "Prediction": ["Attack" if p == 1 else "Normal" for p in preds]
            })
            st.dataframe(result_df)
            st.pyplot(plot_prediction_distribution(preds))

# Footer
st.markdown("---")
st.markdown("**SentinelNet IDS** © 2025 | Powered by Scapy + Streamlit + ML")
