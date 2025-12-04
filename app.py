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
if mode == "Live Predicting (Real-Time)":
    st.header("Live Network Monitoring")
    st.info("Capturing real packets using **Scapy** → Random Forest predictions every few seconds.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Controls")
        interface_input = st.text_input("Interface", value="auto", help="Use 'auto' or full name")
        start_btn = st.button("Start Live Detection", type="primary")
        stop_btn = st.button("Stop Detection")
        refresh_btn = st.button("Force Refresh Now")
        interval = st.slider("Auto-refresh (seconds)", 10, 120, 20)

        if st.button("List Interfaces (tshark)"):
            import subprocess
            try:
                out = subprocess.check_output("tshark -D", shell=True, text=True)
                st.code(out)
            except:
                st.error("tshark not found")

    with col2:
        status = st.empty()
        result_box = st.empty()
        table = st.empty()
        chart = st.empty()

    # Session state
    if "sniffer" not in st.session_state:
        st.session_state.sniffer = None
    if "last_run" not in st.session_state:
        st.session_state.last_run = 0

    # Load model & scaler
    rf_model = safe_load(AVAILABLE_MODELS_FILES["Random Forest"])
    scaler = safe_load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    # START
    if start_btn:
        stop_sniffing()
        start_sniffing(live_packet_callback, interface="auto")
        st.success("Sniffer started!")
        st.rerun()

    # STOP
    if stop_btn and st.session_state.sniffer:
        stop_sniffing()
        st.session_state.sniffer = None
        st.warning("Sniffer stopped")
        st.rerun()

    # PROCESS PACKETS
    if rf_model and (refresh_btn or time.time() - st.session_state.last_run >= interval):
        st.session_state.last_run = time.time()

        with buffer_lock:
            packets = live_buffer.copy()
            live_buffer.clear()

        if not packets:
            status.info("Waiting for packets... Open a website or ping google.com")
        else:
            df = pd.DataFrame([p["features"] for p in packets])
            X = preprocess_live_data(df)
            X_scaled = scaler.transform(X) if scaler else X.values
            preds = predict_with_model(rf_model, X_scaled)

            attacks = int(preds.sum())
            normal = len(preds) - attacks

            status.success(f"Processed {len(preds)} packets")
            result_box.markdown(f"""
                ### Live Result
                - **Attacks Detected**: {attacks}  
                - **Normal Traffic**: {normal}  
                - **Total Packets**: {len(preds)}
            """)

            recent = df.tail(10).copy()
            recent["Prediction"] = ["**ATTACK**" if x == 1 else "Normal" for x in preds[-10:]]
            table.dataframe(recent, use_container_width=True)
            chart.pyplot(plot_prediction_distribution(preds))

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
