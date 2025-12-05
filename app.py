# app.py
import streamlit as st
import pandas as pd
import time
import os

from utils.realtime_sniffer import packet_queue, start_sniffer_thread
from utils.model_loader import load_model
from utils.predictions import predict_with_model
from utils.metrics_visualizer import plot_prediction_distribution

st.set_page_config(page_title="SentinelNet IDS", layout="wide")

# ---------------- Session state ----------------
if "sniffer_thread" not in st.session_state:
    st.session_state.sniffer_thread = None
if "sniffer_running" not in st.session_state:
    st.session_state.sniffer_running = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

# ---------------- Load models (cached) ----------------
@st.cache_resource
def load_resources():
    model = load_model("models/random_forest.pkl")
    scaler = load_model("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
    return model, scaler

rf_model, scaler = load_resources()

FEATURE_COLUMNS = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'hot', 'logged_in',
    'num_compromised', 'root_shell', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'protocol_type_encoded',
    'service_encoded', 'flag_encoded'
]

# ---------------- UI ----------------
st.title("🔐 SentinelNet IDS — Real-Time Intrusion Detection")
mode = st.selectbox("Choose Mode", ["Live Predicting (Real-Time)", "CSV Predicting (Offline)"])

# Model list for CSV mode
MODELS = {
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "SVM": "models/svm.pkl",
    "Decision Tree": "models/decision_tree.pkl"
}

# ---------------- LIVE MODE ----------------
if mode == "Live Predicting (Real-Time)":
    st.subheader("Live Network Protection")
    col1, col2 = st.columns([1, 3])

    with col1:
        start_btn = st.button("START LIVE DETECTION", use_container_width=True)
        stop_btn = st.button("STOP DETECTION", use_container_width=True)
        refresh_btn = st.button("Refresh Now", use_container_width=True)
        interval = st.slider("Auto-refresh (sec)", 5, 60, 15)

    with col2:
        status_box = st.empty()
        results_box = st.empty()
        chart_box = st.empty()

    # Start
    if start_btn and not st.session_state.sniffer_running:
        st.session_state.sniffer_thread = start_sniffer_thread(
            iface=r"\Device\NPF_{C313C7E2-0BB6-422F-A1B9-C94F462988FD}"
            )
        st.session_state.sniffer_running = True
        st.success("Sniffer started — generate traffic (open YouTube / browse)")

    # Stop
    if stop_btn and st.session_state.sniffer_running:
        if st.session_state.sniffer_thread:
            st.session_state.sniffer_thread.stop()
            st.session_state.sniffer_thread = None
        st.session_state.sniffer_running = False
        st.warning("Sniffer stopped")

    # Process queued packets on refresh/interval
    if st.session_state.sniffer_running:
        if refresh_btn or (time.time() - st.session_state.last_refresh >= interval):
            st.session_state.last_refresh = time.time()

            packets = []
            while not packet_queue.empty():
                packets.append(packet_queue.get())

            if packets:
                df = pd.DataFrame(packets)
                # Ensure columns exist
                X = df[FEATURE_COLUMNS].fillna(0)
                X_in = scaler.transform(X) if scaler else X.values
                preds = predict_with_model(rf_model, X_in)

                attacks = int(sum(preds))
                normal = len(preds) - attacks

                status_box.success(f"Analyzed {len(preds)} packets")
                results_box.markdown(f"""
                    <div style="text-align:center; padding:20px;">
                        <h2>Live Threat Report</h2>
                        <h1 style="color:red;">⚠ {attacks} Attacks Detected</h1>
                        <h3 style="color:lightgreen;">✔ {normal} Normal Packets</h3>
                    </div>
                """, unsafe_allow_html=True)

                fig = plot_prediction_distribution(preds)
                chart_box.pyplot(fig, use_container_width=True)
            else:
                status_box.info("Waiting for packets... generate traffic (open YouTube / ping google.com).")

# ---------------- CSV MODE ----------------
else:
    st.subheader("Offline CSV Analysis")
    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        model_choice = st.selectbox("Choose Model", list(MODELS.keys()))
        if st.button("Run Prediction"):
            model_path = MODELS[model_choice]
            model = load_model(model_path)
            scaler_local = load_model("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None

            X = df[FEATURE_COLUMNS].fillna(0)
            X_in = scaler_local.transform(X) if scaler_local else X.values
            preds = predict_with_model(model, X_in)

            attacks = int(sum(preds))
            st.success(f"Detected {attacks} attacks out of {len(preds)} rows")

            fig = plot_prediction_distribution(preds)
            st.pyplot(fig)
