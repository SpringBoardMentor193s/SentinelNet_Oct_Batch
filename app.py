import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import os
from utils.realtime_sniffer import start_packet_capture, stop_packet_capture
from utils.model_loader import load_trained_model
from utils.predictions import generate_predictions
from utils.metrics_visualizer import display_prediction_stats

# Page Configuration
st.set_page_config(
    page_title="CyberGuard IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=100) # Assuming logo exists, otherwise it will break gracefully or show broken image
    st.title("CyberGuard")
    st.markdown("### Intelligent Intrusion Detection System")
    st.markdown("---")
    mode = st.radio("Select Operation Mode", ["🛡️ Real-Time Monitoring", "📂 Offline Analysis"])
    st.markdown("---")
    st.info("System Status: **Active**")

# Global State
if "packet_buffer" not in st.session_state:
    st.session_state.packet_buffer = []
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False

buffer_lock = threading.Lock()

def packet_callback(data):
    with buffer_lock:
        st.session_state.packet_buffer.append(data)

# Main Content
st.title("🛡️ CyberGuard Dashboard")

# Model Configuration
MODEL_PATHS = {
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "SVM": "models/svm.pkl"
}
SCALER_PATH = "models/scaler.pkl"

# Load Scaler
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = load_trained_model(SCALER_PATH)

# Feature Columns (Must match training)
FEATURE_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "hot", "logged_in",
    "num_compromised", "root_shell", "num_root", "num_file_creations", 
    "num_shells", "num_access_files", "is_guest_login", "count", "srv_count", 
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate", "protocol_type_encoded", "service_encoded", "flag_encoded"
]

if mode == "🛡️ Real-Time Monitoring":
    st.subheader("Live Network Traffic Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### Control Panel")
        selected_model_name = st.selectbox("Select Detection Model", list(MODEL_PATHS.keys()))
        
        if st.button("▶ Start Monitoring", type="primary"):
            stop_packet_capture()
            st.session_state.packet_buffer = []
            start_packet_capture(packet_callback)
            st.session_state.monitoring_active = True
            st.toast("Monitoring Started", icon="🚀")
            
        if st.button("⏹ Stop Monitoring"):
            stop_packet_capture()
            st.session_state.monitoring_active = False
            st.toast("Monitoring Stopped", icon="🛑")

    with col2:
        st.markdown("### Live Statistics")
        stats_container = st.container()
        
    with col3:
        st.markdown("### Alerts")
        alert_box = st.empty()

    # Real-time Update Loop
    if st.session_state.monitoring_active:
        time.sleep(1) # Refresh rate
        
        # Load Model
        model = load_trained_model(MODEL_PATHS[selected_model_name])
        
        with buffer_lock:
            current_batch = st.session_state.packet_buffer.copy()
            st.session_state.packet_buffer = []
            
        if current_batch:
            df = pd.DataFrame(current_batch)
            # Ensure columns match
            X = df[FEATURE_COLUMNS] if set(FEATURE_COLUMNS).issubset(df.columns) else df.iloc[:, :len(FEATURE_COLUMNS)]
            
            # Scale if needed
            if scaler:
                try:
                    X = scaler.transform(X)
                except:
                    pass # Handle mismatch gracefully
            
            preds = generate_predictions(model, X)
            
            # Update Stats
            total_packets = len(preds)
            threats = np.sum(preds == 1) # Assuming 1 is attack
            normal = total_packets - threats
            
            with stats_container:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Packets", total_packets)
                c2.metric("Normal Traffic", normal, delta_color="normal")
                c3.metric("Threats Detected", int(threats), delta_color="inverse")
                
                display_prediction_stats(preds)
                
            if threats > 0:
                alert_box.error(f"⚠️ {threats} Malicious Packets Detected!")
            else:
                alert_box.success("✅ System Secure")
        else:
            with stats_container:
                st.info("Waiting for traffic...")

elif mode == "📂 Offline Analysis":
    st.subheader("Historical Data Analysis")
    
    uploaded_file = st.file_uploader("Upload Network Log (CSV)", type="csv")
    selected_model_name = st.selectbox("Select Analysis Model", list(MODEL_PATHS.keys()))
    
    if uploaded_file is not None:
        if st.button("Analyze Log"):
            with st.spinner("Analyzing traffic patterns..."):
                df = pd.read_csv(uploaded_file)
                
                # Preprocessing (Simplified for demo)
                # In production, apply same encoding/cleaning as training
                X = df.select_dtypes(include=[np.number]).fillna(0)
                
                # Align columns
                # This is a placeholder; real implementation needs robust alignment
                X = X.iloc[:, :len(FEATURE_COLUMNS)] 
                
                model = load_trained_model(MODEL_PATHS[selected_model_name])
                
                if scaler:
                     X = scaler.transform(X)
                     
                preds = generate_predictions(model, X)
                
                st.success("Analysis Complete")
                
                # Results
                st.markdown("### Analysis Results")
                display_prediction_stats(preds)
                
                results_df = pd.DataFrame({
                    "Packet ID": range(1, len(preds) + 1),
                    "Classification": ["Malicious" if p == 1 else "Normal" for p in preds]
                })
                
                st.dataframe(results_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>CyberGuard IDS v1.0 | Secure Network Monitoring</div>", unsafe_allow_html=True)
