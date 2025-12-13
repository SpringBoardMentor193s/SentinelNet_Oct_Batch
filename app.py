import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# ============================================================
#  LOAD CSS (NEON THEME)
# ============================================================
def load_css():
    css_path = os.path.join("assets", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ============================================================
#  NSL-KDD LABEL GROUPING (IMPORTANT)
# ============================================================
def map_nsl_label(label):
    label = str(label).lower()

    dos = ["neptune", "smurf", "teardrop", "pod", "land", "back",
           "apache2", "udpstorm", "processtable", "mailbomb"]

    probe = ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]

    r2l = ["guess_passwd", "ftp_write", "imap", "phf", "multihop",
           "warezmaster", "warezclient", "spy", "xlock", "xsnoop", "snmpguess"]

    u2r = ["buffer_overflow", "rootkit", "loadmodule", "perl",
           "sqlattack", "xterm", "ps", "httptunnel"]

    if label in dos:
        return "DoS"
    elif label in probe:
        return "Probe"
    elif label in r2l:
        return "R2L"
    elif label in u2r:
        return "U2R"
    else:
        return "normal"


# ============================================================
#  STREAMLIT SETUP
# ============================================================
st.set_page_config(page_title="Cyber-X IDS Dashboard",
                   page_icon="🛡️", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")

NSL_MODEL_PATH = os.path.join(ART, "nslkdd_grouped_model.pkl")
NSL_ENCODER_PATH = os.path.join(ART, "nsl_label_encoder.pkl")

CIC_MODEL_PATH = os.path.join(ART, "cicids_model.pkl")
CIC_FEATURE_PATH = os.path.join(ART, "cic_features.pkl")

NSL_TEST_PATH = os.path.join(ART, "nslkdd_test.csv")
CIC_TEST_PATH = os.path.join(ART, "cicids_sample_test.csv")

REALTIME_LOG = os.path.join(ART, "realtime_log.csv")
PRED_FILE = os.path.join(BASE, "predictions.csv")


# ============================================================
#  LOAD MODELS
# ============================================================
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

NSL_MODEL = load_model(NSL_MODEL_PATH)
NSL_ENCODER = load_model(NSL_ENCODER_PATH)
CIC_MODEL = load_model(CIC_MODEL_PATH)


# ============================================================
#  Gauge Chart
# ============================================================
def draw_attack_gauge(attack_count, total_count):
    ratio = attack_count / max(total_count, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ratio * 100,
        title={"text": "Attack Intensity (%)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "cyan"}}
    ))
    st.plotly_chart(fig)


# ============================================================
#  SIDEBAR NAVIGATION
# ============================================================
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📤 Upload & Predict",
    "🛡️ CICIDS Attack",
    "📡 Real-Time Monitor",
    "📊 Visualization",
    "📄 PDF Report Export"
])


# ============================================================
#  HOME PAGE
# ============================================================
# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <h1 style='text-align:center; color:#00ffee; text-shadow:0px 0px 20px #00ffff;'>
    AI NETWORK INTRUSION DETECTION SYSTEM
    </h1>
    """, unsafe_allow_html=True)

    st.header("🔷 NSL-KDD (5-Class Model) Evaluation")

    try:
        df = pd.read_csv(NSL_TEST_PATH)

        # Remove unwanted column
        df = df.drop(columns=["attack_type"], errors="ignore")

        # Extract labels
        if "label" in df.columns:
            y_true_raw = df["label"]
            df = df.drop(columns=["label"])
        else:
            y_true_raw = df.iloc[:, -1]
            df = df.iloc[:, :-1]

        # Group labels
        y_true = y_true_raw.apply(map_nsl_label)

        # Encode labels
        y_encoded = NSL_ENCODER.transform(y_true)

        # Features
        X = df.fillna(0)

        # Predict
        pred_encoded = NSL_MODEL.predict(X)
        pred = NSL_ENCODER.inverse_transform(pred_encoded)

        # Accuracy
        acc = accuracy_score(y_true, pred)
        st.metric("NSL-KDD Accuracy", f"{acc * 100:.2f}%")

        # Compact Confusion Matrix
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(confusion_matrix(y_true, pred), cmap="Blues", cbar=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

        with st.expander("📄 Classification Report"):
            st.text(classification_report(y_true, pred))

    except Exception as e:
        st.error(f"NSL Evaluation Error: {e}")


    # =======================================================
    # CICIDS Evaluation
    # =======================================================
    st.header("🟥 CICIDS Evaluation")

    try:
        df = pd.read_csv(CIC_TEST_PATH)
        FEATURES = joblib.load(CIC_FEATURE_PATH)

        y = df["Label"]
        X = df.drop(columns=["Label"], errors="ignore")

        # Add missing CICIDS columns
        for f in FEATURES:
            if f not in X.columns:
                X[f] = 0

        X = X[FEATURES]

        pred = CIC_MODEL.predict(X)
        acc = accuracy_score(y, pred)

        st.metric("CICIDS Accuracy", f"{acc * 100:.2f}%")

        # Compact CICIDS Confusion Matrix
        fig, ax = plt.subplots(figsize=(2.5, 2))
        sns.heatmap(confusion_matrix(y, pred), cmap="Reds", cbar=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

        with st.expander("📄 CICIDS Classification Report"):
            st.text(classification_report(y, pred))

    except Exception as e:
        st.error(f"CICIDS Error: {e}")


# ============================================================
# UPLOAD & PREDICT
# ============================================================
elif page == "📤 Upload & Predict":

    st.title("📤 Upload & Predict")
    file = st.file_uploader("Upload CSV:", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df = df.drop(columns=["label", "Label"], errors="ignore")

        try:
            # NSL dataset (41 features)
            if len(df.columns) == 41:
                st.info("Detected NSL-KDD Dataset")

                X = df.fillna(0)
                pred_enc = NSL_MODEL.predict(X)
                pred = NSL_ENCODER.inverse_transform(pred_enc)

                df["prediction"] = pred
                st.dataframe(df.head())

                df.to_csv(PRED_FILE, index=False)
                st.success("Predictions Saved!")

            # CICIDS dataset
            elif len(df.columns) >= 70:
                FEATURES = joblib.load(CIC_FEATURE_PATH)

                for f in FEATURES:
                    if f not in df.columns:
                        df[f] = 0

                df = df[FEATURES]
                pred = CIC_MODEL.predict(df)

                df["prediction"] = pred
                st.dataframe(df.head())
                df.to_csv(PRED_FILE, index=False)
                st.success("CICIDS Predictions Saved!")

            else:
                st.error("Unknown dataset format")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
# ============================================================
# CICIDS ATTACK PAGE
# ============================================================
elif page == "🛡️ CICIDS Attack":
    st.title("🛡️ CICIDS Attack Exploration")

    if os.path.exists(CIC_TEST_PATH):
        df = pd.read_csv(CIC_TEST_PATH)

        st.subheader("Attack Distribution")
        st.bar_chart(df["Label"].value_counts())

        st.subheader("Sample Attack Records")
        st.dataframe(df.head(20))

    else:
        st.error("CICIDS sample test file not found. Run training or add cicids_sample_test.csv")


# ============================================================
# REAL-TIME MONITOR
# ============================================================
elif page == "📡 Real-Time Monitor":
    st.title("📡 Live Packet Sniffer")
    st.warning("Run this command:\n`py -3.10 src/realtime_sniffer.py`")

    st_autorefresh(interval=2500)

    if os.path.exists(REALTIME_LOG):
        logs = pd.read_csv(REALTIME_LOG)
        st.dataframe(logs.tail(20))

        attack_count = (logs["prediction"] != "normal").sum()
        draw_attack_gauge(attack_count, len(logs))


# ============================================================
# VISUALIZATION
# ============================================================
elif page == "📊 Visualization":
    st.title("📊 Visualization Dashboard")

    if os.path.exists(CIC_TEST_PATH):
        df = pd.read_csv(CIC_TEST_PATH)
        st.bar_chart(df["Label"].value_counts())


# ============================================================
# PDF EXPORT
# ============================================================
elif page == "📄 PDF Report Export":

    st.title("📄 Export Prediction Report")

    if os.path.exists(PRED_FILE):
        df = pd.read_csv(PRED_FILE)

        def make_pdf(df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, "Prediction Report", ln=1, align="C")

            for i in range(min(40, len(df))):
                pdf.multi_cell(0, 6, str(df.iloc[i].to_dict()))

            pdf.output("report.pdf")

        if st.button("Generate PDF"):
            make_pdf(df)
            st.download_button("Download PDF",
                               open("report.pdf", "rb"), "report.pdf")
