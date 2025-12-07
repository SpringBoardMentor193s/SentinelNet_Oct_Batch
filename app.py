import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from streamlit_autorefresh import st_autorefresh

import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components


# ============================================================
#                 ATTACK GAUGE CHART 🔥
# ============================================================
def draw_attack_gauge(attack_count: int, total_count: int = 1):
    ratio = float(attack_count) / max(total_count, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ratio * 100,
        title={"text": "Attack Intensity (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00ff9d"},
            "steps": [
                {"range": [0, 30], "color": "rgba(0,255,163,0.12)"},
                {"range": [30, 70], "color": "rgba(255,180,0,0.12)"},
                {"range": [70, 100], "color": "rgba(255,0,0,0.12)"},
            ],
        }
    ))

    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#            NETWORK GRAPH (PyVis / Plotly fallback) 🔥
# ============================================================
def show_network_graph_from_logs(df_logs):
    G = nx.Graph()

    if "src" in df_logs.columns and "dst" in df_logs.columns:
        edges = df_logs[["src", "dst"]].dropna().astype(str)
        top = edges.groupby(["src", "dst"]).size().reset_index(name="count")
        top = top.sort_values("count", ascending=False).head(150)

        for _, row in top.iterrows():
            G.add_edge(row["src"], row["dst"])

    try:
        net = Network(height="450px", width="100%", bgcolor="#0b0f17", font_color="#d7f3ff")
        net.from_nx(G)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.save_graph(tmp.name)

        components.html(open(tmp.name, "r", encoding="utf-8").read(), height=480)

    except:
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []

        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                 mode="lines",
                                 line=dict(color="#555", width=1)))

        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                 mode="markers+text",
                                 text=list(G.nodes()),
                                 marker=dict(size=10, color="#00ff9d")))

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
#               LOAD CSS + OPTIONAL ANIMATION JS
# ============================================================
def load_css():
    if os.path.exists("assets/custom.css"):
        with open("assets/custom.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_modal_js():
    if os.path.exists("assets/modal.js"):
        st.markdown("<script src='assets/modal.js'></script>", unsafe_allow_html=True)


# ============================================================
#                     STREAMLIT SETUP
# ============================================================
st.set_page_config(page_title="Cyber-X IDS Dashboard", page_icon="🛡️", layout="wide")

load_css()
load_modal_js()

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")

NSL_MODEL_PATH = os.path.join(ART, "nslkdd_model.pkl")
CICIDS_MODEL_PATH = os.path.join(ART, "cicids_model.pkl")
CICIDS_CSV = os.path.join(ART, "cicids_preprocessed.csv")
REALTIME_LOG = os.path.join(ART, "realtime_log.csv")
PRED_FILE = os.path.join(BASE, "predictions.csv")


# ============================================================
#                   LOAD MODELS
# ============================================================
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

NSL_MODEL = load_model(NSL_MODEL_PATH)
CICIDS_MODEL = load_model(CICIDS_MODEL_PATH)


# ============================================================
#              CYBERPUNK MODAL FOR ATTACK DETAILS
# ============================================================
st.markdown("""
<div id="cyber-overlay" class="cyber-overlay"></div>

<div id="attack-modal" class="cyber-modal">
    <h2 style="color:#00ffbf;">⚠️ Attack Details</h2>
    <div id="attack-modal-body" style="color:#d7f3ff; max-height:350px; overflow:auto;"></div>
    <br>
    <button onclick="closeModal('attack-modal')" 
        style="padding:10px 18px; border:none; border-radius:6px;">
        Close
    </button>
</div>
""", unsafe_allow_html=True)


# ============================================================
#                     SIDEBAR NAVIGATION
# ============================================================
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📤 Upload & Predict", "📡 Real-Time Monitor",
    "🛡️ CICIDS Attack", "📊 Visualization", "📄 PDF Report Export"
])


# ============================================================
#                           HEADER
# ============================================================
st.markdown("""
<h1 style='text-align:center; color:#00ffbf; text-shadow:0px 0px 20px #00ffbf;'>
CYBER-X INTRUSION DETECTION SYSTEM
</h1>
""", unsafe_allow_html=True)


# ============================================================
#                          HOME
# ============================================================
if page == "🏠 Home":
    st.success("Welcome to Cyber-X IDS Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("NSL-KDD Model", "Loaded" if NSL_MODEL else "Missing")
    col2.metric("CICIDS Model", "Loaded" if CICIDS_MODEL else "Missing")
    col3.metric("CICIDS Dataset", "Available" if os.path.exists(CICIDS_CSV) else "Missing")


# ============================================================
#                    UPLOAD & PREDICT
# ============================================================
# ============================================================
#                    UPLOAD & PREDICT (FIXED)
# ============================================================
elif page == "📤 Upload & Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📤 Upload & Predict")

    uploaded = st.file_uploader("Upload CSV file:", type=["csv"])

    # Load CICIDS feature list (78 features)
    try:
        CICIDS_FEATURES = joblib.load("artifacts/cic_features.pkl")
    except:
        CICIDS_FEATURES = []

    # NSL-KDD exact 41-feature signature
    NSL_SIGNATURE = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Remove label-type columns
        for col in ["label", "Label", " Label"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        try:
            preds = None

            # ---------------------------------------
            # 1️⃣ Detect NSL-KDD
            # ---------------------------------------
            if all(col in df.columns for col in NSL_SIGNATURE):
                st.info("Dataset Detected: NSL-KDD (Exact signature matched)")
                df = df[NSL_SIGNATURE]   # reorder
                preds = NSL_MODEL.predict(df)

            # ---------------------------------------
            # 2️⃣ Detect CICIDS dataset
            # ---------------------------------------
            elif len(df.columns) >= 70:
                st.info("Dataset Detected: CICIDS (auto feature padding applied)")

                # Add missing columns
                for col in CICIDS_FEATURES:
                    if col not in df.columns:
                        df[col] = 0

                df = df[CICIDS_FEATURES]   # reorder 78 features
                preds = CICIDS_MODEL.predict(df)

            # ---------------------------------------
            # 3️⃣ Unknown dataset
            # ---------------------------------------
            else:
                st.error("❌ Unrecognized dataset format. Unable to predict.")
                preds = None

            # ---------------------------------------
            # SHOW RESULTS
            # ---------------------------------------
            if preds is not None:
                df["prediction"] = preds
                st.subheader("Predictions (sample)")
                st.dataframe(df.head(), use_container_width=True)

                # Attack count
                attack_count = (df["prediction"] != "normal").sum()

                # Flashing alert box
                if attack_count > 0:
                    st.markdown(f"""
                    <div style="
                        padding: 18px;
                        font-size: 22px;
                        background: rgba(255,0,0,0.2);
                        border-left: 6px solid red;
                        color: white;
                        font-weight: bold;
                        border-radius: 10px;
                        animation: flash 1s infinite;
                    ">
                        ⚠️ {attack_count} Intrusions Detected!
                    </div>

                    <style>
                    @keyframes flash {{
                        0% {{opacity:1;}}
                        50% {{opacity:0.3;}}
                        100% {{opacity:1;}}
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                else:
                    st.success("No attacks detected — all traffic is normal.")

                # Save output
                df.to_csv(PRED_FILE, index=False)
                st.success(f"Predictions saved → {PRED_FILE}")

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#                  REAL-TIME MONITOR
# ============================================================
elif page == "📡 Real-Time Monitor":
    st.title("📡 Live Packet Sniffer")

    st.warning("Run sniffer: `py -3.10 src/realtime_sniffer.py`")

    st_autorefresh(interval=2000)

    if not os.path.exists(REALTIME_LOG):
        st.error("Sniffer log not found.")
    else:
        logs = pd.read_csv(REALTIME_LOG)

        st.subheader("Latest Logs")
        st.dataframe(logs.tail(30), use_container_width=True)

        a = (logs["prediction"] != "normal").sum()
        u = logs["src"].nunique() if "src" in logs else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Packets", len(logs))
        col2.metric("Unique Sources", u)
        col3.metric("Detected Attacks", a)

        draw_attack_gauge(a, total_count=len(logs))

        st.subheader("Network Graph")
        show_network_graph_from_logs(logs.tail(150))


# ============================================================
#                 CICIDS ATTACK ANALYSIS
# ============================================================
elif page == "🛡️ CICIDS Attack":

    st.title("🛡️ CICIDS Attack Analysis")

    if os.path.exists(CICIDS_CSV):
        df = pd.read_csv(CICIDS_CSV, usecols=["Label"], nrows=200000)

        counts = df["Label"].value_counts()
        st.bar_chart(counts)
        st.dataframe(counts)
    else:
        st.error("Dataset missing.")


# ============================================================
#                         VISUALIZATION
# ============================================================
elif page == "📊 Visualization":

    st.title("📊 Visualization Dashboard")

    if os.path.exists(CICIDS_CSV):
        df = pd.read_csv(CICIDS_CSV, nrows=150000)

        st.subheader("Label Distribution")
        st.bar_chart(df["Label"].value_counts())

        numeric = df.select_dtypes(include="number").iloc[:, :15]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    else:
        st.error("Dataset missing.")


# ============================================================
#                         PDF EXPORT
# ============================================================
elif page == "📄 PDF Report Export":

    st.title("📄 Export Report to PDF")

    if not os.path.exists(PRED_FILE):
        st.warning("No prediction file found.")
    else:
        df = pd.read_csv(PRED_FILE)
        st.dataframe(df.head())

        def make_pdf(df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)

            pdf.cell(0, 10, "Prediction Report", ln=1, align="C")
            pdf.ln(5)

            for i in range(min(len(df), 40)):
                pdf.multi_cell(0, 6, str(df.iloc[i].to_dict()))

            pdf.output("report.pdf")

        if st.button("Generate PDF"):
            make_pdf(df)
            with open("report.pdf", "rb") as f:
                st.download_button("Download PDF", f, "report.pdf")
