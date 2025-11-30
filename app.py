import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score

# =========================================================
# SESSION STATE
# =========================================================
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = True

if "cleared" not in st.session_state:
    st.session_state.cleared = False

# =========================================================
# PAGE CONFIG & GLOBAL STYLE
# =========================================================
st.set_page_config(page_title="SentinelNet IDS Dashboard", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #020617;  /* very dark */
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .big-header {
        background: linear-gradient(90deg, #0052D4, #4364F7, #6FB1FC);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.45);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #e3edff);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 8px 18px rgba(15,23,42,0.7);
        text-align: center;
        color: #0f172a;             /* dark text so it’s visible */
    }
    .metric-title {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    .pill {
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-blue { background: #dbeafe; color: #1d4ed8; }
    .pill-amber { background: #fffbeb; color: #92400e; }

    /* Recent alerts styling */
    .alerts-card {
        background: #ffe4e6;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.55rem;
        border: 1px solid #fecdd3;
        color: #7f1d1d;
        font-size: 1rem;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    .alerts-conn {
        font-weight: 700;
        color: #7f1d1d;
        margin-right: 0.6rem;
    }
    .alerts-text {
        opacity: 0.95;
        font-weight: 500;
    }
    .alerts-empty {
        color: #e5e7eb;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .alerts-time {
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .sidebar-title {
        font-weight: 700;
        font-size: 0.95rem;
    }
    .sidebar-section {
        margin-bottom: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER: LOAD + PREPARE KDD ASSETS
# =========================================================
@st.cache_resource
def load_kdd_assets():
    train_df = pd.read_csv("kdd_train.csv")
    test_df = pd.read_csv("kdd_test.csv")

    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    categorical_cols = ["protocol_type", "service", "flag"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        encoders[col] = le
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    train_df["attack_binary"] = train_df["labels"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["attack_binary"] = test_df["labels"].apply(lambda x: 0 if x == "normal" else 1)

    X_test = test_df.drop(columns=["labels", "attack_binary"])
    y_test = test_df["attack_binary"].values

    imputer = joblib.load("kdd_imputer.pkl")
    scaler = joblib.load("kdd_scaler.pkl")

    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)

    model = None
    for fname in ["kdd_decision_tree_model.pkl", "kdd_best_model.pkl"]:
        try:
            model = joblib.load(fname)
            break
        except FileNotFoundError:
        # try next name
            continue
    if model is None:
        raise FileNotFoundError("No KDD model file found.")

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    test_display = test_df.copy()
    test_display["prediction"] = y_pred

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "encoders": encoders,
        "categorical_cols": categorical_cols,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "test_display": test_display,
        "accuracy": acc,
        "precision": prec,
    }

# =========================================================
# HELPER: LOAD + PREPARE CIC-DDoS ASSETS
# =========================================================
@st.cache_resource
def load_ddos_assets():
    df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df.columns = df.columns.str.strip()

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols)

    df = df.replace([np.inf, -np.inf], np.nan)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    skew_vals = df[num_cols].skew()
    for col in num_cols:
        if abs(skew_vals[col]) <= 0.5:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    label_encoder = LabelEncoder()
    df["Label_enc"] = label_encoder.fit_transform(df["Label"])

    y = df["Label_enc"].values
    X = df.drop(columns=["Label", "Label_enc"])

    feature_cols = list(X.columns)
    cat_cols_in_X = X.select_dtypes(include=["object"]).columns.tolist()

    encoders = {}
    for col in cat_cols_in_X:
        le = LabelEncoder()
        le.fit(X[col])
        encoders[col] = le
        X[col] = le.transform(X[col])

    scaler = joblib.load("ddos_scaler.pkl")
    model = joblib.load("ddos_best_model.pkl")

    X_scaled = scaler.transform(X.values)

    y_pred = model.predict(X_scaled)
    true_labels = label_encoder.inverse_transform(y)
    pred_labels = label_encoder.inverse_transform(y_pred)

    true_binary = np.array([0 if "BENIGN" in lab.upper() else 1 for lab in true_labels])
    pred_binary = np.array([0 if "BENIGN" in lab.upper() else 1 for lab in pred_labels])

    acc = accuracy_score(true_binary, pred_binary)
    prec = precision_score(true_binary, pred_binary, zero_division=0)

    df_display = df.copy()
    df_display["pred_label"] = pred_labels

    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols_in_X,
        "X_scaled": X_scaled,
        "y_true": true_binary,
        "df_display": df_display,
        "accuracy": acc,
        "precision": prec,
    }

# =========================================================
# DATASET SELECTOR & LOAD
# =========================================================
DATASET_OPTIONS = {
    "KDD (NSL-KDD)": "kdd",
    "CIC-DDoS (CIC-IDS2017)": "ddos"
}

# =========================================================
# SIDEBAR UI
# =========================================================
st.sidebar.markdown("<div class='sidebar-title'>⚙ Configuration</div>", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Detection Mode",
    options=["Live Monitoring", "File Analysis"],
    index=0
)

st.sidebar.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-title'>📊 Dataset</div>", unsafe_allow_html=True)
dataset_label = st.sidebar.selectbox("Select Dataset", list(DATASET_OPTIONS.keys()))
dataset_key = DATASET_OPTIONS[dataset_label]

st.sidebar.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-title'>🧠 Algorithm</div>", unsafe_allow_html=True)

# Algorithm lists
kdd_algorithms = [
    "Decision Tree (Recommended)",
    "Random Forest",
    "Logistic Regression",
    "Gradient Boosting",
    "SVM",
    "KNN",
    "Gaussian NB"
]

ddos_algorithms = [
    "Random Forest (Recommended)",
    "Gradient Boosting",
    "Decision Tree",
    "Logistic Regression",
    "SVM"
]

if dataset_key == "kdd":
    algo = st.sidebar.selectbox("Select Algorithm", kdd_algorithms)
else:
    algo = st.sidebar.selectbox("Select Algorithm", ddos_algorithms)

# Load assets for selected dataset
if dataset_key == "kdd":
    kdd_assets = load_kdd_assets()
    acc = kdd_assets["accuracy"]
    prec = kdd_assets["precision"]
    total_packets = len(kdd_assets["test_display"])
    intrusions = int((kdd_assets["test_display"]["prediction"] == 1).sum())
    normals = total_packets - intrusions
else:
    ddos_assets = load_ddos_assets()
    acc = ddos_assets["accuracy"]
    prec = ddos_assets["precision"]
    total_packets = len(ddos_assets["df_display"])
    pred_labels = ddos_assets["df_display"]["pred_label"]
    benign_mask = np.array([("BENIGN" in lab.upper()) for lab in pred_labels])
    intrusions = int((~benign_mask).sum())
    normals = int(benign_mask.sum())

intrusion_rate = (intrusions / total_packets * 100) if total_packets > 0 else 0.0

# Apply "Clear Data" effect
if st.session_state.cleared:
    total_packets = 0
    intrusions = 0
    normals = 0
    intrusion_rate = 0.0

st.sidebar.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-title'>📈 Model Details</div>", unsafe_allow_html=True)
st.sidebar.metric("Accuracy", f"{acc*100:.1f}%")
st.sidebar.metric("Precision", f"{prec*100:.1f}%")

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown(f"""
<div class="big-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:1.9rem; font-weight:800; letter-spacing:0.03em;">SentinelNet</div>
            <div style="font-size:0.9rem; opacity:0.85; margin-top:0.15rem;">
                Unified AI-Powered Network Intrusion Detection Dashboard
            </div>
        </div>
        <div style="font-size:0.85rem; text-align:right; opacity:0.9;">
            Dataset: <b>{dataset_label}</b><br/>
            Mode: <b>{mode}</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP CONTROLS & METRICS
# =========================================================
st.subheader("🌐 Live Network Monitoring" if mode == "Live Monitoring" else "📁 File-Based Analysis")

btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 2])

# 1) Stop / Start monitoring
with btn_col1:
    stop_label = "⏹ Stop Monitoring" if st.session_state.monitoring_active else "▶ Start Monitoring"
    stop_clicked = st.button(stop_label, key="monitor_btn", use_container_width=True)
    if stop_clicked:
        st.session_state.monitoring_active = not st.session_state.monitoring_active
        if st.session_state.monitoring_active:
            st.success("Monitoring resumed.")
        else:
            st.warning("Monitoring paused.")

# 2) Clear data
with btn_col2:
    clear_clicked = st.button("🧹 Clear Data", key="clear_btn", use_container_width=True)
    if clear_clicked:
        st.session_state.cleared = True
        st.info("Dashboard data cleared for this session.")

# 3) Export summary
with btn_col3:
    summary_df = pd.DataFrame([{
        "Dataset": dataset_label,
        "Mode": mode,
        "Algorithm": algo,
        "Total_Packets": total_packets,
        "Intrusions": intrusions,
        "Normal": normals,
        "Intrusion_Rate_percent": intrusion_rate,
        "Accuracy_percent": acc * 100,
        "Precision_percent": prec * 100
    }])
    csv_data = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📤 Export Summary",
        data=csv_data,
        file_name="sentinelnet_summary.csv",
        mime="text/csv",
        use_container_width=True,
        key="export_btn"
    )

st.markdown("")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        "<div class='metric-card'>"
        "<div class='metric-title'>Total Packets</div>"
        f"<div class='metric-value'>{total_packets}</div>"
        "<div class='metric-sub'>Analyzed in current session</div>"
        "</div>",
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        "<div class='metric-card'>"
        "<div class='metric-title'>Intrusions</div>"
        f"<div class='metric-value'>{intrusions}</div>"
        "<div class='metric-sub'>Predicted malicious connections</div>"
        "</div>",
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        "<div class='metric-card'>"
        "<div class='metric-title'>Normal</div>"
        f"<div class='metric-value'>{normals}</div>"
        "<div class='metric-sub'>Predicted safe connections</div>"
        "</div>",
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        "<div class='metric-card'>"
        "<div class='metric-title'>Intrusion Rate</div>"
        f"<div class='metric-value'>{intrusion_rate:.1f}%</div>"
        "<div class='metric-sub'>Intrusions / Total packets</div>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("")
monitor_text = "● Monitoring Active" if st.session_state.monitoring_active else "⏸ Monitoring Paused"
monitor_class = "pill-green" if st.session_state.monitoring_active else "pill-amber"

st.markdown(
    f"<span class='pill {monitor_class}'>{monitor_text}</span>"
    f"<span class='pill pill-blue'>Using: {algo}</span>"
    f"<span class='pill pill-amber'>Stats: {intrusions} intrusions ({intrusion_rate:.1f}%)</span>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# RECENT ALERTS SECTION
# =========================================================
st.subheader("⚠ Recent Alerts")

if st.session_state.cleared:
    st.markdown("<div class='alerts-empty'>Alerts cleared for this session.</div>", unsafe_allow_html=True)
else:
    if dataset_key == "kdd":
        df_alerts = kdd_assets["test_display"]
        df_alerts = df_alerts[df_alerts["prediction"] == 1].copy().head(8)
        if df_alerts.empty:
            st.markdown("<div class='alerts-empty'>No intrusions detected in the sampled KDD test data.</div>",
                        unsafe_allow_html=True)
        else:
            for idx, row in df_alerts.iterrows():
                conn_id = idx
                proto = row["protocol_type"]
                service = row.get("service", "")
                alert_msg = f"Attack detected – protocol {proto}, service {service}"
                st.markdown(
                    f"""
                    <div class='alerts-card'>
                        <span class='alerts-conn'>Conn #{conn_id}</span>
                        <span class='alerts-text'>{alert_msg}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        df_alerts = ddos_assets["df_display"]
        df_attack = df_alerts[~df_alerts["pred_label"].str.upper().str.contains("BENIGN")].copy()
        df_attack = df_attack.head(8)

        if df_attack.empty:
            st.markdown("<div class='alerts-empty'>No intrusions detected in the sampled CIC-DDoS data.</div>",
                        unsafe_allow_html=True)
        else:
            for idx, row in df_attack.iterrows():
                conn_id = idx
                label = row["pred_label"]
                alert_msg = f"{label} traffic detected."
                st.markdown(
                    f"""
                    <div class='alerts-card'>
                        <span class='alerts-conn'>Flow #{conn_id}</span>
                        <span class='alerts-text'>{alert_msg}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
