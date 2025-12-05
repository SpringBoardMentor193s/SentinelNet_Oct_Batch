import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="SentinelNet IDS", layout="wide")

# -----------------------------------------------------------
# CSS THEME
# -----------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #060A19;
    color: #E8ECF2;
    font-family: 'Segoe UI', sans-serif;
}

/* Glowing Header */
h1 {
    color: #7EC8FF !important;
    text-shadow: 0 0 12px #1E90FF;
    animation: glowPulse 2.5s infinite alternate;
}

@keyframes glowPulse {
    from { text-shadow: 0 0 4px #1E90FF; }
    to   { text-shadow: 0 0 18px #61DBFB; }
}

/* Section Titles */
h2, h3, h4 {
    color: #9BD1FF !important;
    text-shadow: 0 0 6px #2271B3;
}

/* Main container */
.block-container {
    background-color: #0B1027;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 0 20px #12172E;
}

/* Buttons */
.stButton>button {
    background-color: #1E90FF !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    font-size: 1rem;
    box-shadow: 0 0 10px #1E90FF;
    transition: 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #5AB3FF !important;
    box-shadow: 0 0 18px #61DBFB;
}

/* Table stylings */
thead tr th {
    background-color: #1A2038 !important;
    color: #A6CEFF !important;
}
tbody tr {
    background-color: #10162E !important;
    color: #D6E4FF !important;
}

/* Metrics glowing cards */
[data-testid="stMetric"] {
    background-color: #151B36;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #1E90FF33;
    box-shadow: 0 0 12px #1E90FF33;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
MODELS_DIR = "models"

# Binary model files (your existing names)
BINARY_MODEL_FILES = {
    "Logistic Regression": "nslkdd_binary_logistic_regression.pkl",
    "Decision Tree": "nslkdd_binary_decision_tree_classifier.pkl",
    "Random Forest": "nslkdd_binary_random_forest.pkl",
    "Gradient Boosting": "nslkdd_binary_gradient_boosting.pkl",
    "KNN": "nslkdd_binary_knn.pkl",
    "SVM (RBF)": "nslkdd_binary_svm__rbf_.pkl",
    "XGBoost": "nslkdd_binary_xgboost.pkl"
}

# Multiclass model files (5-class: dos, normal, probe, r2l, u2r)
MULTI_MODEL_FILES = {
    "Decision Tree": "nslkdd_multi_decision_tree.pkl",
    "Random Forest": "nslkdd_multi_random_forest.pkl",
    "Extra Trees": "nslkdd_multi_extra_trees.pkl",
    "Naive Bayes": "nslkdd_multi_naive_bayes.pkl",
    "Logistic Regression": "nslkdd_multi_logistic_regression.pkl"
}

# Load scalers + feature lists
binary_scaler = joblib.load(f"{MODELS_DIR}/nslkdd_scaler.pkl")
binary_features = [str(c) for c in joblib.load(f"{MODELS_DIR}/nslkdd_binary_features.pkl")]

multi_scaler = joblib.load(f"{MODELS_DIR}/nslkdd_multi_scaler.pkl")
multi_features = [str(c) for c in joblib.load(f"{MODELS_DIR}/nslkdd_multi_features.pkl")]

LABEL_COL = 41
DIFFICULTY_COL = 42

# NSL-KDD column names (43-col)
NSL_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

# -----------------------------------------------------------
# Multiclass mapping (supports both numeric + string labels)
# -----------------------------------------------------------

# String attack → 5-class
dos_attacks = [
    "back","land","neptune","pod","smurf","teardrop",
    "mailbomb","processtable","udpstorm","apache2","worm"
]
probe_attacks = [
    "satan","ipsweep","nmap","portsweep","mscan","saint"
]
r2l_attacks = [
    "guess_passwd","ftp_write","imap","phf","multihop","warezmaster",
    "warezclient","spy","xlock","xsnoop","snmpguess","snmpgetattack",
    "httptunnel","sendmail","named"
]
u2r_attacks = [
    "buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps"
]

def map_string_label(v: str):
    v = str(v).strip().lower().replace(".", "")
    if v == "normal":
        return "normal"
    if v in dos_attacks:
        return "dos"
    if v in probe_attacks:
        return "probe"
    if v in r2l_attacks:
        return "r2l"
    if v in u2r_attacks:
        return "u2r"
    return "unknown"

# Old NSL numeric → 5-class
numeric_to_5class = {
    0:"normal", 14:"normal", 16:"normal", 21:"normal",

    # DOS
    1:"dos", 2:"dos", 3:"dos", 4:"dos", 5:"dos",
    6:"dos", 7:"dos", 8:"dos", 9:"dos", 10:"dos",
    15:"dos", 18:"dos", 34:"dos",

    # PROBE
    11:"probe", 12:"probe", 13:"probe", 17:"probe",
    22:"probe", 23:"probe", 24:"probe", 27:"probe",
    33:"probe", 36:"probe", 37:"probe", 39:"probe",

    # R2L
    19:"r2l", 20:"r2l", 26:"r2l", 38:"r2l", 35:"r2l",

    # U2R
    25:"u2r", 28:"u2r", 29:"u2r", 30:"u2r",
    31:"u2r", 32:"u2r"
}

# 0..4 encoding → 5-class (NSL-KDD 5-class version)
INT_TO_CLASS = {
    0: "normal",
    1: "dos",
    2: "probe",
    3: "r2l",
    4: "u2r"
}

def to_main_class(label):
    """
    Convert ANY label format into one of:
    'dos', 'normal', 'probe', 'r2l', 'u2r', or 'unknown'.

    Handles:
    - 0..4 numeric (5-class NSL)
    - NSL original numeric IDs (0,1,2,...,39)
    - String attack names: 'neptune', 'smurf', 'normal', etc.
    """
    s = str(label).strip().lower()

    # numeric style
    if s.isdigit():
        n = int(s)
        if n in INT_TO_CLASS:
            return INT_TO_CLASS[n]
        if n in numeric_to_5class:
            return numeric_to_5class[n]
        return "unknown"

    # string attack / normal
    return map_string_label(s)

MULTI_CLASSES = ["dos", "normal", "probe", "r2l", "u2r"]
MULTI_CLASS_TO_ID = {c: i for i, c in enumerate(MULTI_CLASSES)}

# -----------------------------------------------------------
# PREPROCESS: BINARY
# -----------------------------------------------------------
def preprocess_binary(df_raw: pd.DataFrame):
    """
    Preprocessing for binary models (KDDTrain+/KDDTest+ style):
    - label at col 41
    - optional difficulty at col 42
    - sparse dummies to avoid memory issues
    """
    df = df_raw.copy()

    # labels from column 41
    y_raw = df.iloc[:, LABEL_COL].astype(str)

    # drop label + optional difficulty
    df = df.drop(columns=[LABEL_COL], errors="ignore")
    if df.shape[1] > DIFFICULTY_COL:
        df = df.drop(columns=[DIFFICULTY_COL], errors="ignore")

    df.columns = df.columns.astype(str)

    # encode categorical with SPARSE dummies
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True,
        sparse=True
    )
    df_encoded.columns = df_encoded.columns.astype(str)

    # ensure all training features exist
    for col in binary_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # keep only features used during training
    df_encoded = df_encoded[binary_features].astype(float)

    # scale
    X_scaled = binary_scaler.transform(df_encoded)

    return X_scaled, y_raw, df_raw

# -----------------------------------------------------------
# PREPROCESS: MULTICLASS (NO get_dummies)
# -----------------------------------------------------------
# -----------------------------------------------------------
# SERVICE → INTEGER ID MAP (70 services)
# -----------------------------------------------------------
SERVICE_MAP = {
    "IRC": 0,
    "X11": 1,
    "Z39_50": 2,
    "aol": 3,
    "auth": 4,
    "bgp": 5,
    "courier": 6,
    "csnet_ns": 7,
    "ctf": 8,
    "daytime": 9,
    "discard": 10,
    "domain": 11,
    "domain_u": 12,
    "echo": 13,
    "eco_i": 14,
    "ecr_i": 15,
    "efs": 16,
    "exec": 17,
    "finger": 18,
    "ftp": 19,
    "ftp_data": 20,
    "gopher": 21,
    "harvest": 22,
    "hostnames": 23,
    "http": 24,
    "http_2784": 25,
    "http_443": 26,
    "http_8001": 27,
    "imap4": 28,
    "iso_tsap": 29,
    "klogin": 30,
    "kshell": 31,
    "ldap": 32,
    "link": 33,
    "login": 34,
    "mtp": 35,
    "name": 36,
    "netbios_dgm": 37,
    "netbios_ns": 38,
    "netbios_ssn": 39,
    "netstat": 40,
    "nnsp": 41,
    "nntp": 42,
    "ntp_u": 43,
    "other": 44,
    "pm_dump": 45,
    "pop_2": 46,
    "pop_3": 47,
    "printer": 48,
    "private": 49,
    "red_i": 50,
    "remote_job": 51,
    "rje": 52,
    "shell": 53,
    "smtp": 54,
    "sql_net": 55,
    "ssh": 56,
    "sunrpc": 57,
    "supdup": 58,
    "systat": 59,
    "telnet": 60,
    "tftp_u": 61,
    "tim_i": 62,
    "time": 63,
    "urh_i": 64,
    "urp_i": 65,
    "uucp": 66,
    "uucp_path": 67,
    "vmnet": 68,
    "whois": 69
}
# Base NSL-KDD 41 feature names (same as your training notebook)
base_features = [
 'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
 'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
 'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
 'num_shells','num_access_files','num_outbound_cmds','is_host_login',
 'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
 'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
 'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
 'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
 'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
 'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]
# Numeric → 5-class mapping used during training
label_to_5class = {}

# NORMAL
label_to_5class[0] = "normal"

# DOS
for x in [1,2,3,4,5,6,7,8,9,10,11]:
    label_to_5class[x] = "dos"

# PROBE
for x in [12,13,14,15]:
    label_to_5class[x] = "probe"

# R2L
for x in [16,17,18,19,20,21,22,23,24,25,26,27]:
    label_to_5class[x] = "r2l"

# U2R
for x in [28,29,30,31,32,33,34,35,36,37,38,39]:
    label_to_5class[x] = "u2r"


def preprocess_multiclass(df_raw):
    df = df_raw.copy()

    # Fix columns (your notebook used only 42 + 1 label)
    if df.shape[1] == 42:
        df.columns = base_features + ["label"]
    else:
        df = df.iloc[:, :43]
        df.columns = base_features + ["label"]

    # Convert label to numeric (same as notebook)
    df["label_num"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label_num"]).reset_index(drop=True)
    df["label_num"] = df["label_num"].astype(int)

    # Map to 5 classes
    df["target"] = df["label_num"].map(label_to_5class)

    # ---------- INPUT FEATURES ----------
    X = df.drop(columns=["label", "label_num", "target"])

    cat_cols = ["protocol_type", "service", "flag"]

    # One-hot categorical
    cat_test = pd.get_dummies(X[cat_cols], prefix=cat_cols)
    cat_test = cat_test.loc[:, ~cat_test.columns.duplicated()]

    # Align with training
    all_cat_cols = [
        c for c in multi_features
        if any(c.startswith(p) for p in ["protocol_type_", "service_", "flag_"])
    ]
    cat_test = cat_test.reindex(columns=all_cat_cols, fill_value=0)

    # Numeric portion
    num_test = X.drop(columns=cat_cols).astype(float)

    # Combine in correct order
    X_final = pd.concat(
        [num_test.reset_index(drop=True), cat_test.reset_index(drop=True)],
        axis=1
    )

    # Add missing training columns
    for col in multi_features:
        if col not in X_final.columns:
            X_final[col] = 0

    # Reorder
    X_final = X_final[multi_features].astype(float)

    # Scale using training scaler
    X_scaled = multi_scaler.transform(X_final)

    # RETURN 3 VALUES — THE FIX
    return X_scaled, df["target"].values, df
# UI
# -----------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>⚡ SentinelNet IDS — Binary + Multiclass ⚡</h1>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload NSL-KDD / KDD CSV file", type=["csv"])

mode = st.sidebar.selectbox(
    "Select Task",
    ["Binary Classification", "Multiclass Classification"]
)

if uploaded is not None:
    # clean read, no python engine nonsense
    df_raw = pd.read_csv(
        uploaded,
        header=None,
        low_memory=False
    )

    st.success(f"Loaded: {uploaded.name} | Shape: {df_raw.shape}")
    st.dataframe(df_raw.head(10), width="stretch")

    st.subheader("⚙ Classification Mode")
    mode = st.radio(
        "Select Mode:",
        ["Binary Classification", "Multiclass Classification"],
        horizontal=True
    )

    # -------------------------- BINARY -------------------------
    if mode == "Binary Classification":
        X, y_raw, base_df = preprocess_binary(df_raw)
        y_true = np.array([0 if v.lower() == "normal" else 1 for v in y_raw])

        model_name = st.selectbox("Select Binary Model", list(BINARY_MODEL_FILES.keys()))
        model = joblib.load(f"{MODELS_DIR}/{BINARY_MODEL_FILES[model_name]}")

        y_pred = model.predict(X)
        try:
            y_score = model.predict_proba(X)[:, 1]
        except Exception:
            y_score = None

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        st.subheader("📊 Binary Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc*100:.2f}%")
        c2.metric("Precision", f"{prec*100:.2f}%")
        c3.metric("Recall", f"{rec*100:.2f}%")
        c4.metric("F1 Score", f"{f1*100:.2f}%")

        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["normal", "attack"],
            y=["normal", "attack"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}"
        ))
        fig.update_layout(title="Confusion Matrix — Binary")
        st.plotly_chart(fig, width="stretch")

        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"AUC={auc_val:.3f}"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dash")))
            roc_fig.update_layout(title="ROC Curve — Binary",
                                  xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(roc_fig, width="stretch")

    # -------------------------- MULTICLASS -------------------------
    # -------------------------- MULTICLASS -------------------------
    # -------------------------- MULTICLASS -------------------------
# -------------------------- MULTICLASS -------------------------
if mode == "Multiclass Classification" and uploaded is not None:

    # 1) Preprocess
    X, y_true_raw, df_named = preprocess_multiclass(df_raw)

    # 2) Load model
    model_name = st.selectbox("Select Multiclass Model", list(MULTI_MODEL_FILES.keys()))
    model = joblib.load(f"{MODELS_DIR}/{MULTI_MODEL_FILES[model_name]}")

    # 3) True labels
    y_true = np.array(y_true_raw, dtype=str)

    # 4) Predictions
    raw_pred = model.predict(X)
    y_pred = np.array(raw_pred, dtype=str)

    # 5) Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    st.subheader("📊 Multiclass Metrics (5-class)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall", f"{rec*100:.2f}%")
    c4.metric("F1 Score", f"{f1*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred,
                          labels=["normal", "dos", "probe", "r2l", "u2r"])

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["normal", "dos", "probe", "r2l", "u2r"],
        y=["normal", "dos", "probe", "r2l", "u2r"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}"
    ))
    fig.update_layout(title="Confusion Matrix — Multiclass (5-Class)")
    st.plotly_chart(fig, width="stretch")

    # ---------------- ROC CURVE (Inside the block!) ----------------
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)

            CLASS_ORDER = ["normal", "dos", "probe", "r2l", "u2r"]
            y_true_ids = np.array([CLASS_ORDER.index(v) for v in y_true])

            fig_roc = go.Figure()

            for i, cls in enumerate(CLASS_ORDER):
                fpr, tpr, _ = roc_curve((y_true_ids == i).astype(int), y_proba[:, i])
                roc_auc = auc(fpr, tpr)

                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{cls} (AUC={roc_auc:.3f})"
                ))

            fig_roc.update_layout(
                title="ROC Curve — Multiclass (One-vs-Rest)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        else:
            st.info("Selected model does not support probability outputs (predict_proba).")

    except Exception as e:
        st.error(f"ROC Curve could not be generated: {e}")
