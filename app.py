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
def preprocess_multiclass(df_raw: pd.DataFrame):
    """
    Preprocessing for NSLKDDTest.csv (5-class numeric labels 0..4):
    - 42 or 43 columns
    - label column is numeric 0..4
    - output labels as strings: dos, normal, probe, r2l, u2r
    """
    df = df_raw.copy()

    # Fix shape: if 42 cols, add dummy difficulty
    if df.shape[1] == 42:
        df[42] = 0
    elif df.shape[1] != 43:
        raise ValueError(f"Expected 42 or 43 columns, got {df.shape[1]}")

    # Proper column names
    df.columns = NSL_COLUMNS

    # Remove accidental header row
    if not str(df.loc[0, "duration"]).replace(".", "", 1).isdigit():
        df = df.iloc[1:].reset_index(drop=True)

    # ---- LABELS: numeric 0..4 → 5-class strings ----
    y_raw_numeric = (
    df["label"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .astype(int)
)

    y_labels = y_raw_numeric.map(INT_TO_CLASS)   # normal/dos/probe/r2l/u2r

    # ---- FEATURES ----
    cat_cols = ["protocol_type", "service", "flag"]
    drop_cols = cat_cols + ["label", "difficulty"]
    numeric_cols = [c for c in df.columns if c not in drop_cols]

    numeric_part = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Manual one-hot encoding based ONLY on training feature list
    df_cat = pd.DataFrame(index=df.index)

    for col in cat_cols:
        prefix = col + "_"
        available_features = [f for f in multi_features if f.startswith(prefix)]
        values = df[col].astype(str)

        for feat in available_features:
            cat_value = feat[len(prefix):]
            df_cat[feat] = (values == cat_value).astype(int)

    full = pd.concat([numeric_part, df_cat], axis=1)

    # Ensure all model features exist
    for col in multi_features:
        if col not in full.columns:
            full[col] = 0

    full = full[multi_features].astype(float)

    # Scale
    X_scaled = multi_scaler.transform(full)

    # Return X + string labels
    return X_scaled, y_labels.values, df
# -----------------------------------------------------------
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
    if mode == "Multiclass Classification":
        X, y_true, df_named = preprocess_multiclass(df_raw)  # y_true are strings

        model_name = st.selectbox("Select Multiclass Model", list(MULTI_MODEL_FILES.keys()))
        model = joblib.load(f"{MODELS_DIR}/{MULTI_MODEL_FILES[model_name]}")

        # model predictions (normally 0..4 integers)
        raw_pred = model.predict(X)

        # Map predictions to 5-class strings
        y_pred = []
        for v in raw_pred:
            if isinstance(v, (int, np.integer, np.int64, np.int32)):
                y_pred.append(INT_TO_CLASS.get(int(v), "unknown"))
            else:
                # if model somehow outputs string labels (rare), just pass through
                y_pred.append(str(v).lower())

        y_pred = np.array(y_pred)

        # We expect no 'unknown' for NSLKDDTest; but just in case:
        valid_mask = y_pred != "unknown"
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            st.error("All predictions mapped to 'unknown'. Cannot compute metrics.")
        else:
            # METRICS
            acc = accuracy_score(y_true_valid, y_pred_valid)
            prec = precision_score(y_true_valid, y_pred_valid,
                                   average="weighted", zero_division=0)
            rec = recall_score(y_true_valid, y_pred_valid,
                               average="weighted", zero_division=0)
            f1 = f1_score(y_true_valid, y_pred_valid,
                          average="weighted", zero_division=0)

            st.subheader("📊 Multiclass Metrics (5-class)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc*100:.2f}%")
            c2.metric("Precision", f"{prec*100:.2f}%")
            c3.metric("Recall", f"{rec*100:.2f}%")
            c4.metric("F1 Score", f"{f1*100:.2f}%")

            # CONFUSION MATRIX in fixed class order
            cm = confusion_matrix(
                y_true_valid,
                y_pred_valid,
                labels=MULTI_CLASSES
            )

            fig = go.Figure(go.Heatmap(
                z=cm,
                x=MULTI_CLASSES,
                y=MULTI_CLASSES,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}"
            ))
            fig.update_layout(title="Confusion Matrix — Multiclass (5-Class)")
            st.plotly_chart(fig, use_container_width=True)
