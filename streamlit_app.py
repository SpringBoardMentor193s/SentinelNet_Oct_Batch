# SentinelNet – AI-Powered Network Intrusion Detection System
# Two-page SaaS-style Streamlit app (Dashboard + Analysis)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Plotly for SaaS-style charts
import plotly.express as px
import plotly.graph_objects as go

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import datetime
import io

# Optional PCAP support
try:
    import dpkt
    import socket
    HAS_DPKT = True
except Exception:
    HAS_DPKT = False

# -------------------------------------------------------
# PAGE CONFIG + DARK SAAS THEME CSS
# -------------------------------------------------------
st.set_page_config(
    page_title="SentinelNet – AI-Powered NIDS",
    page_icon="🛡️",
    layout="wide",
)

DARK_SAAS_CSS = """
<style>
body {
    background-color: #020617;
    color: #e5e7eb;
}
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#020617,#020617,#0b1120);
    color: #e5e7eb;
    border-right: 1px solid #111827;
}

/* Header */
.header-container {
    background: radial-gradient(circle at top left,#0f172a 0,#020617 60%);
    padding: 18px 20px;
    border-radius: 22px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    box-shadow: 0 18px 40px rgba(0,0,0,0.9);
}
.header-title {
    font-size: 24px;
    font-weight: 900;
    color: #f9fafb;
    margin-left: 15px;
}
.header-subtitle {
    font-size: 12px;
    color: #9ca3af;
    margin-left: 15px;
    margin-top: 2px;
}
.logo-img {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    border: 2px solid #22d3ee;
    box-shadow: 0 0 18px rgba(34,211,238,0.8);
}

/* Stat cards */
.stats-card {
    background: radial-gradient(circle at top left,#0b1120,#020617);
    padding: 16px 18px;
    border-radius: 20px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.9);
    border: 1px solid #1f2937;
}
.stats-value {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 4px;
    color: #e5e7eb;
}
.stats-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
}

/* Long horizontal cards (like your mentor's File Analysis bars) */
.long-card {
    border-radius: 18px;
    padding: 10px 16px;
    margin-bottom: 8px;
    color: #e5e7eb;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 24px rgba(0,0,0,0.9);
}
.long-red   { background: linear-gradient(90deg,#7f1d1d,#b91c1c); }
.long-green { background: linear-gradient(90deg,#064e3b,#047857); }
.long-blue  { background: linear-gradient(90deg,#0f172a,#1d4ed8); }
.long-gray  { background: linear-gradient(90deg,#111827,#374151); }

/* Metric bar */
.sent-card {
    border-radius: 22px;
    padding: 14px 18px;
    background: linear-gradient(135deg,#020617,#020617,#0f172a);
    box-shadow: 0 20px 45px rgba(0,0,0,1);
    border: 1px solid #111827;
    color: #e5e7eb;
}
.sent-title {
    font-size: 0.78rem;
    opacity: 0.85;
}
.sent-value {
    font-size: 1.4rem;
    font-weight: 700;
}

.metric-green { color:#22c55e; }
.metric-red   { color:#ef4444; }
.metric-amber { color:#f59e0b; }

hr {
    border: none;
    border-top: 1px solid #111827;
    margin: 0.8rem 0;
}
</style>
"""
st.markdown(DARK_SAAS_CSS, unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER BAR
# -------------------------------------------------------
st.markdown(
    f"""
    <style>

    .sentinel-banner {{
        width:100%;
        padding: 90px 0 60px; /* increased top padding since no logo */
        border-radius: 0 0 36px 36px;
        background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1650&q=80');
        background-size: cover;
        background-position: center;
        text-align: center;
        box-shadow: 0px 20px 70px rgba(0,0,0,0.85);
        border-bottom: 1px solid rgba(0,200,255,0.18);
        position: relative;
        overflow: hidden;
    }}

    .sentinel-banner::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: rgba(2,6,23,0.60);
        backdrop-filter: blur(8px);
        z-index: 1;
    }}

    /* Pulse animation */
    @keyframes titlePulse {{
        0% {{
            text-shadow: 
                0 0 25px rgba(56,189,248,0.35),
                0 0 45px rgba(56,189,248,0.15);
            transform: scale(1);
        }}
        50% {{
            text-shadow: 
                0 0 35px rgba(56,189,248,0.55),
                0 0 60px rgba(56,189,248,0.25);
            transform: scale(1.02);
        }}
        100% {{
            text-shadow: 
                0 0 25px rgba(56,189,248,0.35),
                0 0 45px rgba(56,189,248,0.15);
            transform: scale(1);
        }}
    }}

    .sentinel-title {{
        font-size: 64px;
        font-weight: 900;
        z-index: 2;
        position: relative;
        background: linear-gradient(90deg,#38bdf8,#7dd3fc,#bae6fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titlePulse 4.2s ease-in-out infinite;
        margin-top: 20px; /* adjust vertical centering */
    }}

    .sentinel-subtitle {{
        font-size: 18px;
        color: #d1d5db;
        letter-spacing: 1px;
        margin-top: 15px;
        z-index: 2;
        position: relative;
    }}

    </style>

    <div class="sentinel-banner">
        <div class="sentinel-title">SENTINELNET</div>
        <div class="sentinel-subtitle">
            AI-Powered Cyber Defense • Enterprise-Grade Intrusion Detection System
        </div>
    </div>

    """,
    unsafe_allow_html=True
)
# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def auto_encode(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True)
    if name == "XGBoost":
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        else:
            return None
    return None


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()
    if class_counts.min() < 2 or class_counts.shape[0] == 1:
        stratify_arg = None
        st.warning(
            "Some classes have fewer than 2 samples or only one class present. "
            "Using random train/test split (no stratify)."
        )
    else:
        stratify_arg = y

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )


def train_one_model(X_train, y_train, X_test, y_test, model_name):
    model = get_model(model_name)
    if model is None:
        raise RuntimeError(f"Model '{model_name}' not available (maybe XGBoost missing).")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        proba = model.predict_proba(X_test)
    except Exception:
        proba = None

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    return {
        "model": model,
        "y_true": y_test,
        "y_pred": y_pred,
        "proba": proba,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def plot_confusion(cm, labels=None, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels is not None:
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)
    st.pyplot(fig)
    return fig


def plot_roc_curve(y_true, proba, title="ROC Curve"):
    fig, ax = plt.subplots()
    if proba is None or proba.shape[1] < 2:
        ax.text(0.2, 0.5, "ROC Not Available", fontsize=14, color="white")
        ax.set_axis_off()
    else:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    st.pyplot(fig)
    return fig


def generate_pdf_report(
    model_name,
    accuracy,
    precision,
    recall,
    f1,
    cm_fig,
    roc_fig,
    classification_rep,
    abnormal_pct,
    student_name="Your Name",
    college_name="Your College / University",
    github_link="https://github.com/your-github/SentinelNet",
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_temp:
        pdf_path = pdf_temp.name

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "SentinelNet – Intrusion Detection Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 85, f"Model Used: {model_name}")
    c.drawString(50, height - 100, f"Student: {student_name}")
    c.drawString(50, height - 115, f"Institution: {college_name}")
    c.drawString(50, height - 130, f"GitHub: {github_link}")

    # Metrics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 155, "Model Performance Metrics:")
    c.setFont("Helvetica", 10)
    c.drawString(70, height - 170, f"Accuracy: {accuracy:.4f}")
    c.drawString(70, height - 185, f"Precision: {precision:.4f}")
    c.drawString(70, height - 200, f"Recall: {recall:.4f}")
    c.drawString(70, height - 215, f"F1 Score: {f1:.4f}")
    c.drawString(70, height - 230, f"Abnormal Traffic: {abnormal_pct:.2f}%")

    # Confusion Matrix
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 255, "Confusion Matrix:")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_temp:
        cm_fig.savefig(img_temp.name, bbox_inches="tight")
        cm_img_path = img_temp.name
    c.drawImage(ImageReader(cm_img_path), 50, height - 470, width=260, height=180)

    # ROC
    c.setFont("Helvetica-Bold", 12)
    c.drawString(330, height - 255, "ROC Curve:")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img2_temp:
        roc_fig.savefig(img2_temp.name, bbox_inches="tight")
        roc_img_path = img2_temp.name
    c.drawImage(ImageReader(roc_img_path), 330, height - 470, width=230, height=180)

    # Classification report
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 495, "Classification Report:")
    c.setFont("Helvetica", 8)
    y_offset = height - 510
    for line in classification_rep.split("\n"):
        c.drawString(50, y_offset, line)
        y_offset -= 10
        if y_offset < 40:
            c.showPage()
            y_offset = height - 50

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, "Generated by SentinelNet IDS – Academic Project")
    c.save()
    return pdf_path


def firewall_style(val):
    if val == "Normal":
        return "background-color: #064e3b; color: white;"
    else:
        return "background-color: #7f1d1d; color: white;"


# -------------------------------------------------------
# SIDEBAR – NAVIGATION + SETTINGS
# -------------------------------------------------------
with st.sidebar:
    st.title("🧭 SentinelNet Panel")

    page = st.radio(
        "Go to",
        ["📊 Dashboard", "🧪 Analysis Workspace"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📂 Main Dataset Upload")

    dataset_type = st.selectbox(
        "Dataset type (info only)",
        ["NSL-KDD", "CICIDS", "Custom"],
    )

    train_file = st.file_uploader(
        "Upload main dataset (CSV/TXT/XLSX)",
        type=["csv", "txt", "xlsx"],
        help="This single file is used for EDA, training and predictions.",
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Settings")

    model_name = st.selectbox(
        "Model",
        ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (RBF)"]
        + (["XGBoost"] if HAS_XGB else []),
    )

    test_size = st.slider("Internal test size", 0.1, 0.4, 0.2, step=0.05)
    max_features = st.slider("Max features to use", 5, 80, 30, step=5)

    st.markdown("---")
    run_pipeline = st.button("🚀 Run Full Pipeline")

st.caption(
    "Upload once → entire pipeline (EDA, preprocessing, training, evaluation, SOC view) runs from this dataset."
)

# -------------------------------------------------------
# LOAD DATA (fix for NoneType issue)
# -------------------------------------------------------
if train_file is None:
    st.error("⚠️ No dataset uploaded yet. Please upload a CSV/TXT/XLSX file from the sidebar.")
    st.stop()

try:
    if train_file.name.endswith(".xlsx"):
        df_raw = pd.read_excel(train_file)
    else:
        df_raw = pd.read_csv(train_file)
except Exception:
    df_raw = pd.read_csv(train_file, header=None)

# -------------------------------------------------------
# LABEL CONFIG (used by both pages)
# -------------------------------------------------------
st.subheader("Label Configuration")

label_col = st.selectbox(
    "Select label/target column (e.g., label, Label, class)",
    df_raw.columns.tolist(),
    index=len(df_raw.columns) - 1,
)

unique_labels = df_raw[label_col].dropna().unique().tolist()
default_normal = None
for v in unique_labels:
    if str(v).upper() in ["BENIGN", "NORMAL"]:
        default_normal = v
        break

normal_label = st.selectbox(
    "Which value represents NORMAL traffic?",
    unique_labels,
    index=unique_labels.index(default_normal) if default_normal in unique_labels else 0,
)

st.caption(
    "All rows with this label are treated as Normal (0). All others become Abnormal (1)."
)

label_counts = df_raw[label_col].value_counts()
total_records = len(df_raw)
num_features = df_raw.shape[1] - 1

normal_count = (df_raw[label_col] == normal_label).sum()
abnormal_count = total_records - normal_count
abnormal_pct_dataset = (abnormal_count / total_records * 100) if total_records else 0.0

# -------------------------------------------------------
# RUN PIPELINE (when button clicked)
# -------------------------------------------------------
if run_pipeline:
    with st.spinner("Running preprocessing + training + predictions..."):
        y_binary = (df_raw[label_col] != normal_label).astype(int)
        X_raw = df_raw.drop(columns=[label_col])

        X_encoded, cat_encoders = auto_encode(X_raw)
        if max_features < X_encoded.shape[1]:
            X_encoded = X_encoded.iloc[:, :max_features]
        feature_cols = X_encoded.columns.tolist()

        X_train, X_test, y_train, y_test = safe_train_test_split(
            X_encoded, y_binary, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        metrics = train_one_model(X_train_s, y_train, X_test_s, y_test, model_name)
        model = metrics["model"]

        X_full_s = scaler.transform(X_encoded)
        preds_full = model.predict(X_full_s)
        try:
            proba_full = model.predict_proba(X_full_s)
        except Exception:
            proba_full = None

        pred_labels_full = np.where(preds_full == 0, "Normal", "Abnormal")
        df_pred_full = df_raw.copy()
        df_pred_full["Prediction"] = pred_labels_full
        if proba_full is not None and proba_full.shape[1] >= 2:
            df_pred_full["Abnormal_Prob"] = proba_full[:, 1]

        st.session_state["pipeline_ran"] = True
        st.session_state["metrics"] = metrics
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["cat_encoders"] = cat_encoders
        st.session_state["feature_cols"] = feature_cols
        st.session_state["df_pred_full"] = df_pred_full
        st.session_state["y_binary"] = y_binary
        st.session_state["X_encoded"] = X_encoded
        st.session_state["label_col"] = label_col
        st.session_state["normal_label"] = normal_label
        st.success("Pipeline completed successfully!")

# -------------------------------------------------------
# PAGE 1 – DASHBOARD (SOC LOOK)
# -------------------------------------------------------
if page == "📊 Dashboard":
    st.subheader("SentinelNet Dashboard")

    # Metric tiles
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">Total Records</div>
            <div class="stats-value">{total_records}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">Features</div>
            <div class="stats-value">{num_features}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">Unique Labels</div>
            <div class="stats-value">{label_counts.nunique()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">Abnormal Share (Dataset)</div>
            <div class="stats-value">{abnormal_pct_dataset:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Layout similar to mentor's: left charts + right "File Analysis" vertical bars
    left, right = st.columns([2.3, 1])
    # ============================
    # LEFT PANEL
    # ============================
    with left:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Label Distribution</div>', unsafe_allow_html=True)

        label_df = label_counts.rename_axis("Label").reset_index(name="Count")
        fig_label = px.bar(label_df, x="Label", y="Count", template="plotly_dark")
        fig_label.update_layout(height=600)
        st.plotly_chart(fig_label, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # trend chart...
        # histogram chart...

    # ============================
    # RIGHT PANEL
    # ============================

    # === LABEL DISTRIBUTION ===
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Label Distribution</div>', unsafe_allow_html=True)

    label_df = label_counts.rename_axis("Label").reset_index(name="Count")
    fig_label = px.bar(
        label_df,
        x="Label",
        y="Count",
        color="Label",
        template="plotly_dark",
    )
    fig_label.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        title=None
    )
    st.plotly_chart(fig_label, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # === NUMERIC CHARTS ===
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        ca, cb = st.columns(2)

        # === TREND CHART ===
        with ca:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Feature Trend</div>', unsafe_allow_html=True)

            feat_line = st.selectbox("Trend feature", num_cols, index=0, key="dash_line")
            fig_trend = px.line(df_raw.head(700), y=feat_line, template="plotly_dark")
            fig_trend.update_traces(line=dict(width=4))
            fig_trend.update_layout(height=600, title=None)
            st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # === HISTOGRAM ===
        with cb:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Histogram</div>', unsafe_allow_html=True)

            feat_hist = st.selectbox("Histogram feature", num_cols, index=0, key="dash_hist")
            fig_hist = px.histogram(df_raw, x=feat_hist, nbins=50, template="plotly_dark")
            fig_hist.update_layout(height=600, title=None)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


    # ---- RIGHT: vertical long cards (File Analysis-like) ----
    with right:
        st.markdown("### File Analysis")

        st.markdown(
            f"""
            <div class="long-card long-blue">
                <span>Live Network Monitoring</span>
                <span>{total_records} packets</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="long-card long-green">
                <span>Normal Sessions</span>
                <span>{normal_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="long-card long-red">
                <span>Abnormal / Attack Sessions</span>
                <span>{abnormal_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="long-card long-gray">
                <span>Abnormal Share</span>
                <span>{abnormal_pct_dataset:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Donut chart
        share_df = pd.DataFrame(
            {
                "Type": ["Normal", "Abnormal"],
                "Count": [normal_count, abnormal_count],
            }
        )
        fig_pie = px.pie(
            share_df,
            values="Count",
            names="Type",
            template="plotly_dark",
            hole=0.6,
            title="Normal vs Abnormal",
        )
        fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Threat Level & Stream")

    if not st.session_state.get("pipeline_ran", False):
        st.info("Run the full pipeline from the sidebar to see model-based threat analytics.")
    else:
        df_pred_full = st.session_state["df_pred_full"]
        total = len(df_pred_full)
        abnormal = (df_pred_full["Prediction"] == "Abnormal").sum()
        normal = total - abnormal
        abnormal_pct = abnormal / total * 100 if total else 0.0

        st.markdown('<div class="sent-card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown('<div class="sent-title">Total Records (Predicted)</div>', unsafe_allow_html=True)
        c1.markdown(f'<div class="sent-value">{total}</div>', unsafe_allow_html=True)
        c2.markdown('<div class="sent-title">Normal</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="sent-value metric-green">{normal}</div>', unsafe_allow_html=True)
        c3.markdown('<div class="sent-title">Abnormal</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="sent-value metric-red">{abnormal}</div>', unsafe_allow_html=True)
        c4.markdown('<div class="sent-title">Abnormal %</div>', unsafe_allow_html=True)
        c4.markdown(
            f'<div class="sent-value metric-amber">{abnormal_pct:.2f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Threat gauge
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=abnormal_pct,
                title={"text": "Abnormal Traffic %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#22d3ee"},
                    "steps": [
                        {"range": [0, 10], "color": "#064e3b"},
                        {"range": [10, 30], "color": "#f59e0b"},
                        {"range": [30, 60], "color": "#f97316"},
                        {"range": [60, 100], "color": "#7f1d1d"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=10), height=260)
        st.plotly_chart(fig_gauge, use_container_width=True)

        if "Abnormal_Prob" in df_pred_full.columns:
            fig_stream = px.line(
                df_pred_full.head(500),
                y="Abnormal_Prob",
                template="plotly_dark",
                title="Abnormal Probability – first 500 samples",
            )
            fig_stream.update_traces(line=dict(width=2))
            fig_stream.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_stream, use_container_width=True)

# -------------------------------------------------------
# PAGE 2 – ANALYSIS WORKSPACE
# -------------------------------------------------------
if page == "🧪 Analysis Workspace":
    tabs = st.tabs(["🔍 EDA & Preprocessing", "🧠 Training & Evaluation", "📈 Detection Results & New Data"])

    # ---------- TAB 0 ----------
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Preprocessing Pipeline (Steps)")
        steps = [
            "Use selected label column as target (y).",
            f"Map labels to binary: 0 = Normal ({normal_label}), 1 = Abnormal (others).",
            "Drop label column to form feature matrix X.",
            "Encode categorical features using LabelEncoder() for each categorical column.",
            f"Keep at most {max_features} features (based on column order).",
            f"Split into train/test with test size = {test_size}.",
            "Scale numerical features using StandardScaler().",
            f"Train selected model: {model_name}.",
            "Evaluate on internal test set (accuracy, precision, recall, F1, confusion matrix, ROC).",
            "Use trained model to predict on FULL dataset and compute threat statistics.",
        ]
        for s_step in steps:
            st.write("✔ " + s_step)

        st.markdown("---")
        st.subheader("Optional: PCAP to CSV Preview (not used for training)")
        if HAS_DPKT:
            pcap_file = st.file_uploader("Upload PCAP/PCAPNG (optional)", type=["pcap", "pcapng"])
            if pcap_file is not None:
                try:
                    st.info("Converting PCAP → simple flow table (timestamp, src_ip, dst_ip, packet_size)...")
                    pcap_data = pcap_file.read()
                    pcap = dpkt.pcap.Reader(io.BytesIO(pcap_data))

                    rows = []
                    for ts, buf in pcap:
                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                            if isinstance(eth.data, dpkt.ip.IP):
                                ip = eth.data
                                src = socket.inet_ntoa(ip.src)
                                dst = socket.inet_ntoa(ip.dst)
                                rows.append([ts, src, dst, len(buf)])
                        except Exception:
                            continue

                    if rows:
                        pcap_df = pd.DataFrame(rows, columns=["timestamp", "src_ip", "dst_ip", "packet_size"])
                        st.dataframe(pcap_df.head(), use_container_width=True)
                    else:
                        st.warning("No IP packets extracted from this PCAP.")
                except Exception as e:
                    st.error(f"PCAP parsing failed: {e}")
        else:
            st.info("PCAP preview requires 'dpkt' library (optional).")

    # ---------- TAB 1 ----------
    with tabs[1]:
        st.subheader("Internal Model Training & Evaluation")

        if not st.session_state.get("pipeline_ran", False):
            st.info("Run the full pipeline from the sidebar to train and evaluate the model.")
        else:
            metrics = st.session_state["metrics"]
            acc = metrics["accuracy"]
            prec = metrics["precision"]
            rec = metrics["recall"]
            f1 = metrics["f1"]
            y_true = metrics["y_true"]
            y_pred = metrics["y_pred"]
            proba = metrics["proba"]

            st.markdown('<div class="sent-card">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown('<div class="sent-title">Model</div>', unsafe_allow_html=True)
            m1.markdown(f'<div class="sent-value">{model_name}</div>', unsafe_allow_html=True)

            color_class = "metric-green" if acc >= 0.9 else "metric-amber" if acc >= 0.8 else "metric-red"
            m2.markdown('<div class="sent-title">Accuracy (Test)</div>', unsafe_allow_html=True)
            m2.markdown(
                f'<div class="sent-value {color_class}">{acc*100:.2f}%</div>',
                unsafe_allow_html=True,
            )

            m3.markdown('<div class="sent-title">Precision / Recall</div>', unsafe_allow_html=True)
            m3.markdown(
                f'<div class="sent-value">{prec:.2f} / {rec:.2f}</div>',
                unsafe_allow_html=True,
            )

            m4.markdown('<div class="sent-title">F1 Score</div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="sent-value">{f1:.2f}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            cm = confusion_matrix(y_true, y_pred)
            col_cm, col_roc = st.columns(2)
            with col_cm:
                cm_fig = plot_confusion(
                    cm,
                    labels=["Normal", "Abnormal"],
                    title="Internal Confusion Matrix",
                )
            with col_roc:
                roc_fig = plot_roc_curve(y_true, proba, title="Internal ROC Curve")

            st.subheader("Classification Report")
            classification_rep_text = classification_report(
                y_true, y_pred, target_names=["Normal", "Abnormal"], zero_division=0
            )
            st.text(classification_rep_text)

            st.subheader("📄 PDF Report")
            if st.button("Generate PDF Report", key="pdf_btn"):
                pdf_path = generate_pdf_report(
                    model_name,
                    acc,
                    prec,
                    rec,
                    f1,
                    cm_fig,
                    roc_fig,
                    classification_rep_text,
                    abnormal_pct_dataset,
                    student_name="Your Name",
                    college_name="Your College / University",
                    github_link="https://github.com/your-github/SentinelNet",
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=f,
                        file_name="SentinelNet_Report.pdf",
                        mime="application/pdf",
                    )

    # ---------- TAB 2 ----------
    with tabs[2]:
        st.subheader("Detection Results on Uploaded Dataset & New Traffic")

        if not st.session_state.get("pipeline_ran", False):
            st.info("Run the full pipeline first to see predictions.")
        else:
            df_pred_full = st.session_state["df_pred_full"]

            st.markdown("### Detection Results (first 80 rows)")
            st.dataframe(df_pred_full.head(80), use_container_width=True)

            csv_data = df_pred_full.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions as CSV",
                data=csv_data,
                file_name="SentinelNet_Predictions.csv",
                mime="text/csv"
            )

            total = len(df_pred_full)
            abnormal = (df_pred_full["Prediction"] == "Abnormal").sum()
            normal = total - abnormal
            abnormal_pct = abnormal / total * 100 if total else 0.0

            st.markdown('<div class="sent-card">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown('<div class="sent-title">Total Records</div>', unsafe_allow_html=True)
            c1.markdown(f'<div class="sent-value">{total}</div>', unsafe_allow_html=True)
            c2.markdown('<div class="sent-title">Normal</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="sent-value metric-green">{normal}</div>', unsafe_allow_html=True)
            c3.markdown('<div class="sent-title">Abnormal</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="sent-value metric-red">{abnormal}</div>', unsafe_allow_html=True)
            c4.markdown('<div class="sent-title">Abnormal %</div>', unsafe_allow_html=True)
            c4.markdown(
                f'<div class="sent-value metric-amber">{abnormal_pct:.2f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("🚨 Automated Alert")
            if abnormal_pct < 10:
                st.success("🟢 Low-level activity — No immediate threat.")
            elif abnormal_pct < 30:
                st.warning("🟡 Moderate-level threat detected.")
            elif abnormal_pct < 60:
                st.error("🟠 HIGH THREAT — Investigate suspicious traffic.")
            else:
                st.error("🔴 CRITICAL ATTACK — SYSTEM UNDER ATTACK!")
                st.write("⚠ Most recent 10 abnormal rows:")
                st.dataframe(df_pred_full[df_pred_full["Prediction"] == "Abnormal"].head(10))

            st.subheader("🛡 Firewall-style Table (first 200 rows)")
            styled_fw = df_pred_full.head(200).style.applymap(firewall_style, subset=["Prediction"])
            st.dataframe(styled_fw, use_container_width=True)

            if "Abnormal_Prob" in df_pred_full.columns:
                st.subheader("📉 Abnormal Probability Trend (first 500 samples)")
                fig_as, ax_as = plt.subplots()
                ax_as.plot(df_pred_full["Abnormal_Prob"].values[:500])
                ax_as.set_title("Abnormal Probability Trend")
                ax_as.set_xlabel("Record Index")
                ax_as.set_ylabel("Abnormal Probability")
                st.pyplot(fig_as)

            st.markdown("---")
            st.subheader("📂 Optional: Upload NEW Traffic for Prediction (same trained model)")

            new_file = st.file_uploader(
                "Upload new CSV/TXT/XLSX (optional)",
                type=["csv", "txt", "xlsx"],
                key="new_test_file",
            )

            if new_file is not None:
                try:
                    if new_file.name.endswith(".xlsx"):
                        df_new = pd.read_excel(new_file)
                    else:
                        df_new = pd.read_csv(new_file)
                except Exception:
                    df_new = pd.read_csv(new_file, header=None)

                st.markdown("#### New Traffic Preview")
                st.dataframe(df_new.head(10), use_container_width=True)

                feature_cols = st.session_state["feature_cols"]
                cat_encoders_new = st.session_state["cat_encoders"]
                scaler_new = st.session_state["scaler"]
                model_new = st.session_state["model"]

                missing_cols = [c for c in feature_cols if c not in df_new.columns]
                for col in missing_cols:
                    df_new[col] = 0

                df_new = df_new[feature_cols]

                for col, enc in cat_encoders_new.items():
                    if col in df_new.columns:
                        try:
                            df_new[col] = enc.transform(df_new[col].astype(str))
                        except Exception:
                            df_new[col] = 0

                X_new_s = scaler_new.transform(df_new)
                preds_new = model_new.predict(X_new_s)
                try:
                    probs_new = model_new.predict_proba(X_new_s)
                except Exception:
                    probs_new = None

                labels_new = np.where(preds_new == 0, "Normal", "Abnormal")
                df_new_out = df_new.copy()
                df_new_out["Prediction"] = labels_new
                if probs_new is not None and probs_new.shape[1] >= 2:
                    df_new_out["Abnormal_Prob"] = probs_new[:, 1]

                st.markdown("#### Predictions on New File (first 80 rows)")
                st.dataframe(df_new_out.head(80), use_container_width=True)