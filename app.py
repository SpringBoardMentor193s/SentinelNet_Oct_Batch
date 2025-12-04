# ---------------------------------------
# SentinelNet – CSV-Only Intrusion Detection Dashboard
# Supports: NSL-KDD + CICIDS
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# PAGE CONFIG + CUSTOM CSS
# -------------------------------
st.set_page_config(page_title="SentinelNet - CSV IDS", layout="wide")

st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.metric-card{
    background: #ffffff15;
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 20px;
    text-align:center;
    box-shadow:0 4px 10px rgba(0,0,0,0.15);
}
.metric-value{
    font-size: 30px;
    font-weight: 800;
}
.metric-label{
    font-size: 14px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# UTILS
# -------------------------------
def load_obj(path):
    if not os.path.exists(path):
        return None
    try:
        return pickle.load(open(path, "rb"))
    except:
        try:
            return joblib.load(path)
        except:
            return None


# Base model path
BASE = "C:/SentinalNet/SentinelNet_Oct_Batch/models"


# -------------------------------
# LOAD PIPELINE
# -------------------------------
def load_pipeline(dataset, encoding, task, model_name):

    pipe = {}

    task_folder = "binary" if task == "Binary" else "multiclass"

    if dataset == "NSL-KDD":
        model_path = f"{BASE}/nsl_kdd/{task_folder}/{encoding.lower()}/{model_name}.pkl"
        base = f"{BASE}/nsl_kdd/{task_folder}/{encoding.lower()}"

        pipe["model"] = load_obj(model_path)
        pipe["ohe"] = load_obj(f"{base}/ohe_encoder.pkl")
        pipe["target_enc"] = load_obj(f"{base}/target_encoder.pkl")
        pipe["scaler"] = load_obj(f"{base}/scaler.pkl")
        pipe["label_enc"] = load_obj(f"{base}/label_encoder.pkl")

    else:  # CICIDS
        model_path = f"{BASE}/cicids/{task_folder}/models/{model_name}.pkl"
        base = f"{BASE}/cicids/{task_folder}/preprocess"

        pipe["model"] = load_obj(model_path)
        pipe["scaler"] = load_obj(f"{base}/scaler.pkl")
        pipe["pca"] = load_obj(f"{base}/pca.pkl")
        pipe["label_enc"] = load_obj(f"{base}/label_encoder.pkl")

    return pipe


# -------------------------------
# PREPROCESSORS
# -------------------------------
def preprocess_nsl(df, encoding, pipe):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["class", "label", "attack_class", "attack_binary"], errors="ignore")

    if encoding == "OHE":
        ohe = pipe["ohe"]
        needed = list(ohe.feature_names_in_)

        # Auto-add missing categorical columns
        for col in needed:
            if col not in df.columns:
                df[col] = "missing"

        df = df[needed]
        X = ohe.transform(df)

    else:  # Target Encoding
        te = pipe["target_enc"]
        needed = list(te.feature_names_in_)

        for col in needed:
            if col not in df.columns:
                df[col] = "missing"

        df = df[needed]
        X = te.transform(df)

    scaler = pipe["scaler"]
    if scaler:
        try:
            X = scaler.transform(X)
        except:
            pass

    return X


def preprocess_cicids(df, pipe):
    df = df.copy()
    df = df.drop(columns=["Label"], errors="ignore")

    scaler = pipe["scaler"]
    pca = pipe["pca"]

    # Auto-align columns
    needed = list(scaler.feature_names_in_)
    for col in needed:
        if col not in df.columns:
            df[col] = 0

    df = df[needed]

    Xs = scaler.transform(df)
    Xp = pca.transform(Xs)
    return Xp


# -------------------------------
# PREDICT + METRICS
# -------------------------------
def get_predictions(model, X, label_enc):

    preds = model.predict(X)

    # decode
    if label_enc is not None:
        try:
            preds_disp = label_enc.inverse_transform(preds)
        except:
            preds_disp = preds
    else:
        preds_disp = preds

    # probabilities
    try:
        probs = model.predict_proba(X)
    except:
        probs = None

    return preds, preds_disp, probs


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro")
    }


def plot_confusion(cm):
    fig = px.imshow(cm,
                    x=["Pred Normal", "Pred Attack"],
                    y=["True Normal", "True Attack"],
                    color_continuous_scale="Blues",
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    return fig


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_score = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc_score:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve")
    return fig


# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="header-box">
    <h1 style="font-weight:900; margin-bottom:4px;">🛡 SentinelNet IDS Dashboard</h1>
    <p>AI-powered CSV-based intrusion detection for NSL-KDD and CICIDS datasets</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# SIDEBAR CONFIG
# -------------------------------
dataset = st.sidebar.selectbox("Dataset", ["NSL-KDD", "CICIDS"])
task = st.sidebar.selectbox("Task Type", ["Binary", "Multi-Class"])

encoding = "PCA" if dataset == "CICIDS" else st.sidebar.selectbox("Encoding", ["OHE", "TE"])

# Load model list dynamically
task_folder = "binary" if task == "Binary" else "multiclass"
if dataset == "NSL-KDD":
    model_dir = f"{BASE}/nsl_kdd/{task_folder}/{encoding.lower()}"
else:
    model_dir = f"{BASE}/cicids/{task_folder}/models"

models = [f.replace(".pkl","") for f in os.listdir(model_dir)]
model_name = st.sidebar.selectbox("Choose Model", models)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# -------------------------------
# MAIN
# -------------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview of Uploaded CSV")
    st.dataframe(df.head(), use_container_width=True)

    pipeline = load_pipeline(dataset, encoding, task, model_name)
    model = pipeline["model"]

    # PREPROCESS
    if dataset == "NSL-KDD":
        X = preprocess_nsl(df, encoding, pipeline)
    else:
        X = preprocess_cicids(df, pipeline)

    # Extract label 
    y_true = None
    for c in ["Label", "class", "attack_binary", "attack_class"]:
        if c in df.columns:
            y_true = df[c].copy()
            if pipeline["label_enc"]:
                try:
                    y_true = pipeline["label_enc"].transform(y_true)
                except:
                    pass
            break

    # PREDICT
    y_pred, display_pred, probs = get_predictions(model, X, pipeline["label_enc"])

    st.subheader("Predictions")
    st.dataframe(pd.DataFrame({"Prediction": display_pred}).head(10))

    # METRICS
    if y_true is not None:
        st.subheader("Metrics")
        m = compute_metrics(y_true, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m['accuracy']:.3f}")
        col2.metric("Precision", f"{m['precision']:.3f}")
        col3.metric("Recall", f"{m['recall']:.3f}")
        col4.metric("F1 Score", f"{m['f1']:.3f}")

        # Confusion matrix
        st.plotly_chart(plot_confusion(confusion_matrix(y_true, y_pred)), use_container_width=True)

        # ROC Curve (binary only)
        if probs is not None and task == "Binary":
            st.plotly_chart(plot_roc_curve(y_true, probs), use_container_width=True)

else:
    st.info("Upload a CSV to begin intrusion detection.")














# # -----------------------------
# # SentinelNet - STREAMLIT APP
# # FINAL CORRECTED VERSION
# # -----------------------------
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import joblib
# import os
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score,
#     confusion_matrix, roc_curve, auc
# )
# from sklearn.preprocessing import LabelEncoder
# import plotly.express as px
# import plotly.graph_objects as go
# import io

# # -----------------------------------------------------------------
# # BASIC CONFIG
# # -----------------------------------------------------------------
# st.set_page_config(page_title="SentinelNet - Model Inference", layout="wide")

# BASE = "C:/SentinalNet/SentinelNet_Oct_Batch/models"


# # -----------------------------------------------------------------
# # Safe Loader for Pickle/Joblib
# # -----------------------------------------------------------------
# def load_file(path):
#     if not os.path.exists(path):
#         return None
#     try:
#         return pickle.load(open(path, "rb"))
#     except:
#         try:
#             return joblib.load(path)
#         except:
#             return None


# # -----------------------------------------------------------------
# # UI - LEFT SIDEBAR
# # -----------------------------------------------------------------
# st.sidebar.header("Configuration")

# dataset = st.sidebar.selectbox(
#     "Dataset",
#     ["NSL-KDD", "CICIDS"],
#     key="dataset",
#     disabled=False
# )

# task = st.sidebar.radio(
#     "Task Type",
#     ["Binary", "Multi-Class"],
#     key="task"
# )

# # FIX: Correct folder names
# task_folder = "binary" if task == "Binary" else "multiclass"

# # Choose Encoding
# if dataset == "NSL-KDD":
#     encoding = st.sidebar.selectbox(
#         "Preprocessing",
#         ["OHE", "TE"],
#         key="encoding",
#         disabled=False
#     )
#     base_model_path = f"{BASE}/nsl_kdd/{task_folder}/{encoding.lower()}"

# else:  # CICIDS
#     encoding = st.sidebar.selectbox(
#         "Preprocessing",
#         ["PCA"],
#         key="encoding_cicids",
#         disabled=True
#     )
#     base_model_path = f"{BASE}/cicids/{task_folder}/models"
#     preprocess_path = f"{BASE}/cicids/{task_folder}/preprocess"


# # -----------------------------------------------------------------
# # List Available Models
# # -----------------------------------------------------------------
# def list_models(path):
#     if not os.path.exists(path):
#         return []
#     return [
#         os.path.splitext(f)[0]
#         for f in os.listdir(path)
#         if f.endswith(".pkl") or f.endswith(".joblib")
#     ]


# available_models = list_models(base_model_path)

# if not available_models:
#     st.sidebar.warning(f"No models found in {base_model_path}")

# selected_model = st.sidebar.selectbox(
#     "Select Model",
#     available_models,
#     key="selected_model",
#     disabled=len(available_models) == 0
# )


# # Upload CSV
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# # -----------------------------------------------------------------
# # Load Pipeline Components
# # -----------------------------------------------------------------
# @st.cache_resource
# def load_pipeline(dataset, task_folder, encoding, model_name):
#     pipe = {}

#     # Build correct path
#     if dataset == "NSL-KDD":
#         model_path = f"{BASE}/nsl_kdd/{task_folder}/{encoding.lower()}/{model_name}.pkl"
#     else:
#         model_path = f"{BASE}/cicids/{task_folder}/models/{model_name}.pkl"

#     pipe["model"] = load_file(model_path)

#     # Load preprocessors
#     if dataset == "NSL-KDD":
#         base = f"{BASE}/nsl_kdd/{task_folder}/{encoding.lower()}"

#         pipe["scaler"] = load_file(f"{base}/scaler.pkl")
#         pipe["ohe"] = load_file(f"{base}/ohe_encoder.pkl")
#         pipe["target_enc"] = load_file(f"{base}/target_encoder.pkl")
#         pipe["label_enc"] = load_file(f"{base}/label_encoder.pkl")

#     else:  # CICIDS
#         base = f"{BASE}/cicids/{task_folder}/preprocess"

#         pipe["scaler"] = load_file(f"{base}/scaler.pkl")
#         pipe["pca"] = load_file(f"{base}/pca.pkl")
#         pipe["label_enc"] = load_file(f"{base}/label_encoder.pkl")

#     return pipe


# pipeline = None
# if selected_model:
#     pipeline = load_pipeline(dataset, task_folder, encoding, selected_model)


# # -----------------------------------------------------------------
# # Preprocessing Functions
# # -----------------------------------------------------------------
# def preprocess_nsl(df: pd.DataFrame, encoding, pipe):
#     df = df.copy()
#     df.columns = df.columns.str.strip()

#     # drop labels
#     for c in ["class", "attack_class", "attack_binary", "label", "Label"]:
#         df = df.drop(columns=[c], errors="ignore")

#     ohe = pipe.get("ohe")
#     te = pipe.get("target_enc")

#     if encoding == "OHE":
#         if ohe is None:
#             st.error("Missing OHE encoder")
#             return None

#         # Ensure same feature names
#         required = list(ohe.feature_names_in_)
#         for col in required:
#             if col not in df.columns:
#                 df[col] = 0  

#         df = df[required]

#         # Apply OHE
#         X = ohe.transform(df)

#         # Apply scaler
#         scaler = pipe.get("scaler")
#         if scaler is not None:
#             try:
#                 X = scaler.transform(X)
#             except:
#                 st.warning("Skipping scaler – mismatch")
#         return X

#     elif encoding == "TE":
#         if te is None:
#             st.error("Missing Target Encoder")
#             return None

#         required = list(te.feature_names_in_)
#         for col in required:
#             if col not in df.columns:
#                 df[col] = 0
#         df = df[required]

#         X = te.transform(df)

#         scaler = pipe.get("scaler")
#         if scaler is not None:
#             try:
#                 X = scaler.transform(X)
#             except:
#                 st.warning("Skipping TE scaler – mismatch")

#         return np.array(X)

#     return None



# def preprocess_cicids(df, pipe):
#     df = df.copy()
#     df.columns = df.columns.str.strip()
#     df = df.drop(columns=["Label"], errors="ignore")

#     scaler = pipe.get("scaler")
#     pca = pipe.get("pca")

#     if scaler is None or pca is None:
#         st.error("Missing CICIDS preprocess files.")
#         return None

#     Xs = scaler.transform(df)
#     Xp = pca.transform(Xs)
#     return Xp


# # -----------------------------------------------------------------
# # Prediction
# # -----------------------------------------------------------------
# def predict_and_metrics(model, X, y_true=None, label_enc=None):
#     y_pred = model.predict(X)

#     # Decode multiclass labels
#     if label_enc is not None:
#         try:
#             y_display = label_enc.inverse_transform(y_pred)
#         except:
#             y_display = y_pred
#     else:
#         y_display = y_pred

#     # Probabilities if available
#     try:
#         probs = model.predict_proba(X)
#     except:
#         probs = None

#     metrics = None
#     if y_true is not None:
#         metrics = {
#             "accuracy": accuracy_score(y_true, y_pred),
#             "precision": precision_score(y_true, y_pred, average="macro"),
#             "recall": recall_score(y_true, y_pred, average="macro"),
#             "f1": f1_score(y_true, y_pred, average="macro"),
#         }

#     return y_pred, probs, metrics, y_display


# # -----------------------------------------------------------------
# # MAIN OUTPUT AREA
# # -----------------------------------------------------------------
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("Preview of Uploaded Data")
#     st.dataframe(df.head(), use_container_width=True)

#     if pipeline is None or pipeline.get("model") is None:
#         st.error("Model not loaded. Check folder paths.")
#     else:
#         model = pipeline["model"]

#         # Preprocess based on dataset
#         if dataset == "NSL-KDD":
#             X = preprocess_nsl(df, encoding, pipeline)
#         else:
#             X = preprocess_cicids(df, pipeline)

#         if X is not None:
#             # Extract true labels if present
#             y_true = None
#             for c in ["class", "attack_class", "attack_binary", "Label"]:
#                 if c in df.columns:
#                     y_true = df[c].copy()
#                     if pipeline.get("label_enc") is not None:
#                         try:
#                             y_true = pipeline["label_enc"].transform(y_true)
#                         except:
#                             pass
#                     break

#             y_pred, probs, metrics, y_display = predict_and_metrics(
#                 model, X, y_true, pipeline.get("label_enc")
#             )

#             st.subheader("Predictions (first 10 rows)")
#             st.dataframe(pd.DataFrame({"Prediction": y_display}).head(10))

#             # Metrics
#             if metrics:
#                 st.subheader("Metrics")
#                 st.metric("Accuracy", f"{metrics['accuracy']:.3%}")
#                 st.write(f"Precision: {metrics['precision']:.3%}")
#                 st.write(f"Recall: {metrics['recall']:.3%}")
#                 st.write(f"F1 Score: {metrics['f1']:.3%}")

#             # Download predictions
#             out_df = df.copy()
#             out_df["_prediction"] = y_display

#             buf = io.StringIO()
#             out_df.to_csv(buf, index=False)
#             buf.seek(0)

#             st.download_button(
#                 "Download Prediction CSV",
#                 buf.getvalue(),
#                 "predictions.csv",
#                 "text/csv"
#             )


# else:
#     st.info("Upload a CSV file to begin inference.")
