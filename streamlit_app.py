
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(layout="wide", page_title="SentinelNet - Model Demo")

st.title("SentinelNet — Model Inference App")

# Utility: find latest run folder
def latest_run_folder(base="./outputs"):
    runs = sorted(glob.glob(os.path.join(base, "run_*")), reverse=True)
    return runs[0] if runs else None

run_folder = latest_run_folder()
if run_folder is None:
    st.error("No saved runs found in ./outputs. Run training first and save artifacts.")
    st.stop()

st.write("**Using run folder:**", run_folder)

# Load metadata
meta_file = os.path.join(run_folder, "feature_names.json")
if not os.path.exists(meta_file):
    st.error("feature_names.json not found in run folder.")
    st.stop()
meta = json.load(open(meta_file, "r"))
FEATURES = meta["feature_columns"]
TARGET = meta["target_column"]

# Load scaler
scaler_file = os.path.join(run_folder, "scaler.joblib")
scaler = joblib.load(scaler_file)

# List models available
models_dir = os.path.join(run_folder, "models")
model_files = sorted(glob.glob(os.path.join(models_dir, "*.joblib")))
model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_files]

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose model", model_names)
model_path = os.path.join(models_dir, f"{model_choice}.joblib")
model = joblib.load(model_path)

st.sidebar.write("FEATURES count:", len(FEATURES))
st.sidebar.caption("Upload CSV with same feature columns or use sample test.")

# Upload area
uploaded = st.file_uploader("Upload CSV for prediction (must include same feature columns)", type=["csv"])
use_sample_btn = st.button("Use sample test file from run (if present)")

df_input = None
if uploaded is not None:
    df_input = pd.read_csv(uploaded)
    st.success("File uploaded.")
elif use_sample_btn:
    # try to find predictions folder and a preds file
    preds_folder = os.path.join(run_folder, "predictions")
    sample_files = glob.glob(os.path.join(preds_folder, "*_preds.csv"))
    if sample_files:
        df_input = pd.read_csv(sample_files[0])
        st.info(f"Loaded sample preds file: {os.path.basename(sample_files[0])}")
    else:
        st.warning("No sample prediction file found in run folder.")
        df_input = None

if df_input is not None:
    st.subheader("Preview uploaded data (first 5 rows)")
    st.dataframe(df_input.head())

    # Check for feature columns presence
    missing = [c for c in FEATURES if c not in df_input.columns]
    if missing:
        st.error(f"Uploaded file is missing required feature columns. Missing: {missing[:10]}{'...' if len(missing)>10 else ''}")
    else:
        # select features and scale
        X = df_input[FEATURES].copy()
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            st.error(f"Scaler transform failed: {e}")
            st.stop()

        # predict
        y_pred = model.predict(X_scaled)
        df_out = df_input.copy()
        df_out["y_pred"] = y_pred

        st.write("Prediction counts:")
        st.write(df_out["y_pred"].value_counts().rename_axis("class").reset_index(name="count"))

        # if true labels present, show confusion
        if TARGET in df_out.columns:
            y_true = df_out[TARGET].values
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_true, df_out["y_pred"])
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y_true, df_out["y_pred"], digits=4))

        # allow download
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.write("---")
st.caption("App auto-loads the latest saved run from ./outputs. To use different run, move its folder to top or modify code.")
