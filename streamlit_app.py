import streamlit as st
import os, glob, json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="SentinelNet IDS", layout="wide")


# locate latest run folder

def get_latest_run():
    runs = glob.glob("outputs/run_*")
    return sorted(runs, reverse=True)[0] if runs else None

RUN_FOLDER = get_latest_run()
if RUN_FOLDER is None:
    st.error("No saved run found under ./outputs/run_*. Train/save models first.")
    st.stop()


# load artifacts (features, scaler, models)

def load_artifacts(run_folder):
    feature_file = os.path.join(run_folder, "feature_names.json")
    scaler_file  = os.path.join(run_folder, "scaler.joblib")
    models_nsl   = {}
    models_cic   = {}

    if not os.path.exists(feature_file):
        st.error("feature_names.json missing in run folder.")
        st.stop()

    with open(feature_file, "r") as f:
        meta = json.load(f)
    FEATURES = meta.get("feature_columns", None)
    if not FEATURES:
        st.error("feature_columns not found in feature_names.json.")
        st.stop()

    SCALER = joblib.load(scaler_file) if os.path.exists(scaler_file) else None

    def load_models_from(path):
        md = {}
        if os.path.exists(path):
            for p in glob.glob(os.path.join(path, "*.joblib")):
                name = os.path.splitext(os.path.basename(p))[0]
                try:
                    md[name] = joblib.load(p)
                except Exception:
                    continue
        return md

    models_nsl = load_models_from(os.path.join(run_folder, "models"))
    models_cic = load_models_from(os.path.join(run_folder, "cic-ids"))

    return FEATURES, SCALER, models_nsl, models_cic

FEATURES, SCALER, MODELS_NSL, MODELS_CIC = load_artifacts(RUN_FOLDER)


# Sidebar controls

st.sidebar.title("Configuration")
dataset_choice = st.sidebar.selectbox("Dataset", ["NSL-KDD", "CIC-IDS-2017"])
model_dict = MODELS_NSL if dataset_choice == "NSL-KDD" else MODELS_CIC

if not model_dict:
    st.sidebar.warning("No models found for selected dataset (check outputs/run_*/models or cic-ids).")
    model_choice = None
else:
    model_choice = st.sidebar.selectbox("Select model", sorted(model_dict.keys()))

mode = st.sidebar.radio("Mode", ["Live Monitoring", "File Analysis"])


# helper: metrics card

def stats_box(total, intrusions):
    normal = total - intrusions
    rate = (intrusions / total * 100) if total else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Packets", f"{total:,}")
    c2.metric("Intrusions", f"{intrusions:,}")
    c3.metric("Normal", f"{normal:,}")
    c4.metric("Intrusion Rate", f"{rate:.2f}%")

 
# predictor (validates features)

def validate_and_predict(df, model):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Wrong file — missing columns: {missing[:8]}{'...' if len(missing)>8 else ''}")
    X = df[FEATURES].fillna(0)
    X_in = SCALER.transform(X) if SCALER is not None else X.values
    preds = model.predict(X_in)
    return preds.astype(int)

# header
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#2563eb,#7c3aed);padding:14px;border-radius:10px;color:#fff">
      <h2 style="margin:0">SentinelNet — IDS </h2>
      <div style="opacity:0.9;margin-top:-6px">Upload CSV (must contain required features) — choose dataset & model from sidebar</div>
    </div>
    """,
    unsafe_allow_html=True
)


# Model existence check

if model_choice is None:
    st.error("No model available for the selected dataset. Place model .joblib files under outputs/run_*/models or outputs/run_*/cic-ids.")
    st.stop()

MODEL = model_dict[model_choice]


# Live Monitoring (chunked)

if mode == "Live Monitoring":
    st.subheader("Live Monitoring (CSV stream by chunks)")
    uploaded = st.file_uploader("Upload CSV stream source (rows = flows)", type=["csv"])
    chunk_size = st.slider("Rows per step", 50, 1000, 200)

    if uploaded:
        try:
            df_stream = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Failed to read uploaded CSV.")
            st.stop()

        # validate column presence once
        missing = [c for c in FEATURES if c not in df_stream.columns]
        if missing:
            st.error(f"Wrong file — missing {len(missing)} required columns. Example missing: {missing[:8]}")
            st.stop()

        if "stream_idx" not in st.session_state:
            st.session_state.stream_idx = 0
            st.session_state.total = 0
            st.session_state.intrusions = 0

        if st.button("Next ▶ Process chunk"):
            i = st.session_state.stream_idx
            j = min(i + chunk_size, len(df_stream))
            chunk = df_stream.iloc[i:j].copy()
            preds = validate_and_predict(chunk, MODEL)
            st.session_state.stream_idx = j
            st.session_state.total += len(preds)
            st.session_state.intrusions += int((preds == 1).sum())

            st.write(f"Processed rows {i} → {j-1}")
            out = chunk.head(10).copy()
            out["predicted_label"] = preds[:len(out)]
            st.dataframe(out)

            # charts
            n_norm = int((preds==0).sum()); n_intr = int((preds==1).sum())
            fig, ax = plt.subplots(1,2, figsize=(8,3))
            ax[0].bar(["Normal","Attack"], [n_norm, n_intr], color=["#10b981","#ef4444"])
            ax[1].pie([n_norm,n_intr], labels=["Normal","Attack"], autopct="%1.1f%%")
            st.pyplot(fig)

        stats_box(st.session_state.total, st.session_state.intrusions)


# File Analysis (single file)

else:
    st.subheader("File Analysis")
    uploaded = st.file_uploader("Upload CSV for full-file prediction", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("Failed to read uploaded CSV.")
            st.stop()

        # quick column check
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Wrong file — missing required columns. Example missing: {missing[:10]}")
        else:
            if st.button("Run Prediction"):
                try:
                    preds = validate_and_predict(df, MODEL)
                except ValueError as e:
                    st.error(str(e)); st.stop()

                df_out = df.copy()
                df_out["predicted_label"] = preds
                total = len(preds); intr = int((preds==1).sum())
                stats_box(total, intr)

                st.markdown("#### Prediction counts")
                st.write(df_out["predicted_label"].value_counts())

                csv = df_out.to_csv(index=False).encode()
                st.download_button("Download predictions (CSV)", csv, "predictions.csv", "text/csv")
