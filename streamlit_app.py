# streamlit_monitor_dashboard.py (updated UI: header hidden, single start/stop toggle)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, glob, os, json
import matplotlib.pyplot as plt

st.set_page_config(page_title="SentinelNet IDS Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Styling: dark theme + hide Streamlit top header (Deploy etc.)
# -------------------------
st.markdown(
    """
    <style>
    /* page theme */
    .stApp { background-color: #0b0f1a; color: #e6eef6; }
    .sidebar .sidebar-content { background-color: #071028; color: #e6eef6; }
    .card { background: linear-gradient(180deg,#111827,#0b1220); border-radius:12px; padding:14px; color:#e6eef6; box-shadow: 0 6px 12px rgba(0,0,0,0.5); }
    .small-muted { color:#a8b0bf; font-size:0.9rem; }

    /* hide Streamlit top header (this removes Deploy / menu area) */
    header { display: none !important; }

    /* reduce sidebar padding so it looks tight */
    .sidebar .block-container { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- helpers ----------
def latest_run_folder(base="./outputs"):
    runs = sorted(glob.glob(os.path.join(base, "run_*")), reverse=True)
    return runs[0] if runs else None

def load_artifacts(run_folder):
    models = {}
    scaler = None
    meta = None
    if not run_folder:
        return models, scaler, meta
    # load meta
    meta_path = os.path.join(run_folder, "feature_names.json")
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, "r"))
        except:
            meta = None
    # load scaler
    sc_path = os.path.join(run_folder, "scaler.joblib")
    if os.path.exists(sc_path):
        try:
            scaler = joblib.load(sc_path)
        except:
            scaler = None
    # load models
    models_dir = os.path.join(run_folder, "models")
    if os.path.exists(models_dir):
        for f in glob.glob(os.path.join(models_dir, "*.joblib")):
            name = os.path.splitext(os.path.basename(f))[0]
            try:
                models[name] = joblib.load(f)
            except Exception:
                continue
    return models, scaler, meta

# ---------- sidebar ----------
st.sidebar.title("SentinelNet - Config")
mode = st.sidebar.radio("Mode", ["Live Monitoring", "File Analysis"])
dataset_choice = st.sidebar.selectbox("Dataset", ["NSL-KDD (example)", "Upload CSV"])
task_type = st.sidebar.radio("Task Type", ["Binary", "Multiclass"])

run_folder = latest_run_folder()
available_models, loaded_scaler, meta = load_artifacts(run_folder)
model_list = list(available_models.keys())
model_choice = st.sidebar.selectbox("Model", options=(model_list if model_list else ["Fallback detector"]))

chunk_size = st.sidebar.slider("Rows per step", 1, 500, 50)

# ---------- header (no Run info shown) ----------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='color:#e6eef6'>SentinelNet</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Network Intrusion Detection - Live Demo</div>", unsafe_allow_html=True)
with col2:
    # intentionally left blank to hide 'Run: ...' line (user requested it hidden)
    st.write("")

st.markdown("---")

# ---------- upload & buttons ----------
upload_col, control_col = st.columns([3,1])
with upload_col:
    uploaded_file = st.file_uploader("Upload CSV (treated as stream)", type=["csv"])
    df_full = None
    if uploaded_file:
        df_full = pd.read_csv(uploaded_file)
        st.success(f"CSV loaded: {len(df_full)} rows")
    elif dataset_choice.startswith("NSL-KDD") and run_folder:
        preds_folder = os.path.join(run_folder, "predictions")
        sample_files = glob.glob(os.path.join(preds_folder, "*_preds.csv"))
        if sample_files:
            try:
                df_full = pd.read_csv(sample_files[0])
                st.info("Using sample file from saved run as stream source.")
            except:
                df_full = None

with control_col:
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False
        st.session_state.index = 0
        st.session_state.stats = {"total":0, "intrusions":0, "normal":0}
        st.session_state.processed = []

    # Single toggle: show Start if stopped, Stop if running
    if not st.session_state.monitoring:
        start_clicked = st.button("▶ Start Monitoring", key="start")
        stop_clicked = False
    else:
        start_clicked = False
        stop_clicked = st.button("⏹ Stop Monitoring", key="stop")

    next_btn = st.button("Next ▶", key="next")
    clear = st.button("🧹 Clear", key="clear")
    export = st.button("📤 Export Summary", key="export")

    # handle toggles
    if start_clicked:
        st.session_state.monitoring = True
        st.session_state.index = 0
        st.session_state.stats = {"total":0, "intrusions":0, "normal":0}
        st.session_state.processed = []
    if stop_clicked:
        st.session_state.monitoring = False
    if clear:
        st.session_state.index = 0
        st.session_state.stats = {"total":0, "intrusions":0, "normal":0}
        st.session_state.processed = []
    if export:
        out = pd.DataFrame([st.session_state.stats]).to_csv(index=False).encode()
        st.download_button("Download summary.csv", data=out, file_name="sentinel_summary.csv", mime="text/csv")

# ---------- model reference ----------
loaded_model = available_models.get(model_choice) if model_choice in available_models else None

# ---------- predictor ----------
def predict_chunk(chunk):
    # use model+scaler if available
    if loaded_model is not None and loaded_scaler is not None and meta is not None:
        feature_cols = meta.get("feature_columns", None)
        if feature_cols:
            missing = [c for c in feature_cols if c not in chunk.columns]
            if missing:
                X = chunk.select_dtypes(include=[np.number])
            else:
                X = chunk[feature_cols]
        else:
            X = chunk.select_dtypes(include=[np.number])
        try:
            Xs = loaded_scaler.transform(X)
            preds = loaded_model.predict(Xs)
            return np.asarray(preds)
        except Exception:
            pass

    # fallback simple rule
    preds = []
    mean_src = chunk["src_bytes"].mean() if "src_bytes" in chunk.columns else 0
    mean_dst = chunk["dst_bytes"].mean() if "dst_bytes" in chunk.columns else 0
    for _, r in chunk.iterrows():
        if "label" in r.index:
            preds.append(0 if str(r["label"]).strip().lower()=="normal" else 1)
            continue
        if ("src_bytes" in r.index and pd.notna(r["src_bytes"]) and r["src_bytes"] > mean_src*10) or \
           ("dst_bytes" in r.index and pd.notna(r["dst_bytes"]) and r["dst_bytes"] > mean_dst*10):
            preds.append(1)
        else:
            preds.append(0)
    return np.array(preds)

# ---------- metric cards ----------
def metric_cards(total, intrusions, normal):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><h3>{total:,}</h3><div class='small-muted'>Total Packets</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h3>{intrusions:,}</h3><div class='small-muted'>Intrusions</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><h3>{normal:,}</h3><div class='small-muted'>Normal</div></div>", unsafe_allow_html=True)
    with c4:
        rate = (intrusions/total*100) if total>0 else 0.0
        st.markdown(f"<div class='card'><h3>{rate:.2f}%</h3><div class='small-muted'>Intrusion Rate</div></div>", unsafe_allow_html=True)

metric_cards(st.session_state.stats["total"], st.session_state.stats["intrusions"], st.session_state.stats["normal"])

# ---------- processing logic ----------
if df_full is None:
    st.warning("No CSV loaded. Upload a CSV to stream rows.")
else:
    st.info(f"Stream source ready ({len(df_full)} rows). Click Start then Next to process.")
    if (st.session_state.monitoring and next_btn) or (next_btn and not st.session_state.monitoring):
        idx = st.session_state.index
        start = idx
        end = min(idx + chunk_size, len(df_full))
        chunk = df_full.iloc[start:end].copy()
        if chunk.empty:
            st.warning("No more rows to process.")
        else:
            preds = predict_chunk(chunk)
            n_total = len(chunk)
            n_intr = int((preds==1).sum())
            n_norm = int((preds==0).sum())
            st.session_state.stats["total"] += n_total
            st.session_state.stats["intrusions"] += n_intr
            st.session_state.stats["normal"] += n_norm
            st.session_state.processed.append((start, end-1, n_total, n_intr))
            st.session_state.index = end

            chunk_out = chunk.reset_index(drop=True)
            chunk_out["predicted_label"] = preds
            st.subheader(f"Processed rows {start} → {end-1}")
            st.dataframe(chunk_out.head(10))

            fig, ax = plt.subplots(1,2, figsize=(8,3))
            ax[0].bar(["Normal","Attack"], [n_norm, n_intr], color=["#67b7d1","#f26d6d"])
            ax[0].set_title("Chunk Distribution")
            ax[1].pie([n_norm, n_intr], labels=["Normal","Attack"], autopct="%1.1f%%", colors=["#67b7d1","#f26d6d"])
            ax[1].set_title("Chunk Pie")
            st.pyplot(fig)

# history & export
if st.session_state.processed:
    hist_df = pd.DataFrame(st.session_state.processed, columns=["start","end","rows","intrusions"])
    st.markdown("### Processing history")
    st.dataframe(hist_df)

if st.session_state.stats["total"]>0:
    if st.button("Download processed summary"):
        out = pd.DataFrame([st.session_state.stats]).to_csv(index=False).encode()
        st.download_button("Download CSV", data=out, file_name="sentinel_summary_export.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div class='small-muted'>This app simulates streaming by stepping through the uploaded CSV in chunks. To use trained models place run folders under ./outputs/run_* with models/*.joblib and scaler.joblib.</div>", unsafe_allow_html=True)
