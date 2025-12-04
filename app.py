# app.py — SentinelNet Gradio Dashboard (with Binary/Multiclass toggle)

import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os
import tempfile

from functools import lru_cache
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score

# =========================================================
# DATASET OPTIONS
# =========================================================
DATASET_OPTIONS = {
    "KDD (NSL-KDD)": "kdd",
    "CIC-DDoS (CIC-IDS2017)": "ddos"
}

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

# =========================================================
# HELPERS: LOAD + PREPARE KDD ASSETS
# =========================================================
@lru_cache(maxsize=1)
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
# HELPERS: LOAD + PREPARE CIC-DDoS ASSETS
# =========================================================
@lru_cache(maxsize=1)
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
# CORE DASHBOARD COMPUTATION
# =========================================================
def compute_dashboard(dataset_label, mode, algo, task_type, monitoring_active, cleared):
    dataset_key = DATASET_OPTIONS[dataset_label]

    if dataset_key == "kdd":
        assets = load_kdd_assets()
        acc = assets["accuracy"]
        prec = assets["precision"]
        df_display = assets["test_display"]
        total_packets = len(df_display)
        intrusions = int((df_display["prediction"] == 1).sum())
        normals = total_packets - intrusions
    else:
        assets = load_ddos_assets()
        acc = assets["accuracy"]
        prec = assets["precision"]
        df_display = assets["df_display"]
        total_packets = len(df_display)
        pred_labels = df_display["pred_label"]
        benign_mask = np.array([("BENIGN" in lab.upper()) for lab in pred_labels])
        intrusions = int((~benign_mask).sum())
        normals = int(benign_mask.sum())

    intrusion_rate = (intrusions / total_packets * 100) if total_packets > 0 else 0.0

    # Apply clear effect
    if cleared:
        total_packets = 0
        intrusions = 0
        normals = 0
        intrusion_rate = 0.0

    # Header
    header_md = (
        "# SentinelNet IDS Dashboard\n\n"
        f"*Dataset:* {dataset_label}  |  *Mode:* {mode}  |  *Task:* {task_type}"
    )

    # Status / pills style text
    monitor_text = "Monitoring Active" if monitoring_active else "Monitoring Paused"
    status_md = (
        f"*Status:* {monitor_text}  \n"
        f"*Algorithm:* {algo}  \n"
        f"*Task Type:* {task_type}  \n"
        f"*Stats:* {intrusions} intrusions ({intrusion_rate:.1f}%)"
    )

    # Alerts
    if cleared:
        alerts_md = "> Alerts cleared for this session."
    else:
        lines = []
        if dataset_key == "kdd":
            df_alerts = assets["test_display"]
            df_alerts = df_alerts[df_alerts["prediction"] == 1].copy().head(8)
            if df_alerts.empty:
                alerts_md = "> No intrusions detected in the sampled KDD test data."
            else:
                for idx, row in df_alerts.iterrows():
                    proto = row["protocol_type"]
                    service = row.get("service", "")
                    lines.append(f"- *Conn #{idx}* — Attack detected (protocol {proto}, service {service})")
                alerts_md = "\n".join(lines)
        else:
            df_alerts = assets["df_display"]
            df_attack = df_alerts[~df_alerts["pred_label"].str.upper().str.contains("BENIGN")].copy()
            df_attack = df_attack.head(8)
            if df_attack.empty:
                alerts_md = "> No intrusions detected in the sampled CIC-DDoS data."
            else:
                for idx, row in df_attack.iterrows():
                    label = row["pred_label"]
                    lines.append(f"- *Flow #{idx}* — {label} traffic detected.")
                alerts_md = "\n".join(lines)

    acc_pct = round(acc * 100, 2)
    prec_pct = round(prec * 100, 2)

    return (
        header_md,
        status_md,
        int(total_packets),
        int(intrusions),
        int(normals),
        round(intrusion_rate, 2),
        acc_pct,
        prec_pct,
        alerts_md,
    )

# =========================================================
# EXPORT SUMMARY CSV
# =========================================================
def export_summary(dataset_label, mode, algo, task_type, monitoring_active, cleared):
    (
        _header,
        _status,
        total_packets,
        intrusions,
        normals,
        intrusion_rate,
        acc_pct,
        prec_pct,
        _alerts,
    ) = compute_dashboard(
        dataset_label, mode, algo, task_type, monitoring_active, cleared
    )

    summary_df = pd.DataFrame(
        [{
            "Dataset": dataset_label,
            "Mode": mode,
            "Algorithm": algo,
            "Task_Type": task_type,
            "Total_Packets": total_packets,
            "Intrusions": intrusions,
            "Normal": normals,
            "Intrusion_Rate_percent": intrusion_rate,
            "Accuracy_percent": acc_pct,
            "Precision_percent": prec_pct,
        }]
    )

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="sentinelnet_summary_")
    os.close(fd)
    summary_df.to_csv(path, index=False)
    return path

# =========================================================
# UI CALLBACKS
# =========================================================
def update_algorithms(dataset_label):
    dataset_key = DATASET_OPTIONS[dataset_label]
    if dataset_key == "kdd":
        opts = kdd_algorithms
    else:
        opts = ddos_algorithms
    return gr.Dropdown(choices=opts, value=opts[0])

def on_controls_change(dataset_label, mode, algo, task_type, monitoring_active, cleared):
    return compute_dashboard(dataset_label, mode, algo, task_type, monitoring_active, cleared)

def toggle_monitoring(dataset_label, mode, algo, task_type, monitoring_active, cleared):
    new_state = not monitoring_active
    result = compute_dashboard(dataset_label, mode, algo, task_type, new_state, cleared)
    return (new_state, *result)

def clear_data(dataset_label, mode, algo, task_type, monitoring_active, cleared):
    # Set cleared = True
    new_cleared = True
    result = compute_dashboard(dataset_label, mode, algo, task_type, monitoring_active, new_cleared)
    return (new_cleared, *result)

# =========================================================
# BUILD GRADIO APP
# =========================================================
with gr.Blocks(title="SentinelNet IDS Dashboard") as demo:
    gr.Markdown("## SentinelNet — Unified AI-Powered NIDS Dashboard")

    monitoring_state = gr.State(True)
    cleared_state = gr.State(False)

    with gr.Row():
        dataset_dd = gr.Dropdown(
            label="Dataset",
            choices=list(DATASET_OPTIONS.keys()),
            value="KDD (NSL-KDD)"
        )
        mode_radio = gr.Radio(
            label="Detection Mode",
            choices=["Live Monitoring", "File Analysis"],
            value="Live Monitoring"
        )
        algo_dd = gr.Dropdown(
            label="Algorithm",
            choices=kdd_algorithms,
            value=kdd_algorithms[0]
        )
        task_radio = gr.Radio(
            label="Classification Type",
            choices=["Binary", "Multiclass"],
            value="Binary"
        )

    with gr.Row():
        toggle_btn = gr.Button("Toggle Monitoring")
        clear_btn = gr.Button("Clear Data")
        export_btn = gr.Button("Export Summary CSV")

    header_md = gr.Markdown()
    status_md = gr.Markdown()

    with gr.Row():
        total_out = gr.Number(label="Total Packets", interactive=False)
        intru_out = gr.Number(label="Intrusions", interactive=False)
        normals_out = gr.Number(label="Normal", interactive=False)
        rate_out = gr.Number(label="Intrusion Rate (%)", interactive=False)

    with gr.Row():
        acc_out = gr.Number(label="Accuracy (%)", interactive=False)
        prec_out = gr.Number(label="Precision (%)", interactive=False)

    alerts_md = gr.Markdown()
    summary_file = gr.File(label="Download Summary", interactive=False)

    # Dataset change -> update algo list, then recompute dashboard
    dataset_dd.change(
        update_algorithms,
        inputs=dataset_dd,
        outputs=algo_dd
    ).then(
        on_controls_change,
        inputs=[dataset_dd, mode_radio, algo_dd, task_radio, monitoring_state, cleared_state],
        outputs=[header_md, status_md, total_out, intru_out, normals_out, rate_out, acc_out, prec_out, alerts_md]
    )

    # Other controls change -> recompute dashboard
    for ctrl in [mode_radio, algo_dd, task_radio]:
        ctrl.change(
            on_controls_change,
            inputs=[dataset_dd, mode_radio, algo_dd, task_radio, monitoring_state, cleared_state],
            outputs=[header_md, status_md, total_out, intru_out, normals_out, rate_out, acc_out, prec_out, alerts_md]
        )

    # Toggle Monitoring
    toggle_btn.click(
        toggle_monitoring,
        inputs=[dataset_dd, mode_radio, algo_dd, task_radio, monitoring_state, cleared_state],
        outputs=[monitoring_state, header_md, status_md, total_out, intru_out, normals_out, rate_out, acc_out, prec_out, alerts_md]
    )

    # Clear Data
    clear_btn.click(
        clear_data,
        inputs=[dataset_dd, mode_radio, algo_dd, task_radio, monitoring_state, cleared_state],
        outputs=[cleared_state, header_md, status_md, total_out, intru_out, normals_out, rate_out, acc_out, prec_out, alerts_md]
    )

    # Export Summary CSV
    export_btn.click(
        export_summary,
        inputs=[dataset_dd, mode_radio, algo_dd, task_radio, monitoring_state, cleared_state],
        outputs=summary_file
    )

    # Initial load
    demo.load(
        on_controls_change,
        inputs=[dataset_dd, mode_radio, algo_dd, task_radio, monitoring_state, cleared_state],
        outputs=[header_md, status_md, total_out, intru_out, normals_out, rate_out, acc_out, prec_out, alerts_md]
    )

if __name__ == "__main__":
    demo.launch()