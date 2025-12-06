import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------- Load Pickles --------------------
def load_pickle(path):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def try_load_pickle(path):
    try:
        return load_pickle(path)
    except FileNotFoundError:
        return None

# -------------------- Preprocessing --------------------
def preprocess_nsl(df, feature_columns, scaler, names):
    df_proc = df.copy()
    if "difficulty" in df_proc.columns:
        df_proc = df_proc.drop(columns=["difficulty"])
    categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in df_proc.columns]
    if categorical_cols:
        df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = 0
    df_proc = df_proc[feature_columns]
    X_scaled= scaler.transform(df_proc.values)
    X_df=pd.DataFrame(X_scaled, columns=df_proc.columns)
    return X_df, df_proc
def preprocess_cicids(df, feature_columns, scaler):
    df_proc = df.copy()
    for col in feature_columns:
        if col not in df_proc.columns:
            df_proc[col] = 0
    df_proc = df_proc[feature_columns]
    X = scaler.transform(df_proc.values)
    X_df=pd.DataFrame(X, columns=df_proc.columns)
    return X_df, df_proc

# -------------------- Models Map --------------------
MODELS_MAP = {
    "NSL-KDD": {
        "Models": {
            "Logistic_Regression": "Models/NSL_KDD/Logistic_Regression_Model.pkl",
            "Decision_Tree": "Models/NSL_KDD/Decision_Tree_Model.pkl",
            "Random_Forest": "Models/NSL_KDD/Random_Forest_Model.pkl",
            "XGBoost": "Models/NSL_KDD/XGBoost_Model.pkl",
            "LightGBM": "Models/NSL_KDD/LightGBM_Model.pkl"
        },
        "Scaler": "Models/NSL_KDD/scaler.pkl",
        "Feature_Columns": "Models/NSL_KDD/Features.pkl"
    },
    "CICIDS-2017": {
        "Binary": {
            "Logistic_Regression": "Models/CICIDS-2017/Binary_Class/Logistic_Regression.pkl",
            "Decision_Tree": "Models/CICIDS-2017/Binary_Class/Decision_Tree.pkl",
            "Random_Forest": "Models/CICIDS-2017/Binary_Class/Random_Forest.pkl",
            "XGBoost": "Models/CICIDS-2017/Binary_Class/XGBoost.pkl",
            "LightGBM": "Models/CICIDS-2017/Binary_Class/LightGBM.pkl"
        },
        "Multiclass": {
            "Logistic_Regression": "Models/CICIDS-2017/Multi_Class/Logistic_Regression.pkl",
            "Decision_Tree": "Models/CICIDS-2017/Multi_Class/Decision_Tree.pkl",
            "Random_Forest": "Models/CICIDS-2017/Multi_Class/Random_Forest.pkl",
            "XGBoost": "Models/CICIDS-2017/Multi_Class/XGBoost.pkl",
            "LightGBM": "Models/CICIDS-2017/Multi_Class/LightGBM.pkl"
        },
        "Scaler": "Models/CICIDS-2017/Scaler.pkl",
        "Feature_Columns": "Models/CICIDS-2017/Feature_Columns.pkl",
        "Label_Encoder": "Models/CICIDS-2017/Label_Encoder.pkl"
    }
}

# -------------------- Models Metrics --------------------
Models_Metrics={
    "NSL-KDD": {
        "Logistic_Regression": {"accuracy": 0.8564, "precision": 0.9154 },
        "Decision_Tree": {"accuracy": 0.9395, "precision": 0.9654},
        "Random_Forest": {"accuracy": 0.9599, "precision": 0.9661},
        "XGBoost": {"accuracy": 0.8204, "precision": 0.9699},
        "LightGBM": {"accuracy": 0.8909, "precision": 0.9697},
    },
    "CICIDS-2017": {
        "Binary": {
            "Logistic_Regression": {"accuracy": 0.9264, "precision": 0.9154 },
            "Decision_Tree": {"accuracy": 0.9695, "precision": 0.8154},
            "Random_Forest": {"accuracy": 0.9799, "precision": 0.8661},
            "XGBoost": {"accuracy": 0.9204, "precision": 0.8699},
            "LightGBM": {"accuracy": 0.9109, "precision": 0.8797},
        },
        "Multiclass": {
            "Logistic_Regression": {"accuracy": 0.9764, "precision": 0.9654 },
            "Decision_Tree": {"accuracy": 0.9895, "precision": 0.9754},
            "Random_Forest": {"accuracy": 0.9199, "precision": 0.9561},
            "XGBoost": {"accuracy": 0.9504, "precision": 0.9899},
            "LightGBM": {"accuracy": 0.9609, "precision": 0.9617},
        }    
    }
}

# -------------------- Column Names --------------------
NSL_KDD_COLUMN_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'class','difficulty'
]

# -------------------- UI --------------------
st.set_page_config(page_title="SentinelNet - NIDS", layout="wide")
st.title("SentinelNet — AI-Powered Network Intrusion Detection System (NIDS)")

# -------------------- SideBar --------------------
st.sidebar.header("Configuration")
st.sidebar.subheader("Detection Mode")
mode = st.sidebar.radio("Select Mode",["File Analysis", "Live Monitoring"])
st.sidebar.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
st.sidebar.subheader("Dataset")
dataset_choice = st.sidebar.selectbox("Choose Dataset:", ["NSL-KDD", "CICIDS-2017"])
class_type = None
if dataset_choice == "CICIDS-2017":
    st.sidebar.subheader("Classification Type")
    class_type = st.sidebar.selectbox("Choose Type:", ["Binary", "Multiclass"])
if dataset_choice == "NSL-KDD":
    available_models = list(MODELS_MAP["NSL-KDD"]["Models"].keys())
else:
    available_models = list(MODELS_MAP["CICIDS-2017"][class_type].keys())
st.sidebar.subheader("Algorithm")
algorithm = st.sidebar.selectbox("Select Algorithm:", available_models)
st.sidebar.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
st.sidebar.subheader("Model Details")
if dataset_choice == "NSL-KDD":
    metrics= Models_Metrics["NSL-KDD"][algorithm]
else:
    metrics = Models_Metrics["CICIDS-2017"][class_type][algorithm]
st.markdown("""
    <style>
        .metric-value {
            font-size: 35px !important;
            font-weight: 400 !important;
            margin-top: -10px !important;
        }
    </style>
""", unsafe_allow_html=True)
st.sidebar.subheader("Accuracy:")
st.sidebar.markdown(f"<div class='metric-value'>{metrics['accuracy']:.2f}%</div>", unsafe_allow_html=True)
st.sidebar.subheader("Precision:")
st.sidebar.markdown(f"<div class='metric-value'>{metrics['precision']:.2f}%</div>", unsafe_allow_html=True)
try:
    if dataset_choice == "NSL-KDD":
        model_path = MODELS_MAP["NSL-KDD"]["Models"][algorithm]
        scaler_path = MODELS_MAP["NSL-KDD"]["Scaler"]
        features_path = MODELS_MAP["NSL-KDD"]["Feature_Columns"]
    else:
        model_path = MODELS_MAP["CICIDS-2017"][class_type][algorithm]
        scaler_path = MODELS_MAP["CICIDS-2017"]["Scaler"]
        features_path = MODELS_MAP["CICIDS-2017"]["Feature_Columns"]
        label_encoder_path = MODELS_MAP["CICIDS-2017"].get("Label_Encoder")
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# -------------------- Live Monitoring --------------------
if mode == "Live Monitoring":
    st.subheader("Live Network Monitoring")
    col1, col2, col3= st.columns(3)
    with col1:
        start=st.button("Start Monitoring")
    with col2:
        stop=st.button("Stop Monitoring")
    with col3:
        clear=st.button("Clear Data") 
    if start:
        st.success("Live monitoring started...")
        # Placeholder for live monitoring logic
    if stop:
        st.success("Live Monitoring Stoped.")
    if clear:
        st.success("Data cleared.")
    m1, m2, m3, m4=st.columns(4)
    with m1:
        st.metric("Total Packets", "0")
    with m2:
        st.metric("Normal", "0")
    with m3:
        st.metric("Intrusions", "0")
    with m4:
        st.metric("Intrusion Rate", "0%")
    
    st.subheader("Recent Alerts")
    st.info("No alerts to display.")
    st.stop()

# -------------------- File Analysis --------------------
st.subheader("File Analysis")
uploaded_file = st.file_uploader("Upload CSV (files)", type=["csv"])
if uploaded_file is None:
    st.stop()
def load_uploaded_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        df=pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df=pd.read_csv(uploaded_file, header=None)
    if dataset_choice =="NSL-KDD":
        if df.shape[1] == len(NSL_KDD_COLUMN_NAMES):
            df.columns = NSL_KDD_COLUMN_NAMES
        else:
            st.error(
                f"Uploaded CSV has {df.shape[1]} columns,"
                f"but NSL-KDD requires {len(NSL_KDD_COLUMN_NAMES)} columns."
            )
            st.stop()
    return df
df = load_uploaded_csv(uploaded_file)
st.write("### Uploaded Data Review")
st.dataframe(df.head())

try:
    model = load_pickle(model_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
scaler = try_load_pickle(scaler_path)
if scaler is None:
    st.error(f"Scaler not found at `{scaler_path}`.")
    st.stop()
feature_columns = try_load_pickle(features_path)
if feature_columns is None:
    st.error(f"Feature columns not found at `{features_path}`.")
    st.stop()
label_encoder = None
if dataset_choice == "CICIDS-2017":
    label_encoder = try_load_pickle(label_encoder_path)
if st.button("Evaluate"):
    t0 = datetime.now()
    st.info("Preprocessing and predicting...")
    try:
        if dataset_choice == "NSL-KDD":
            X, aligned_df = preprocess_nsl(df, feature_columns, scaler, names=NSL_KDD_COLUMN_NAMES)
        else:
            X, aligned_df = preprocess_cicids(df, feature_columns, scaler)
        preds = model.predict(X)
        if label_encoder is not None:
            try:
                preds_decoded = label_encoder.inverse_transform(preds)
            except Exception:
                preds_decoded = preds.astype(str)
        else:
            preds_decoded = preds.astype(str)
        results_df = df.reset_index(drop=True).copy()
        results_df["Prediction"] = preds_decoded
        st.success("Prediction finished.")
        st.write("### Results")
        st.dataframe(results_df.head())
    except Exception as err:
        st.error(f"Error during evaluation: {err}")

# # import streamlit as st
# import pandas as pd
# import numpy as np
# import time
# import pickle
# import random

# # ----------------------
# # Load model components
# # ----------------------
# model = pickle.load(open("Models/NSL_KDD/Random_Forest_Model.pkl", "rb"))
# scaler = pickle.load(open("Models/NSL_KDD/scaler.pkl", "rb"))
# encoder = pickle.load(open("Models/NSL_KDD/One_Hot_Encoded.pkl", "rb"))

# # ----------------------
# # Load dataset for simulation
# # ----------------------
# df_live = pd.read_csv("Dataset/KDD_Train.csv")   # use a small CSV

# categorical_cols = ["flag"]
# numeric_cols = ["packet_length","src_port","dst_port","protocol"]

# # ----------------------
# # Preprocessing function
# # ----------------------
# def preprocess_live(df):
#     df = df.copy()

#     # --------------------------------------
#     # Case 1 → NSL-KDD
#     # --------------------------------------
#     if "flag" in df.columns:
#         categorical_cols = ["protocol_type", "service", "flag"]

#         # Apply one-hot encoder used during training
#         df_encoded = pd.get_dummies(df[categorical_cols])

#         # Align columns with training columns
#         for col in feature_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0

#         df_encoded = df_encoded[feature_columns]

#         # Scale numerical values
#         df_num = scaler.transform(df[numeric_cols])

#         final = pd.concat([
#             pd.DataFrame(df_num, columns=numeric_cols),
#             df_encoded
#         ], axis=1)

#         return final

#     # --------------------------------------
#     # Case 2 → CICIDS
#     # --------------------------------------
#     # else:
#     #     # numeric only
#     #     df_num = Scaler.transform(df[numerical_cols])

#     #     # apply label encoder for multiclass if needed
#     #     # NOTE: label encoder is only for labels, not features
#     #     final = pd.DataFrame(df_num, columns=numerical_cols_cicids)

#     #     return final

# # def preprocess_live(df):
# #     df_cat = encoder.transform(df[categorical_cols])
# #     df_num = scaler.transform(df[numeric_cols])

# #     final = pd.concat([
# #         pd.DataFrame(df_num, columns=numeric_cols),
# #         pd.DataFrame(df_cat, columns=encoder.get_feature_names_out())
# #     ], axis=1)

# #     return final

# # ----------------------
# # Streamlit UI
# # ----------------------
# st.title("🟢 Live Network Monitoring (Simulation Mode)")

# placeholder = st.empty()

# if st.button("Start Live Monitoring"):
#     st.success("Simulation Started...")

#     for i in range(200):          # number of packets to simulate
#         # pick a random row from dataset
#         row = df_live.sample(1).reset_index(drop=True)

#         processed = preprocess_live(row)
#         pred = model.predict(processed)[0]

#         with placeholder.container():
#             st.write("### Latest Packet")
#             st.json(row.to_dict(orient="records")[0])

#             if pred == 1:
#                 st.error("🚨 Intrusion Detected!")
#             else:
#                 st.success("✔ Normal Traffic")

#         time.sleep(0.4)



    #     # If ground truth present, compute metrics
    #     ground_truth_col = None
    #     for cand in ["class", "label", "Label", "CLASS", "Attack", "attack", "binary_attack"]:
    #         if cand in df.columns:
    #             ground_truth_col = cand
    #             break

    #     if ground_truth_col is not None:
    #         y_true = df[ground_truth_col].values
    #         y_pred = preds_decoded

    #         # make sure arrays have same dtype shape for metrics (cast to str for safety)
    #         y_true_s = np.array(y_true).astype(str)
    #         y_pred_s = np.array(y_pred).astype(str)

    #         # choose averaging
    #         avg = "binary" if len(np.unique(y_true_s)) == 2 else "weighted"

    #         try:
    #             acc = accuracy_score(y_true_s, y_pred_s)
    #             prec = precision_score(y_true_s, y_pred_s, average=avg, zero_division=0)
    #             rec = recall_score(y_true_s, y_pred_s, average=avg, zero_division=0)
    #             f1 = f1_score(y_true_s, y_pred_s, average=avg, zero_division=0)

    #             st.write("### Evaluation Metrics")
    #             st.write(f"**Accuracy:** {acc:.4f}")
    #             st.write(f"**Precision ({avg}):** {prec:.4f}")
    #             st.write(f"**Recall ({avg}):** {rec:.4f}")
    #             st.write(f"**F1-score ({avg}):** {f1:.4f}")

    #             # Confusion matrix
    #             labels = np.unique(np.concatenate([y_true_s, y_pred_s]))
    #             cm = confusion_matrix(y_true_s, y_pred_s, labels=labels)
    #             fig, ax = plt.subplots(figsize=(6, 5))
    #             im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    #             ax.set_title("Confusion Matrix")
    #             plt.colorbar(im, ax=ax)
    #             ax.set_xticks(np.arange(len(labels)))
    #             ax.set_yticks(np.arange(len(labels)))
    #             ax.set_xticklabels(labels, rotation=45, ha="right")
    #             ax.set_yticklabels(labels)
    #             for i in range(len(labels)):
    #                 for j in range(len(labels)):
    #                     ax.text(j, i, format(cm[i, j], 'd'),
    #                             ha="center", va="center",
    #                             color="white" if cm[i, j] > cm.max()/2 else "black")
    #             st.pyplot(fig)
    #         except Exception as exc:
    #             st.warning(f"Could not compute metrics: {exc}")
    #     else:
    #         st.info("No ground-truth label column found; skipping metric computation.")

    #     # Download results
    #     csv_bytes = results_df.to_csv(index=False).encode()
    #     st.download_button("Download predictions CSV", csv_bytes, "nids_predictions.csv", "text/csv")

    #     t1 = datetime.now()
    #     st.write(f"Completed in {(t1 - t0).total_seconds():.2f} seconds.")

    
        # do not re-raise so app stays up
