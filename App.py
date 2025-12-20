import os
import io
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc

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

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# -------------------- Preprocessing --------------------
def preprocess_nsl(df, feature_columns, scaler, names):
    df_proc = df.copy()
    categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in df_proc.columns]

    if "difficulty" in df_proc.columns:
        df_proc = df_proc.drop(columns=["difficulty"])

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
        "Scaler": "Models/CICIDS-2017/Multi_Class/Scaler.pkl",
        "Feature_Columns": "Models/CICIDS-2017/Feature_Columns.pkl",
        "Label_Encoder": "Models/CICIDS-2017/Multi_Class/Label_Encoder.pkl"
    }
}

def convert_to_binary(label):
        label = str(label).upper()
        if label == "BENIGN":
            return "Normal"
        elif label == "NORMAL":  
            return "Normal"
        else:
            return "Intrusion"

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
st.sidebar.markdown(f"<div class='metric-value'>{metrics['accuracy']*100:.2f}%</div>", unsafe_allow_html=True)
st.sidebar.subheader("Precision:")
st.sidebar.markdown(f"<div class='metric-value'>{metrics['precision']*100:.2f}%</div>", unsafe_allow_html=True)

if dataset_choice == "NSL-KDD":
        model_path = MODELS_MAP["NSL-KDD"]["Models"][algorithm]
        scaler_path = MODELS_MAP["NSL-KDD"]["Scaler"]
        features_path = MODELS_MAP["NSL-KDD"]["Feature_Columns"]
else:
        model_path = MODELS_MAP["CICIDS-2017"][class_type][algorithm]
        scaler_path = MODELS_MAP["CICIDS-2017"]["Scaler"]
        features_path = MODELS_MAP["CICIDS-2017"]["Feature_Columns"]
        label_encoder_path = MODELS_MAP["CICIDS-2017"].get("Label_Encoder")

 
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

            if X.shape[0] == 0:
                 st.error("Error: Preprocessing resulted in 0 samples. Check your feature columns and input data.")
                 st.stop()
            preds = model.predict(X)

            if label_encoder is not None:
                try:
                    preds_decoded = label_encoder.inverse_transform(preds)
                except Exception:
                    preds_decoded = preds.astype(str)
            else:
                preds_decoded = preds.astype(str)

            if dataset_choice == "CICIDS-2017" and class_type == "Multiclass":
                preds_decoded = [
                    "Benign" if str(p).strip().lower() == "benign"
                    else "DoS" if str(p).strip() == "DoS"
                    else "DDoS" if str(p).strip() == "DDoS"
                    else "Intrusion"
                    for p in preds_decoded
                ]

            elif dataset_choice == "CICIDS-2017" and class_type == "Binary":
                preds_decoded = ["Normal" if str(p).lower() in ["0", "benign", "BENIGN","normal"] else "Intrusion" for p in preds_decoded]

            elif dataset_choice=="NSL-KDD":
                preds_decoded = ["Normal" if str(p).lower() in ["0", "normal"] else "Intrusion" for p in preds_decoded]

            results_df = df.reset_index(drop=True).copy()
            results_df["Prediction"] = preds_decoded
            
            st.success("Prediction finished.")
            st.write("### Results")
            st.dataframe(results_df.head())
          
            st.subheader("Dataset Traffic Overview")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Traffic Distribution:")
                if dataset_choice == "NSL-KDD":
                    traffic_counts = results_df["Prediction"].value_counts()
                    pie_color={
                        "Normal": "Green",
                        "Intrusion":"Red"
                    }
                    fig_traffic = px.pie(
                        names=traffic_counts.index,
                        values=traffic_counts.values,  
                        hole=0.3
                    )
                    fig_traffic.update_traces(
                        marker=dict(
                        colors=[pie_color[label_encoder]for label_encoder in traffic_counts.index]
                        )
                    )
                    fig_traffic.update_layout(
                        height=250,
                        width=250,
                        margin=dict(l=0,r=0,t=40,b=0)
                    )
                    st.plotly_chart(fig_traffic, config={"responsive": True},width=500)                                

                elif dataset_choice=="CICIDS-2017" and class_type=="Binary":
                        traffic_counts = results_df["Prediction"].value_counts()
                        pie_color={
                            "Normal": "Green",
                            "Intrusion":"Red"
                        }
                        fig_traffic = px.pie(
                            names=traffic_counts.index,
                            values=traffic_counts.values,  
                            hole=0.3
                        )
                        fig_traffic.update_traces(
                            marker=dict(
                            colors=[pie_color[label_encoder]for label_encoder in traffic_counts.index]
                            )
                        )
                        fig_traffic.update_layout(
                            height=250,
                            width=250,
                            margin=dict(l=0,r=0,t=40,b=0)
                        )
                        st.plotly_chart(fig_traffic, config={"responsive": True},width=500)

                else:
                    attack_classes=["Benign","DoS","DDoS"]
                    traffic_counts = results_df["Prediction"].value_counts().reindex(attack_classes, fill_value=0)
                    pie_color={
                        "Benign":"Teal",
                        "DoS": "Blue",
                        "DDoS":"Violet"
                    }
                    fig_traffic = px.pie(
                        names=traffic_counts.index,
                        values=traffic_counts.values,  
                        hole=0.3,
                        color=traffic_counts.index,
                        color_discrete_map=pie_color
                    )
                    fig_traffic.update_traces(
                        sort=False
                    )
                    fig_traffic.update_layout(
                        height=250,
                        width=250,
                        showlegend=True,
                        margin=dict(l=0,r=0,t=40,b=0)
                    )
                    st.plotly_chart(fig_traffic, config={"responsive": True},width=500)
     
            with col2:
                if dataset_choice == 'NSL-KDD':
                    st.write("Protocol Distribution:")
                    protocol_col = None
                    possible_cols = ["protocol_type", "protocol", "Protocol", "Protocol_Type"]

                    for c in possible_cols:
                        if c in df.columns:
                            protocol_col = c
                            break

                    if protocol_col is None:
                        st.error("❌ No protocol column found in uploaded dataset.")
                    else:
                        protocol_counts = df[protocol_col].value_counts().reset_index()
                        protocol_counts.columns=["Protocol","Count"]
                        protocol_color={
                            "tcp": "Blue",
                            "udp": "Yellow",
                            "icmp": "Violet"
                        }
                        fig_protocol = px.bar(
                            protocol_counts,
                            x="Protocol",
                            y="Count",   
                        )
                        fig_protocol.update_layout(
                            height=300,  
                            uniformtext_minsize=12,
                            uniformtext_mode="hide",
                            margin=dict(t=20,b=20)
                        )
                        fig_protocol.update_traces(
                            marker_color=[
                                protocol_color.get(proto, "#1f77b4")
                                for proto in protocol_counts["Protocol"]
                            ],
                        textposition="outside" 
                        )
                        st.plotly_chart(fig_protocol, config={"responsive": True}, width=500)

                elif dataset_choice=="CICIDS-2017" and class_type=="Binary":
                    st.write("Label Distribution:")
                    label_col = None

                    for c in ["Label", "label"]:
                        if c in df.columns:
                            label_col = c
                            break

                    if label_col is None:
                        st.error("❌ No Label column found in CICIDS dataset.")
                    else:
                        df_labels = df[[label_col]].copy()
                        def map_cicids_labels(label):
                            label = str(label).upper()
                            if label == "BENIGN":
                                return "Benign"
                            elif "DDOS" in label:
                                return "DDoS"
                            elif "DOS" in label:
                                return "DoS"
                            else:
                                return None 

                        df_labels["Mapped_Label"] = df_labels[label_col].apply(map_cicids_labels)
                        df_labels = df_labels[df_labels["Mapped_Label"].notna()]

                        label_counts = df_labels["Mapped_Label"].value_counts().reset_index()
                        label_counts.columns = ["Label", "Count"]
                        color_map = {
                            "Benign":"Blue",
                            "DoS": "Yellow",
                            "DDoS": "Violet"
                        }
                        fig = px.bar(
                            label_counts,
                            x="Label",
                            y="Count",
                            color="Label",
                            color_discrete_map=color_map
                        )
                        fig.update_traces(
                            textposition="outside"
                            )
                        fig.update_layout(
                            showlegend=False,
                            height=300,
                            uniformtext_minsize=12,
                            uniformtext_mode="hide",
                            margin=dict(t=20, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.write("Label Distribution:")
                    label_col = None

                    for c in ["Label", "label"]:
                        if c in df.columns:
                            label_col = c
                            break

                    if label_col is None:
                        st.error("❌ No Label column found in CICIDS dataset.")
                    else:
                        df_labels = df[[label_col]].copy()
                        def map_cicids_labels(label):
                            label = str(label).upper()
                            if label == "BENIGN":
                                return "Benign"
                            elif "DDOS" in label:
                                return "DDoS"
                            elif "DOS" in label:
                                return "DoS"
                            else:
                                return "Intrusion"  

                        df_labels["Mapped_Label"] = df_labels[label_col].apply(map_cicids_labels)
                        df_labels = df_labels[df_labels["Mapped_Label"].notna()]

                        label_counts = df_labels["Mapped_Label"].value_counts().reset_index()
                        label_counts.columns = ["Label", "Count"]
                        color_map = {
                            "Benign": "Blue",
                            "DoS": "Yellow",
                            "DDoS": "Violet",
                            "Intrusion": "Red"
                        }
                        fig = px.bar(
                            label_counts,
                            x="Label",
                            y="Count",
                            color="Label",
                            color_discrete_map=color_map
                        )
                        fig.update_traces(
                            textposition="outside"
                            )
                        fig.update_layout(
                            showlegend=False,
                            height=300,
                            uniformtext_minsize=12,
                            uniformtext_mode="hide",
                            margin=dict(t=20, b=20,l=0,r=0)
                        )
                        st.plotly_chart(fig,width=500, config={"responsive":True})
        
            st.subheader("Classification Summary")
            c1, c2 = st.columns(2)
            with c1:
                if dataset_choice=="NSL-KDD" or dataset_choice=="CICIDS-2017" and class_type=="Binary":
                    st.write("Normal vs Intrusion Distribution:")
                    bar_counts = results_df["Prediction"].value_counts().reset_index()
                    bar_counts.columns = ["Label", "Count"]
                    colors = {
                        "Normal": "Teal",      
                        "Intrusion": "Red"
                    }   
                    fig_bar = px.bar(
                        bar_counts,
                        x="Label",
                        y="Count",
                        color="Label",
                        color_discrete_map=colors
                    )
                    fig_bar.update_layout(
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, width=500, config={"responsive": True})

                else:
                    st.write("Multiclass Distribution:")
                    classes = ["Benign", "DoS", "DDoS", "Intrusion"]
                    def normalize_label(label):
                        label = str(label).lower()
                        if label == "benign":
                            return "Benign"
                        elif label == "dos":
                            return "DoS"
                        elif label == "ddos":
                            return "DDoS"
                        else:
                            return "Intrusion"
                    results_df["Label"] = results_df["Label"].apply(normalize_label)
                    results_df["Prediction"] = results_df["Prediction"].apply(normalize_label)
                    results_df=results_df.dropna(subset=["Label","Prediction"])
                    actual_counts = results_df["Label"].value_counts().reindex(classes, fill_value=0)
                    pred_counts = results_df["Prediction"].value_counts().reindex(classes, fill_value=0)

                    bar_df = pd.DataFrame({
                        "Class": classes,
                        "Actual": actual_counts.values,
                        "Predicted": pred_counts.values
                    })
                    bar_long = bar_df.melt(
                        id_vars="Class",
                        value_vars=["Actual", "Predicted"],
                        var_name="Type",
                        value_name="Count"
                    )
                    fig=go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=classes,
                            y=actual_counts.values,
                            name="Actual",
                            marker_color="Red"
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            x=classes,
                            y=pred_counts.values,
                            name="Predicted",
                            marker_color="Blue"
                        )
                    )
                    fig.update_layout(
                        barmode="group",
                        height=350,
                        xaxis_title="Traffic Class",
                        yaxis_title="Count",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, width=500,config={"Responsive":True})
                    
            with c2:
                st.write("Confusion Matrix:")
                if dataset_choice=="NSL-KDD":
                    possible_actual_cols =["Label", "label", "Class", "class","Actual","actual"]
                    actual_col=None

                    for col in possible_actual_cols:
                        if col in df.columns:
                            actual_col=col
                            break

                    if actual_col is None:
                        st.error("Error: No actual label column found in uploaded dataset.")
                        st.stop()

                    y_true=df[actual_col].astype(str)
                    mapping={
                        0:"Normal",
                        1:"Intrusion",
                        "0": "Normal",
                        "1":"Intrusion",
                        "normal": "Normal", "anomaly":"Intrusion",
                        "attack": "Intrusion"
                    }
                    y_true=y_true.replace(mapping)
                    y_pred=results_df["Prediction"].astype(str).replace(mapping)
                    labels=["Normal","Intrusion"]
                    y_true=y_true.apply(lambda x: x if x in labels else "Intrusion")
                    y_pred=y_pred.apply(lambda x: x if x in labels else "Intrusion")

                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

                    fig_cm = px.imshow(
                        cm_df,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count")
                    )
                    fig_cm.update_layout(
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_cm, width=500, config={"responsive": True})

                elif dataset_choice=="CICIDS-2017" and class_type == "Binary":
                    possible_actual_cols = ["Label", "label", "Class", "class", "Actual", "actual"]
                    actual_col = None

                    for col in possible_actual_cols:
                        if col in df.columns:
                            actual_col = col
                            break

                    if actual_col is None:
                        st.error("Error: No actual label column found in uploaded dataset.")
                        st.stop()

                    y_true = df[actual_col].astype(str)
                    mapping = {
                        0: "Normal",
                        1: "Intrusion",
                        "0": "Normal",
                        "1": "Intrusion",
                        "BENIGN": "Normal",
                        "benign": "Normal",
                        "Attack": "Intrusion",
                        "attack": "Intrusion"
                    }
                    y_true = y_true.replace(mapping)
                    y_pred = results_df["Prediction"].astype(str).replace(mapping)
                    labels = ["Normal", "Intrusion"]
                    y_true = y_true.apply(lambda x: x if x in labels else "Intrusion")
                    y_pred = y_pred.apply(lambda x: x if x in labels else "Intrusion")

                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

                    fig_cm = px.imshow(
                        cm_df,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count")
                    )
                    fig_cm.update_layout(
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_cm, width=500, config={"responsive": True})
                            
                elif dataset_choice == "CICIDS-2017" and class_type == "Multiclass":
                    labels = ["Benign", "DoS", "DDoS", "Intrusion"]
                    def map_cicids_label(label):
                        if pd.isna(label):
                            return None
                        label = str(label).strip().upper()
                        if label == "BENIGN":
                            return "Benign"
                        elif label == "DOS":
                            return "DoS"
                        elif label == "DDOS":
                            return "DDoS"
                        else:
                            return "Intrusion"
                    possible_actual_cols = ["Label", "label", "Class", "class", "Actual", "actual"]
                    actual_col = None

                    for col in possible_actual_cols:
                        if col in df.columns:
                            actual_col = col
                            break

                    if actual_col is None:
                        st.error("❌ No actual label column found in CICIDS dataset.")
                        st.stop()
                        
                    y_true = df[actual_col].apply(map_cicids_label)
                    y_pred = results_df["Prediction"].apply(map_cicids_label)

                    mask = y_true.notna() & y_pred.notna()
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]

                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

                    fig_cm = px.imshow(
                        cm_df,
                        text_auto=True,
                        color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual", color="Count")
                    )
                    fig_cm.update_layout(
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_cm, config={"Resposive":True},width=500)   

            if dataset_choice == "NSL-KDD" or dataset_choice == "CICIDS-2017" and class_type == "Binary":
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]  
                else:
                    st.warning("Model does not support probability prediction. ROC curve cannot be computed.")
                    y_prob = None

                if y_prob is not None:
                    y_true_bin = y_true.map({"Normal":0, "Intrusion":1}).values

                    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob)
                    roc_auc = auc(fpr, tpr)

                    st.write("ROC Curve:")
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        name='ROC Curve',
                        line=dict(width=2)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Random',
                        line=dict(dash='dash', color='gray')
                    ))
                    fig_roc.add_annotation(
                        x=fpr[-1],
                        y=tpr[-1],
                        text=f"AUC = {roc_auc:.3f}",
                        showarrow=True,
                        arrowhead=2
                    )
                    fig_roc.update_layout(
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        width=600,
                        height=400,
                        margin=dict(l=10, r=10, t=40, b=10),
                        template='plotly_white',
                        showlegend =False
                    )
                    st.plotly_chart(fig_roc, config={"responsive": True})
            
            else:
                X_test = df.drop(columns=["Label"])
                X_test_scaled = scaler.transform(X_test)
                y_test_str = df["Label"]
                y_test_str = y_test_str.apply(
                    lambda x: "Benign" if str(x).upper() == "BENIGN"
                    else "DoS" if "DOS" in str(x).upper() and "DDOS" not in str(x).upper()
                    else "DDoS" if "DDOS" in str(x).upper()
                    else "Intrusion"
                )
                y_test = label_encoder.transform(y_test_str)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_scaled)
                    class_names = label_encoder.classes_
                    n_classes = len(class_names)
                    y_test_bin = label_binarize(
                        y_test,
                        classes=range(n_classes)
                    )
                    st.write("ROC Curve:")
                    fig_roc = go.Figure()
                    class_colors = {
                        "Benign": "Blue",      
                        "DoS": "Yellow",         
                        "DDoS": "Violet",        
                        "Intrusion": "Red"   
                    }
                    for i, class_name in enumerate(class_names):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)

                        fig_roc.add_trace(
                            go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode="lines",
                                name=f"{class_name} (AUC = {roc_auc:.3f})",
                                line=dict(
                                    width=2,
                                    color=class_colors.get(class_name, "#000000")
                                )
                            )
                        )
                    fig_roc.add_shape(
                        type="line",
                        x0=0, x1=1, y0=0, y1=1,
                        line=dict(dash="dash", color="gray")
                    )
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        width=600,
                        height=400,
                        legend_title="Classes",
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_roc, config={"responsive": True})
                else:
                    st.warning("Model does not support probability prediction. ROC curve cannot be computed.")

            st.subheader("Full Predicted Dataset")
            st.dataframe(results_df)  

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="Download",
                data=csv_data,
                file_name="predicted_dataset.csv",
                mime="text/csv"
            )
        except Exception as err:
            st.error(f"Error during evaluation: {err}")