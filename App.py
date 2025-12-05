import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
import time
from sklearn.metrics import (
    confusion_matrix, roc_curve,accuracy_score, 
    precision_score, recall_score, f1_score
)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model 

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_feature_columns(columns_path):
    with open(columns_path, 'rb') as f:
        feature_columns = pickle.load(f)
    return feature_columns

def preprocess_input(input_df, feature_columns, scaler):
    if 'difficulty' in input_df.columns:
        input_df = input_df.drop(['difficulty'], axis=1)
    
    categorical_cols=['protocol_type', 'service', 'flag']
    categorical_cols=[col for col in categorical_cols if col in input_df.columns]
    
    input_df_encoded=pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0 
    input_df_encoded = input_df_encoded[feature_columns]

    input_df_encoded = input_df_encoded[feature_columns]
    input_scaled = scaler.transform(input_df_encoded)
    return input_scaled


column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

Models_map = {
    "NSL-KDD": {
        "Logistic_Regression": "Models/NSL_KDD/Logistic_Regression.pkl",
        "Gradient_Boosting": "Models/NSL_KDD/Gradient_Boosting.pkl",
        "K-Nearest_Neighbors": "Models/NSL_KDD/KNN.pkl",
        "Decision_Tree": "Models/NSL_KDD/Decision_Tree.pkl",
        "Random_Forest": "Models/NSL_KDD/Random_Forest.pkl",
        "XGBoost": "Models/NSL_KDD/XGBoost.pkl",
        "LightGBM": "Models/NSL_KDD/LightGBM.pkl",
        "AdaBoost": "Models/NSL_KDD/AdaBoost.pkl",
        "Scaler": "Models/NSL_KDD/scaler.pkl",
        "Feature_Columns": "Models/NSL_KDD/Features.pkl"
    },

    "CICIDS-2017": {
        "Binary":{
            "Logistic_Regression": "Models/CICIDS-2017/Binary_Class/Logistic_Regression.pkl",
            "Decision_Tree": "Models/CICIDS-2017/Binary_Class/Decision_Tree.pkl",
            "Random_Forest": "Models/CICIDS-2017/Binary_Class/Random_Forest.pkl",
            "K-Nearest_Neighbors": "Models/CICIDS-2017/Binary_Class/KNN.pkl",
            "XGBoost": "Models/CICIDS-2017/Binary_Class/XGBoost.pkl",
            "LightGBM": "Models/CICIDS-2017/Binary_Class/LightGBM.pkl"
        },
        "Multiclass":{
            "Logistic_Regression": "Models/CICIDS-2017/Multi_Class/Logistic_Regression.pkl",
            "Decision_Tree": "Models/CICIDS-2017/Multi_Class/Decision_Tree.pkl",
            "Random_Forest": "Models/CICIDS-2017/Multi_Class/Random_Forest.pkl",
            "XGBoost": "Models/CICIDS-2017/Multi_Class/XGBoost.pkl",
            "LightGBM": "Models/CICIDS-2017/Multi_Class/LightGBM.pkl",    
        },
        "Scaler": "Models/CICIDS-2017/Scaler.pkl",
        "Feature_Columns": "Models/CICIDS-2017/Feature_Columns.pkl",
    }
}

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

st.title("SentinelNet - AI-powered Intrusion Detection System (NIDS)")
st.markdown("""
This application allows you to upload network traffic data and evaluate it using pre-trained machine learning models for intrusion detection.
""")

st.sidebar.header("Configuration")

st.sidebar.subheader("Detection Mode")
detection_mode = st.sidebar.radio("Select Mode:", ["Live Monitoring", "File Analysis"])

st.sidebar.subheader("Dataset Selection")
dataset_choice = st.sidebar.selectbox("Choose Dataset:", ["NSL-KDD", "CICIDS-2017"])

if dataset_choice == "CICIDS-2017":
    class_type = st.sidebar.selectbox("Choose Classification Type:", ["Binary", "Multiclass"])
    model_paths = Models_map[dataset_choice]
    model_subpaths = model_paths[class_type]
else:
    model_paths = Models_map[dataset_choice]
    model_subpaths = model_paths

st.sidebar.subheader("Model Selection")
if dataset_choice == "NSL-KDD":
    algorithm = st.sidebar.selectbox("Select Algorithm:", [
        "Logistic_Regression", "Gradient_Boosting", "K-Nearest_Neighbors",
        "Decision_Tree", "Random_Forest", "XGBoost", "LightGBM", "AdaBoost"
    ])
else:
    algorithm = st.sidebar.selectbox("Select Algorithm:", [
        "Logistic_Regression", "Decision_Tree", "Random_Forest",
        "K-Nearest_Neighbors", "XGBoost", "LightGBM"
    ])

if dataset_choice == "NSL-KDD":
    model_path = model_subpaths[algorithm]
    scaler_path = model_paths["Scaler"]
    feature_columns_path = model_paths["Feature_Columns"]
    one_hot_encoder_path = None
else:
    model_path = model_subpaths[algorithm]
    scaler_path = model_paths["Scaler"]
    feature_columns_path = model_paths["Feature_Columns"]
    one_hot_encoder_path = None  

model = load_model(model_path)
scaler = load_scaler(scaler_path)
feature_columns = load_feature_columns(feature_columns_path)

if detection_mode == "Live Monitoring":
    st.subheader("Live Network Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        start=st.button("Start Monitoring")
    with col2:
        clear=st.button("Clear Logs")
    st.markdown("### Monitoring Logs")
    if start:
        st.success("Live monitoring started...")
        # Placeholder for live monitoring logic
    if clear:
        st.success("Logs cleared.")
    m1,m2,m3,m4=st.columns(4)
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

else:
    st.subheader("File-based Traffic Analysis")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file, names=column_names, header=None)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
        time.sleep(1)

        if st.button("Evaluate"):
            st.write("Running model.... Please wait.") 

            X=preprocess_input(df, feature_columns, scaler)
            predictions=model.predict(X)

            df['Prediction'] = predictions
            st.write("### Evaluation Results")
            st.dataframe(df.head())
            
