import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="SentinelNet - Cyber Threat Detector", layout="wide")

# ---------- HEADER ----------
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50; font-size:50px;'>SentinelNet - Cyber Threat Detection</h1>",
    unsafe_allow_html=True,
)

st.write("")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Control Panel")
option = st.sidebar.selectbox("Choose Action:", ["View Dataset", "Select Model & Predict"])

model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ["RandomForest", "SVM", "LogisticRegression", "DecisionTree", "KNN"]
)

# ---------- LOAD DATA ----------
df = pd.read_csv("models/nsl_kdd_sample.csv")

enc = LabelEncoder()
for col in ["protocol_type", "service", "flag", "label"]:
    df[col] = enc.fit_transform(df[col])

# ---------- VIEW DATA ----------
if option == "View Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

# ---------- PREDICTION UI ----------
if option == "Select Model & Predict":
    st.subheader("🔍 Make Prediction")

    duration = st.number_input("Duration", min_value=0, max_value=10000)
    src_bytes = st.number_input("Source Bytes", min_value=0, max_value=20000)
    dst_bytes = st.number_input("Destination Bytes", min_value=0, max_value=20000)

    protocol = st.selectbox("Protocol Type", enc.classes_)
    service = st.selectbox("Service", enc.classes_)
    flag = st.selectbox("Flag Type", enc.classes_)

    if st.button("Predict"):
        model_path = f"models/{model_choice}.pkl"
        model = joblib.load(model_path)

        data = pd.DataFrame([[duration, protocol, service, flag, src_bytes, dst_bytes]],
                            columns=["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"])
        for col in ["protocol_type", "service", "flag"]:
            data[col] = enc.transform(data[col])

        pred = model.predict(data)[0]

        if pred == 0:
            st.success("Result: ✔ NORMAL TRAFFIC")
        else:
            st.error("Result: ❌ ANOMALY DETECTED")

