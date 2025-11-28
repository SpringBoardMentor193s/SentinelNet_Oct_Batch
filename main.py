import streamlit as st

st.set_page_config(
    page_title=" Default Risk Predictor",
    layout="wide"
)

st.title("SentinelNet Network Intrusion Detection using AI")

st.markdown("Enter Network Traffic Details")

st.sidebar.header("Input Parameters")

duration = st.sidebar.slider(
    "Duration (in seconds)",
    min_value=0,
    max_value=5000,
    value=10
)

protocol = st.sidebar.selectbox(
    "Protocol Type",
    ("tcp", "udp", "icmp")
)

st.subheader("Selected Input Values")
st.write(f"**Duration:** {duration} sec")
st.write(f"**Protocol:** {protocol}")

if st.button("Predict Intrusion"):
    st.success("Prediction: Normal Traffic")
