
import streamlit as st

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="-",
    layout="wide"
)

st.title("SentinelNet Network Intrusion Detection Using AI")
st.markdown("Enter Network Traffic details")


st.sidebar.header("Enter the N/W Traffic Details")

duration = st.sidebar.slider("duration (sec)", 1, 1000)

protocol = st.sidebar.selectbox("protocol", ['None', 'tcp', 'udp', 'icmp'])

auth = st.sidebar.radio("Auth", ['Owner', 'guest', 'Temporary', 'Admin'])