# src/realtime_sniffer.py
import os
import time
import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, Raw

ARTIFACTS = "artifacts"
LOG_FILE = os.path.join(ARTIFACTS, "realtime_log.csv")

# load models (if exist)
nsl_model_path = os.path.join(ARTIFACTS, "nslkdd_model.pkl")
cic_model_path = os.path.join(ARTIFACTS, "cicids_model.pkl")

nsl_model = joblib.load(nsl_model_path) if os.path.exists(nsl_model_path) else None
cic_model = joblib.load(cic_model_path) if os.path.exists(cic_model_path) else None

# helper: take model, expected features, and a partial row (dict) -> full DF row
def build_row_for_model(model, partial_dict):
    features = list(model.feature_names_in_)
    row = {c: 0 for c in features}
    for k, v in partial_dict.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row], columns=features)

# ensure artifacts dir exists
os.makedirs(ARTIFACTS, exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp","src","dst","proto","pkt_len","dataset","prediction"]).to_csv(LOG_FILE,index=False)

def packet_handler(pkt):
    try:
        if not pkt.haslayer(IP):
            return

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto  # integer protocol
        pkt_len = len(pkt)

        # Build minimal partial features. Map a few names that match NSL features if possible.
        partial = {}
        # Many NSL features are flow-level; we can't reconstruct them, set common ones:
        # set src_bytes / dst_bytes to pkt_len as approximation
        partial["src_bytes"] = pkt_len
        partial["dst_bytes"] = 0  # unknown
        # If model feature names include 'protocol_type' we need numeric encoding used in preprocessing.
        # We'll try to set some common fields if present in model
        # Choose dataset based on available model (prefer NSL first)
        model = nsl_model if nsl_model is not None else cic_model
        dataset = "NSL-KDD" if nsl_model is not None else "CICIDS"

        row_df = build_row_for_model(model, partial)
        pred = model.predict(row_df)[0]
        # convert label to human
        pred_label = str(pred)
        # Save to log
        rec = {"timestamp": ts, "src": src, "dst": dst, "proto": proto, "pkt_len": pkt_len, "dataset": dataset, "prediction": pred_label}
        pd.DataFrame([rec]).to_csv(LOG_FILE, mode="a", header=False, index=False)
        print(f"{ts} {src} -> {dst} proto={proto} len={pkt_len} -> {pred_label}")
    except Exception as e:
        print("Error handling packet:", e)

if __name__ == "__main__":
    print("Starting packet sniffing (press Ctrl+C to stop). Requires admin privileges.")
    sniff(prn=packet_handler, store=False)
