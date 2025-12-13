import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "artifacts")
INPUT = os.path.join(ART, "nsl_train.csv")
OUTPUT = os.path.join(ART, "nsl_train_small.csv")

CHUNK = 5000  # small chunk

use_cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]

print("Processing NSL Train with Python engine...")

first = True
for chunk in pd.read_csv(
        INPUT,
        chunksize=CHUNK,
        usecols=use_cols,
        engine="python"  # FIX: Python engine safe for big files
):
    chunk.to_csv(OUTPUT, index=False, mode='w' if first else 'a', header=first)
    first = False

print("✔ Reduced dataset saved at:", OUTPUT)

