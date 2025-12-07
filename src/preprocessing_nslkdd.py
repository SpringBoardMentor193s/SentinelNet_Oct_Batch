import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

print("Running NSL-KDD preprocessing...")

RAW_TRAIN = "data_nslkdd/KDDTrain+.txt"
RAW_TEST = "data_nslkdd/KDDTest+.txt"
OUTPUT_DIR = "artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

# Load
train_df = pd.read_csv(RAW_TRAIN, names=columns)
test_df = pd.read_csv(RAW_TEST, names=columns)

df = pd.concat([train_df, test_df], ignore_index=True)

# Remove difficulty column
df.drop("difficulty", axis=1, inplace=True)

# -----------------------
# CREATE attack categories
# -----------------------
attack_mapping = {
    # DoS Attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS",
    
    # Probe Attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    
    # R2L Attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L",
    
    # U2R Attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R",
    "perl": "U2R", "rootkit": "U2R"
}

# Add attack_type column
df["attack_type"] = df["label"].apply(lambda x: attack_mapping.get(x, "normal"))

# -----------------------------------------
# LABEL ENCODING FOR CATEGORICAL FEATURES
# -----------------------------------------
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Save file
output_path = os.path.join(OUTPUT_DIR, "nslkdd_preprocessed.csv")
df.to_csv(output_path, index=False)

print("NSL-KDD Preprocessing Completed!")
print(f"Saved → {output_path}")
