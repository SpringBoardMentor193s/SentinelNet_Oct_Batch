# src/preprocess_nsl_grouped.py
import os
import pandas as pd

IN = "artifacts/nsl_train.csv"
OUT = "artifacts/nsl_train_grouped.csv"

if not os.path.exists(IN):
    raise FileNotFoundError(f"Input not found: {IN}")

print("Loading:", IN)
df = pd.read_csv(IN)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:8], "...")

# ------- mapping function -------
def map_attack(label):
    label = str(label).strip().lower()
    dos = {"back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","mailbomb"}
    probe = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
    r2l = {"guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","spy","xlock","xsnoop"}
    u2r = {"buffer_overflow","loadmodule","rootkit","perl","sqlattack","ps","xterm"}

    if label == "normal":
        return "normal"
    if label in dos:
        return "DoS"
    if label in probe:
        return "Probe"
    if label in r2l:
        return "R2L"
    if label in u2r:
        return "U2R"
    return "Other"

# ------- apply mapping -------
if "label" not in df.columns:
    raise KeyError("Expected column 'label' in input CSV")

df["class"] = df["label"].apply(map_attack)

# Show distribution before removing Other
print("Class distribution (before):")
print(df["class"].value_counts())

# Option: remove 'Other' to avoid extremely small categories
df = df[df["class"] != "Other"].copy()
print("After dropping 'Other' rows:", df.shape)

# Save
df.to_csv(OUT, index=False)
print("✔ Grouped file saved to:", OUT)
print("Final class counts:")
print(df["class"].value_counts())
