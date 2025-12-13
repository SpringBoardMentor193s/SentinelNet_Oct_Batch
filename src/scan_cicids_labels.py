import pandas as pd

print("🔍 Scanning CICIDS label classes...")

DATA_PATH = "artifacts/cicids_preprocessed.csv"
CHUNK_SIZE = 200000

labels = set()

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    labels.update(chunk["Label"].astype(str).unique())

print("\n✔ ALL UNIQUE LABELS FOUND:")
for lbl in sorted(labels):
    print(lbl)

print("\nTotal:", len(labels))
