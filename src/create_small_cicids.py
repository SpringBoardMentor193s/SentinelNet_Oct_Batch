import pandas as pd

INPUT = "artifacts/cicids_preprocessed.csv"
OUTPUT = "artifacts/cicids_small.csv"

print("Loading only LABEL column...")
labels = pd.read_csv(INPUT, usecols=["Label"])
print("Total rows in dataset:", len(labels))

# Sample 150k rows safely
sample_indices = labels.sample(n=150000, random_state=42).index

print("Reading only selected rows...")
df = pd.read_csv(INPUT, skiprows=lambda x: x not in sample_indices and x != 0)

print("Saving small dataset...")
df.to_csv(OUTPUT, index=False)

print("✔ Small CICIDS dataset created:", OUTPUT)
