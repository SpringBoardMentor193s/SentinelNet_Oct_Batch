import pandas as pd
import os

print("Merging NSL-KDD + CICIDS datasets...")

path_nsl = "artifacts/nslkdd_preprocessed.csv"
path_cic = "artifacts/cicids_preprocessed.csv"

df_nsl = pd.read_csv(path_nsl)
df_cic = pd.read_csv(path_cic)

# Convert all column names to lowercase (standardization)
df_nsl.columns = df_nsl.columns.str.lower()
df_cic.columns = df_cic.columns.str.lower()

# Ensure both have a label column
if "label" not in df_cic.columns:
    df_cic.rename(columns={df_cic.columns[-1]: "label"}, inplace=True)

# Get common columns
common_columns = list(set(df_nsl.columns) & set(df_cic.columns))

df_nsl = df_nsl[common_columns]
df_cic = df_cic[common_columns]

# Combine datasets
df_merged = pd.concat([df_nsl, df_cic], ignore_index=True)

# Save combined dataset
output_path = "artifacts/combined_dataset.csv"
df_merged.to_csv(output_path, index=False)

print("Merged Successfully!")
print(f"Saved to: {output_path}")
print("Final Shape:", df_merged.shape)
