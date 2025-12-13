import pandas as pd
import numpy as np
import os

print("Running CICIDS preprocessing...")

DATA_DIR = "data_cicids"
OUTPUT_DIR = "artifacts"
OUTPUT_FILE = f"{OUTPUT_DIR}/cicids_preprocessed.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# All CICIDS files
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

df_list = []

for file in files:
    path = f"{DATA_DIR}/{file}"
    print(f"Processing: {path}")

    try:
        # Load CSV safely
        df = pd.read_csv(path, low_memory=False)

        # Strip spaces from column names (IMPORTANT!)
        df.columns = df.columns.str.strip()

        # Ensure Label column exists
        if "Label" not in df.columns:
            raise Exception(f"'Label' column missing in: {file}")

        # Drop unnamed empty columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Clean missing / infinite
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        df_list.append(df)

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

# Merge all CICIDS CSVs
if len(df_list) > 0:
    final_df = pd.concat(df_list, ignore_index=True)
    print(f"\nFinal CICIDS shape: {final_df.shape}")

    # Save cleaned output
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"CICIDS Preprocessing Completed Successfully!\nSaved to: {OUTPUT_FILE}")

else:
    print("❌ No valid CSV files found or all failed.")
