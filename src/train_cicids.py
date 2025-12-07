import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Training CICIDS model...")

DATA_PATH = "artifacts/cicids_preprocessed.csv"
CHUNK_SIZE = 150000

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=18,
    n_jobs=-1
)

first_batch = True
chunk_id = 1

# --- UNIVERSAL LABEL COLUMN DETECTION ---
def find_label_column(columns):
    for col in columns:
        if col.strip().lower() == "label":
            return col
    raise Exception("Label column not found in dataset!")

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    print(f"\nTraining on chunk {chunk_id}...")

    # Remove unnamed columns
    chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]

    # Detect label column
    label_col = find_label_column(chunk.columns)

    # Convert to numeric smoothly
    chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = chunk.drop(label_col, axis=1)
    y = chunk[label_col]

    if first_batch:
        model.fit(X, y)
        first_batch = False
    else:
        model.fit(X, y)

    print(f"Chunk {chunk_id} training complete.")
    chunk_id += 1

# Save final model
joblib.dump(model, "artifacts/cicids_model.pkl")
print("\nCICIDS Model Saved Successfully at artifacts/cicids_model.pkl")
