# src/train_nslkdd.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "artifacts")

os.makedirs(ART, exist_ok=True)

TRAIN_PATH = os.path.join(ART, "nsl_train.csv")
TEST_PATH  = os.path.join(ART, "nsl_test.csv")

# --- NSL feature list (41) - fallback if nsl_features.pkl missing
FALLBACK_NSL_FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# Try load features file if present
FEATURES_PKL = os.path.join(ART, "nsl_features.pkl")
if os.path.exists(FEATURES_PKL):
    try:
        NSL_FEATURES = joblib.load(FEATURES_PKL)
        print("Loaded feature list from", FEATURES_PKL)
    except Exception as e:
        print("Failed to load nsl_features.pkl, using fallback:", e)
        NSL_FEATURES = FALLBACK_NSL_FEATURES
else:
    NSL_FEATURES = FALLBACK_NSL_FEATURES

# Check files
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")
print("Train file found:", TRAIN_PATH)

# Load
df = pd.read_csv(TRAIN_PATH)
print("Raw train shape:", df.shape)

# Expect label column name 'label' (lowercase). If not present, try common variants.
label_col = None
for c in ["label", "Label", " attack", "attack", "class"]:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    raise ValueError("No label column found in training data. Expected 'label' or 'Label'.")

print("Using label column:", label_col)

# Keep only expected feature columns + label (if dataset has extra cols, drop them)
missing_feats = [c for c in NSL_FEATURES if c not in df.columns]
if missing_feats:
    print("Warning - some NSL features missing in train file. Missing count:", len(missing_feats))
    print("Missing example features (up to 10):", missing_feats[:10])
    # For missing numeric features we'll fill later. For critical categorical missing -> raise.
    # We'll create columns of zeros for missing numeric features to keep ordering.
    for c in missing_feats:
        df[c] = 0.0

# Ensure correct order
X = df[NSL_FEATURES].copy()
y = df[label_col].astype(str).copy()

print("Prepared X shape:", X.shape, "y shape:", y.shape)
print("Label distribution before resampling:\n", y.value_counts())

# Split into train/validation for early evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Categorical and numeric columns
cat_cols = [c for c in ["protocol_type", "service", "flag"] if c in NSL_FEATURES]
num_cols = [c for c in NSL_FEATURES if c not in cat_cols]

print("Cat cols:", cat_cols)
print("Num cols:", len(num_cols))

# Preprocessing
ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, cat_cols),
        ("num", scaler, num_cols)
    ],
    remainder="drop",
    sparse_threshold=0
)

# Build imbalanced-learn pipeline: preprocess -> SMOTE -> classifier
rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=None)

pipe = ImbPipeline(steps=[
    ("pre", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", rf)
])

# Grid search for RF hyperparams (keeps runtime reasonable)
param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [12, 20],
    "clf__min_samples_split": [2, 5]
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)

print("Starting GridSearchCV training (this may take a while)...")
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# Evaluate on holdout validation
best_pipe = grid.best_estimator_

y_pred_val = best_pipe.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
print(f"Validation accuracy: {acc_val:.4f}")
print("Classification report (val):\n", classification_report(y_val, y_pred_val))

# Optionally evaluate on test file if present
if os.path.exists(TEST_PATH):
    print("Test file found:", TEST_PATH)
    test_df = pd.read_csv(TEST_PATH)
    if label_col not in test_df.columns:
        # attempt to find label alternative
        for alt in ["Label", "label", "class"]:
            if alt in test_df.columns:
                label_col_test = alt
                break
        else:
            label_col_test = None
    else:
        label_col_test = label_col

    if label_col_test:
        X_test = test_df[NSL_FEATURES].copy()
        y_test = test_df[label_col_test].astype(str).copy()
        # fill missing columns if any
        for c in NSL_FEATURES:
            if c not in X_test.columns:
                X_test[c] = 0.0
        y_pred_test = best_pipe.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        print("Test accuracy:", acc_test)
        print("Test classification report:\n", classification_report(y_test, y_pred_test))
    else:
        print("Test label column not found in test CSV, skipping test evaluation.")

# Save the final pipeline
OUT_MODEL = os.path.join(ART, "nslkdd_model.pkl")
joblib.dump(best_pipe, OUT_MODEL)
print("Saved model pipeline to", OUT_MODEL)

# Also save canonical features list (ordered)
joblib.dump(NSL_FEATURES, FEATURES_PKL)
print("Saved feature list to", FEATURES_PKL)

print("Training complete.")
