# Imports and configuration

import json
import joblib
import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

TRAIN_CSV = "kdd_train.csv"
TEST_CSV = "kdd_test.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
BINARY_TARGET = True
USE_SMOTE = False
USE_LIGHTGBM = True
N_RANDOM_SEARCH = 6
DO_FEATURE_SELECTION = True
FEATURE_IMPORTANCE_THRESHOLD = 1e-3

# File existence validation

for f in (TRAIN_CSV, TEST_CSV):
    if not os.path.exists(f):
        raise FileNotFoundError(f"Required file not found in current folder: {f}")

# Load and combine datasets

print("[INFO] Loading CSV files from current folder...")
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
print(f"[INFO] train: {df_train.shape}   test: {df_test.shape}")

df = pd.concat([df_train, df_test], ignore_index=True)
print("[INFO] combined shape:", df.shape)

# Target column preparation

label_col = df.columns[-1]
print("[INFO] Using last column as label:", label_col)

df[label_col] = df[label_col].astype(str)

if BINARY_TARGET:
    print("[INFO] Converting to binary target: normal vs attack")
    df["target_binary"] = df[label_col].apply(
        lambda x: "normal"
        if str(x).strip().lower() in ("normal", "normal.", "0", "benign")
        else "attack"
    )
    target_source = "target_binary"
else:
    target_source = label_col

# Encode target labels

le = LabelEncoder()
df["target_enc"] = le.fit_transform(df[target_source].astype(str))
joblib.dump(le, "label_encoder.joblib")
print("[INFO] Label encoder saved. Classes:", list(le.classes_))

# Feature matrix construction

X = df.drop(columns=[label_col, target_source, "target_enc"], errors="ignore")
y = df["target_enc"].astype(int)

for c in ["Unnamed: 0", "num_outbound_cmds", "difficulty", "service_flags"]:
    if c in X.columns:
        X = X.drop(columns=c)

# Feature type detection

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for c in X.select_dtypes(include=[np.number]).columns:
    if X[c].nunique() <= 10 and c not in cat_cols:
        cat_cols.append(c)
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"[INFO] Detected {len(num_cols)} numeric cols and {len(cat_cols)} categorical cols")

# Missing value handling

if num_cols:
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
if cat_cols:
    X[cat_cols] = X[cat_cols].fillna("missing")

# Ordinal encoding for categoricals

ordinal = None
if len(cat_cols) > 0:
    print("[INFO] Fitting OrdinalEncoder for categorical columns...")
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = X[cat_cols].astype(str)
    X_cat_enc = ordinal.fit_transform(X_cat)
    X_enc = X.copy()
    X_enc[cat_cols] = X_cat_enc
    joblib.dump(ordinal, "ordinal_encoder.joblib")
    print("[INFO] Saved ordinal_encoder.joblib")
else:
    X_enc = X.copy()

feature_list = X_enc.columns.tolist()
with open("feature_list.json", "w") as f:
    json.dump(feature_list, f)
print("[INFO] Saved feature_list.json (features count: %d)" % len(feature_list))

# Train / test split

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("[INFO] X_train:", X_train.shape, " X_test:", X_test.shape)

# Optional SMOTE balancing

if USE_SMOTE:
    print("[INFO] Applying SMOTE to training set (this may take time)...")
    try:
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("[INFO] After SMOTE, X_train:", X_train.shape)
    except Exception as e:
        print("[WARN] SMOTE failed or imblearn not installed:", e)

# Model choice (LightGBM / RF)

use_lgb = False
lgbm_clf = None
if USE_LIGHTGBM:
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier

        lgbm_clf = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        use_lgb = True
        print("[INFO] LightGBM detected and will be considered for tuning (recommended).")
    except Exception as e:
        use_lgb = False
        print(
            "[WARN] LightGBM not available, falling back to RandomForest. "
            "Install lightgbm for better speed/accuracy."
        )

# Pipeline and hyperparameter search

if use_lgb:
    from lightgbm import LGBMClassifier

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_dist = {
        "clf__num_leaves": [31, 50, 80],
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [-1, 10, 20],
        "clf__learning_rate": [0.01, 0.05, 0.1],
    }
else:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    param_dist = {
        "clf__n_estimators": [100, 150, 250],
        "clf__max_depth": [None, 10, 20, 40],
        "clf__min_samples_split": [2, 4, 8],
    }

rs = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=N_RANDOM_SEARCH,
    scoring="f1",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
print("[INFO] Running RandomizedSearchCV (this may take several minutes depending on dataset size)...")
rs.fit(X_train, y_train)
best = rs.best_estimator_
print("[INFO] Best params:", rs.best_params_)

best.fit(X_train, y_train)

# Optional feature selection

if DO_FEATURE_SELECTION:
    try:
        print("[INFO] Performing feature selection based on importance...")
        clf = best.named_steps["clf"]
        importances = None

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "booster_") and hasattr(clf.booster_, "feature_importance"):
            importances = clf.booster_.feature_importance(importance_type="gain")

        if importances is not None:
            feat_imp = dict(zip(feature_list, importances))
            keep = [f for f, imp in feat_imp.items() if imp >= FEATURE_IMPORTANCE_THRESHOLD]
            if len(keep) < len(feature_list):
                print(
                    f"[INFO] Reducing features from {len(feature_list)} -> {len(keep)} "
                    f"based on importance threshold={FEATURE_IMPORTANCE_THRESHOLD}"
                )
                Xr_train = X_train[keep]
                Xr_test = X_test[keep]

                if use_lgb:
                    from lightgbm import LGBMClassifier

                    clf2 = LGBMClassifier(**best.named_steps["clf"].get_params())
                    pipeline2 = Pipeline([("scaler", StandardScaler()), ("clf", clf2)])
                else:
                    clf2 = RandomForestClassifier(**best.named_steps["clf"].get_params())
                    pipeline2 = Pipeline([("scaler", StandardScaler()), ("clf", clf2)])

                pipeline2.fit(Xr_train, y_train)
                best = pipeline2
                feature_list = keep
                with open("feature_list.json", "w") as f:
                    json.dump(feature_list, f)
                print("[INFO] Re-trained on reduced feature set and updated feature_list.json")
        else:
            print(
                "[WARN] Could not compute feature importances for this classifier; "
                "skipping feature selection."
            )
    except Exception as e:
        print("[WARN] Feature selection failed:", e)

# Evaluation on test split

X_test_eval = X_test[feature_list]
y_pred = best.predict(X_test_eval)

print("\n[RESULT] Classification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("[RESULT] Confusion matrix:\n", cm)

if len(np.unique(y_test)) == 2:
    try:
        y_prob = best.predict_proba(X_test_eval)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("[RESULT] ROC AUC:", round(auc, 4))
    except Exception:
        pass

# Save all training artifacts

joblib.dump(best, "model.joblib")
print("[INFO] Saved model to model.joblib")

joblib.dump(le, "label_encoder.joblib")
if ordinal is not None:
    joblib.dump(ordinal, "ordinal_encoder.joblib")

with open("feature_list.json", "w") as f:
    json.dump(feature_list, f)

print("[OK] Training complete — artifacts saved in current folder.")