import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


# ========= HELPER FUNCTIONS =========

def detect_label_column(df: pd.DataFrame, dataset_name: str):
    """
    Detect the label column.
    - First try common names
    - For KDD: take last object column from the right (label like "normal.", "neptune.", etc.)
    """
    candidates = ["Label", "label", "Class", "class", "target", "Target"]
    for c in candidates:
        if c in df.columns:
            return c

    if dataset_name.startswith("KDD"):
        # Go from right to left and pick the last object-type col
        for col in reversed(df.columns):
            if df[col].dtype == "object":
                return col
        # Fallback: last column
        return df.columns[-1]

    # Generic fallback for other datasets
    last_col = df.columns[-1]
    if df[last_col].dtype == "object":
        return last_col

    raise ValueError(
        "Could not find label column automatically. "
        f"Available columns (first 20): {list(df.columns)[:20]}..."
    )


def make_binary_label(y_raw: pd.Series, dataset_name: str) -> pd.Series:
    """Convert labels to 'Normal' vs 'Attack'."""
    if dataset_name == "CICIDS2017":
        normal_values = ["BENIGN", "BENIGN "]
    else:
        normal_values = ["normal.", "normal", "NORMAL", "Normal"]

    return np.where(y_raw.astype(str).isin(normal_values), "Normal", "Attack")


def preprocess_data(df: pd.DataFrame, dataset_name: str, task_type: str):
    """Full preprocessing pipeline."""
    label_col = detect_label_column(df, dataset_name)
    st.write(f"**Detected label column:** `{label_col}`")

    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    st.write("Sample raw label values:", list(y_raw.astype(str).unique()[:10]))

    # Binary or multiclass
    if task_type == "Binary (Normal vs Attack)":
        y_str = make_binary_label(y_raw, dataset_name)
    else:
        y_str = y_raw.astype(str)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # ---- Remove classes with < 2 samples (for stratify split) ----
    unique, counts = np.unique(y, return_counts=True)
    rare_classes = unique[counts < 2]

    if len(rare_classes) > 0:
        rare_labels = [label_encoder.classes_[c] for c in rare_classes]
        st.warning(
            "These classes have fewer than 2 samples and will be removed "
            f"before splitting: {rare_labels}"
        )
        mask = ~np.isin(y, rare_classes)
        X = X.loc[mask]
        y = y[mask]

        # Recompute distribution
        unique, counts = np.unique(y, return_counts=True)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "After removing extremely rare classes, only one class remains. "
            "Cannot train a classifier. Please use Binary mode or a larger dataset."
        )

    st.write("Class distribution after cleaning (encoded):")
    st.write(dict(zip(unique.tolist(), counts.tolist())))

    # Mapping (encoded -> original)
    label_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    st.write("Classes after encoding:", list(label_encoder.classes_))

    # Encode categorical feature columns
    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        le_feat = LabelEncoder()
        X[col] = le_feat.fit_transform(X[col].astype(str))

    # Handle missing / infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, label_encoder, label_mapping, scaler, X.columns.tolist()


def get_models_dict():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", probability=True),
    }


# ========= STREAMLIT APP =========

def main():
    st.set_page_config(page_title="Sentinel IDS", layout="wide")

    st.title("🛡️ Sentinel IDS – Intrusion Detection System")
    st.write(
        "Train and evaluate models on **KDD** or **CICIDS2017** datasets. "
        "Supports binary (Normal vs Attack) and multiclass classification."
    )

    # ----- SIDEBAR -----
    st.sidebar.header("⚙️ Configuration")

    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["KDD / KDDCup", "CICIDS2017"],
    )

    task_type = st.sidebar.selectbox(
        "Select Task Type",
        ["Binary (Normal vs Attack)", "Multiclass"],
    )

    models_base = get_models_dict()
    algo_options = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest (recommended)",
        "Gradient Boosting",
        "Naive Bayes",
        "KNN",
        "SVM (RBF)",
    ]
    algo_choice_ui = st.sidebar.selectbox("Select Algorithm", algo_options)
    algo_key = algo_choice_ui.split(" (")[0]   # remove " (recommended)"
    model = models_base[algo_key]

    use_smote = st.sidebar.checkbox("Use SMOTE (for class imbalance)", value=True)
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5)
    random_state = st.sidebar.number_input("Random State", min_value=0, value=42, step=1)

    st.sidebar.write("---")
    st.sidebar.write("For KDD, the app auto-detects the label near the end (not the difficulty column).")

    # ----- FILE UPLOAD -----
    st.subheader("1️⃣ Upload Dataset File")

    uploaded_file = st.file_uploader(
        "Upload your dataset file (CSV / TXT / DATA)",
        type=["csv", "txt", "data"],
    )

    if uploaded_file is None:
        st.info("👆 Upload a KDD or CICIDS dataset file to begin.")
        return

    # Try reading as normal CSV first
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            # For whitespace-separated KDD .data files
            df = pd.read_csv(uploaded_file, sep=r"\s+", engine="python")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

    st.write("✅ Dataset Loaded!")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # ----- TRAINING -----
    st.write("---")
    st.subheader("2️⃣ Preprocess, Train & Evaluate")

    if st.button("🚀 Train Model"):
        with st.spinner("Preprocessing and training the model..."):
            try:
                X_scaled, y, label_encoder, label_mapping, scaler, feature_names = preprocess_data(
                    df, dataset_name, task_type
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled,
                    y,
                    test_size=test_size / 100.0,
                    random_state=random_state,
                    stratify=y,
                )

                st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

                # ----- DYNAMIC SMOTE FIX -----
                if use_smote:
                    unique_tr, counts_tr = np.unique(y_train, return_counts=True)
                    min_count = counts_tr.min()

                    if min_count < 2:
                        st.warning(
                            "SMOTE skipped: at least one class in the training set "
                            "has only 1 sample."
                        )
                    else:
                        if min_count <= 6:
                            k_neighbors = max(1, min_count - 1)
                            st.warning(
                                "Training set has very small classes "
                                f"(min_count={min_count}). "
                                f"Using SMOTE with k_neighbors={k_neighbors}."
                            )
                            sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                        else:
                            sm = SMOTE(random_state=random_state)

                        X_train, y_train = sm.fit_resample(X_train, y_train)
                        st.write("SMOTE applied on training data.")
                        st.write("Resampled train shape:", X_train.shape)

                # ----- FIT & EVALUATE MODEL -----
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average="macro")
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=False)

                st.success("Training & evaluation completed.")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{acc * 100:.2f}%")
                with col2:
                    st.metric("Macro F1-score", f"{macro_f1:.4f}")

                st.write("### Label Mapping (Encoded)")
                st.json(label_mapping)

                st.write("### Classification Report")
                st.text(report)

                st.write("### Confusion Matrix")
                cm_df = pd.DataFrame(cm)
                st.dataframe(cm_df)

                # Save for prediction
                st.session_state["trained_model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["feature_names"] = feature_names
                st.session_state["label_encoder"] = label_encoder

            except Exception as e:
                st.error(f"Error during training: {e}")

    # ----- PREDICTION -----
    st.write("---")
    st.subheader("3️⃣ (Optional) Predict on New Data")

    if "trained_model" not in st.session_state:
        st.info("Train a model first to enable prediction.")
        return

    pred_file = st.file_uploader(
        "Upload new traffic data for prediction (same features, without label)",
        type=["csv", "txt", "data"],
        key="pred_uploader",
    )

    if pred_file is None:
        return

    try:
        df_new = pd.read_csv(pred_file)
    except Exception:
        pred_file.seek(0)
        try:
            df_new = pd.read_csv(pred_file, sep=r"\s+", engine="python")
        except Exception as e:
            st.error(f"Failed to read prediction file: {e}")
            return

    st.write("New data shape:", df_new.shape)
    st.dataframe(df_new.head())

    feature_names = st.session_state["feature_names"]
    missing_cols = [c for c in feature_names if c not in df_new.columns]
    if missing_cols:
        st.error(f"Missing columns in new data: {missing_cols}")
        return

    df_new = df_new[feature_names]

    for col in df_new.select_dtypes(include=["object"]).columns:
        le_feat = LabelEncoder()
        df_new[col] = le_feat.fit_transform(df_new[col].astype(str))

    df_new = df_new.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = st.session_state["scaler"]
    X_new_scaled = scaler.transform(df_new)

    model_trained = st.session_state["trained_model"]
    label_encoder = st.session_state["label_encoder"]

    y_new_pred = model_trained.predict(X_new_scaled)
    y_new_labels = label_encoder.inverse_transform(y_new_pred)

    st.write("### Predictions")
    st.dataframe(pd.DataFrame({"Prediction": y_new_labels}))


if __name__ == "__main__":
    main()
