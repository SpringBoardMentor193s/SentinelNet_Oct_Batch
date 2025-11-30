import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="SentinelNet - Intrusion Detection", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #fff; }
    .sidebar .sidebar-content { background-color: #161b22; color: white; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# FRIENDLY DISPLAY NAMES + DESCRIPTIONS
# (used only for features we show to the user)
# =========================================================
friendly_names = {
    "duration": ("Connection Duration (sec)", "How long the connection lasted (seconds)."),
    "protocol_type": ("Protocol Type", "Network protocol used (e.g. TCP, UDP, ICMP)."),
    "service": ("Service Type", "Requested service such as HTTP, FTP, SMTP, etc."),
    "flag": ("Connection Status Flag", "Status flag from the server's response."),

    "src_bytes": ("Source Bytes", "Amount of data sent from source to destination (bytes)."),
    "dst_bytes": ("Destination Bytes", "Amount of data sent from destination to source (bytes)."),

    "num_failed_logins": ("Failed Login Attempts", "How many times login failed."),
    "logged_in": ("User Logged In", "1 if login was successful, 0 otherwise."),
    "num_compromised": ("Compromised Conditions", "Number of compromised/suspicious conditions on the host."),
    "root_shell": ("Root/Admin Shell Access", "1 if attacker got root/admin shell on the system."),
    "su_attempted": ("Admin Login (su) Attempts", "1 if 'su' (admin) login was attempted."),
    "num_root": ("Root Access Count", "Number of root-level actions performed."),

    "count": ("Recent Connections to Same Host", "Connections to the same host in the last 2 seconds."),
    "srv_count": ("Recent Connections to Same Service", "Connections to the same service in last 2 seconds."),

    "same_srv_rate": ("Same Service Rate", "Percentage of connections to the same service."),
    "diff_srv_rate": ("Different Service Rate", "Percentage of connections to different services."),

    "dst_host_count": ("Destination Host Connection Count", "Connections to the same destination host."),
    "dst_host_srv_count": ("Destination Host-Service Count", "Connections to the same host & service."),
    "dst_host_same_srv_rate": ("Host Same Service Rate", "Host connections using the same service."),
    "dst_host_diff_srv_rate": ("Host Different Service Rate", "Host connections using different services."),
    "dst_host_serror_rate": ("Host SYN Error Rate", "SYN error rate for the destination host."),
    "dst_host_srv_serror_rate": ("Host-Service SYN Error Rate", "SYN error rate for this host-service pair."),
}

# Binary (0/1) numeric features we will show as Yes/No
binary_features = {
    "land",
    "logged_in",
    "root_shell",
    "su_attempted",
    "is_host_login",
    "is_guest_login"
}

# Columns that must always be integers (counts)
integer_features = {
    "num_failed_logins",
    "num_compromised",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "urgent",
    "hot",
    "wrong_fragment",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_host_srv_count"
}

# IMPORTANT FEATURES to show in UI; others will be auto-filled
important_features = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "count",
    "srv_count",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
]

# =========================================================
# LOAD TRAIN DATA + BUILD LABEL ENCODERS EXACTLY LIKE TRAINING
# =========================================================
@st.cache_resource
def load_dataset_and_encoders():
    train_df = pd.read_csv("kdd_train.csv")
    test_df = pd.read_csv("kdd_test.csv")

    # same binary label as your model
    train_df["attack_binary"] = train_df["labels"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["attack_binary"] = test_df["labels"].apply(lambda x: 0 if x == "normal" else 1)

    # categorical columns from your training code
    categorical_cols = ["protocol_type", "service", "flag"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        encoders[col] = le

    # feature columns used for training
    feature_cols = [c for c in train_df.columns if c not in ["labels", "attack_binary"]]

    # numeric medians for auto-filling hidden numeric features
    numeric_medians = train_df[feature_cols].median(numeric_only=True)

    # modes for categorical for auto-filling
    cat_modes = {col: train_df[col].mode()[0] for col in categorical_cols}

    return train_df, test_df, encoders, feature_cols, categorical_cols, numeric_medians, cat_modes


(
    train_df,
    test_df,
    encoders,
    feature_cols,
    categorical_cols,
    numeric_medians,
    cat_modes,
) = load_dataset_and_encoders()

# Keep only important features that actually exist
important_features_available = [f for f in important_features if f in feature_cols]

# =========================================================
# LOAD TRAINED MODEL + IMPUTER + SCALER
# (tries both filenames so it works with your setup)
# =========================================================
model = None
for fname in ["kdd_decision_tree_model.pkl", "kdd_best_model.pkl"]:
    try:
        model = joblib.load(fname)
        break
    except FileNotFoundError:
        continue

if model is None:
    st.error(
        "Model file not found. Ensure 'kdd_decision_tree_model.pkl' or 'kdd_best_model.pkl' "
        "is in the same folder as app.py."
    )
    st.stop()

imputer = joblib.load("kdd_imputer.pkl")
scaler = joblib.load("kdd_scaler.pkl")

# =========================================================
# SIDEBAR INPUT UI (ONLY IMPORTANT FEATURES)
# =========================================================
st.sidebar.title("Network Traffic Input (Key Features Only)")
st.sidebar.markdown("Less important technical fields are auto-filled with typical values.")

user_input = {}

# ---- Main categorical controls first ----
for cat_col in ["protocol_type", "service", "flag"]:
    if cat_col in important_features_available:
        le = encoders[cat_col]
        label, help_text = friendly_names.get(cat_col, (cat_col, ""))
        user_input[cat_col] = st.sidebar.selectbox(
            label,
            list(le.classes_),
            help=help_text
        )

st.sidebar.markdown("---")
st.sidebar.subheader("Other Key Features")

# ---- Remaining important features ----
for col in important_features_available:
    if col in user_input:
        continue  # already handled as categorical above

    label, help_text = friendly_names.get(col, (col, ""))

    # Binary yes/no features
    if col in binary_features:
        choice = st.sidebar.selectbox(
            label,
            ["No", "Yes"],
            help=help_text
        )
        user_input[col] = 1 if choice == "Yes" else 0

    # Numeric integer features (counts)
    elif col in integer_features:
        col_min = int(train_df[col].min())
        col_max = int(train_df[col].max())
        default = int(round(train_df[col].median()))
        user_input[col] = st.sidebar.number_input(
            label,
            min_value=col_min,
            max_value=col_max,
            value=default,
            step=1,
            format="%d",
            help=help_text
        )

    # Numeric float features (rates, percentages, etc.)
    else:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        default = float(train_df[col].median())
        user_input[col] = st.sidebar.number_input(
            label,
            min_value=col_min,
            max_value=col_max,
            value=default,
            step=0.01,
            help=help_text
        )

# =========================================================
# MAIN PAGE
# =========================================================
st.markdown(
    "<h1 style='text-align:center;'>SentinelNet Network Intrusion Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-based detection of Normal vs Attack using key KDD network traffic features.</p>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input (Key Features)")
    st.json(user_input)

with col2:
    st.subheader("Prediction")

    if st.button("🔍 Analyze Traffic"):
        # Build full feature dictionary (all features, including hidden ones)
        full_values = {}

        for col in feature_cols:
            # If user provided a value, use it
            if col in user_input:
                full_values[col] = user_input[col]
            # If it's categorical but not shown, use most frequent value
            elif col in categorical_cols:
                full_values[col] = cat_modes[col]
            # For numeric features, use median from train_df
            else:
                # integer features -> int median
                if col in integer_features or col in binary_features:
                    median_val = numeric_medians.get(col, 0)
                    full_values[col] = int(round(median_val))
                else:
                    median_val = numeric_medians.get(col, 0.0)
                    full_values[col] = float(median_val)

        # Create DF in correct order
        input_df = pd.DataFrame([full_values], columns=feature_cols)

        # Encode categorical columns like in training
        for col in categorical_cols:
            input_df[col] = encoders[col].transform(input_df[col])

        # Apply imputer & scaler
        input_imputed = imputer.transform(input_df.values)
        input_scaled = scaler.transform(input_imputed)

        # Predict
        pred = model.predict(input_scaled)[0]
        try:
            prob = float(model.predict_proba(input_scaled)[0][pred])
        except Exception:
            prob = None

        label = "Normal" if pred == 0 else "Attack"

        if pred == 0:
            st.success(f"✅ Prediction: **{label}**")
        else:
            st.error(f"⚠️ Prediction: **{label}**")

        if prob is not None:
            st.write(f"Model Confidence: **{prob * 100:.2f}%**")

        st.caption("Class mapping used in training: 0 = Normal, 1 = Attack")
    else:
        st.info("Set the key inputs on the left and click **Analyze Traffic** to classify the connection.")
