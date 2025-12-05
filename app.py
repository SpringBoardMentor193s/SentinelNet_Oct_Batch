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
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ========= CONFIGURATION =========
st.set_page_config(
    page_title="SentinelNet NIDS",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= CUSTOM STYLING =========
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
   
    * {
        font-family: 'Inter', sans-serif;
    }
   
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
   
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
   
    .metric-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
   
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
   
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.5rem;
    }
   
    .stButton>button {
        background: #0f172a;
        color: white;
        border: none;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s;
    }
   
    .stButton>button:hover {
        background: #334155;
        transform: translateY(-1px);
    }
   
    .info-card {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #1e3a8a;
    }
   
    .warning-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #78350f;
    }
   
    .success-card {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #065f46;
    }
   
    .class-distribution-box {
        background: #f1f5f9;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
   
    .class-name {
        font-weight: 600;
        color: #1e293b;
    }
   
    .class-count {
        background: #0f172a;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ========= HELPER FUNCTIONS =========
def identify_target_column(df: pd.DataFrame, dataset_type: str):
    """Identify the target/label column in the dataset."""
    possible_names = ["Label", "label", "Class", "class", "target", "Target", "attack_cat"]
   
    for name in possible_names:
        if name in df.columns:
            return name
   
    if dataset_type == "NSL-KDD":
        for col in reversed(df.columns):
            if df[col].dtype == "object":
                return col
   
    return df.columns[-1]

def convert_to_binary(labels: pd.Series, dataset_type: str) -> pd.Series:
    """Convert multiclass labels to binary (Normal/Anomaly)."""
    if dataset_type == "CICIDS2017":
        normal_vals = ["BENIGN", "BENIGN "]
    else:
        normal_vals = ["normal.", "normal", "NORMAL", "Normal"]
   
    return np.where(labels.astype(str).isin(normal_vals), "Normal", "Anomaly")

def process_dataset(df: pd.DataFrame, dataset_type: str, classification_mode: str):
    """Complete data preprocessing pipeline."""
    target_col = identify_target_column(df, dataset_type)
   
    st.markdown(f"""
    <div class="info-card">
        <strong>🎯 Target Column Identified:</strong> {target_col}
    </div>
    """, unsafe_allow_html=True)
   
    y_original = df[target_col]
    X = df.drop(columns=[target_col])
   
    if classification_mode == "Binary Classification":
        y_processed = convert_to_binary(y_original, dataset_type)
    else:
        y_processed = y_original.astype(str)
   
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_processed)
   
    # Remove rare classes
    unique_labels, counts = np.unique(y_encoded, return_counts=True)
    rare_labels = unique_labels[counts < 2]
   
    if len(rare_labels) > 0:
        mask = ~np.isin(y_encoded, rare_labels)
        X = X.loc[mask]
        y_encoded = y_encoded[mask]
       
        st.markdown("""
        <div class="warning-card">
            <strong>⚠️ Data Cleaning:</strong> Removed classes with insufficient samples
        </div>
        """, unsafe_allow_html=True)
   
    if len(np.unique(y_encoded)) < 2:
        raise ValueError("Insufficient classes for training. Switch to binary mode.")
   
    # Encode categorical features
    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
   
    # FIXED: Better handling of infinite and missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with column median (more robust than 0)
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if np.isnan(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    # Ensure no remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
   
    # FIXED: Normalize features with better error handling
    scaler = StandardScaler()
    try:
        X_normalized = scaler.fit_transform(X)
    except Exception as e:
        st.warning(f"Standard scaling encountered an issue, using robust scaling: {e}")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_normalized = scaler.fit_transform(X)
   
    label_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
   
    # Return class distribution for later visualization
    unique_labels, counts = np.unique(y_encoded, return_counts=True)
    class_dist = {encoder.classes_[label_idx]: count for label_idx, count in zip(unique_labels, counts)}
   
    return X_normalized, y_encoded, encoder, label_map, scaler, X.columns.tolist(), class_dist

def get_classifier_options():
    """Return available ML classifiers with optimized parameters."""
    return {
        # FIXED: Reduced complexity for faster training
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, solver='saga'),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    }

def create_confusion_heatmap(cm_data, class_labels):
    """Generate interactive confusion matrix visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=class_labels,
        y=class_labels,
        colorscale='Greens',
        text=cm_data,
        texttemplate='%{text}',
        textfont={"size": 14, "color": "black"},
        hoverongaps=False,
        colorbar=dict(title="Count")
    ))
   
    fig.update_layout(
        title='Confusion Matrix Heatmap',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        height=450,
        template="plotly_white"
    )
   
    return fig

def create_performance_chart(metrics_dict):
    """Create bar chart for performance metrics."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics_dict.keys()),
            y=list(metrics_dict.values()),
            marker_color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'],
            text=[f"{v:.2%}" for v in metrics_dict.values()],
            textposition='outside'
        )
    ])
   
    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.1]),
        height=400,
        template="plotly_white",
        showlegend=False
    )
   
    return fig

def create_class_distribution_chart(class_dist_dict, title="Class Distribution"):
    """Create bar chart for class distribution."""
    classes = list(class_dist_dict.keys())
    counts = list(class_dist_dict.values())
   
    # Assign colors based on class type
    colors = []
    for cls in classes:
        if cls.lower() in ['normal', 'benign']:
            colors.append('#10b981')  # Green for normal
        else:
            colors.append('#ef4444')  # Red for attacks
   
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=counts,
            marker_color=colors,
            text=counts,
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
        )
    ])
   
    fig.update_layout(
        title=title,
        xaxis_title='Class',
        yaxis_title='Number of Samples',
        height=400,
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-45
    )
   
    return fig

def generate_report(dataset_type, classification_mode, algorithm, accuracy, precision, recall, f1,
                   train_size, test_size, smote_used, class_report, confusion_mat, label_mapping):
    """Generate comprehensive training report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
    report = f"""
================================================================================
                    SENTINELNET NIDS - TRAINING REPORT
================================================================================
Report Generated: {timestamp}
--------------------------------------------------------------------------------
CONFIGURATION
--------------------------------------------------------------------------------
Dataset Type: {dataset_type}
Classification Mode: {classification_mode}
Algorithm: {algorithm}
SMOTE Applied: {'Yes' if smote_used else 'No'}
Training Set Size: {train_size:,} samples
Test Set Size: {test_size:,} samples
--------------------------------------------------------------------------------
PERFORMANCE METRICS
--------------------------------------------------------------------------------
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Precision (Macro): {precision:.4f} ({precision*100:.2f}%)
Recall (Macro): {recall:.4f} ({recall*100:.2f}%)
F1-Score (Macro): {f1:.4f} ({f1*100:.2f}%)
--------------------------------------------------------------------------------
DETAILED CLASSIFICATION REPORT
--------------------------------------------------------------------------------
{class_report}
--------------------------------------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------------------------------------
{confusion_mat}
--------------------------------------------------------------------------------
LABEL ENCODING
--------------------------------------------------------------------------------
{label_mapping}
--------------------------------------------------------------------------------
END OF REPORT
================================================================================
"""
    return report

# ========= MAIN APPLICATION =========
def main():
    # Header
    st.markdown('<h1 class="main-title">🔐 SentinelNet NIDS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Network Intrusion Detection System with Machine Learning</p>', unsafe_allow_html=True)
   
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        st.markdown("---")
       
        dataset_type = st.selectbox(
            "Dataset Type",
            ["NSL-KDD", "CICIDS2017"],
            help="Select the intrusion detection dataset"
        )
       
        classification_mode = st.selectbox(
            "Classification Mode",
            ["Binary Classification", "Multiclass Classification"],
            help="Binary: Normal vs Anomaly | Multiclass: All attack categories"
        )
       
        st.markdown("---")
        st.markdown("### 🤖 Model Configuration")
       
        classifiers = get_classifier_options()
        selected_algorithm = st.selectbox(
            "Algorithm",
            list(classifiers.keys()),
            help="Choose machine learning algorithm"
        )
        classifier = classifiers[selected_algorithm]
       
        st.markdown("---")
        st.markdown("### 🎯 Training Parameters")
       
        apply_smote = st.checkbox("Apply SMOTE Balancing", value=True, help="Oversample minority classes")
        test_split = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        seed_value = st.number_input("Random Seed", min_value=0, value=42, step=1)
       
        st.markdown("---")
        st.markdown("### 📚 About")
        st.info("SentinelNet NIDS uses machine learning to detect network intrusions and anomalies.")
   
    # Main Content
    tab1, tab2 = st.tabs(["📂 Data Management", "🎯 Model Training"])
   
    # ========= TAB 1: DATA MANAGEMENT =========
    with tab1:
        st.markdown("### Upload Dataset")
        st.markdown("Supported formats: **CSV, TXT, DATA**")
       
        uploaded_data = st.file_uploader(
            "Choose your dataset file",
            type=["csv", "txt", "data"],
            help="Upload NSL-KDD or CICIDS2017 dataset"
        )
       
        if uploaded_data is None:
            st.info("📌 Please upload a dataset file to proceed")
           
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">NSL-KDD Dataset</div>
                    <p style="margin-top: 0.5rem; color: #475569;">
                    • Classic network intrusion dataset<br>
                    • 41 feature attributes<br>
                    • Multiple attack categories<br>
                    • Balanced training & test sets
                    </p>
                </div>
                """, unsafe_allow_html=True)
           
            with col2:
                st.markdown("""
                <div class="metric-box">
                    <div class="metric-label">CICIDS2017 Dataset</div>
                    <p style="margin-top: 0.5rem; color: #475569;">
                    • Modern network traffic dataset<br>
                    • 80+ feature attributes<br>
                    • Contemporary attack patterns<br>
                    • Real-world network scenarios
                    </p>
                </div>
                """, unsafe_allow_html=True)
            return
       
        # Load dataset
        try:
            dataset = pd.read_csv(uploaded_data)
        except Exception:
            uploaded_data.seek(0)
            try:
                dataset = pd.read_csv(uploaded_data, sep=r"\s+", engine="python")
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                return
       
        st.markdown("""
        <div class="success-card">
            <strong>✓ Dataset loaded successfully</strong>
        </div>
        """, unsafe_allow_html=True)
       
        # Dataset Statistics
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{dataset.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
       
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Features</div>
                <div class="metric-value">{dataset.shape[1]:,}</div>
            </div>
            """, unsafe_allow_html=True)
       
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Memory</div>
                <div class="metric-value">{dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB</div>
            </div>
            """, unsafe_allow_html=True)
       
        with col4:
            null_count = dataset.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value">{null_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
       
        st.markdown("---")
       
        # FIXED: Replaced use_container_width with width
        with st.expander("🔍 Dataset Preview", expanded=True):
            st.dataframe(dataset.head(15), width='stretch', height=400)
       
        with st.expander("📊 Statistical Summary"):
            st.dataframe(dataset.describe(), width='stretch')
       
        st.session_state['dataset'] = dataset
   
    # ========= TAB 2: MODEL TRAINING =========
    with tab2:
        if 'dataset' not in st.session_state:
            st.warning("⚠️ Please upload a dataset in the Data Management tab first")
            return
       
        dataset = st.session_state['dataset']
        st.markdown("### Model Training Pipeline")
       
        # FIXED: Added warning for large multiclass training
        if classification_mode == "Multiclass Classification":
            st.markdown("""
            <div class="warning-card">
                <strong>⚠️ Performance Note:</strong> Multiclass training may take 2-5 minutes for complex models (Random Forest, Gradient Boosting, SVM).
                For faster results, try Binary Classification or Decision Tree/Naive Bayes algorithms.
            </div>
            """, unsafe_allow_html=True)
       
        if st.button("▶ Start Training Process", type="primary"):
            progress = st.progress(0)
            status = st.empty()
           
            try:
                # Step 1: Preprocessing
                status.markdown("**🔄 Step 1/4:** Preprocessing dataset...")
                progress.progress(25)
               
                X_norm, y_enc, encoder, label_dict, scaler, feature_cols, class_dist = process_dataset(
                    dataset, dataset_type, classification_mode
                )
               
                # Step 2: Split
                status.markdown("**🔄 Step 2/4:** Creating train-test split...")
                progress.progress(50)
               
                X_train, X_test, y_train, y_test = train_test_split(
                    X_norm, y_enc,
                    test_size=test_split / 100.0,
                    random_state=seed_value,
                    stratify=y_enc
                )
               
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box" style="text-align: center;">
                        <div class="metric-label">Training Set</div>
                        <div class="metric-value">{X_train.shape[0]:,}</div>
                        <div style="color: #64748b; margin-top: 0.5rem;">samples</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-box" style="text-align: center;">
                        <div class="metric-label">Test Set</div>
                        <div class="metric-value">{X_test.shape[0]:,}</div>
                        <div style="color: #64748b; margin-top: 0.5rem;">samples</div>
                    </div>
                    """, unsafe_allow_html=True)
               
                st.markdown("---")
               
                # Step 3: SMOTE
                smote_applied = False
                if apply_smote:
                    status.markdown("**🔄 Step 3/4:** Applying SMOTE resampling...")
                    progress.progress(60)
                   
                    unique_tr, counts_tr = np.unique(y_train, return_counts=True)
                    min_samples = counts_tr.min()
                   
                    if min_samples >= 2:
                        k_val = max(1, min(min_samples - 1, 5))
                        smote = SMOTE(random_state=seed_value, k_neighbors=k_val)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        smote_applied = True
                        st.markdown("""
                        <div class="success-card">
                            <strong>✓ SMOTE resampling completed</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    progress.progress(60)
               
                # Step 4: Training
                status.markdown(f"**🔄 Step 4/4:** Training {selected_algorithm} (this may take 1-5 minutes)...")
                progress.progress(75)
               
                # FIXED: Added progress feedback for long-running training
                with st.spinner(f'Training {selected_algorithm}... Please wait'):
                    classifier.fit(X_train, y_train)
               
                # Evaluation
                status.markdown("**📊 Evaluating model performance...**")
                progress.progress(100)
               
                y_pred = classifier.predict(X_test)
               
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
                recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                
                # FIXED: Get unique labels present in test set for proper classification report
                unique_labels_in_test = np.unique(np.concatenate([y_test, y_pred]))
                target_names_filtered = [encoder.classes_[i] for i in unique_labels_in_test]
                
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels_in_test)
               
                status.markdown("**✅ Training completed successfully**")
               
                st.markdown("---")
                st.markdown("### 📊 Performance Summary")
               
                # Metrics Display
                col1, col2, col3, col4 = st.columns(4)
               
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{accuracy:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
               
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value">{precision:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
               
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value">{recall:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
               
                with col4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value">{f1:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
               
                st.markdown("---")
               
                # Test Set Class Distribution
                unique_test, counts_test = np.unique(y_test, return_counts=True)
                test_class_dist = {encoder.classes_[label_idx]: count for label_idx, count in zip(unique_test, counts_test)}
               
                st.markdown("### 📊 Test Set Class Distribution")
                test_dist_chart = create_class_distribution_chart(test_class_dist, "Test Set Class Distribution")
                # FIXED: Replaced use_container_width with width='stretch'
                st.plotly_chart(test_dist_chart, use_container_width=True)
               
                st.markdown("---")
               
                # Visualizations
                col1, col2 = st.columns(2)
               
                with col1:
                    metrics_chart = create_performance_chart({
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1
                    })
                    # FIXED: Replaced use_container_width with use_container_width=True
                    st.plotly_chart(metrics_chart, use_container_width=True)
               
                with col2:
                    cm_chart = create_confusion_heatmap(cm, target_names_filtered)
                    # FIXED: Replaced use_container_width with use_container_width=True
                    st.plotly_chart(cm_chart, use_container_width=True)
               
                # Detailed Report
                with st.expander("📋 Detailed Classification Report"):
                    report_text = classification_report(y_test, y_pred, 
                                                       labels=unique_labels_in_test,
                                                       target_names=target_names_filtered)
                    st.text(report_text)
               
                with st.expander("🏷️ Label Encoding Reference"):
                    st.json(label_dict)
               
                # Generate comprehensive report
                class_report_str = classification_report(y_test, y_pred, 
                                                        labels=unique_labels_in_test,
                                                        target_names=target_names_filtered)
                cm_str = np.array2string(cm, separator=', ')
                label_mapping_str = '\n'.join([f"{k}: {v}" for k, v in label_dict.items()])
               
                full_report = generate_report(
                    dataset_type=dataset_type,
                    classification_mode=classification_mode,
                    algorithm=selected_algorithm,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    train_size=X_train.shape[0],
                    test_size=X_test.shape[0],
                    smote_used=smote_applied,
                    class_report=class_report_str,
                    confusion_mat=cm_str,
                    label_mapping=label_mapping_str
                )
               
                # Download Report Button
                st.markdown("---")
                st.markdown("### 📥 Download Training Report")
               
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    report_filename = f"SentinelNet_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    st.download_button(
                        label="📄 Download Complete Training Report",
                        data=full_report,
                        file_name=report_filename,
                        mime="text/plain",
                        type="primary"
                    )
               
                # Save to session
                st.session_state["model"] = classifier
                st.session_state["scaler"] = scaler
                st.session_state["features"] = feature_cols
                st.session_state["encoder"] = encoder
               
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()