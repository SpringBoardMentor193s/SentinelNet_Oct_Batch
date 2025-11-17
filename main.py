import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

train_df=pd.read_csv("Dataset/KDD_Train.csv", header=None, names=column_names)
train_df.drop("difficulty", axis=1, inplace=True)
print("Train Dataset Shape:", train_df.shape)

test_df=pd.read_csv("Dataset/KDD_Test.csv", header=None, names=column_names)
test_df.drop("difficulty", axis=1, inplace=True)
print("Test Dataset Shape:", test_df.shape)

# -------------------- Creating Binary Attack Column in Train dataset --------------------
train_df["binary_attack"]=train_df["class"].apply(lambda x:'0'if x == 'normal'else '1')
# print(train_df[['class','binary_attack']])

# -------------------- Creating Binary Attack Column in Test dataset --------------------
test_df["binary_attack"]=test_df["class"].apply(lambda x:'0'if x == 'normal'else '1')
# print(test_df[['class','binary_attack']])

# -------------------- One-Hot Encoding ---------------------
X_train=train_df.drop(['class', 'binary_attack'], axis=1)
Y_train=train_df['binary_attack']

X_test=test_df.drop(['class', 'binary_attack'], axis=1)
Y_test=test_df['binary_attack']

categorical_cols=['protocol_type', 'service', 'flag']
numerical_cols=X_train.columns.drop(categorical_cols)

X_train_encoded=pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded=pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

train_cols=X_train_encoded.columns
test_cols=X_test_encoded.columns

for col in train_cols:
    if col not in test_cols:
        X_test_encoded[col]=0
        
X_test_encoded=X_test_encoded[train_cols]

print("Train dataset after encoding:", X_train_encoded.shape)
print("Test dataset after encoding:", X_test_encoded.shape)

# -------------------- Standard Scaler ---------------------
from sklearn.preprocessing import StandardScaler

numerical_cols = X_train.columns.drop(categorical_cols)

X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()

scaler = StandardScaler()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])

# -------------------- SMOTE Analysis ---------------------
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_scaled, Y_train)

print("Resampled Train dataset shape:", X_train_resampled.shape)
print("Original Train dataset value counts:", Y_train.value_counts())
print("Resampled Train dataset value counts:", Y_train_resampled.value_counts())

# -------------------- Model Training ---------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

models = {
    'Logistic Regression': LogisticRegression(random_state = 42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

for name,model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, Y_train_resampled)
    print(name + " model trained successfully.")
    
for name,model in models.items():
    print(f"\nEvaluating {name}...")

    predictions = model.predict(X_test_scaled)
    accuracy=accuracy_score(Y_test, predictions)
    print(f"{name} Accuracy: {accuracy:.2f}%")
    precision=precision_score(Y_test, predictions, pos_label='1')
    print(f"{name} Precision: {precision:.2f}%")
    recall=recall_score(Y_test, predictions, pos_label='1')
    print(f"{name} Recall: {recall:.2f}%")
    f1=f1_score(Y_test, predictions, pos_label='1')
    print(f"{name} F1-Score: {f1:.2f}%")
  
# -------------------- Confusion Matrix ---------------------
for name, model in models.items():
    print(f"\n----Confusion Matrix for {name}----")
    predictions = model.predict(X_test_scaled)
    cm = confusion_matrix(Y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    disp.plot(cmap=plt.cm.Greens)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
  