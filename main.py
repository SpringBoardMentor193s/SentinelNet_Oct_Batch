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
print("\nTrain Dataset Shape:", train_df.shape)

test_df=pd.read_csv("Dataset/KDD_Test.csv", header=None, names=column_names)
test_df.drop("difficulty", axis=1, inplace=True)
# print("Test Dataset Shape:", test_df.shape)

# -------------------- Creating Binary Attack Column in Train dataset --------------------
train_df["binary_attack"]=train_df["class"].apply(lambda x : 0 if x == 'normal'else 1)
# print(train_df[['class','binary_attack']])

# -------------------- Creating Binary Attack Column in Test dataset --------------------
test_df["binary_attack"]=test_df["class"].apply(lambda x : 0 if x == 'normal'else 1)
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

print("\nTrain dataset after encoding:", X_train_encoded.shape)
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

# print("\nResampled Train dataset shape:", X_train_resampled.shape)
# print("\nOriginal Train dataset value counts:", Y_train.value_counts())
# print("\nResampled Train dataset value counts:", Y_train_resampled.value_counts())

# -------------------- Models Training ---------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate

models = {
    'Logistic Regression': LogisticRegression(
        random_state = 42, 
        max_iter=1000,
        C=10,
        solver='liblinear'
        ),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=2, 
        random_state=42,
        criterion='gini'
        ),

    'Random Forest': RandomForestClassifier(
        n_estimators = 300,
        max_depth =10,
        min_samples_split=5, 
        min_samples_leaf=2
        ),

    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=3, 
        metric='minkowski', 
        weights='distance'
        ),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
        ),

    'XGBoost': XGBClassifier(
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
        ),

    'LightGBM': LGBMClassifier(
       num_leaves=20, 
       learning_rate=0.05, 
       n_estimators=300, 
       force_row_wise = True, 
       verbose = -1,
       random_state=42
       ),

    'AdaBoost': AdaBoostClassifier(
        n_estimators=150,
        learning_rate=1.5,
        random_state=42
    )   
}

# -------------------- Model Training ---------------------
for name,model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, Y_train_resampled)
    print(name + " model trained successfully.")

# -------------------- Testing the Models using Train Dataset ---------------------
results = []
for name,model in models.items():
    predictions = model.predict(X_train_scaled)
    accuracy=accuracy_score(Y_train, predictions)*100
    precision=precision_score(Y_train, predictions, pos_label=1) 
    recall=recall_score(Y_train, predictions, pos_label=1)
    f1=f1_score(Y_train, predictions, pos_label=1)
    results.append([name, accuracy, precision, recall, f1])

results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
results_df['Accuracy'] = results_df['Accuracy'].map(lambda x: f"{x:.2f}%")

for col in ['Precision', 'Recall', 'F1-Score']:
    results_df[col] = results_df[col].map(lambda x: f"{x:.4f}")

print("\nModel Evaluation Results on Train Dataset:\n")

table_str = tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False)
print(table_str)

with open('Train Dataset Results.txt', 'w', encoding='utf-8') as f:
    f.write(table_str)
print("Evaluation metrics saved as 'Train Dataset Results.txt'")

# -------------------- Testing the Models using Test Dataset ---------------------   
Results = []
for name,model in models.items():
    predictions = model.predict(X_test_scaled)
    accuracy=accuracy_score(Y_test, predictions)*100
    precision=precision_score(Y_test, predictions, pos_label=1) 
    recall=recall_score(Y_test, predictions, pos_label=1)
    f1=f1_score(Y_test, predictions, pos_label=1)
    Results.append([name, accuracy, precision, recall, f1])

Results_df = pd.DataFrame(Results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
Results_df['Accuracy'] = Results_df['Accuracy'].map(lambda x: f"{x:.2f}%")

for col in ['Precision', 'Recall', 'F1-Score']:
    Results_df[col] = Results_df[col].map(lambda x: f"{x:.4f}")

print("\nModel Evaluation Results on Test Dataset:\n")

table = tabulate(Results_df, headers='keys', tablefmt='fancy_grid', showindex=False)
print(table)

with open('Test Dataset Results.txt', 'w', encoding='utf-8') as f:
    f.write(table)
print("Evaluation metrics saved as 'Test Dataset Results.txt'")