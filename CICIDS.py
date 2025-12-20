import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train_df=pd.read_csv("Dataset/CICIDS.csv")
train_df.drop_duplicates(inplace=True)
train_df=train_df.sample(n=200282, random_state=42)
# print("\nTrain Dataset Shape:", train_df.shape)

# -------------------- Creating Binary classification in Train dataset --------------------
# train_df["binary_attack"]=train_df["Label"].apply(lambda x:0 if x == 'BENIGN'else 1)
# print(train_df[['Label','binary_attack']])

# --------------------- Splitting the dataset --------------------
from sklearn.model_selection import train_test_split
# x=train_df.drop(columns=["Label"])
# y=train_df["Label"]
# X=train_df.drop(['Label', 'binary_attack'], axis=1)
# Y=train_df['binary_attack']

# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5, random_state=42, stratify=y)
# print("Train dataset shape:", X_train.shape)
# print("Test dataset shape:", X_test.shape)
# Train_df=pd.concat([X_train, Y_train], axis=1)
# Test_df=pd.concat([X_test], axis=1)

# Train_df.to_csv("CICIDS Train.csv", index=False)
# Test_df.to_csv("CICIDS Test.csv", index=False)

# -------------------- Standardscaler --------------------
# from sklearn.preprocessing import StandardScaler

# numerical_cols=X_train.columns

# X_train_scaled = X_train.copy()
# X_test_scaled = X_test.copy()

# scaler = StandardScaler()
# X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# --------------------- SMOTE Analysis --------------------
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)
# X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_scaled, Y_train)

# print("\nResampled Train dataset shape:", X_train_resampled.shape)
# print("\nOriginal Train dataset value counts:", Y_train.value_counts())
# print("\nResampled Train dataset value counts:", Y_train_resampled.value_counts())

# with open("SMOTE.pkl", "wb") as f:
#     pickle.dump(smote, f)

# with open("Scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open("Feature_Columns.pkl", 'wb') as f:
#     pickle.dump(X_train.columns, f)

# -------------------- Model Training --------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate

# models = { 
#     "Logistic Regression": LogisticRegression(
#         random_state=42,
#         max_iter=1000,
#         C=10,
#         solver='liblinear'
#         ),
#     "Decision Tree": DecisionTreeClassifier(
#         random_state=42,
#         max_depth=None, 
#         min_samples_split=2, 
#         min_samples_leaf=2,criterion='gini'
#         ),
#     "Random Forest": RandomForestClassifier(
#         random_state=42,
#         n_estimators = 300,
#         max_depth =10,
#         min_samples_split=5, 
#         min_samples_leaf=2
#         ),
#     "XGBoost": XGBClassifier(
#         eval_metric='logloss',
#         random_state=42,
#         n_estimators=200, 
#         learning_rate=0.1, 
#         max_depth=5
#         ),
#     "LightGBM": LGBMClassifier(
#         random_state=42,
#         um_leaves=20, 
#         learning_rate=0.05, 
#         n_estimators=300, 
#         force_row_wise = True, 
#         verbose = -1
#         ),
#     "K-Nearest Neighbors": KNeighborsClassifier(
#         n_neighbors=3, 
#         metric='minkowski', 
#         weights='distance'
#         )
# }

# for name, model in models.items():
#     print(f"\nTraining {name}...")
#     model.fit(X_train_resampled, Y_train_resampled)
#     print(name + " trained.")

# --------------------- Testing the Models using Train Dataset ---------------------
# Results = []
# for name, model in models.items():
#     print(f"\nEvaluating {name}...")
#     predictions = model.predict(X_train_scaled)
#     accuracy = accuracy_score(Y_train, predictions)
#     precision = precision_score(Y_train, predictions)
#     recall = recall_score(Y_train, predictions)
#     f1 = f1_score(Y_train, predictions)
#     Results.append([name, accuracy, precision, recall, f1])

# Results_df = pd.DataFrame(Results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
# Results_df['Accuracy'] = Results_df['Accuracy'].map(lambda x: f"{x:.2f}%")

# for col in ['Precision', 'Recall', 'F1-Score']:
#     Results_df[col] = Results_df[col].map(lambda x: f"{x:.4f}")

# print("\nModel Evaluation Results on Train Dataset:\n")

# table = tabulate(Results_df, headers='keys', tablefmt='fancy_grid', showindex=False)
# print(table)

# with open('Train Dataset Results.txt', 'w', encoding='utf-8') as f:
#     f.write(table)
# print("Evaluation metrics saved as 'Train Dataset Results.txt'")

# -------------------- Testing the Models using Test Dataset ---------------------
# results = []
# for name, model in models.items():
#     print(f"\nEvaluating {name}...")
#     predictions = model.predict(X_test_scaled)
#     accuracy = accuracy_score(Y_test, predictions)
#     precision = precision_score(Y_test, predictions)
#     recall = recall_score(Y_test, predictions)
#     f1 = f1_score(Y_test, predictions)
#     results.append([name, accuracy, precision, recall, f1])

# results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
# results_df['Accuracy'] = results_df['Accuracy'].map(lambda x: f"{x:.2f}%")

# for col in ['Precision', 'Recall', 'F1-Score']:
#     results_df[col] = results_df[col].map(lambda x: f"{x:.4f}")

# print("\nModel Evaluation Results on Test Dataset:\n")

# table_str = tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False)
# print(table_str)

# with open('Test Dataset Results.txt', 'w', encoding='utf-8') as f:
#     f.write(table_str)
# print("Evaluation metrics saved as 'Test Dataset Results.txt'")

# -------------------- Multiclass Classification ---------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from imblearn.over_sampling import SMOTE


# Target normalization
def normalize_label(label):
    label = str(label).strip().upper()
    if label == "BENIGN":
        return "Benign"
    elif "DOS" in label and "DDOS" not in label:
        return "DoS"
    elif "DDOS" in label:
        return "DDoS"
    else:
        return "Intrusion"

train_df["Label"] = train_df["Label"].apply(normalize_label)

# Separate features and target
X = train_df.drop(columns=["Label"])
y = train_df["Label"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(label_encoder.classes_)

smote = SMOTE(random_state=42)
x_train , Y_train =smote.fit_resample(X_train_scaled, y_train)

# Model training
# model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1
# )
# model.fit(X_train_scaled, y_train)

# Evaluation
# y_pred = model.predict(X_test_scaled)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # Save model and preprocessors
# with open("multiclass_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# with open("label_encoder.pkl", "wb") as f:
#     pickle.dump(label_encoder, f)
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# x=train_df.drop(['Label', 'binary_attack'], axis=1)
# y=train_df['Label']

# counts = train_df['Label'].value_counts()
# mask = train_df['Label'].isin(counts[counts>1].index)
# x=x[mask]
# y=y[mask]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42, stratify=y)

# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_test = label_encoder.transform(y_test)
# # print(label_encoder.classes_)

# --------------------- Standardscaler --------------------
# scaler = StandardScaler()

# x_train_scaled = x_train.copy()
# x_test_scaled = x_test.copy()
# numerical_cols = x_train.columns

# x_train_scaled[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
# x_test_scaled[numerical_cols] = scaler.transform(x_test[numerical_cols])

# --------------------- Model Training --------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=10,
        solver='lbfgs',
        multi_class="multinomial"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=2,
        criterion='gini'
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=10,
        min_samples_split=5, 
        min_samples_leaf=2
    ),
    "XGBoost": XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5,
        objective="multi:softmax",
        num_class =len(label_encoder.classes_)
    ),
    "LightGBM": LGBMClassifier(
        random_state=42,
        num_leaves=20, 
        learning_rate=0.05, 
        n_estimators=300, 
        force_row_wise=True, 
        verbose=-1,
        objective="multiclass", class_weight="balanced"
    )
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    print(name + " trained.")

# --------------------- Testing the Models using Train Dataset ---------------------
Results = []
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    predictions = model.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, predictions)
    precision = precision_score(y_train, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_train, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_train, predictions, average='weighted', zero_division=0)
    Results.append([name, accuracy, precision, recall, f1])
    predictions=label_encoder.inverse_transform(predictions)
    df=X_train.copy()
    df["Actual_Label"]=label_encoder.inverse_transform(y_train)
    df["Predicted_Label"]=predictions
print(df)
Results_df = pd.DataFrame(Results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
Results_df['Accuracy'] = Results_df['Accuracy'].map(lambda x: f"{x:.2f}%")

for col in ['Precision', 'Recall', 'F1-Score']:
    Results_df[col] = Results_df[col].map(lambda x: f"{x:.4f}")
print("\nModel Evaluation Results on Train Dataset:\n")

table = tabulate(Results_df, headers='keys', tablefmt='fancy_grid', showindex=False)
print(table)

# with open('Multiclass Train Dataset Results.txt', 'w', encoding='utf-8') as f:
#     f.write(table)
# print("Evaluation metrics saved as 'Multiclass Train Dataset Results.txt'")

# -------------------- Testing the Models using Test Dataset ---------------------
results = []
for name, model in models.items():
    # print(f"\nEvaluating {name}...")
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    results.append([name, accuracy, precision, recall, f1])

results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
results_df['Accuracy'] = results_df['Accuracy'].map(lambda x: f"{x:.2f}%")

for col in ['Precision', 'Recall', 'F1-Score']:
    results_df[col] = results_df[col].map(lambda x: f"{x:.4f}")
print("\nModel Evaluation Results on Test Dataset:\n")

table_str = tabulate(results_df, headers='keys', tablefmt='fancy_grid', showindex=False)
print(table_str)

# with open('Multiclass Test Dataset Results.txt', 'w', encoding='utf-8') as f:
#     f.write(table_str)
# print("Evaluation metrics saved as 'Multiclass Test Dataset Results.txt'")

with open("Random_Forest.pkl", "wb") as f:
    pickle.dump(models["Random Forest"], f)

with open("Logistic_Regression.pkl", "wb") as f:
    pickle.dump(models["Logistic Regression"], f)

with open("XGBoost.pkl", "wb") as f:
    pickle.dump(models["XGBoost"], f)

with open("LightGBM.pkl", "wb") as f:
    pickle.dump(models["LightGBM"], f)

with open("Decision_Tree.pkl", "wb") as f:
    pickle.dump(models["Decision Tree"], f)

# with open("KNN.pkl", "wb") as f:
#     pickle.dump(models["K-Nearest Neighbors"], f)

with open("Scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("Label_Encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("SMOTE.pkl", "wb") as f:
    pickle.dump(smote, f)