# 1. IMPORTS
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. LOAD DATA
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
df.columns = df.columns.str.strip()
print("Loaded:", df.shape)


# 3. REMOVE NON-FEATURE COLUMNS
remove_list = ["Flow ID", "Timestamp", "Source IP", "Destination IP"]
df = df.drop([c for c in remove_list if c in df.columns], axis=1)


# 4. CLEAN DATA
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print("Cleaned:", df.shape)


# 5. ENCODE LABEL
enc = LabelEncoder()
df["Attack"] = enc.fit_transform(df["Label"])
df = df.drop(columns=["Label"])
print("Classes:", enc.classes_)


# 6. SPLIT FEATURES/TARGET
X = df.drop(columns=["Attack"])
y = df["Attack"]


# 7. SMOTE BALANCING
sm = SMOTE(random_state=11)
X_bal, y_bal = sm.fit_resample(X, y)
print("Balanced:", X_bal.shape)


# 8. SCALING
sc = StandardScaler()
X_norm = sc.fit_transform(X_bal)


# 9. PCA REDUCTION
pca = PCA(n_components=30, random_state=11)
X_pca = pca.fit_transform(X_norm)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_bal, test_size=0.25, random_state=11, stratify=y_bal
)


# 10. DEFINE MODELS
models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "DecisionTree": DecisionTreeClassifier(max_depth=20),
    "RandomForest": RandomForestClassifier(n_estimators=350),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", gamma="scale"),
    "XGBoost": XGBClassifier(
        n_estimators=350,
        max_depth=7,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss"
    )
}


# 11. TRAIN & EVALUATE
scores = {}

for name, clf in models.items():
    print(f"\nMODEL: {name}")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    scores[name] = acc * 100

    print("Accuracy:", round(acc * 100, 3), "%")
    print(classification_report(y_test, preds))


# 12. FINAL RESULT TABLE
results = (
    pd.DataFrame(scores.items(), columns=["Model", "Accuracy (%)"])
      .sort_values(by="Accuracy (%)", ascending=False)
)

print("\nFINAL MODEL SCORES")
print(results)
