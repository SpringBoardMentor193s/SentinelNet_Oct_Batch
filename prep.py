import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

train = pd.read_csv('kdd_train.csv')
test = pd.read_csv('kdd_test.csv')

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

X_train = train.drop('labels', axis=1)
y_train = train['labels']
X_test = test.drop('labels', axis=1)
y_test = test['labels']

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

X_train[cat_cols] = X_train[cat_cols].fillna('missing')
X_test[cat_cols] = X_test[cat_cols].fillna('missing')

for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Class distribution before SMOTE:", Counter(y_train))

class_counts = y_train.value_counts()
valid_classes = class_counts[class_counts > 1].index
mask = y_train.isin(valid_classes)
X_train_filtered = X_train_scaled[mask]
y_train_filtered = y_train[mask]

smote = SMOTE(random_state=42, k_neighbors=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train_filtered)

print("Class distribution after SMOTE:", Counter(y_train_balanced))

X_train_final = pd.DataFrame(X_train_balanced)
y_train_final = pd.DataFrame(y_train_balanced)
X_test_final = pd.DataFrame(X_test_scaled)
y_test_final = pd.DataFrame(y_test)

X_train_final.to_csv("X_train_preprocessed.csv", index=False)
y_train_final.to_csv("y_train_preprocessed.csv", index=False)
X_test_final.to_csv("X_test_preprocessed.csv", index=False)
y_test_final.to_csv("y_test_preprocessed.csv", index=False)

print("Preprocessing complete.")
