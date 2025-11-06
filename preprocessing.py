# ================================
# KDD Dataset Preprocessing Script
# ================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter

# ================================
# Load Train and Test Data
# ================================
train = pd.read_csv('kdd_train.csv')
test = pd.read_csv('kdd_test.csv')

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# ================================
# Split Features and Labels
# ================================
X_train = train.drop('labels', axis=1)
y_train = train['labels']
X_test = test.drop('labels', axis=1)
y_test = test['labels']

# ================================
# Identify Numeric and Categorical Columns
# ================================
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

# ================================
# Handle Missing Values
# ================================

# For numeric columns → fill missing values with mean
num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# For categorical columns → fill missing with 'missing'
X_train[cat_cols] = X_train[cat_cols].fillna('missing')
X_test[cat_cols] = X_test[cat_cols].fillna('missing')

# ================================
# Encode Categorical Columns
# ================================
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# ================================
# Scale Numeric Features
# ================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# Handle Class Imbalance with SMOTE
# ================================
print("Class distribution before SMOTE:", Counter(y_train))

# Remove rare classes (with only 1 sample)
class_counts = y_train.value_counts()
valid_classes = class_counts[class_counts > 1].index
mask = y_train.isin(valid_classes)
X_train_filtered = X_train_scaled[mask]
y_train_filtered = y_train[mask]

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train_filtered)

print("Class distribution after SMOTE:", Counter(y_train_balanced))

# ================================
# Convert Back to DataFrames
# ================================
X_train_final = pd.DataFrame(X_train_balanced)
y_train_final = pd.DataFrame(y_train_balanced, columns=['labels'])
X_test_final = pd.DataFrame(X_test_scaled)
y_test_final = pd.DataFrame(y_test, columns=['labels'])

# ================================
# Save Preprocessed Data
# ================================
X_train_final.to_csv("X_train_preprocessed.csv", index=False)
y_train_final.to_csv("y_train_preprocessed.csv", index=False)
X_test_final.to_csv("X_test_preprocessed.csv", index=False)
y_test_final.to_csv("y_test_preprocessed.csv", index=False)

print("\n Preprocessing complete!")
print("Files saved:")
print(" - X_train_preprocessed.csv")
print(" - y_train_preprocessed.csv")
print(" - X_test_preprocessed.csv")
print(" - y_test_preprocessed.csv")
