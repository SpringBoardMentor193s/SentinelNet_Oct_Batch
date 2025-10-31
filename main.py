import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

path = r"C://Users//S Rakshita//Desktop//SENITNELNET//SentinelNet_Oct_Batch//KDDTrain+.csv"
test_path = r"C://Users//S Rakshita//Desktop//SENITNELNET//SentinelNet_Oct_Batch//KDDTest+.csv"
output_path = r"C://Users//S Rakshita//Desktop//SENITNELNET//SentinelNet_Oct_Batch//modified_data.csv"

df = pd.read_csv(path)
print("Before deleting column:")
print(df.head())

if 'labels' in df.columns:
    target_col = 'labels'
    y = df[target_col]
    df = df.drop(target_col, axis=1)
else:
    target_col = df.columns[-1]
    y = df[target_col]
    df = df.drop(target_col, axis=1)

print("\nAfter deleting column:")
print(df.head())

print("\nTail (last 5 rows):")
print(df.tail())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

df_onecol = pd.read_csv(path, usecols=[0])
print("\nLoaded 1 column only:")
print(df_onecol.head())

df_multicol = pd.read_csv(path, usecols=[0, 1, 2])
print("\nLoaded multiple columns:")
print(df_multicol.head())

df["new_col"] = range(1, len(df) + 1)
print("\nAfter adding new column:")
print(df.head())

first_col = df.columns[0]
df_filtered = df[df[first_col] > 10]
print(f"\nFiltered rows where {first_col} > 10:")
print(df_filtered.head())

output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(output_path, index=False)
print(f"\nModified data stored successfully at: {output_path}")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Histogram of Numeric Columns", fontsize=16)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f"Boxplot of {col}")
    plt.show()

sns.pairplot(df.sample(min(200, len(df))), diag_kind='kde')
plt.suptitle("Pairplot of Numeric Columns", y=1.02)
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(6, 3))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Bar Chart of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=(5, 5))
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribution of {col}")
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap", fontsize=16)
plt.show()

df_encoded = pd.get_dummies(df, drop_first=True)
print("\nEncoded dataset shape:", df_encoded.shape)

try:
    df_test = pd.read_csv(test_path)
    print("\nTest dataset loaded:", df_test.shape)
    df_test_encoded = pd.get_dummies(df_test, drop_first=True)
    missing_cols = set(df_encoded.columns) - set(df_test_encoded.columns)
    for col in missing_cols:
        df_test_encoded[col] = 0
    df_test_encoded = df_test_encoded[df_encoded.columns]
    print("\nColumn consistency maintained between train and test datasets.")
except FileNotFoundError:
    print("\nTest dataset not found.")

encoded_output = r"C://Users//S Rakshita//Desktop//SENITNELNET//SentinelNet_Oct_Batch//encoded_data.csv"
df_encoded.to_csv(encoded_output, index=False)
print(f"\nEncoded dataset saved to: {encoded_output}")


# ---------------------------------------------------
# 🧩 NEW ADDITIONS: Train-Test Split, Imputer, SMOTE, Scaler
# ---------------------------------------------------

print("\n--- Data Preprocessing Steps ---")

# 1️⃣ Train-Test Split
print("\nPerforming Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 2️⃣ SimpleImputer for Missing Values
print("\nApplying SimpleImputer (strategy='mean') on numeric columns...")
imputer = SimpleImputer(strategy='mean')
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
print("Missing values handled successfully!")

# 3️⃣ SMOTE for Class Imbalance
print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("After SMOTE: X_train =", X_train_resampled.shape, ", y_train =", y_train_resampled.shape)

# 4️⃣ StandardScaler for Numerical Features
print("\nApplying StandardScaler on numeric features...")
scaler = StandardScaler()
X_train_resampled[numeric_cols] = scaler.fit_transform(X_train_resampled[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
print("Standardization completed successfully!")

print("\n✅ All preprocessing steps (Imputer, SMOTE, Scaler) applied successfully!")
