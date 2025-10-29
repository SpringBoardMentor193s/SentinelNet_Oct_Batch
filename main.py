import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "C:\\Users\\S Rakshita\\Downloads\\nslkdd.csv"
df = pd.read_csv(path)

print("Before deleting column:")
print(df.head())

if 'label' in df.columns:
    df = df.drop('label', axis=1)
else:
    df = df.drop(df.columns[-1], axis=1)

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
if pd.api.types.is_numeric_dtype(df[first_col]):
    df_filtered = df[df[first_col] > 10]
    print(f"\nFiltered rows where {first_col} > 10:")
    print(df_filtered.head())

output_path = "C:\\Users\\S Rakshita\\Downloads\\modified_data.csv"
df.to_csv(output_path, index=False)
print(f"\nModified data stored successfully at: {output_path}")

df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Histogram of Numeric Columns", fontsize=16)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f"Boxplot of {col}")
    plt.show()

sns.pairplot(df.sample(min(200, len(df))), diag_kind='kde')
plt.suptitle("Pairplot of Numeric Columns", y=1.02)
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap", fontsize=16)
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, palette="viridis")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

for col in categorical_cols:
    if 'attack' in df.columns or 'class' in df.columns:
        target = 'attack' if 'attack' in df.columns else 'class'
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, hue=target, data=df, palette="Set2")
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

df_encoded = pd.get_dummies(df, drop_first=True)
print("\nEncoded Feature Set:")
print(df_encoded.head())

test_path = "C:\\Users\\S Rakshita\\Downloads\\nslkdd_test.csv"
try:
    df_test = pd.read_csv(test_path)
    df_test_encoded = pd.get_dummies(df_test, drop_first=True)
    missing_cols = set(df_encoded.columns) - set(df_test_encoded.columns)
    for col in missing_cols:
        df_test_encoded[col] = 0
    df_test_encoded = df_test_encoded[df_encoded.columns]
    print("\nColumn consistency check successful between train and test data.")
except FileNotFoundError:
    print("\nTest dataset not found for column consistency check.")
