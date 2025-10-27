import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("kdd_train.csv")
df_test = pd.read_csv("kdd_test.csv")

print("Train Shape:", df_train.shape)
print("Test Shape:", df_test.shape)

df = pd.concat([df_train, df_test], ignore_index=True)
print("Combined Shape:", df.shape)

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

if 'label' in df.columns:
    target = 'label'
elif 'attack' in df.columns:
    target = 'attack'
elif df.columns[-1] not in df.select_dtypes(include=np.number).columns:
    target = df.columns[-1]
else:
    target = None

if target:
    print("Target Column:", target)
    print(df[target].value_counts())
    plt.figure(figsize=(12,5))
    sns.countplot(data=df, x=target, order=df[target].value_counts().index, palette="coolwarm")
    plt.xticks(rotation=90)
    plt.title("Attack Type Distribution")
    plt.show()

for col in ['protocol_type', 'service', 'flag']:
    if col in df.columns:
        plt.figure(figsize=(10,4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10], palette="Set2")
        plt.title(f"Top 10 {col.capitalize()} Types")
        plt.xticks(rotation=45)
        plt.show()

numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

for col in numeric_cols[:5]:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

df[numeric_cols].hist(figsize=(15,12), bins=30)
plt.suptitle("Numeric Feature Distributions")
plt.show()

if target and 'protocol_type' in df.columns:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x='protocol_type', hue=target, palette="Spectral")
    plt.title("Protocol vs Attack Type")
    plt.show()

print("EDA Completed Successfully")
