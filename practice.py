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

test_df=pd.read_csv("Dataset/KDD_Test.csv", header=None, names=column_names)
test_df.drop("difficulty", axis=1, inplace=True)

# Basic pandas functions in Train dataset
print("\n----Number of rows and columns in Train_df----")
print(train_df.shape)

print("\n----First 5 rows of Train_df----")
print(train_df.head())

print("\n----Last 5 rows of Train_df----")
print(train_df.tail())

print("\n----Structure of the Train_df----")
print(train_df.info())

print("\n----Statistical summary of numeric columns----")
print(train_df.describe())

print("\n----Missing values----")
print(train_df.isnull())

print("\n----Sum of a column----")
print(train_df['duration'].sum())

# Basic pandas functions in Test dataset
print("\n----First 5 rows of Test_df----")
print(test_df.head())

print("\n----Last 5 rows of Test_df----")
print(test_df.tail())

print("\n----Structure of the Test_df----")
print(test_df.info())

print("\n----Statistical summary of numeric colums----")
print(test_df.describe())

print("\n----Count----")
print(train_df['class'].value_counts())

# Creating Binary classification in Train dataset
train_df["binary_attack"]=train_df["class"].apply(lambda x:'0'if x == 'normal'else '1')
print("\nclass and binary attack in Train_df")
print(train_df[['class','binary_attack']])

# Creating binary classification in Test dataset
test_df["binary_attack"]=test_df["class"].apply(lambda x:'0'if x == 'normal'else '1')
print("\nclass and binary attack in Test_df")
print(test_df[['class','binary_attack']])

# Exploring the training dataset using basic plots
# -------------------- Line Plot --------------------
plt.figure(figsize=(8,5))
sns.lineplot(x='duration', y='src_bytes', data=train_df, color='blue')
plt.title('Duration vs Source Bytes')  
plt.xlabel('Duration')
plt.ylabel('Source Bytes')
plt.show()

# -------------------- Countplot--------------------
plt.figure(figsize=(6,5))
sns.countplot(x='binary_attack', hue='binary_attack', legend=False, data=train_df, palette=['cadetblue', 'crimson'])
plt.title('Normal vs Attack Distribution')
plt.xlabel('Binary Attack (0=Normal, 1=Attack)')
plt.ylabel('Count')
plt.show()