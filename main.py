import pandas as pd
import numpy as np
train_df = pd.read_csv('/content/Kdd_testing.csv')
test_df = pd.read_csv('/content/kdd_training.csv')

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print(train_df.tail())

print(test_df.tail())

print("Training Data Info:")
train_df.info()
print("\n Testing Data Info:")
test_df.info()

print(train_df.isnull().sum())

print(test_df.isnull().sum())

if 'duration' in train_df.columns:
    print(train_df['duration'].head())
else:
    print(" Column 'duration' not found in dataset.")

cols = [col for col in ['duration', 'protocol_type', 'service'] if col in train_df.columns]
if cols:
    print(train_df[cols].head())
else:
    print(" Columns not found in dataset.")

if 'service' in train_df.columns:
    train_df.drop('service', axis=1, inplace=True)
    print(" Column 'service' deleted successfully!")
else:
    print("Column 'service' not found, skipping delete.")
print("Updated columns after delete:")
print(train_df.columns.tolist())

train_df['new_constant'] = 0
train_df['remarks'] = ['normal'] * len(train_df)
print(train_df.head())

column_names = [
'duration', 'protocol_type', 'service', 'flag',
'src_bytes', 'dst_bytes',
'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins',
'logged_in', 'num_compromised',
'root_shell', 'su_attempted',
'num_root', 'num_file_creations',
'num_shells', 'num_access_files',
'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count',
'srv_count', 'serror_rate', 'srv_serror_rate',
'rerror_rate',
'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate',
'dst_host_count', 'dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'class', 'difficulty' ]
print("Train Data Head:")
print(train_df.head())
print("\nTrain Data Info:")
print(train_df.info())
print("\nTest Data Head:")
print(test_df.head())
print("\nTest Data Info:")
print(test_df.info())
