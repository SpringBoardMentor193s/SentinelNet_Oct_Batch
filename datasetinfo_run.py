import pandas as pd

# Read KDD datasets from current directory
train_path = 'kdd_train.csv'
test_path = 'kdd_test.csv'

print('\n--- Running datasetinfo_run.py ---\n')

# Load train
try:
    df_train = pd.read_csv(train_path)
except Exception as e:
    print(f"Failed to read {train_path}: {e}")
    raise

print('Dataset Train-\n')
print(f"size:-{df_train.size}\n")
print(f"shape:-{df_train.shape}\n")
print('Null values:-')
print(df_train.isna().sum())
print('\n')
print(f"total missing value percentage:-{(df_train.isnull().sum().sum()/df_train.size ) * 100}%\n")
print('preview:-\n')
print(df_train.head())

# Load test
try:
    df_test = pd.read_csv(test_path)
except Exception as e:
    print(f"Failed to read {test_path}: {e}")
    raise

print('\nDataset Test-\n')
print(f"size:-{df_test.size}\n")
print(f"shape:-{df_test.shape}\n")
print('Null values:-')
print(df_test.isna().sum())
print('\n')
print(f"total missing value percentage:-{(df_test.isnull().sum().sum()/df_test.size ) * 100}%\n")
print('preview:-\n')
print(df_test.head())

print('\n--- Done ---\n')
