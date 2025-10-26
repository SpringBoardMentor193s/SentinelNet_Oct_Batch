import pandas as pd

train_data = pd.read_csv("kdd_train.csv")
print(train_data)

test_data = pd.read_csv("kdd_test.csv")
print(test_data)

print(f"Shape: {train_data.shape}")
print(train_data.head())

print(f"Shape: {test_data.shape}")
print(test_data.head())


train_data.drop('flag', axis=1, inplace=True) 

print(train_data.head())

print(train_data.tail(7))

print(train_data.info())

total_nulls = 0
for column in train_data.columns:
    total_nulls += train_data[column].isnull().sum()

print(total_nulls)

print(train_data.describe())