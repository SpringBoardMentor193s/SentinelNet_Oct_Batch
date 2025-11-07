import pandas as pd


train_data = pd.read_csv("kdd_train.csv")
test_data = pd.read_csv("kdd_test.csv")

print("Train Data Shape:", train_data.shape)
print(train_data.head(), "\n")

print("Test Data Shape:", test_data.shape)
print(test_data.head(), "\n")


if 'flag' in train_data.columns:
    train_data.drop('flag', axis=1, inplace=True)
    print("'flag' column dropped.\n")


print("Dataset Info:")
print(train_data.info(), "\n")


total_nulls = train_data.isnull().sum().sum()
print("Total Missing Values:", total_nulls, "\n")

print("Statistical Summary:")
print(train_data.describe(), "\n")

# 
print("Last 7 Rows:")
print(train_data.tail(7))
