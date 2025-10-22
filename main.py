import pandas as pd

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
print(data)

data.drop('Idle Min', axis=1, inplace=True) 

print(data.head())

print(data.tail())

print(data.info())

total_nulls = 0
for column in data.columns:
    total_nulls += data[column].isnull().sum()

print(total_nulls)