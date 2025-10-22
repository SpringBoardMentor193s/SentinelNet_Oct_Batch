import pandas as pd

data = pd.read_csv("kdd_train.csv")
print(data)

#data.drop('Flag', axis=1, inplace=True) 