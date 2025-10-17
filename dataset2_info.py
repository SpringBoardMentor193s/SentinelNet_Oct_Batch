import pandas as pd
import numpy as np

#loading the dataset
df = pd.read_csv('kdd_train.csv')
print("Dataset 2 loaded successfully...")
#displaying basic information about the dataset
print(f"the size of dataset 2 is:{df.size}")
print(f"the shape of dataset 2 is:{df.shape}")
print(f"the columns of dataset 2 are:{df.columns.tolist()}")

num_missing = df.isnull().sum().sum()
print(f"the number of missing values in dataset 2 is:{num_missing}")

print(f"The dataset 2 info\n:{df.info()}")
