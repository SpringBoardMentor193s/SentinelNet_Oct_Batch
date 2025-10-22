import pandas as pd
import numpy as np

#loading the dataset
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print("Dataset 1 loaded successfully...")
#displaying basic information about the dataset
print(f"the size of dataset 1 is:{df.size}")
print(f"the shape of dataset 1 is:{df.shape}")
print(f"the columns of dataset 1 are:{df.columns.tolist()}")

num_missing = df.isnull().sum().sum()
print(f"the number of missing values in dataset 1 is:{num_missing}")

print(f"The dataset 1 info\n:{df.info()}")
