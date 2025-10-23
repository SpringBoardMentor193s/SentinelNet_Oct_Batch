import pandas as pd

#loading the dataset
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print("Dataset 1 loaded successfully...\n\n")
#displaying basic information about the dataset
print(f"the size of dataset 1 is:{df.size}\n\n")
print(f"the shape of dataset 1 is:{df.shape}\n\n")
print(f"the columns of dataset 1 are:{df.columns.tolist()}\n\n")

num_missing = df.isnull().sum().sum()
print(f"the number of missing values in dataset 1 is:{num_missing}\n\n")

print(f"The dataset 1 info\n:{df.info()}")
