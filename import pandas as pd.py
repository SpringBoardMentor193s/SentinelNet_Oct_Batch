import pandas as pd # type: ignore

#loading the dataset
df = pd.read_csv('Wednesday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print("Dataset 1 loaded successfully...\n\n")

#displaying basic information about the dataset
print(f"the size of dataset 1 is:{df.size}\n\n")
print(f"the shape of dataset 1 is:{df.shape}\n\n")
print(f" the columns of dtaaset 1 are:{df.columns.tolist()}\n\n")

#counting missing values
num_missing = df.isnull() .sum().sum()
print(f"the number of missing values in dataset 1 is:{num_missing}\n\n")

#display dataset info
print(f" The dataset 1 info\n:{df.info()}")
df.info()