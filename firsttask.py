import pandas as pd

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
print(data)

#data.drop('Idle Min', axis=1, inplace=True) 