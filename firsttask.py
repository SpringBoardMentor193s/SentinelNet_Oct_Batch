import pandas as pd

data = pd.read_csv(r"C:\Users\Dev Bhatt\OneDrive\Desktop\loan_data_set.csv")
#print(data)

data.drop('Education', axis=1, inplace=True) 