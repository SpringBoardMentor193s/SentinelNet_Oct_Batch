import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

data = pd.read_csv("kdd_test.csv")

print("Original Dataset:")
print(data.head())

column_to_delete = "hot"
if column_to_delete in data.columns:
    data.drop(column_to_delete, axis=1, inplace=True)
    print(f"\nColumn '{column_to_delete}' deleted successfully.")
else:
    print(f"\nColumn '{column_to_delete}' not found in dataset.")

print("\nDataset after deletion:")
print(data.head())

data.to_csv("updated_kdd_test.csv", index=False)
print("\nUpdated dataset saved as 'updated_kdd_test.csv' in the same folder.")
