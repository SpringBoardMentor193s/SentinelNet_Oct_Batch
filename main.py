import pandas as pd

df = pd.read_csv("kdd_test.csv")

print("Initial Dataset:")
print(df.head())

col_to_remove = "urgent"
if col_to_remove in df.columns:
    df.drop(col_to_remove, axis=1, inplace=True)
    print(f"\nColumn '{col_to_remove}' removed successfully.")
else:
    print(f"\nColumn '{col_to_remove}' not found in dataset.")

print("\nDataset after removing the column:")
print(df.head())

df.to_csv("cleaned_kdd_test.csv", index=False)
print("\nModified dataset saved as 'cleaned_kdd_test.csv' in the same folder.")
