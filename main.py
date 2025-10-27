import pandas as pd

path = "C:\\Users\\S Rakshita\\Downloads\\nslkdd.csv"
df = pd.read_csv(path)

print("Before deleting column:")
print(df.head())

if 'label' in df.columns:
    df = df.drop('label', axis=1)
else:
    df = df.drop(df.columns[-1], axis=1)

print("\nAfter deleting column:")
print(df.head())

print("\nTail (last 5 rows):")
print(df.tail())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

df_onecol = pd.read_csv(path, usecols=[0])
print("\nLoaded 1 column only:")
print(df_onecol.head())

df_multicol = pd.read_csv(path, usecols=[0, 1, 2])
print("\nLoaded multiple columns:")
print(df_multicol.head())

df["new_col"] = range(1, len(df) + 1)
print("\nAfter adding new column:")
print(df.head())

first_col = df.columns[0]
df_filtered = df[df[first_col] > 10]
print(f"\nFiltered rows where {first_col} > 10:")
print(df_filtered.head())

output_path = "C:\\Users\\91900\\Downloads\\modified_data.csv"
df.to_csv(output_path, index=False)
print(f"\nModified data stored successfully at: {output_path}")