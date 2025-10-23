import pandas as pd

df = pd.read_csv('kdd_train.csv') 

# View the first few rows: Display the top 5 rows of the DataFrame.
print(df.head())

# View the last few rows: Display the bottom 5 rows
print(df.tail())

# Get a summary of the data: Show the data types, non-null values, and memory usage for each column.
print(df.info())
print(df.describe(include = 'all'))

print(df.shape)
print(df.columns)

print(df['land'].value_counts)

df.dropna(inplace=True)
df.rename(columns={'land': 'chikky'}, inplace=True)
df.drop(columns=['hot'], inplace=True)
df['pappu'] = df['chikky'] * 2

df.to_csv('task_2_output.csv', index = False)