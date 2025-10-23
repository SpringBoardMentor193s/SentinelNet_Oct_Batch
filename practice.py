import pandas as pd

df = pd.read_csv('kdd_train.csv')
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

durations = df['duration']
src_and_dst_bytes = df[['src_bytes', 'dst_bytes']]
short_duration = df[df['duration'] < 100]

df.loc[df['protocol_type'] == 'tcp', ['service', 'flag']]
df = df.rename(columns={'hot': 'hot_connections'})
df['duration_in_seconds'] = df['duration'] / 60
df = df.drop(columns=['num_outbound_cmds'])
print(df.isnull().sum())
df['duration'] = df['duration'].fillna(df['duration'].mean())
df.to_csv('output.csv', index=False)
