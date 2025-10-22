import pandas as pd
path = "C:\\Users\\91900\\Downloads\\infsprnslkdddata.csv"

df = pd.read_csv(path)


print("Before deleting column:")
print(df.head())


if 'label' in df.columns:
    df = df.drop('label', axis=1)
else:
    df = df.drop(df.columns[-1], axis=1) 

print("\nAfter deleting column:")
print(df.head())
 


