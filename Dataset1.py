import pandas as pd
from scipy.io import arff
import pandas as pd

# Read and clean the file before parsing
with open("KDD_train+.arff", "r", encoding="utf-8") as f:
    content = f.read()

# Clean up messy quotes and spaces
content = content.replace(" 'icmp'", "icmp").replace("'icmp'", "icmp")

# Write it back to a temp file
with open("clean_KDD_train+.arff", "w", encoding="utf-8") as f:
    f.write(content)

# Now load it safely
data, meta = arff.loadarff("clean_KDD_train+.arff")
df_train = pd.DataFrame(data)
df_train.head()

with open("KDD_test+.arff", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(" 'icmp'", "icmp").replace("'icmp'", "icmp")

with open("clean_KDD_test+.arff", "w", encoding="utf-8") as f:
    f.write(content)

data, meta = arff.loadarff("clean_KDD_test+.arff")
df_test = pd.DataFrame(data)
df_test.head()

print(f'''
Dataset Train-\n
size:-{df_train.size}\n
shape:-{df_train.shape}\n
Null values:-
{df_train.isna().sum()}\n\n
total missing value percentage:-{(df_train.isnull().sum().sum()/df_train.size ) * 100}%

preview:-\n
{df_train.head()}
''')
print(f"Columns with object datatype:- {df_train.dtypes[df_train.dtypes == 'object'].index.tolist()}")

