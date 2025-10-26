import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Column names for the dataset
column_names = [
 'duration', 'protocol_type', 'service', 'flag',
 'src_bytes', 'dst_bytes',
 'land', 'wrong_fragment', 'urgent', 'hot',
 'num_failed_logins',
 'logged_in', 'num_compromised',
 'root_shell', 'su_attempted',
 'num_root', 'num_file_creations',
 'num_shells', 'num_access_files',
 'num_outbound_cmds', 'is_host_login',
 'is_guest_login', 'count',
 'srv_count', 'serror_rate', 'srv_serror_rate',
 'rerror_rate',
 'srv_rerror_rate', 'same_srv_rate',
 'diff_srv_rate', 'srv_diff_host_rate',
 'dst_host_count', 'dst_host_srv_count',
 'dst_host_same_srv_rate',
 'dst_host_diff_srv_rate',
 'dst_host_same_src_port_rate',
 'dst_host_srv_diff_host_rate',
 'dst_host_serror_rate',
 'dst_host_srv_serror_rate',
 'dst_host_rerror_rate',
 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

# Read CSV file
train_df = pd.read_csv('kdd_train.csv', header=None, names=column_names)

# Drop unnecessary column
train_df.drop(['difficulty'], axis=1, inplace=True)

# Convert class labels into binary (0 = normal, 1 = attack)
train_df['attack_binary'] = train_df['class'].apply(lambda x: 0 if x == 'normal' else 1)

# Count Normal vs Attack
counts = train_df['attack_binary'].value_counts()
labels = ['Normal', 'Attack']

# Create a figure with 3 subplots
plt.figure(figsize=(18, 5))

# 1️⃣ Vertical Bar Chart
plt.subplot(1, 3, 1)
sns.barplot(x=labels, y=counts.values, palette='pastel')
plt.title('Vertical Bar: Network Traffic', fontsize=14)
plt.ylabel('Count', fontsize=12)
for i, val in enumerate(counts.values):
    plt.text(i, val + 500, str(val), ha='center', fontsize=10)

# 2️⃣ Horizontal Bar Chart
plt.subplot(1, 3, 2)
plt.barh(labels, counts.values, color=['#1f77b4', '#ff7f0e'])
plt.title('Horizontal Bar: Network Traffic', fontsize=14)
plt.xlabel('Count', fontsize=12)
for i, val in enumerate(counts.values):
    plt.text(val + 500, i, str(val), va='center', fontsize=10)

# 3️⃣ Pie Chart
plt.subplot(1, 3, 3)
plt.pie(counts.values, labels=labels, autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff'], startangle=90, explode=(0.05,0.05), shadow=True)
plt.title('Pie Chart: Network Traffic', fontsize=14)

plt.tight_layout()

# 🔥 Save the figure to a file
plt.savefig('task3_output.png', dpi=300)  # PNG file
# plt.savefig('network_traffic_distribution.pdf')  # Optional: PDF file

plt.show()
