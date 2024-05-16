import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess data
df = pd.read_csv('cleaned_final_data.csv')

# Select the desired features
selected_features = [
    'Flow IAT Mean', 'Fwd Pkts/s', 'Init Fwd Win Byts', 'Flow Duration', 
    'Fwd IAT Mean', 'Dst Port', 'Fwd IAT Tot', 'Fwd IAT Max', 'Flow IAT Max', 
    'Bwd Pkts/s', 'Flow Pkts/s', 'Fwd IAT Min', 'Fwd Header Len', 
    'Flow IAT Std', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 
    'Pkt Size Avg', 'Flow IAT Min', 'Fwd IAT Std', 'Fwd Pkt Len Mean', 
    'Fwd Seg Size Avg', 'Bwd Pkt Len Mean', 'Bwd Seg Size Avg', 
    'Pkt Len Max', 'TotLen Fwd Pkts', 'Subflow Fwd Byts', 
    'Bwd IAT Std', 'TotLen Bwd Pkts', 'Subflow Bwd Byts', 
    'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 'Bwd IAT Mean', 
    'Fwd Pkt Len Std', 'Bwd Header Len', 'Bwd Pkt Len Std', 
    'Bwd IAT Tot', 'Bwd IAT Max', 'Init Bwd Win Byts', 
    'Fwd Seg Size Min', 'Flow Byts/s', 'Tot Bwd Pkts', 
    'Subflow Bwd Pkts', 'Bwd IAT Min', 'Tot Fwd Pkts', 
    'Subflow Fwd Pkts'
]

# Drop the remaining columns
df = df[selected_features + ['Label']]

# Label encoding for target variable
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(df[selected_features])
df[selected_features] = X_normalized

# Group data by class label
grouped = df.groupby('Label')

# Perform KS test for each feature
ks_results = []
class_labels = df['Label'].unique()
for feature in selected_features:
    for i in range(len(class_labels)):
        for j in range(i + 1, len(class_labels)):
            class_i = class_labels[i]
            class_j = class_labels[j]
            data_i = grouped.get_group(class_i)[feature]
            data_j = grouped.get_group(class_j)[feature]
            ks_stat, p_value = ks_2samp(data_i, data_j)
            ks_results.append({
                'Feature': feature,
                'Class 1': class_i,
                'Class 2': class_j,
                'KS Statistic': ks_stat,
                'P-Value': p_value
            })

ks_df = pd.DataFrame(ks_results)

# Display the results as a table image
pivot_table = ks_df.pivot_table(values='KS Statistic', index='Feature', columns=['Class 1', 'Class 2'])
plt.figure(figsize=(20, 15))
sns.set(font_scale=1.2)
ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'KS Statistic'})
plt.title('KS Test Results')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the table as an image
plt.savefig('ks_test_results.png')
plt.show()

# Display the dataframe
print(ks_df)
