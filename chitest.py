import pandas as pd
from scipy.stats import chi2_contingency

# Load your data into a DataFrame
data = pd.read_csv("cleaned_final_data.csv")

# Define the target variable
target = 'Label'

# Drop the target variable from the DataFrame
features = data.drop(columns=[target])

# Initialize an empty dictionary to store chi-square values for each feature
chi2_values = {}

# Iterate over each feature and calculate the chi-square value
for feature in features.columns:
    contingency_table = pd.crosstab(data[target], data[feature])
    if contingency_table.size == 0:
        print(f"Skipping {feature} due to empty contingency table")
        continue
    chi2, _, _, _ = chi2_contingency(contingency_table)
    chi2_values[feature] = chi2

# Sort the chi-square values in descending order
sorted_chi2_values = sorted(chi2_values.items(), key=lambda x: x[1], reverse=True)

# Print the chi-square values with the column name in descending order
for feature, chi2 in sorted_chi2_values:
    print(f"{feature}: {chi2}")
