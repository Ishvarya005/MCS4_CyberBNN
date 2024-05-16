import pandas as pd
import numpy as np

# Load your CSV file into a DataFrame
df = pd.read_csv("final_data.csv", dtype=str)

# Convert all columns except the last one to numeric
numeric_columns = df.columns[:-1]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Iterate through each column (except the last one)
for column in numeric_columns:  # Exclude the last column
    # Replace NaN values with the mean of the column
    df[column].fillna(df[column].mean(), inplace=True)

    # Replace infinite (inf) values with the maximum value of the column
    max_value = df[column].replace([np.inf, -np.inf], np.nan).max()
    df[column] = df[column].replace([np.inf, -np.inf], max_value)

# Save the DataFrame back to a CSV file
df.to_csv("cleaned_final_data.csv", index=False)

print("NaN and inf values handled and saved to 'cleaned_final_data.csv'.")
