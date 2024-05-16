import matplotlib.pyplot as plt

# Define the labels and their frequencies
labels = ['Infiltration', 'Benign', 'Bot', 'DoS']
frequencies = [150000, 150000, 150000, 150000]

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.bar(labels, frequencies, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Distribution of Labels')
plt.show()
