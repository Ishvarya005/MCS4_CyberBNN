import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load and preprocess your training data
df = pd.read_csv('final_4_h.csv')

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit the label encoder to the training data labels
df['Label'] = label_encoder.fit_transform(df['Label'])

# Save the label encoder
label_encoder_path = 'label_encoder.pkl'
import joblib
joblib.dump(label_encoder, label_encoder_path)
