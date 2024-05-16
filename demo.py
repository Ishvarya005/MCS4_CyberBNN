import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import numpy as np
import joblib  

test_data = pd.read_csv('output.pcam')

X_test = test_data.drop(columns=['Label', 'Timestamp']).values

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = X_test.shape[1]  
hidden_dim = 100  
output_dim = 4  

model = BayesianNN(input_dim, hidden_dim, output_dim)

model.load_state_dict(torch.load('bayesian_nn_model.pth'))  

model.eval()

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

test_data['Predicted_Label'] = predicted.numpy()
label_encoder = joblib.load('label_encoder.pkl')
predicted_class_labels = label_encoder.inverse_transform(predicted.numpy())
test_data['Predicted_Class'] = predicted_class_labels

print(test_data[['Predicted_Class']])
