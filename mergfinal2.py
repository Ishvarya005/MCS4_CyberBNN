import pandas as pd
from collections import defaultdict


sampled_data = defaultdict(pd.DataFrame)


classes_to_sample = ['Bot', 'Infilteration', 'DoS']


desired_samples = {class_label: 150000 for class_label in classes_to_sample}


for i in range(1, 4):  
    
    dataset = pd.read_csv(f'E:\\Sharanya\\Hackathon\\Cyberset\\file{i}.csv')
    
    
    classes_present = dataset['Label'].unique()
    
    
    for class_label in classes_to_sample:
        if class_label in classes_present:
            class_data = dataset[dataset['Label'] == class_label]
            if len(class_data) > desired_samples[class_label]:
                sampled_data[class_label] = pd.concat([sampled_data[class_label], class_data.sample(n=desired_samples[class_label], replace=False)])
            else:
                sampled_data[class_label] = pd.concat([sampled_data[class_label], class_data])
        else:
            print(f"Class '{class_label}' not found in dataset. Skipping sampling for this class.")


final_dataset = pd.concat(sampled_data.values())

classes_to_sample = ['Benign']

desired_samples = {class_label: 50000 for class_label in classes_to_sample}


for i in range(1, 4):  
    
    dataset = pd.read_csv(f'E:\\Sharanya\\Hackathon\\Cyberset\\file{i}.csv')
    
    
    classes_present = dataset['Label'].unique()
    
   
    for class_label in classes_to_sample:
        if class_label in classes_present:
            class_data = dataset[dataset['Label'] == class_label]
            if len(class_data) > desired_samples[class_label]:
                sampled_data[class_label] = pd.concat([sampled_data[class_label], class_data.sample(n=desired_samples[class_label], replace=False)])
            else:
                sampled_data[class_label] = pd.concat([sampled_data[class_label], class_data])
        else:
            print(f"Class '{class_label}' not found in dataset. Skipping sampling for this class.")


final_dataset = pd.concat(sampled_data.values())

final_dataset = final_dataset.sample(frac=1).reset_index(drop=True)


final_dataset.to_csv('E:\\Sharanya\\Hackathon\\final_dataset4.csv', index=False)
