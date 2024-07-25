import json
import random
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
import glob
import pandas as pd
import csv
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
class MyDataSet(IterableDataset):
    def __init__(self, projects):
        super(MyDataSet, self).__init__()
        self.projects = projects

    def __iter__(self):
        for project in self.projects:
            with open(project, 'r+') as file:
                text_data = file.read()
            text_data = json.loads(text_data)
            csv_data = pd.read_csv(project.replace('.json','.csv'),header=0,index_col=0)
            if csv_data.shape[0] != len(text_data):
                print(rf'{project} error!~')
                continue
            for index, text in enumerate(text_data):
                yield text, torch.from_numpy(csv_data.iloc[index,1:].to_numpy()), int(csv_data.iloc[index,0])

    def __len__(self):
        return len(self.projects)

class FullDataset(Dataset):

    def __init__(self, projects,stage='train'):
        super(FullDataset, self).__init__()
        self.projects = projects
        self.text_data = []
        self.features = []
        self.label = []
        self.stage = stage
        scaler = StandardScaler()
        # the data reading part
        if stage == 'train':
            text_data = json.load(open('./train_fail.json', 'r'))
            feature_data = pd.read_csv('./train_fail.csv',index_col=0)
            self.text_data.extend(text_data)
            self.features.extend(feature_data.iloc[:,1:-1].values.tolist())
            data = np.array(self.features)
            self.features = scaler.fit_transform(data)
            self.label.extend(feature_data.iloc[:,-1].to_list())
        elif stage == 'test':
            text_data = json.load(open('./test_fail.json', 'r'))
            feature_data = pd.read_csv('./test_fail.csv',index_col=0)
            self.text_data.extend(text_data)
            self.features.extend(feature_data.iloc[:,1:-1].values.tolist())
            data = np.array(self.features)
            self.features = scaler.fit_transform(data)
            self.label.extend(feature_data.iloc[:,-1].to_list())

        print(len(self.text_data), len(self.features), len(self.label))
        print(len(self.text_data), sum(self.label))
        self.label = [1 if item == 0 else 0 for item in self.label] 
        print(len(self.text_data), sum(self.label))
        # open csv file to save the label
        filename = f'./Results/{stage}_labels.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['label'])  

            for i in range(len(self.label)):
                writer.writerow([int(self.label[i])])

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        return self.text_data[idx], np.array(self.features[idx]), int(self.label[idx])

def load_data(args):
    projects = glob.glob(rf'/data/*.json')
    projects = list(set(projects))
    
    train_dataset, test_dataset = FullDataset(projects,stage='train'), FullDataset(projects,stage='test')
    # define weights for each class
    label_weights = [0.8,0.2]
    # a simple example to define the weights for each class
    #label_weights = [sum(train_dataset.label) / len(train_dataset.label), 1 - sum(train_dataset.label) / len(train_dataset.label)]
    # define the weights for each sample
    sample_weights = [label_weights[label] for _, _, label in train_dataset]
    #sample_weights = [label_weights[label] for _, label in train_dataset]
    # create WeightedRandomSampler to handle imbalanced dataset
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,sampler=sampler), DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

# the test code. You can simply ingore it.
# if __name__ == '__main__':
#     project = 'Achilles'
#     with open(rf'./{project}.json', 'r+') as file:
#         text_data = file.read()
#     text_data = json.loads(text_data)  # change json file to dict
#     csv_data = pd.read_csv(rf'./zg_dataset/{project}.csv', header=0, index_col=0)

#     print(1)