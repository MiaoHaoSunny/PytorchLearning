import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as torchdata
from PIL import Image
import torch.nn as nn
from sklearn.utils import shuffle

# rawdata_root = './'
# all_pd = pd.read_csv('./train.csv', sep=",")
# all_pd = all_pd.drop(['Id'], axis=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class dataset(torchdata.Dataset):
    def __init__(self, csv_file, mode=None, transform=None):
        self.all_pd = pd.read_csv(csv_file).drop(['Id'], axis=1)
        self.transform = transform
        self.mode = mode
        # print(self.mode)
        # self.all_pd = shuffle(self.all_pd)
        self.split_length = int(len(self.all_pd) / 3)
        # print(split_length)
        self.val_data = self.all_pd[: self.split_length]
        self.train_data = self.all_pd[self.split_length:]
        self.test_data = self.all_pd

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test':
            return len(self.test_data)

    def __getitem__(self, idx):
        image = None
        labels = None
        if self.mode == 'train':
            image = self.train_data.iloc[idx, :-1].values
            labels = self.train_data.iloc[idx, -1]
        elif self.mode == 'val':
            image = self.val_data.iloc[idx, :-1].values
            labels = self.val_data.iloc[idx, -1]
        elif self.mode == 'test':
            image = self.test_data.iloc[idx, :-1].values
            labels = self.test_data.iloc[idx, -1]
        else:
            print('Wrong mode!!!')
        # image = self.all_pd.iloc[idx, :-1].values
        # # image = Image.fromarray()
        # labels = self.all_pd.iloc[idx, -1]
        sample = {'image': image, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


train_dataset = dataset(csv_file='./train.csv', mode='train')
val_dataset = dataset(csv_file='./train.csv', mode='val')
test_dataset = dataset(csv_file='./test.csv', mode='test')
# print(train_dataset[10080])
train_loader = torchdata.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
val_loader = torchdata.DataLoader(dataset=val_dataset, batch_size=100, shuffle=True)
test_loader = torchdata.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
# print(train_loader.data)
# model = nn.Conv1d(54, 8, kernel_size=3)
# optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# criterion = nn.CrossEntropyLoss()
i = 0
for sample in train_loader:
    i += 1
    print("The {}th, data:{}".format(i, sample['labels']))
