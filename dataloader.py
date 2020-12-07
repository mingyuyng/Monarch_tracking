import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import os
import glob


class dataloader_light(data.Dataset):

    def __init__(self, filename):

        self.mat_data = sio.loadmat(filename)
        self.data = self.mat_data['data']
        self.labels = self.mat_data['label']

    def __getitem__(self, index):

        intensity = self.data[index].astype('double')
        label = self.labels[index]

        pair = {'intensity': intensity, 'label': label}
        return pair

    def __len__(self):
        return self.mat_data['data'].shape[0]


class dataloader_tmp(data.Dataset):

    def __init__(self, filename):

        self.mat_data = sio.loadmat(filename)
        self.data = self.mat_data['tmp_set']
        self.labels = self.mat_data['labels']

    def __getitem__(self, index):

        temp = self.data[index].astype('double')
        label = self.labels[index][0]

        pair = {'temp': temp, 'label': label}
        return pair

    def __len__(self):
        return self.mat_data['tmp_set'].shape[0]
