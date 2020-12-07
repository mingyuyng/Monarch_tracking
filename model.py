import numpy as np
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import torch
import torch.nn as nn
from torch.autograd import Variable


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


class Sandwich(nn.Module):
    def __init__(self, c_in, c_out, filter_size):
        super(Sandwich, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, filter_size, stride=1, padding=(filter_size - 1) // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x)


class CNN_Light(nn.Module):
    def __init__(self, length, channel, num_layers, num_neu, pdrop):
        super(CNN_Light, self).__init__()

        self.len = length

        blocks = [Sandwich(1, channel, 7)]
        self.len = self.len // 2
        for _ in range(num_layers - 1):
            blocks.append(Sandwich(channel, channel, 3))
            self.len = self.len // 2

        blocks.append(Flatten())
        blocks.append(nn.Linear(self.len * channel, num_neu))
        blocks.append(nn.ReLU())
        blocks.append(nn.Dropout(p=pdrop))
        blocks.append(nn.Linear(num_neu, num_neu))
        blocks.append(nn.ReLU())
        blocks.append(nn.Dropout(p=pdrop))
        blocks.append(nn.Linear(num_neu, 1))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):

        return self.net(x)


class CNN_Light_lite(nn.Module):
    def __init__(self, length, channel, num_layers, num_neu, pdrop):
        super(CNN_Light_lite, self).__init__()

        self.len = length

        blocks = [Sandwich(1, channel, 3)]
        self.len = self.len // 2
        for _ in range(num_layers - 1):
            blocks.append(Sandwich(channel, channel, 3))
            self.len = self.len // 2

        blocks.append(nn.Conv1d(channel, 1, 1, stride=1))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):

        return torch.mean(self.net(x).squeeze(), dim=1, keepdim=True)


class CNN_Temp(nn.Module):
    def __init__(self, size, pdrop, num_neu):
        super(CNN_Temp, self).__init__()

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=pdrop)

        self.fc1 = nn.Linear(size, num_neu)
        self.fc2 = nn.Linear(num_neu, num_neu)
        self.fc3 = nn.Linear(num_neu, num_neu)
        self.fc4 = nn.Linear(num_neu, 1)

    def forward(self, x):

        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)            # Din = 1024, Dout = 1024
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x


class CNN_hybrid(nn.Module):
    def __init__(self, size, isdrop=0):
        super(CNN_hybrid, self).__init__()
        # Cin = 1, Cout = 256, Kernel_size = 11
        self.relu = nn.ReLU()
        self.isdrop = isdrop
        #self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        if self.isdrop == 1:
            self.drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.fc2(x)            # Din = 1024, Dout = 1024
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.fc3(x)

        return x


class CNN_large(nn.Module):
    def __init__(self, length, isdrop):
        super(CNN_large, self).__init__()
        # Cin = 1, Cout = 256, Kernel_size = 11
        self.isdrop = isdrop

        self.conv1 = nn.Conv1d(1, 64, 3, stride=1, padding=1)
        # Cin = 256, Cout = 256, Kernel_size = 5
        self.conv2 = nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 128, 3, stride=1, padding=1)

        # Batch Nromalization
        self.batnorm1 = nn.BatchNorm1d(64)
        self.batnorm2 = nn.BatchNorm1d(128)
        self.batnorm3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        if self.isdrop == 1:
            self.drop = nn.Dropout(p=0.25)

        self.len = length

        self.fc1 = nn.Linear(int(self.len / 8) * 128, 128)

        #self.fc1 = nn.Linear(self.len, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.conv1(x)          # Cin = 1, Cout = 64, Kernel_size = 11
        x = self.batnorm1(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool1(x)

        x = self.conv2(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm2(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool2(x)

        x = self.conv3(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm3(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool3(x)

        x = x.view(-1, int(self.len / 8) * 128)
        #x = x.squeeze(1)
        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.fc2(x)            # Din = 1024, Dout = 1024
        x = self.relu(x)
        x = self.fc3(x)            # Din = 1024, Dout = 1

        return x
