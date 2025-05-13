import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)  # z 28x28 -> 28x28
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0) # z 14x14 -> 10x10
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding=0) # z 5x5 -> 1x1
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.flat = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.05)
        self.dense1 = torch.nn.Linear(in_features=120, out_features=84)
        self.dense2 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # B,1,28,28
        x = self.relu(self.conv1(x))    # B,6,28,28
        x = self.pool(x)                # B,6,14,14
        x = self.relu(self.conv2(x))    # B,16,10,10
        x = self.pool(x)                # B,16,5,5
        x = self.relu(self.conv3(x))    # B,120,1,1
        x = x.squeeze(-1).squeeze(-1)   # B,120
        x = self.relu(self.dense1(x))   # B,84
        x = self.dropout(x)
        x = self.dense2(x)              # B,10
        return x