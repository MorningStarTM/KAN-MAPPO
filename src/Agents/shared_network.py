import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import os
from src.Utils.logger import logger
from src.Utils.memory import RolloutBuffer
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)   # (210, 160, 3) -> (32, 52, 39)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # (32, 52, 39) -> (64, 25, 18)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # (64, 25, 18) -> (64, 23, 16)
        
        # Calculate the flattened size after convs (do a dummy forward pass or calculate manually)
        self.flattened_size = 64 * 23 * 16

    def forward(self, x):
        # x is already (B, 1, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return x

