import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import os
from src.Utils.logger import logger
from src.Utils.memory import RolloutBuffer
import torch.nn.functional as F
from convkan import ConvKAN, LayerNorm2D

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



class KANConvNet(nn.Module):
    def __init__(self):
        super(KANConvNet, self).__init__()
        # Assuming ConvKAN has the same interface as nn.Conv2d
        self.kan_conv1 = ConvKAN(1, 32, kernel_size=8, stride=4, padding=0)  # Match original ConvNet
        self.norm1 = LayerNorm2D(32)
        
        self.kan_conv2 = ConvKAN(32, 64, kernel_size=4, stride=2, padding=0)
        self.norm2 = LayerNorm2D(64)
        
        self.kan_conv3 = ConvKAN(64, 64, kernel_size=3, stride=1, padding=0)
        self.norm3 = LayerNorm2D(64)
        
        # Output size after convolutions (same as original ConvNet)
        self.flattened_size = 64 * 23 * 16  # (Double-check this matches your actual KANConvNet output)

    def forward(self, x):
        x = F.relu(self.norm1(self.kan_conv1(x)))
        x = F.relu(self.norm2(self.kan_conv2(x)))
        x = F.relu(self.norm3(self.kan_conv3(x)))
        x = x.reshape(x.size(0), -1)
        return x
