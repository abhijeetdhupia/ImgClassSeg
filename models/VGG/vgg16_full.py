"""
A simple VGG16 model implementation.

Input = [224, 224]
Filter size = 3x3 and 1x1, stride = 1
MaxPool 2x2, stride = 2

vgg16 = [Input, conv3-64 * 2, Max, conv3-128 * 2, Max, conv3-256 * 3, Max, conv3-512 * 3, Max, conv3-512 * 3, Max,
 FC-4096 * 2, FC-1000, SoftMax]

In addition, BatchNorm2d module are also added. 
"""

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# To supress unneccsary warnings
import warnings
warnings.filterwarnings("ignore")

class VGG(nn.Module):
    """ 16 Layers VGG Network"""
    def __init__(self, in_channels, num_classes):
        super(VGG, self).__init__()
        self.block1 = nn.Sequential(

            # Block 1 
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

if __name__ == "__main__":
    model = VGG(in_channels=3, num_classes=1000)
    x = torch.randn((1,3,224,224)) # 512*7*7 
    # x = torch.randn((1,3,256,256)) # 512*8*8
    print(f'Final Shape: {model(x).shape}')