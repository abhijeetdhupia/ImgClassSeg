"""
VGG16 model implementation using functions defined in vgg_func.py

Input = [224, 224]
Filter size = 3x3 and 1x1, stride = 1
MaxPool 2x2, stride = 2

vgg16 = [Input, conv3-64 * 2, Max, conv3-128 * 2, Max, conv3-256 * 3, Max, conv3-512 * 3, Max, conv3-512 * 3, Max,
 FC-4096 * 2, FC-1000, SoftMax]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg_funcs import * 

# To supress unneccsary warnings
import warnings
warnings.filterwarnings("ignore")

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes 

        self.block1 = DoubleConv(in_channels, 64)
        self.block2 = DoubleConv(64, 128)
        self.block3 = ThreeConv(128, 256)
        self.block4 = ThreeConv(256, 512)
        self.block5 = ThreeConv(512, 512)

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
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x 

if __name__ == "__main__":
    model = VGG_net(in_channels=3, num_classes=1000)
    x = torch.randn((1,3,224,224)) # 512*7*7 
    # x = torch.randn((1,3,256,256)) # 512*8*8
    print(f'Final Shape: {model(x).shape}')