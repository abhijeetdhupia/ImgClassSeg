import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """ Double Conv Block: Conv2d --> ReLU --> Conv2d --> ReLU """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels= mid_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Down sampling block: {DoubleConv --> MaxPool2d} * 5 """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # Need to verify stride=2
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool(x)

class Up(nn.Module):
    """Up Sampling block"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
        self.final_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x2.size()[2] - x1.size()[2]  
        diffY = x2.size()[3] - x1.size()[3] 

        x1 = F.pad(x1, [diffX//2, diffX - diffY//2, diffY//2, diffY-diffY//2])
        
        x = torch.cat((x2, x1), dim=1)
        return self.final_conv(x)

class OutConv(nn.Module):
    """ Conv2d 1x1 """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.out_conv = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size=1)
    def forward(self, x):

        return self.out_conv(x)