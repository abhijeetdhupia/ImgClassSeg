"""
It is a slightly different approach from the original UNet implementation. 
Instead of cropping, padding is utilized before the concatenation of layers while Upsampling.
"""

import torch 
import torch.nn as nn
from unet_funcs import * 

# To supress unneccsary warnings
import warnings
warnings.filterwarnings("ignore")

class UNet(nn.Module):
    """Main UNET class"""
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

    # Encoder Part
        self.input = DoubleConv(num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256) 
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    # Decoder Part
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.output = OutConv(64, num_classes)

    def forward(self, x):

        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x = self.output(x9)
        return x 


if __name__ == "__main__":
    image = torch.randn(1, 1, 572, 572)
    model = UNet(num_channels=1, num_classes=2)
    print(model(image).shape)