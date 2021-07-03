from unet_parts import DoubleConv
import torch 
import torch.nn as nn
import numpy as np 
from unet_parts import * 
import matplotlib.pyplot as plt 
from torchvision import transforms

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

        print(x.size())
        x1 = self.input(x)
        print(f'x1 = {x1.size()}')
        x2 = self.down1(x1)
        print(f'x2 = {x2.size()}')
        x3 = self.down2(x2) 
        print(f'x3 = {x3.size()}')
        x4 = self.down3(x3)
        print(f'x4 = {x4.size()}')
        x5 = self.down4(x4)
        print(f'x5 = {x5.size()}')

        x6 = self.up1(x5, x4)
        print(f'x6 = {x6.size()}')
        x7 = self.up2(x6, x3)
        print(f'x7 = {x7.size()}')
        x8 = self.up3(x7, x2)
        print(f'x8 = {x8.size()}')
        x9 = self.up4(x8, x1)
        print(f'x9 = {x9.size()}')
        x = self.output(x9)
        print(f'x = {x.size()}')
        return x 


if __name__ == "__main__":
    image = torch.randn(1, 1, 572, 572)
    #  plt.imshow(image.permute(1,2,0))

    model = UNet(num_channels=1, num_classes=2)
    #  print(f'Image Shape= {image.size()}')
    #  model(image)
    #  print(model(image))
    print(model)

    #Plot the image

    #  new_img = torch.reshape(image,(1,572,572))
    #  print(new_img.shape)
    #  img = new_img.numpy().squeeze()
    #  plt.imshow(img)
    #  plt.show()

# RuntimeError: Given transposed=1, weight of size [1024, 512, 2, 2], expected input[1, 768, 136, 136] to 
# have 1024 channels, but got 768 channels instead

#   File "/Users/abhijeetdhupia/Documents/Unet/unet_model.py", line 52, in forward
#     x8 = self.up1(x7, x2)
#   File "/Users/abhijeetdhupia/Documents/miniconda3/envs/py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "/Users/abhijeetdhupia/Documents/Unet/unet_parts.py", line 46, in forward
#     x1 = self.up(x1)