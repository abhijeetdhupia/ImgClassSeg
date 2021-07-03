from tqdm import tqdm 

import torch
from torch.utils import data
from  torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor 

root = './data'
batch_size = 64

# N(dim=0), C(dim=1), H(dim=2), W(dim=3)

# Load Data
train_dataset = datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)

def get_mean_sd(loader):
    channels_sum, channel_squared_sum, num_batches = 0, 0, 0
    loop = tqdm(loader)

    loop.set_description(f"Calculating Mean and SD")
    
    # Accuracy and Loss values should be replaced by the respective values 
    # loop.set_postfix(Accuracy=torch.rand(1).item(), Loss=torch.randn(1).item())
    
    for data, _ in loop:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channel_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches +=1 
    mean = channels_sum/num_batches
    sd = (channel_squared_sum/num_batches - mean**2)**0.5
    return mean, sd

mean, sd = get_mean_sd(train_loader)
print(f'Mean = {mean}')
print(f'SD = {sd}')
