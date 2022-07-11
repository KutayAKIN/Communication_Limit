import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# Dataset Class for the CIFAR Dataset
class AdverseWeatherDataset(Dataset):
    def __init__(self,X,y,transform=None,cache_all=False):
        if cache_all:
            self.img_locs = X.numpy()
        else:
            self.img_locs = X
        self.label = y
        self.transform = transform
        self.cache_all = cache_all


    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):

        if self.cache_all:
            x = Image.fromarray(self.img_locs[index])
        else:
            x = Image.open(self.img_locs[index])
        
        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)