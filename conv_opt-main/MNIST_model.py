import torch.nn as nn
import torch
from torch.utils.data import Dataset
from PIL import Image

# Classifier Model for the MNIST Dataset
class MNISTClassifier(nn.Module):

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.EmbeddingLearner = nn.Sequential(
            nn.Conv2d(1,16,3,padding=(1,1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(16,32,3,padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(7*7*32, 128),
            nn.ReLU(True),
            nn.Linear(128,10)
        )
        
    def forward(self, input):
        x = self.EmbeddingLearner(input)
        flat_x = torch.flatten(x,1)
        out = self.fc(flat_x)
        return out

# Dataset Class for the MNIST Dataset
class MNISTDataset(Dataset):
    def __init__(self,X,y,transform=None):
        self.data = X.numpy()
        self.label = y.numpy()
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        x = Image.fromarray(self.data[index], mode='L')

        if self.transform:
            x = self.transform(x)

        y = self.label[index]

        return (x,y)