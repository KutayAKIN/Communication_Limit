import torch.nn as nn
import torch

from utils import create_dataset

# Classifier Model for the Synthetic Dataset
class Classifier(nn.Module):
    def __init__(self,n_features,n_class):
        super(Classifier, self).__init__()
        self.EmbeddingLearner = nn.Sequential(
            nn.Linear(n_features, 7),
            nn.ReLU(True),
            nn.Linear(7, 8),
            nn.ReLU(True),
            nn.Linear(8, n_class)

        )
    def forward(self, input):
        output = self.EmbeddingLearner(input)
        return output

# Function to initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)

