import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np
from PIL import Image
import tqdm
from glob import glob

# Mingxing Tan, Quoc V. Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019
# pip install --upgrade efficientnet-pytorch
# https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # model with randomly intialized weights
        self.effNet = EfficientNet.from_name('efficientnet-b2')
        self.MLP = nn.Sequential(
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 43)
        )
            
    def forward(self, x):
        x = self.effNet(x)
        x = self.MLP(x)
        return x

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("yes")
    else:
        device = torch.device('cpu')
        
    # initialize with Net()
    # images are resized and transformed according to the notebook ipynb
    model = Net()
    model.to(device)
