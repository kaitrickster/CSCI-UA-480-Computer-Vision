import os
import numpy as np
from glob import glob
from PIL import Image
from scipy.io import loadmat
from matplotlib.path import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

import transforms as T
import utils
from engine import train_one_epoch, evaluate
import utils

# import variable
from main import val_set


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("./model.pth")
    model.eval()
    sample_inds = np.random.randint(low=0, high=len(dataset_val) - 1, size=5)
    sample_imgs = []
    
    predictions = []
    with torch.no_grad():
    for ind in sample_inds:
        predictions.append(model([val_set[ind][0].to(device)]))
        sample_imgs.append(val_set[ind][0])
