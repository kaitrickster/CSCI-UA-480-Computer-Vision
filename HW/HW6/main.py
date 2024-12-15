import os
import numpy as np
from glob import glob
from PIL import Image
from scipy.io import loadmat
from matplotlib.path import Path

import torch
import torchvision.transforms as transforms

class EgoHands(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        folders = sorted(glob(os.path.join(self.path, "*")))
        self.imgs = []
        self.polygons = []
        for folder in folders:
            # Add images
            self.imgs += sorted(glob(os.path.join(folder, "*.jpg")))

            # Add polygons
            polygon_path = glob(os.path.join(folder, "*.mat"))[0]
            polygon = loadmat(polygon_path)['polygons'][0]
            for i in range(len(polygon)):
                self.polygons.append(polygon[i])

        # TODO: use suitable transformations
        self.transform = transform

    def __getitem__(self, index):
        # Load image
        img = np.array(Image.open(self.imgs[index]))

        # Compute mask
        polygons = self.polygons[index]
        gt_mask = []
        x, y = np.meshgrid(
            np.arange(img.shape[1]), np.arange(img.shape[0]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        for i, polygon in enumerate(polygons):
            if polygon.size == 0:
                continue
            path = Path(polygon)
            grid = path.contains_points(points)
            grid = grid.reshape((*img.shape[:2]))
            gt_mask.append(np.expand_dims(grid, axis=-1))
        gt_mask = np.concatenate(gt_mask, axis=-1)

        # TODO: compute minimal bounding boxes
        target = None

        if self.transform:
            img = transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    # TODO: training
    pass



#             images = sorted(glob(os.path.join(folder, "*.jpg")))
#             polygon_path = glob(os.path.join(folder, "*.mat"))[0]
#             polygon = loadmat(polygon_path)['polygons'][0]
#             buffer_2.append(len(polygon))
#             buffer.append(len(images))
            
            
#             for i in range(len(polygon)):
#                 polygons = polygon[i]
#                 gt_mask = []
#                 x, y = np.meshgrid(
#                     np.arange(img.shape[1]), np.arange(img.shape[0]))
#                 x, y = x.flatten(), y.flatten()
#                 points = np.vstack((x, y)).T
#                 for i, polygon in enumerate(polygons):
#                     if polygon.size == 0:
#                         continue
#                     path = Path(polygon)
#                     grid = path.contains_points(points)
#                     grid = grid.reshape((*img.shape[:2]))
#                 try:
#                     gt_mask = np.concatenate(gt_mask, axis=-1)
#                     self.imgs += images[i]
#                     self.polygons.append(polygon[i])
#                 except:
#                     pass
