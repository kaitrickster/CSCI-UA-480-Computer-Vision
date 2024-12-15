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
        self.transform = transforms.Compose([transforms.ToTensor()])

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
        
        try:
            gt_mask = np.concatenate(gt_mask, axis=-1)
            target = {}
            boxes = []
            for i in range(gt_mask.shape[2]):
                pos = np.where(gt_mask[:,:,i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            labels = torch.ones((gt_mask.shape[2],), dtype=torch.int64)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target["boxes"] = boxes
            target["labels"] = labels

            if self.transform:
                img = self.transform(img)

            return img, target
        
        except:
            '''
            return next image that has mask
            it may happen that an image is used multiple times during training
            but the effect is parhaps negligible given the size of dataset.
            '''
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def training():
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 1

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=120)
        lr_scheduler.step()

    print("Finished Training")


if __name__ == "__main__":
    # TODO: training
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # define dataset: 70/30 train/val
    train_set = EgoHands('_LABELLED_SAMPLES/', get_transform(train=True))
    val_set = EgoHands('_LABELLED_SAMPLES/', get_transform(train=False))
    indices = torch.randperm(len(train_set)).tolist()
    train_size = int(len(train_set) * 0.7)
    val_set = torch.utils.data.Subset(val_set, indices[:-train_size])
    train_set = torch.utils.data.Subset(train_set, indices[-train_size:])

    # define dataloader
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

    model.to(device)

    # start training
    training()

    torch.save(model, "./model.pth")
