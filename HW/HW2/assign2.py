import json
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import alexnet
import torchvision.transforms as transforms
import torchvision.models as models


class ImageNet(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.folder_paths = glob("{}/*/".format(self.path))
        self.json_path = "{}/imagenet_class_index.json".format(self.path)

        with open("{}/imagenet_class_index.json".format(self.path), "r") as f:
            self.lbl_dic = json.load(f)
        self.lbl_dic = {v[0]: int(k) for k, v in self.lbl_dic.items()}

        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.imgs = []
        self.lbls = []
        for folder_path in self.folder_paths:
            image_paths = glob("{}/*".format(folder_path))
            self.imgs += image_paths
            self.lbls += [self.lbl_dic[folder_path.split("/")[-2]]] * len(image_paths)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        return img, lbl

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    # Your code goes here
    tiny_image_net = ImageNet("/Users/liaokai/Desktop/CV/HW2/imagenet_12")
    model = alexnet(pretrained=True)
    model.eval()

    # print(tiny_image_net[0][0].shape)

    testloader = torch.utils.data.DataLoader(tiny_image_net, batch_size=1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
