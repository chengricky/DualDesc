"""
The dataset file of West Lake is only used for TEST, so the positive selection is not implemented
Ricky 2019.Dec.10
"""
import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join, exists
import os
from PIL import Image

root_dir = '/localresearch/VisualLocalization/Dataset/Self-collected/HangzhouCity/WestLake'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to West Lake dataset')

# the list of database folder (images)
dbFolder = join(root_dir, 'westlake-5-database')
qFolder = join(root_dir, 'westlake-5-query')


def input_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_whole_val_set(onlyDB=False):
    """
    database + query
    """
    return DatasetFromStruct(dbFolder, qFolder, input_transform=input_transform(), onlyDB=onlyDB)


class DatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolder, input_transform=None, onlyDB=False):
        super().__init__()

# DATABASE 图像
        self.input_transform = input_transform
        listImg = os.listdir(dbFolder)
        listImg.sort()
        listImg = [img for img in listImg if 'color' in img]
        self.images = []
        self.images.extend([join(dbFolder, dbIm) for dbIm in listImg])
        self.numDb = len(self.images)
# QUERY 图像
        if not onlyDB:
            listImg = os.listdir(qFolder)
            listImg.sort()
            listImg = [img for img in listImg if 'color' in img]
            self.images.extend([join(qFolder, qIm) for qIm in listImg])
        self.numQ = len(self.images)-self.numDb

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

