
from operator import mod
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import glob
import os
import random
import zipfile
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split

import torchvision

# -=====load dataset 
class TransformDataset(Dataset):
    def __init__(self, file_list, transform=None) -> None:
        super().__init__()
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data(batch_size):
        ## Training settings

    # image Augmentation 
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # ====loade dataset
    trainset = datasets.MNIST('datasets/mnist_train', train=True, download=True, transform=train_transforms)
    # testset = datasets.MNIST('mnist_test', train=False, download=True, transform=test_transforms)
    testset = datasets.MNIST('datasets/mnist_test', train=False, download=True, transform=test_transforms)
    
    print(f"Train Data: {len(trainset._load_data()[0])}")
    print(f"Test Data: {len(testset._load_data()[0])}")
    return trainset, testset


# 注意一下min train
"""
trainset._load_data()[0].shape --代表的是样本
torch.Size([60000, 28, 28])
trainset._load_data()[1].shape ---代表的式label
torch.Size([60000])
"""


