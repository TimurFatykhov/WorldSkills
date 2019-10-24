from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torch
import PIL
import numpy as np

from albumentations import Compose, HorizontalFlip, Cutout, ShiftScaleRotate, ToGray, ToFloat, Transpose, PadIfNeeded, Resize
from albumentations.pytorch import ToTensor

mnist_dataset_train = MNIST('./mnist/', train=True, download=True)
mnist_dataset_test = MNIST('./mnist/', train=False, download=True)


class MNIST_Dataset(Dataset):
    def __init__(self, train=True, trs=None, size=1.0):
        global mnist_dataset
        self.train = train
        
        if self.train:
            self.data = mnist_dataset_train.data.numpy()
            self.N = len(self.data)
            self.targets = mnist_dataset_train.targets.numpy()
        else:
            self.data = mnist_dataset_test.data.numpy()
            self.N = len(self.data)
            self.targets = mnist_dataset_test.targets.numpy()
            
        self.N = int(self.N * size) 
        self.data = self.data[:self.N]
        self.targets = self.targets[:self.N]
        
        if trs is None:
            self.trs = Compose([Transpose(always_apply=True), ToTensor()])
        else:
            self.trs = trs
            
        
    def __len__(self):
        return self.N
    
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.trs(image=x)['image']
        x = torch.unsqueeze(x, 0)
        
        y = self.targets[idx]
        
        return x, y
    

def get_mnist_data(train=True):
    X, y = None, None
    
    if train:
        X = mnist_dataset_train.data.numpy().reshape(-1, 28*28)
        y = mnist_dataset_train.targets.numpy()
    else:
        X = mnist_dataset_test.data.numpy().reshape(-1, 28*28)
        y = mnist_dataset_test.targets.numpy()
        
    return X, y
        
        

def create_mnist_loader(train=True, trs=None, batch_size=256, size=1.0, num_workers=0):
    dataset = MNIST_Dataset(train=train, trs=trs, size=size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    return loader