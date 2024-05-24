"""
The file contains coloured MNIST dataset.
loading MNIST data and picking colour at random and then colouring the MNIST image with that colour.
"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision



class ColouredMNIST(Dataset):
    def __init__(self, root_dir, transform=None, download=True):
        self.mnist = datasets.MNIST(root=root_dir, train=True, transform=ToTensor(), download=download)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        colour = torch.randint(0, 256, (3,)).float() / 255
        coloured_img = self.colour_image(img, colour)
        if self.transform:
            coloured_img = self.transform(coloured_img)
        return coloured_img, label, colour
    def colour_image(self, img, colour):
        #color all the pixcels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img
