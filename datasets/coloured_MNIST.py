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


class ColouredMNIST(Dataset):
    def __init__(self, root_dir, transform=None, download=True):
        self.mnist = datasets.MNIST(root=root_dir, train=True, transform=ToTensor(), download=download)
        self.transform = transform
        self.colours = torch.tensor([
            [1., 0., 0.],  # red
            [0., 1., 0.],  # green
            [0., 0., 1.],  # blue
            [1., 1., 0.],  # yellow
            [1., 0., 1.],  # magenta
            [0., 1., 1.],  # cyan
            [1., 1., 1.],  # white
            [0., 0., 0.],  # black
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        colour = self.colours[random.randint(0, 7)]
        coloured_img = torch.zeros(3, 28, 28)
        for i in range(3):
            coloured_img[i] = colour[i] * img.squeeze()
        if self.transform:
            coloured_img = self.transform(coloured_img)
        return coloured_img, label