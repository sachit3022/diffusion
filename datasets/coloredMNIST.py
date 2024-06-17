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
from torchvision import transforms


COLOURS =  torch.tensor([(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1),(1,0.98,0.78),(0.5,0.5,0.5),(0,0.5,0.5)])

class Threshold:
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, y):
        if random.uniform(0, 1) <= self.threshold:
            return y
        else:
            z = random.randint(0, 8)
            if z >= y:
                z += 1
            return z

class ColoredMNIST(Dataset):
    def __init__(self, root_dir,train=True, download=False,threshold=0.1):
        self.mnist = datasets.MNIST(root=root_dir, train=train, transform=ToTensor(), download=download)
        self.mean = torch.tensor([0.1050, 0.1151, 0.1105])
        self.std = torch.tensor([0.2823, 0.2891, 0.2787])
        self.transform =transforms.Compose([
            transforms.Normalize(mean=self.mean, std=self.std)])
        
        self.threshold = Threshold(threshold)
        self.colors = COLOURS
        self.randlabels = np.vectorize(self.threshold)(self.mnist.targets.numpy())
        

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]

        colour_idx = self.randlabels[idx]
        colour = self.colors[colour_idx]
        coloured_img = self.colour_image(img, colour)

        if self.transform:
            coloured_img = self.transform(coloured_img)

        return {'X': coloured_img, 'label': label, 'sensitive': colour_idx}
    
    def colour_image(self, img, colour):
        #color all the pixcels that are not black
        img = img.squeeze(0)
        color_indices = img > 0
        img = img.repeat(3, 1, 1) # convert to 3 channel image
        img[:, color_indices] = colour[:, None]
        return img
    
    def sample_labels(self):
        #torch one hot encoding of 10 labels
        labels = torch.linspace(0, 9, 10).long()
        return labels
    
    def sample(self,N):
        #sample N images from the dataset
        indices = np.random.choice(len(self), N)
        images = torch.stack([self[i]['image'] for i in indices])
        labels = torch.tensor([self[i]['label'] for i in indices])
        colors = torch.tensor([self[i]['sensitive'] for i in indices])
        return {'X': images, 'label': labels, 'sensitive': colors}
    
    def inverse_transform(self,img):
        #inverse of self.mean and self.std
        img = img.permute(1,2,0)
        return img * self.std + self.mean



class CounterfactualColoredMNIST(Dataset):
    def __init__(self, root_dir, train=True, download=False, threshold=0.1):
        super().__init__()
        #load all_x.pt, all_y.pt, all_s.pt
        self.all_x = torch.load(f'{root_dir}/all_x.pt')
        self.all_y = torch.load(f'{root_dir}/all_y.pt')
        self.all_s = torch.load(f'{root_dir}/all_s.pt')
        
    def __getitem__(self, idx):
        x = self.all_x[idx]
        y = self.all_y[idx]
        s = self.all_s[idx]
        return {'X': x, 'label': y, 'sensitive': s}
    def __len__(self):
        return len(self.all_x)
        
















def standarisation():
    data = ColoredMNIST(root_dir='/research/hal-gaudisac/Diffusion/image_gen/data', download=False)
    #compute mean and std of the data
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for i in range(len(data)):
        img= data[i]['image']
        mean += img.mean([1,2])
        std += img.var([1,2])
    mean /= len(data)
    std /= len(data)
    print(mean, torch.sqrt(std))


def plot_dataset_digits(datasets,filename='colored_mnist.png'):
  
 
    columns = 10
    rows = 5

    fig, ax = plt.subplots(rows, columns, figsize=(20, 10))
    k=0
    dataset = datasets[1]
    for i in range(5):
        for j in range(10):     
            while True:
                img, label, color = dataset[k]['image'], dataset[k]['label'], dataset[k]['color']
                #print(img[img > 0].unique(),COLOURS[color])
                k+=1
                if label==j:
                    img = dataset.inverse_transform(img)
                    ax[i,j].imshow(img)
                    ax[i,j].set_title(f"Label: {label}, Color: {color}")
                    ax[i,j].axis('off')
                    break
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == "__main__":
    train_dataset = ColoredMNIST(root_dir='/research/hal-gaudisac/Diffusion/image_gen/data', download=False, threshold=1, train=True)
    test_dataset = ColoredMNIST(root_dir='/research/hal-gaudisac/Diffusion/image_gen/data', download=False, threshold=0.1, train=False)
    plot_dataset_digits([train_dataset,test_dataset], filename='colored_mnist.png')
    #standarisation()

