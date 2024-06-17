
# plotting utils

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
import torchvision
import imageio
import os


def make_gif_from_images(imgs_path, gif_path, duration=0.5):
    """
    imgs_path: All the images from the folder make gif based on time it was edited
    gif_path: path to save the gif
    duration: duration of each frame
    """
    images = [os.path.join(imgs_path, f) for f in os.listdir(imgs_path) if f.endswith('.png')]
    images.sort(key=os.path.getmtime)

    


    frames = []
    for image in images:
        frames.append(imageio.imread(image))
    imageio.mimsave(gif_path, frames,duration = 0.5)


    return


def inverse_transform(x,mean=[0.1050, 0.1151, 0.1105],std=[0.2823, 0.2891, 0.2787]):
    """
    x: bx3x28x28
    mean: 3
    std: 3
    """
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]

    return x

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def plot_loss(loss_list,filename):
    sns.lineplot(x=range(len(loss_list)), y=loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(filename)


def plot_samples(samples, filepath='sample_images.png', how='density',show=False, save=True, title=None,ax=None):
    if how == 'density':
        ax = plot_density_from_samples(samples, filepath, show, save, title, ax)
    elif how == 'image':
        ax = show_sample_images(samples, filepath, show, save, title, ax)
    else:
        raise ValueError('Invalid value for how')
    return ax

def show_sample_images(samples, filepath='sample_images.png', how='density',show=False, save=True, title=None,ax=None):
    if save:
        torchvision.utils.save_image(samples, filepath, nrow=int(np.sqrt(samples.size(0))))
    else:
        grid = torchvision.utils.make_grid(samples, nrow=int(np.sqrt(samples.size(0))))
        if ax is None:
            fig = plt.figure()
            plt.imshow(grid)
            plt.axis('off')
            plt.title(title)
            plt.set
            if show:
                plt.show()
            else:
                plt.close(fig)

        else:
            ax.imshow(grid)
            ax.axis('off')
            ax.set_title(title)
            return ax

# sample some data and plot the density
def plot_density_from_samples(samples, filepath='gmm-density-samples.png', show=True, save=True,title='Gaussian Mixture Model Density',ax=None):
    #if torch tensor convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Reds", fill=True, thresh=0, bw_adjust=0.5,ax=ax)
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim(-10,10)
        
        ax.set_ylim(-10,10)
        if show:
            plt.show()
        else:
            plt.close(fig)
        if save:
            fig.savefig(filepath)
    else:
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Reds", fill=True, thresh=0, bw_adjust=0.5,ax=ax)
        ax.set_title(title)
        ax.set_ylabel('x2')
        ax.set_xlabel('x1')
        if save:
            plt.savefig(filepath)
        return ax
        

    
def plot_density(samples, rho, filepath='gmm-density.png', show=True, save=True):
    fig = plt.figure()
    sns.scatterplot(x=samples[:, 0], y=samples[:, 1], hue=rho, palette='Reds')
    plt.title('Gaussian Mixture Model Density')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if show:
        plt.show()
    else:        
        plt.close(fig)
    if save:
        fig.savefig(filepath)
    
def plot_score_function(samples, eta, filepath='gmm-score-function.png', show=True, save=True):
    fig = plt.figure()
    plt.quiver(samples[:, 0], samples[:, 1], eta[:, 0], eta[:, 1])
    plt.title('Gaussian Mixture Model Score Function')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    if show:
        plt.show()
    else:
        plt.close(fig)
    if save:
        fig.savefig(filepath)
        
def plot_loss(losses, filepath='gmm-loss.png', show=True, save=True):
    fig = plt.figure()
    plt.plot(losses)
    plt.title('Score Function Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if show:
        plt.show()
    else:
        plt.close(fig)
    if save:
        fig.savefig(filepath)


def compute_mean_std(torch_datatset):
    data = torch.stack([torch_datatset[i][0] for i in range(len(torch_datatset))])
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0)
    return data_mean, data_std