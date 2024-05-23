
# plotting utils

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch


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

# sample some data and plot the density
def plot_density_from_samples(samples, filepath='gmm-density-samples.png', show=True, save=True,title=None,ax=None):
    if title is  None:
        title = 'Gaussian Mixture Model Density'
        
    if ax is None:
        fig = plt.figure()
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Reds", fill=True, thresh=0, bw_adjust=0.5)
        plt.title(title)
        plt.ylabel('x2')
        plt.xlabel('x1')
        #plt.xlim(-15, 15)
        #plt.ylim(-15, 15)
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
        #ax.set_xlim(-15, 15)
        #ax.set_ylim(-15, 15)
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