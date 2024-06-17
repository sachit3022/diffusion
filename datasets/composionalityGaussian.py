
import torch
from torch.utils.data import  Dataset
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


class ComposionalityGaussian(Dataset):
    def __init__(self, n=10000,train=True):
        #means
        mu = 5.0
        std = mu/5
        self.mus = np.array([(-mu, -mu), (mu, mu), (-mu, mu), (mu, -mu)])
        self.sigma = np.array([std,std])
        self.s = [0,1,0,1]
        self.y = [0,1,1,0]
        self.k = [0.334,0.334,0.334,0]
        self.n = n
        self.samples, self.labels,self.s = self.sample(n)
        #make samples to torch float tensor labels to long tensor
        self.samples = torch.tensor(self.samples).float()
        self.labels = torch.tensor(self.labels).long()
        self.s = torch.tensor(self.s).long()

        #standardize the samples
        self.mean = torch.tensor(self.samples.mean(dim=0))
        self.std = torch.tensor(self.samples.std(dim=0))
        self.samples = (self.samples - self.mean) / self.std

    def inverse_transform(self, x):
        #what if x and mean are on different devices
        x= (x.to(self.mean.device)*self.std + self.mean).to(x.device)
        return x

    def inv_sigmoid(self, value):
        return torch.log(value/(1-value))
    
    def sample(self, n):
        selected_cov = np.diag(np.sqrt(self.sigma))
        samples, labels,sensitive = [], [],[]
        for selected_mu,threshold,_y,_s in zip(self.mus,self.k,self.y,self.s):
            if threshold == 0:
                continue
            samples.append(np.random.multivariate_normal(selected_mu,selected_cov,int(n*threshold)))
            labels.append(np.ones(int(n*threshold))*_y)
            sensitive.append(np.ones(int(n*threshold))*_s)

        
        samples = np.vstack(samples)
        labels = np.hstack(labels)
        sensitive = np.hstack(sensitive)

        #shuffle the indices
        indices = np.arange(n)
        np.random.shuffle(indices)
        samples = samples[indices]
        labels = labels[indices]
        sensitive = sensitive[indices]

        #labels as long
        labels = labels.astype(np.int64)
        sensitive = sensitive.astype(np.int64)

        return samples, labels,sensitive
    
    def inv_transform(self, x):
        return (x.to(self.mean.device) * self.std + self.mean).to(x.device)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {"X": self.samples[idx], "label": self.labels[idx], "sensitive": self.s[idx]}
    
    def get_samples(self,n):
        return {"X": self.samples[:n], "label": self.labels[:n], "sensitive": self.s[:n]}
    


def plot_density_from_samples(samples, filepath='gmm-density-samples.png', show=True, save=True,title='Gaussian Mixture Model Density',ax=None):
    #if torch tensor convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if ax is None:
        fig = plt.figure()
        
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Reds", fill=True, thresh=0, bw_adjust=0.5)
        plt.title(title)
        plt.ylabel('y')
        plt.xlabel('s')
        plt.scatter([5],[5],color='blue',label='(1,1)')
        plt.scatter([-5],[-5],color='yellow',label='(0,0)')
        plt.scatter([5],[-5],color='green',label='(0,1)')
        plt.scatter([-5],[5],color='cyan',label='(1,0)')
        plt.annotate('(1,1)',(5,5),textcoords="offset points", xytext=(0,10), ha='center',color='blue')
        plt.annotate('(0,0)',(-5,-5),textcoords="offset points", xytext=(0,10), ha='center',color='yellow')
        plt.annotate('(0,1)',(-5,5),textcoords="offset points", xytext=(0,10), ha='center',color='cyan')
        plt.annotate('(1,0)',(5,-5),textcoords="offset points", xytext=(0,10), ha='center',color='green')




        if show:
            plt.show()
        else:
            plt.close(fig)
        if save:
            fig.savefig(filepath)
    else:
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], cmap="Reds", fill=True, thresh=0, bw_adjust=0.5,ax=ax)
        ax.set_title(title)
        ax.set_ylabel('s')
        ax.set_xlabel('y')
        if save:
            plt.savefig(filepath)
        return ax
           
if __name__ == "__main__":
    N=1000
    dataset = ComposionalityGaussian(n=N)
    dataset.samples = dataset.inverse_transform(dataset.samples)
    plot_density_from_samples(dataset.samples, title='Composionality Gaussian with (s,y) attribute',filepath='composionality_gaussian.png',show=False, save=True)
    #scatter plot of s 
    fig = plt.figure()
    plt.scatter(dataset.samples[:,0],dataset.samples[:,1],c=dataset.s)
    plt.title('Sensitive Attribute')
    plt.savefig('sensitive_attribute.png')
    #scatter plot of y
    fig = plt.figure()
    plt.scatter(dataset.samples[:,0],dataset.samples[:,1],c=dataset.labels)
    plt.title('Label')
    plt.savefig('label.png')
    
