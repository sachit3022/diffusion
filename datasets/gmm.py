
import torch
from torch.utils.data import  Dataset
import numpy as np
import math

class GaussianMixtureDataset(Dataset):
    def __init__(self, k=8, dim=2, n=1000):
        self.k = k
        self.dim = 2
        self.n = n
        
        radius = 10
        thetas = torch.arange(k) * 2 * np.pi / k
        self.mus = radius * torch.hstack([torch.cos(thetas)[:, None], torch.sin(thetas)[:, None]])
        self.sigmas = torch.stack([2 * torch.rand(self.dim) for _ in range(k)])
        
        temp = torch.rand(k)
        self.alphas = temp / temp.sum()
        
        self.samples,self.labels = self.sample(n)

    def inv_sigmoid(self, value):
        return torch.log(value/(1-value))
    
    def sample(self, n):
        samples = torch.zeros((n, self.dim), dtype=torch.float32)
        labels = torch.zeros(n, dtype=torch.int64)
        for i in range(n):
            # sample uniform
            r = torch.rand(1).item()
            # select gaussian
            k = 0
            for j, threshold in enumerate(self.alphas.cumsum(dim=0).tolist()):
                if r < threshold:
                    k = j
                    break

            selected_mu = self.mus[k]
            selected_cov = self.sigmas[k] * torch.eye(2)

            # sample from selected gaussian
            lambda_, gamma_ = torch.linalg.eig(selected_cov)
            lambda_ = lambda_.real
            gamma_ = gamma_.real
            

            dimensions = len(lambda_)
            # sampling from normal distribution
            y_s = torch.rand((dimensions * 1, 3))
            x_normal = torch.mean(self.inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
            # transforming into multivariate distribution
            samples[i] = (x_normal * lambda_) @ gamma_ + selected_mu
            labels[i] = k
            
        return samples, labels
    
    def get_gaussian_likelihood(self,x ,mu, sigma):
        sigma = sigma.sqrt()
        return torch.exp(-0.5*(((x-mu)/sigma)**2).sum(dim=-1))/(2*math.pi*torch.prod(sigma))

    def rho0(self,samples):
        likelihood = 0
        for i in range(len(self.mus)):
            likelihood += self.alphas[i] * self.get_gaussian_likelihood(samples,self.mus[i],self.sigmas[i])
        return likelihood
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    

