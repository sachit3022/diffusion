import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import pairwise
from models.utils import weights_init,TimestepEmbedder
import math


class MLPEncoder(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden//2),
            nn.GELU(),
            nn.BatchNorm1d(d_hidden//2),
            nn.Linear(d_hidden//2, d_hidden),
            nn.GELU(),
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden)
        )
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.BatchNorm1d(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2),
            nn.GELU(),
            nn.BatchNorm1d(d_hidden//2),
            nn.Dropout(dropout),
            nn.Linear(d_hidden//2, d_hidden//2),
            nn.GELU(),
            nn.Linear(d_hidden//2, d_out),
        )
        self.net.apply(weights_init)
    def forward(self, x):
        return self.net(x)

class LabelEncoder(nn.Module):

    def __init__(self,num_classes,d_latent,dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_classes,d_latent)
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.BatchNorm1d(d_latent),
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        )
        self.net.apply(weights_init)
    def forward(self, x):
        return self.net(self.emb(x))
    
class TimeInputMLP(nn.Module):
    def __init__(self, d_latent, num_layers=4,mlp_ratio=2.0):
        super().__init__()
        layers = []

        mul_fac = 3


        hidden_dims = [d_latent*mul_fac] + [int(d_latent * mlp_ratio *mul_fac) for i in range(num_layers-1)]

        for in_dim, out_dim in pairwise([d_latent*mul_fac]+ hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], d_latent))
        self.net = nn.Sequential(*layers)
        self.input_dims = (d_latent,)
        self.time_embedder = TimestepEmbedder(d_latent)

    def forward(self, x, t,y):
        time_embeds = self.time_embedder(t )         # shape: b x dim
        nn_input = torch.cat([x, time_embeds,y], dim=1) # shape: b x (dim *3)
        return self.net(nn_input)
    
