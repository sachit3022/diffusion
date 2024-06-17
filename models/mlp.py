import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import pairwise
from models.utils import weights_init,TimestepEmbedder
import math


class MLPEncoder(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out//2),
            nn.GELU(),
            nn.BatchNorm1d(d_out//2),
            nn.Linear(d_out//2, d_out),
            nn.GELU(),
            nn.BatchNorm1d(d_out),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out)
        )
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.BatchNorm1d(d_in),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.BatchNorm1d(d_in),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_in//2),
            nn.GELU(),
            nn.BatchNorm1d(d_in//2),
            nn.Dropout(dropout),
            nn.Linear(d_in//2, d_in//2),
            nn.GELU(),
            nn.Linear(d_in//2, d_out),
        )
        self.net.apply(weights_init)
    def forward(self, x):
        return self.net(x)

class LabelEncoder(nn.Module):

    def __init__(self,num_classes,d_latent,dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_classes,d_latent)
        # self.net = nn.Sequential(
        #     nn.Linear(d_latent, d_latent//2),
        #     nn.GELU(),
        #     nn.BatchNorm1d(d_latent//2),
        #     nn.Dropout(dropout),
        #        # self.mlp = nn.Sequential(
        #     nn.Linear(4,16),
        #     nn.GELU(),
        #     nn.BatchNorm1d(16),
        #     nn.Linear(16,16),
        #     nn.GELU(),
        #     nn.BatchNorm1d(16),
        #     nn.Linear(16,2)
        #     )     nn.Linear(d_latent//2, d_latent)
        # )
        # self.net.apply(weights_init)
    def forward(self, x):
        return self.emb(x)
    
class MultiLabelEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb1 = nn.Embedding(11,64)
        self.emb2 = nn.Embedding(11,64)
        # self.mlp = nn.Sequential(
        #     nn.Linear(256,128),
        #     nn.GELU(),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128,128)
        #     )

    def forward(self,y,s):
        y1 = self.emb1(y)
        y2 = self.emb2(s)
        y = torch.cat([y1,y2],dim=1)
        return y
        
class MultiLabelEncoder2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb1 = nn.Embedding(3,2)
        self.emb2 = nn.Embedding(3,2)


    def forward(self,y,s):
        y1 = self.emb1(y)
        y2 = self.emb2(s)
        y = torch.cat([y1,y2],dim=1)
        #self.mlp(y)
        return y


class LatentProjection(nn.Module):
    def __init__(self,d_in,d_latent,dropout=0.1):
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_latent),
            nn.GELU(),
            nn.BatchNorm1d(d_latent),
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        )
        self.net.apply(weights_init)
    def forward(self, x):
        return self.net(x)



class TimeInputMLP(nn.Module):
    def __init__(self, d_latent, num_layers=4,mlp_ratio=2.0):
        super().__init__()
        layers = []

        mul_fac = 4

        hidden_dims = [d_latent*mul_fac] + [int(d_latent * mlp_ratio *mul_fac) for i in range(num_layers-1)]

        for in_dim, out_dim in pairwise([d_latent*mul_fac]+ hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], d_latent))
        self.net = nn.Sequential(*layers)
        self.input_dims = (d_latent,)
        self.time_embedder = TimestepEmbedder(d_latent)

    def forward(self, x, t,y):
        time_embeds = self.time_embedder(t ) # shape: b x dim
        nn_input = torch.cat([x, time_embeds,y], dim=1) # shape: b x (dim *3)
        return self.net(nn_input)
    


class TimeInputMLPImg(nn.Module):
    def __init__(self, d_latent, num_layers=3,mlp_ratio=1.0):
        super().__init__()
        layers = []

        mul_fac = 2
        self.old_d_latent = d_latent
        d_latent = math.prod(d_latent)


        hidden_dims = [d_latent*mul_fac] + [int(d_latent * mlp_ratio *mul_fac) for i in range(num_layers-1)]

        for in_dim, out_dim in pairwise([d_latent*mul_fac]+ hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], d_latent))
        self.net = nn.Sequential(*layers)
        self.input_dims = (d_latent,)
        self.time_embedder = TimestepEmbedder(d_latent)

    def forward(self, x, t,y=None):
        time_embeds = self.time_embedder(t)         # shape: b x dim
        x = x.view(x.size(0),-1)
        nn_input = torch.cat([x, time_embeds], dim=1) # shape: b x (dim *3)
        return self.net(nn_input).view(x.size(0),*self.old_d_latent)
    
