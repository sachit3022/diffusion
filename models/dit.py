from torch import nn
import torch
import torch.nn.functional as F
from models.utils import weights_init, TimestepEmbedder



def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiffMLPBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, mlp_ratio=2.0):
        
        super().__init__()

        hidden_size = hidden_size*2
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.mlp1 = nn.Sequential( 
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.mlp2 = nn.Sequential( 
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.mlp1.apply(weights_init)
        self.mlp2.apply(weights_init)
        self.adaLN_modulation.apply(weights_init)

    def forward(self, x,c):   
        
        shift1_mlp, scale1_mlp, gate1_mlp,shift2_mlp, scale2_mlp, gate2_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate1_mlp * self.mlp1(modulate(self.norm1(x), shift1_mlp, scale1_mlp))
        x = x + gate2_mlp * self.mlp2(modulate(self.norm2(x), shift2_mlp, scale2_mlp))
        return x
    
class DiffMLP(nn.Module):
    def __init__(self, hidden_size,num_layers=2, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiffMLPBlock(hidden_size, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.tspan = (1e-2,5.)
        t_list = torch.linspace(self.tspan[0], self.tspan[1], 500)
        self.register_buffer('t_list', t_list)

    def forward(self, x, t, c):
        t = self.timestep_embedder(self.t_list[t])
        for block in self.blocks:
            x = x + block(x,t)
        return x