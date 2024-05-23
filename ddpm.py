import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging,LearningRateMonitor
from utils import plot_density_from_samples,set_seed
import torchcontrib
from EMA import EMA
from itertools import pairwise



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
    


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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
        
        # x = self.norm1(x)
        # x = self.mlp1(x)
        # x = self.norm2(x)
        # x = self.mlp2(x)
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
 
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    This code is copied from DiT
    """
    def __init__(self, hidden_size, frequency_embedding_size=500):
        super().__init__()
       
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=500):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
      
        return t_emb


#Like clip we project onto the common space.
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

class VAE(pl.LightningModule):
    """
    In the latent diffusion they use VAE as the encoder to project onto the latent space. Any latent space
    can be used. The authors have used VAE, Diagonal Gaussian VAE."""

    def __init__(self,d_in,d_latent,dropout=0.1,beta=0.1):
        super().__init__()
        self.encoder = MLPEncoder(d_in,2*d_latent,dropout=dropout)        
        self.decoder = MLPDecoder(d_latent,d_in,dropout=dropout)

        self.d_latent = d_latent
        self.d_in = d_in
        self.save_hyperparameters(ignore=['encoder','decoder'])
        self.beta = beta


    def reparametrisation(self, mean, log_var):
        """Reparameterization trick: z = mean + std*eps; eps ~ N(0,1)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    @torch.no_grad()
    def sample(self,n=32):
        self.eval()
        z = torch.randn(n,self.d_latent,device=self.device)
        return self.decoder(z)
    
    def forward(self, x):
        mean, log_var = self.encoder(x).chunk(2, dim=-1) 
        z = self.reparametrisation(mean, log_var)
        x_recon = self.decoder(z)
        return x_recon, mean, log_var
    
    def kl_loss(self,mean, log_var):
        """KL divergence between N(mean, var) and N(0,1)"""
        return  torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    def reconstruction_loss(self, x, x_recon):
        return F.mse_loss(x_recon, x)
    
    def loss(self, x, x_recon, mean, log_var):
        return  (1-self.beta)*self.reconstruction_loss(x, x_recon) + self.beta*self.kl_loss(mean, log_var) 
    
    def configure_optimizers(self):
        """ e and m optimizers first train encoder freeze it and then then train decoder alternatively."""
        e_opt = torch.optim.AdamW(list(self.encoder.parameters()) +list(self.decoder.parameters()) , lr=1e-2)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False,
        }

        return [e_opt],[e_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mean, log_var = self(x)
        e_loss = self.loss(x, x_recon, mean, log_var)
        return {"loss": e_loss}

    def validation_step(self, batch, batch_idx):
        x,_ = batch
        x_recon, mean, log_var = self(x)
        kl_loss = self.kl_loss(mean, log_var)
        recon_loss = self.reconstruction_loss(x, x_recon)
        self.log_dict({"kl_loss": kl_loss, "recon_loss": recon_loss})
        return {"kl_loss": kl_loss, "recon_loss": recon_loss}
       

class LabelEncoderModel(nn.Module):

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
    


class LabelCLIP(pl.LightningModule):
    """Train latent space of VAE with label encoding"""
    def __init__(self,encoder, d_latent, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.label_encoder = LabelEncoderModel(num_classes+1,d_latent)
        self.d_latent = d_latent
        self.num_classes = num_classes
        #freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.save_hyperparameters(ignore=['encoder','label_encoder'])

    def forward(self, x, y):
        z = self.encoder(x).chunk(2, dim=-1)[0]
        y = self.label_encoder(y)
        return z, y
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        xf, yf = self(x, y)
        xf = xf/torch.norm(xf, dim=-1, keepdim=True)
        yf = yf/torch.norm(yf, dim=-1, keepdim=True)
        logits = yf @ xf.t()
        labels = torch.arange(len(logits)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        self.log_dict({"loss": loss})
        return {"loss": loss}
    
    def configure_optimizers(self):
        e_opt = torch.optim.AdamW( list(self.label_encoder.parameters()), lr=3e-3)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False
        }
        return [e_opt],[e_scheduler]
    

if __name__ == "__main__":
    #test dataset and build dataloader
    set_seed(42)
    vae_logger = TensorBoardLogger('lightning_logs', name='VAE')
    clip_logger = TensorBoardLogger('lightning_logs', name='clip')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    plots_dir= 'plots'


    train_dataset = GaussianMixtureDataset(n=1000)
    val_dataset = GaussianMixtureDataset(n=1000)
    val_dataset.alphas = train_dataset.alphas
    val_dataset.samples,val_dataset.labels = val_dataset.sample(1000)

    fig,axes = plt.subplots(2,2,figsize=(10,5))


    plot_density_from_samples(val_dataset.samples.detach().numpy(), filepath=f"{plots_dir}/val_input_samples.png", show=False, save=False,title="Val input samples density", ax=axes[0,0])
    plot_density_from_samples(train_dataset.samples.detach().numpy(), filepath=f"{plots_dir}/train_input_samples.png", show=False, save=False,title="Train input samples density", ax=axes[0,1])


    #normalise the data
    data_mean, data_std = train_dataset.samples.mean(dim=0), train_dataset.samples.std(dim=0)
    train_dataset.samples = (train_dataset.samples - data_mean) / data_std
    val_dataset.samples = (val_dataset.samples - data_mean) / data_std


    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4,persistent_workers=True)
    
    #train VAE
    vae = VAE(2,32)

    trainer = pl.Trainer(max_epochs=200, accelerator='gpu',devices=1, log_every_n_steps=10, logger=vae_logger,gradient_clip_val=0.5,callbacks=[lr_monitor,EMA(decay=0.9999,validate_original_weights=True,every_n_steps=10)])# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
    trainer.fit(vae, train_dataloader, val_dataloader)    

    #test VAE
    vae.eval()

    all_x = []
    for batch in val_dataloader:
        x,_ = batch
        x_recon, mean, log_var = vae(x)
        x_recon = x_recon*data_std + data_mean
        all_x.append(x_recon)
    all_x = torch.cat(all_x)
    
    
    plot_density_from_samples(all_x.detach().numpy(), filepath=f"{plots_dir}/vae_reconstructions.png", show=False, save=False,title="vae_reconstructions", ax=axes[1,0])
    
    x_hat = vae.sample(n=1000)*data_std + data_mean
    plot_density_from_samples(x_hat.detach().numpy(), filepath=f"{plots_dir}/vae_samples.png", show=False, save=False, title="vae_samples",ax=axes[1,1])

    #train label encoder
    label_clip = LabelCLIP(vae.encoder, 32, 8)
    trainer = pl.Trainer(max_epochs=200, accelerator='gpu',devices=1, log_every_n_steps=10, logger=clip_logger,gradient_clip_val=0.5,callbacks=[lr_monitor,EMA(decay=0.9999,validate_original_weights=True,every_n_steps=10)])# StochasticWeightAveraging(swa_lrs = 1e-2,swa_epoch_start=0.2),
    trainer.fit(label_clip, train_dataloader, val_dataloader)    


    #save vae and label clip



    #sample form the label encoder
    label_clip.eval()
    #random labels from 0 to 7 grid of 3x3
    


    z = torch.arange(8).to(vae.device)
    z_latent = label_clip.label_encoder(z)
    x_hat = vae.decoder(z_latent)*data_std + data_mean
    #plot the centers of x_hat
    for i in range(8):
        axes[1,0].scatter(x_hat[i,0].item(),x_hat[i,1].item(),c='r',s=100,marker='x')
    
    plt.savefig(f"{plots_dir}/vae_samples.png")
        
    
    



