import torch 
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl



class VAE(pl.LightningModule):

    """
    In the latent diffusion they use VAE as the encoder to project onto the latent space. Any latent space
    can be used. The authors have used VAE, Diagonal Gaussian VAE."""

    def __init__(self,encoder,decoder,d_in,d_latent,beta=0.1):
        """
        encoder: nn.Module - suppose if you want the latent space to be d, then the encoder should output 2*d to account for mu and sigma. we ignore sigma during eval.
        beta: float - we train using beta VAE loss. beta is the weight of the KL divergence loss as if e use large weight for KL then the model will compramise on the latent space.
        """
        super().__init__()
        self.encoder = encoder      
        self.decoder = decoder
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
    @torch.no_grad()
    def encode(self,x):
        self.eval()
        return self.encoder(x).chunk(2, dim=-1)[0]
    @torch.no_grad()
    def decode(self,z):
        self.eval()
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
    
    

class LabelCLIP(pl.LightningModule):
    """Train latent space of VAE with label encoding"""
    def __init__(self,image_encoder,label_encoder, d_latent, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = image_encoder
        self.label_encoder =label_encoder
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