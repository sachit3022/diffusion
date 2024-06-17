import torch 
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl



class VAE(pl.LightningModule):

    """
    In the latent diffusion they use VAE as the encoder to project onto the latent space. Any latent space
    can be used. The authors have used VAE, Diagonal Gaussian VAE."""

    def __init__(self,encoder,decoder,d_in,d_latent,max_beta=0.5,checkpoint=None,lr=3e-4,max_epochs=100):
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
        self.betas = self.constant_beta_schedule(max_beta,max_epochs)
        self.lr = lr
        #if checkpoint is not None:
        #    self.load_state_dict(torch.load(checkpoint))
    def beta_schedule(self,max_beta,N):
        """Anneal beta from 0 to 0.01"""
        return torch.linspace(0,max_beta,N)
    def constant_beta_schedule(self,beta,N):
        return torch.ones(N)*beta
  
    def reparametrisation(self, mean, log_var):
        """Reparameterization trick: z = mean + std*eps; eps ~ N(0,1)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    @torch.no_grad()
    def sample(self,n=32):
        self.eval()
        #self.d_latent can be any size (h,w,c) or (c) or (1)
        if isinstance(self.d_latent, int):
            z = torch.randn(n, self.d_latent).to(self.device)
        else:
            z = torch.randn(n, *self.d_latent).to(self.device)
        return self.decoder(z)
    
    @torch.no_grad()
    def encode(self,x):
        self.eval()
        return self.encoder(x).chunk(2, dim=1)[0]
    
    @torch.no_grad()
    def encode_with_log_var(self,x):
        mean, log_var = self.encoder(x).chunk(2, dim=1) 
        z = self.reparametrisation(mean, log_var)
        return z
    
    @torch.no_grad()
    def decode(self,z):
        self.eval()
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encoder(x).chunk(2, dim=1) 
        z = self.reparametrisation(mean, log_var)
        x_recon = self.decoder(z)
        return x_recon, mean, log_var
    
    def kl_loss(self,mean, log_var):
        """KL divergence between N(mean, var) and N(0,1) sum all dim expect 0"""
        mean_component = torch.mean(0.5*torch.sum(mean ** 2,dim=[x for x in range(1,mean.dim())]),dim=0)
        std_component = torch.mean(-0.5 * torch.sum(log_var - log_var.exp(), dim=[x for x in range(1,mean.dim())]),dim=0 )
        self.log_dict({"mean_comp": mean_component, "std_comp": std_component},on_epoch=True)
        return  mean_component + std_component
    
    def reconstruction_loss(self, x, x_recon):
        return F.mse_loss(x_recon, x)
    
    def loss(self, x, x_recon, mean, log_var):
        beta = self.betas[self.current_epoch]
        return  (1-beta)*self.reconstruction_loss(x, x_recon) + beta*self.kl_loss(mean, log_var) 
    
    def configure_optimizers(self):
        """ e and m optimizers first train encoder freeze it and then then train decoder alternatively."""
        e_opt = torch.optim.AdamW(list(self.encoder.parameters()) +list(self.decoder.parameters()) , lr=self.lr)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False,
        }

        return [e_opt],[e_scheduler]
    
    def training_step(self, batch, batch_idx):
        x =  batch['X']
        x_recon, mean, log_var = self(x)
        e_loss = self.loss(x, x_recon, mean, log_var)
        return {"loss": e_loss}

    def validation_step(self, batch, batch_idx):
        x = batch['X']
        x_recon, mean, log_var = self(x)
        kl_loss = self.kl_loss(mean, log_var)
        recon_loss = self.reconstruction_loss(x, x_recon)
        self.log_dict({"kl_loss": kl_loss, "recon_loss": recon_loss},on_epoch=True)
        return {"kl_loss": kl_loss, "recon_loss": recon_loss}
        
    

class LabelCLIP(pl.LightningModule):
    """Train latent space of VAE with label encoding"""
    def __init__(self,latent_model,label_encoder, d_latent,lr=3e-4,checkpoint=None):
        super().__init__()
        self.latent_model = latent_model
        self.label_encoder =label_encoder
        self.d_latent = d_latent
        self.lr = lr
        #freeze the encoder
        for param in self.latent_model.parameters():
            param.requires_grad = False
        self.latent_model.eval()
        self.save_hyperparameters(ignore=['latent_model','label_encoder'])


    def forward(self, x, y):
        z = self.latent_model.encode(x)
        y = self.label_encoder(y)
        return z.view(z.size(0), -1), y.view(y.size(0), -1)
    
    def training_step(self, batch, batch_idx):
        x, y =  batch['X'],batch['label']
        #sample unique labels
        y_un = y.unique()
        x_un = []
        #sample one x for each y_un
        for y_ in y_un:
            x_ = x[y == y_]
            x_un.append(x_[torch.randint(0,len(x_), (1,))])
        
        x = torch.cat(x_un)
        y = y_un
        
        xf, yf = self(x, y)
        xf = xf/torch.norm(xf, dim=-1, keepdim=True)
        yf = yf/torch.norm(yf, dim=-1, keepdim=True)
        logits = yf @ xf.t()
        labels = torch.arange(len(logits)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        
        self.log_dict({"loss": loss},on_epoch=True)
        return {"loss": loss}
    
    def configure_optimizers(self):
        e_opt = torch.optim.AdamW( list(self.label_encoder.parameters()), lr=self.lr)
        e_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(e_opt, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': False
        }
        return [e_opt],[e_scheduler]